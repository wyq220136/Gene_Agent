"""现场抓碗逐段执行：预测轨迹不能当实测终态，未知夹持不能算成功。"""

from types import SimpleNamespace

import numpy as np
import pytest

from gene_agent.realworld.microwave import pick as bowl_skill
from gene_agent.tasks.checks import CheckResult


def _scene(name, frame, *, q=(0., 0.), width=.04):
    return {
        "snapshot": {"snapshot_sha256": name, "active_group": "right"},
        "provenance": {
            "source_frame_number": frame, "source_timestamp_ms": float(frame * 1000),
            "joint_positions_by_group_rad": {"right": list(q)},
            "gripper_widths_m": {"right": width},
            "aligned_object_poses_world": {"Bowl/Bowl_0": np.eye(4).tolist()},
        },
        "robot_metadata": {"joint_groups": {"right": ["j1", "j2"]}},
    }


def _selection():
    return {
        "grasp_prefix_planned": True, "selected_candidate_id": "bowl_ring_02_00",
        "plan": {"stages": [
            {"name": "open", "status": "planned", "details": {"width_m": .041}},
            {"name": "close", "status": "planned", "details": {"width_m": .007}},
        ]},
    }


def _options(tmp_path, observe, make_runtime, *, execute, check_grasp=None):
    return dict(
        observe_collision_scene=observe, make_runtime=make_runtime,
        planner_pool=object(), output_dir=tmp_path / "grasp",
        execute=execute, check_grasp=check_grasp,
        max_bowl_translation_m=.005, max_bowl_rotation_rad=.02,
        max_start_error_rad=.02, max_gripper_width_error_m=.003,
    )


def _install_selection(monkeypatch):
    monkeypatch.setattr(bowl_skill, "plan_observed_bowl_grasp_stages",
                        lambda *_args, **_kwargs: _selection())


def test_runtime_exception_does_not_report_precommand_state_as_terminal(tmp_path, monkeypatch):
    _install_selection(monkeypatch)
    class Runtime:
        def run_gripper_stage(self, *args, **kwargs):
            raise RuntimeError('SDK command interrupted after dispatch')
    report = bowl_skill.run_observed_bowl_grasp(**_options(
        tmp_path, lambda: _scene('initial', 1), lambda _hash: Runtime(), execute=True))
    assert report['status'] == 'failed'
    assert report['shared_flow']['end_state']['source'] == 'unavailable'
    assert report['shared_flow']['end_state']['state_id'] != 'initial:1'


def test_gripper_width_is_retained_when_postcommand_camera_fails(tmp_path, monkeypatch):
    _install_selection(monkeypatch)
    scenes = [_scene('initial', 1)]
    def observe():
        if scenes: return scenes.pop()
        raise RuntimeError('camera unavailable')
    class Runtime:
        def run_gripper_stage(self, *args, **kwargs):
            return {'success': True, 'status': 'executed', 'measured_end_width_m': .041}
    report = bowl_skill.run_observed_bowl_grasp(**_options(
        tmp_path, observe, lambda _hash: Runtime(), execute=True))
    end = report['shared_flow']['end_state']
    assert report['status'] == 'failed'
    assert end['source'] == 'realworld_sdk_partial'
    assert end['values']['gripper_width_m'] == pytest.approx(.041)
    assert end['values']['joint_state_unknown'] is True


def test_preview_checks_only_open_and_never_uses_predicted_next_state(tmp_path, monkeypatch):
    _install_selection(monkeypatch)
    monkeypatch.setattr(bowl_skill, "plan_observed_bowl_motion_stage",
                        lambda *_args, **_kwargs: (_ for _ in ()).throw(
                            AssertionError("未执行 open 时不得规划 pregrasp")))
    scenes = [_scene("initial", 1)]

    class Runtime:
        def run_gripper_stage(self, stage, **kwargs):
            assert stage == "open" and kwargs["execute"] is False
            assert kwargs["planned_scene_snapshot_sha256"] == "initial"
            return {"source_format": "realworld_gripper_stage", "status": "validated",
                    "success": True}

    report = bowl_skill.run_observed_bowl_grasp(
        **_options(tmp_path, lambda: scenes.pop(0), lambda _hash: Runtime(), execute=False)
    )
    assert report["status"] == "validated_prefix"
    assert report["grasp_success"] is False
    assert report["first_failure"]["code"] == "not_executed"
    assert [row["stage"] for row in report["stages"]] == ["open"]
    assert (tmp_path / "grasp/grasp_result.json").is_file()


@pytest.mark.parametrize("grasp_evidence_frame, expected_success", [
    (None, True),
    (4, False),
])
def test_executed_grasp_replans_each_motion_from_fresh_measured_scene(
    tmp_path, monkeypatch, grasp_evidence_frame, expected_success,
):
    _install_selection(monkeypatch)
    scenes = [
        _scene("initial", 1), _scene("after_open", 2, width=.041),
        _scene("after_pregrasp", 3, q=(.1, .2), width=.041),
        _scene("after_approach", 4, q=(.3, .4), width=.041),
        _scene("after_close", 5, q=(.3, .4), width=.007),
    ]
    seen = []

    def plan(scene, selection, *, stage_name, **kwargs):
        seen.append(("plan", stage_name, scene["snapshot"]["snapshot_sha256"]))
        assert selection["selected_candidate_id"] == "bowl_ring_02_00"
        return {"success": True,
                "stage_collision_snapshot_sha256": f"collision_{stage_name}",
                "stage": SimpleNamespace(name=stage_name, to_dict=lambda: {"name": stage_name})}

    monkeypatch.setattr(bowl_skill, "plan_observed_bowl_motion_stage", plan)

    class Runtime:
        def __init__(self, scene_hash):
            seen.append(("runtime", scene_hash))

        def run_gripper_stage(self, stage, **kwargs):
            assert kwargs["execute"] is True
            if stage == "close":
                assert kwargs["command_mode"] == "grasp"
                assert kwargs["grasp_force"] == .5
            seen.append(("gripper", stage, kwargs["planned_scene_snapshot_sha256"]))
            return {"source_format": "realworld_gripper_stage", "status": "executed",
                    "success": True, "measured_end_width_m": .041 if stage == "open" else .007,
                    "measured_joint_start": kwargs["planned_start"].to_dict()}

        def run_motion_stage(self, stage, *, execute):
            assert execute is True
            seen.append(("motion", stage.name))
            q = (.1, .2) if stage.name == "pregrasp" else (.3, .4)
            return {"source_format": "realworld_motion_stage", "status": "executed",
                    "success": True, "measured_end": {"joint_names": ["j1", "j2"],
                                                      "positions_rad": list(q)}}

    evidence = {"answer": "y"}
    if grasp_evidence_frame is not None:
        evidence["source_frame_number"] = grasp_evidence_frame
    report = bowl_skill.run_observed_bowl_grasp(
        **_options(tmp_path, lambda: scenes.pop(0), Runtime, execute=True,
                   check_grasp=lambda _scene: CheckResult(
                       "pass", "manual_terminal", evidence, "已夹住"))
    )
    assert report["status"] == ("executed" if expected_success else "failed")
    assert report["grasp_success"] is expected_success
    if expected_success:
        assert report["physical_grasp_check"]["source"] == "manual_terminal"
        assert report["physical_grasp_check"]["evidence"]["source_frame_number"] == 5
        assert report['shared_flow']['stages'] == ['open', 'pregrasp', 'approach', 'close', 'grasp_check']
    else:
        assert "现场检查证据与当前观测帧不一致" in report["first_failure"]["message"]
    assert [row["stage"] for row in report["stages"]] == [
        "open", "pregrasp", "approach", "close",
    ]
    assert ("plan", "pregrasp", "after_open") in seen
    assert ("plan", "approach", "after_pregrasp") in seen
    assert ("runtime", "collision_approach") in seen
    assert not scenes


def test_joint_handoff_mismatch_stops_before_approach(tmp_path, monkeypatch):
    _install_selection(monkeypatch)
    scenes = [
        _scene("initial", 1), _scene("after_open", 2, width=.041),
        _scene("wrong_after_pregrasp", 3, q=(.8, .8), width=.041),
    ]
    planned = []

    def plan(_scene, _selection, *, stage_name, **_kwargs):
        planned.append(stage_name)
        return {"success": True, "stage_collision_snapshot_sha256": "strict",
                "stage": SimpleNamespace(name=stage_name, to_dict=lambda: {"name": stage_name})}

    monkeypatch.setattr(bowl_skill, "plan_observed_bowl_motion_stage", plan)

    class Runtime:
        def run_gripper_stage(self, stage, **kwargs):
            return {"source_format": "realworld_gripper_stage", "status": "executed",
                    "success": True, "measured_end_width_m": .041,
                    "measured_joint_start": kwargs["planned_start"].to_dict()}

        def run_motion_stage(self, stage, *, execute):
            return {"source_format": "realworld_motion_stage", "status": "executed",
                    "success": True, "measured_end": {"joint_names": ["j1", "j2"],
                                                      "positions_rad": [.1, .2]}}

    report = bowl_skill.run_observed_bowl_grasp(
        **_options(tmp_path, lambda: scenes.pop(0), lambda _hash: Runtime(), execute=True)
    )
    assert report["status"] == "failed"
    assert report["first_failure"]["code"] == "observed_state_discontinuity"
    assert report['shared_flow']['end_state']['state_id'] == 'wrong_after_pregrasp:3'
    assert planned == ["pregrasp"]


def test_unknown_grasp_check_cannot_claim_success(tmp_path, monkeypatch):
    _install_selection(monkeypatch)
    scenes = [
        _scene("initial", 1), _scene("after_open", 2, width=.041),
        _scene("after_pregrasp", 3, q=(.1, .2), width=.041),
        _scene("after_approach", 4, q=(.3, .4), width=.041),
        _scene("after_close", 5, q=(.3, .4), width=.007),
    ]
    monkeypatch.setattr(bowl_skill, "plan_observed_bowl_motion_stage",
                        lambda _scene, _selection, *, stage_name, **_kwargs: {
                            "success": True, "stage_collision_snapshot_sha256": stage_name,
                            "stage": SimpleNamespace(name=stage_name,
                                                     to_dict=lambda: {"name": stage_name}),
                        })

    class Runtime:
        def run_gripper_stage(self, stage, **kwargs):
            return {"source_format": "realworld_gripper_stage", "status": "executed",
                    "success": True, "measured_end_width_m": .041 if stage == "open" else .007,
                    "measured_joint_start": kwargs["planned_start"].to_dict()}

        def run_motion_stage(self, stage, *, execute):
            q = (.1, .2) if stage.name == "pregrasp" else (.3, .4)
            return {"source_format": "realworld_motion_stage", "status": "executed",
                    "success": True, "measured_end": {"joint_names": ["j1", "j2"],
                                                      "positions_rad": list(q)}}

    report = bowl_skill.run_observed_bowl_grasp(
        **_options(tmp_path, lambda: scenes.pop(0), lambda _hash: Runtime(), execute=True)
    )
    assert report["status"] == "unknown"
    assert report["grasp_success"] is False
    assert report["physical_grasp_check"]["status"] == "unknown"
    assert report["first_failure"]["code"] == "grasp_not_confirmed"


def test_contact_postcheck_failure_stops_before_approach_execution(tmp_path, monkeypatch):
    _install_selection(monkeypatch)
    scenes = [
        _scene("initial", 1), _scene("after_open", 2, width=.041),
        _scene("after_pregrasp", 3, q=(.1, .2), width=.041),
    ]
    executed = []

    def plan(_scene, _selection, *, stage_name, **_kwargs):
        if stage_name == "approach":
            return {"success": False, "failure_code": "collision_postcheck_rejected",
                    "contact_postcheck": {"success": False,
                                          "forbidden_contacts": [{"robot_geom_name": "wrist"}]},
                    "stage": SimpleNamespace(name=stage_name,
                                             to_dict=lambda: {"name": stage_name,
                                                              "status": "failed"})}
        return {"success": True, "stage_collision_snapshot_sha256": "strict",
                "stage": SimpleNamespace(name=stage_name,
                                         to_dict=lambda: {"name": stage_name})}

    monkeypatch.setattr(bowl_skill, "plan_observed_bowl_motion_stage", plan)

    class Runtime:
        def run_gripper_stage(self, stage, **kwargs):
            executed.append(stage)
            return {"source_format": "realworld_gripper_stage", "status": "executed",
                    "success": True, "measured_end_width_m": .041,
                    "measured_joint_start": kwargs["planned_start"].to_dict()}

        def run_motion_stage(self, stage, *, execute):
            executed.append(stage.name)
            return {"source_format": "realworld_motion_stage", "status": "executed",
                    "success": True, "measured_end": {"joint_names": ["j1", "j2"],
                                                      "positions_rad": [.1, .2]}}

    report = bowl_skill.run_observed_bowl_grasp(
        **_options(tmp_path, lambda: scenes.pop(0), lambda _hash: Runtime(), execute=True)
    )
    assert report["status"] == "failed"
    assert report["first_failure"]["code"] == "collision_postcheck_rejected"
    assert report["stages"][-1]["stage"] == "approach"
    assert executed == ["open", "pregrasp"]
