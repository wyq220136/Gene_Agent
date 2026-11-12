"""旋钮整段现场执行：实测交接、物理转角和人工检查不能被视觉提交替代。"""

from types import SimpleNamespace

import numpy as np
import pytest

from gene_agent.realworld.microwave import knob as module
from gene_agent.knowledge.operation import AxisSpec
from gene_agent.skills.library.knob.planning import KnobSkillSettings, RotateKnobSkill
from gene_agent.tasks.checks import CheckResult


def _scene(frame, q=(0., 0.), width=.08, angle=0.):
    return {
        "snapshot": {"snapshot_sha256": f"frame-{frame}", "active_group": "right"},
        "provenance": {
            "source_frame_number": frame, "source_timestamp_ms": frame * 1000.,
            "joint_positions_by_group_rad": {"right": list(q)},
            "gripper_widths_m": {"right": width},
            "knob_angle_rad": angle,
            "aligned_object_poses_world": {"Microwave/Microwave_0": np.eye(4).tolist()},
        },
        "robot_metadata": {"joint_groups": {"right": ["j1", "j2"]}},
    }


def _stages(tip_deg=5.):
    pose = np.eye(4)
    pose[0, 3] = .03
    settings = KnobSkillSettings(.03, .1, .1, .08, .04, .08,
                                 tip_rotations_x_rad=(np.deg2rad(tip_deg),),
                                 grasp_settle_time_s=.01, release_settle_time_s=.01)
    axis = AxisSpec(np.zeros(3), np.array([0., 0., 1.]), "world", "test_axis")
    stages = RotateKnobSkill().targets(pose, axis, start_angle_rad=0., settings=settings)
    source = {"settings": settings, "target_angle_rad": .1,
              "source_collision_snapshot_sha256": "frame-2",
              "microwave_pose_world": np.eye(4),
              "task_config_sha256": "task-config", "grasp_profile_sha256": "grasp-profile"}
    return source, stages


def _setup(monkeypatch, calls, *, last_joint=False):
    source, stages = _stages(0. if last_joint else 5.)
    monkeypatch.setattr(module, "build_observed_knob_targets", lambda scene, **kw: source)

    def select(scene, *, planner_pool, output_dir, task_path, seed):
        assert scene["snapshot"]["snapshot_sha256"] == "frame-2"
        calls.append("select")
        return {"success": True, "source": source, "targets": stages,
                "selected_candidate_id": "knob_choice"}

    def plan(scene, selection, *, stage_name, **kwargs):
        calls.append("plan_" + stage_name)
        stage = SimpleNamespace(name=stage_name, to_dict=lambda: {"name": stage_name})
        return {"success": True, "stage": stage,
                "stage_collision_snapshot_sha256": "collision-" + stage_name,
                "collision_scope": "self_only" if stage_name.startswith("rotate_")
                                   else "full_scene",
                "ignored_world_obstacles": ["microwave"]}

    monkeypatch.setattr(module, "select_observed_knob_grasp", select)
    monkeypatch.setattr(module, "plan_observed_knob_motion_stage", plan)
    monkeypatch.setattr(module.time, "sleep", lambda seconds: calls.append(("wait", seconds)))
    return stages


def _settings(tmp_path, scenes, *, execute, calls):
    current = [None]

    def observe():
        current[0] = scenes.pop(0)
        return current[0]

    targets = {"pregrasp": (.1, .1), "approach": (.2, .2),
               "rotate_06": (.3, .3), "retreat": (.4, .4)}

    class Runtime:
        def run_motion_stage(self, stage, *, execute):
            calls.append("execute_" + stage.name)
            return {"source_format": "realworld_motion_stage",
                    "status": "executed" if execute else "validated",
                    "success": True, "measured_end": None if not execute else {
                        "joint_names": ["j1", "j2"],
                        "positions_rad": targets[stage.name],
                    }}

        def run_last_joint_stage(self, stage, *, execute, joint_delta_rad, **kwargs):
            calls.append("execute_" + stage)
            assert abs(joint_delta_rad - .1) < 1e-9
            return {"source_format": "realworld_motion_stage",
                    "status": "executed" if execute else "validated",
                    "success": True, "collision_scope": "none_legacy_last_joint",
                    "measured_end": {"joint_names": ["j1", "j2"],
                                     "positions_rad": (.2, .3)}}

        def run_gripper_stage(self, stage, *, execute, target_width_m, **kwargs):
            if stage == "close":
                assert kwargs["command_mode"] == "grasp"
                assert kwargs["grasp_force"] == .5
            elif stage == "release":
                assert kwargs["command_mode"] == "release"
            calls.append("execute_" + stage)
            return {"source_format": "realworld_gripper_stage",
                    "status": "executed" if execute else "validated",
                    "success": True,
                    "measured_end_width_m": target_width_m if execute else None,
                    "measured_joint_start": {"joint_names": ["j1", "j2"],
                                             "positions_rad": current[0]["provenance"][
                                                 "joint_positions_by_group_rad"]["right"]}}

    return dict(
        observe_collision_scene=observe,
        make_runtime=lambda scene_hash: Runtime(), planner_pool=object(),
        output_dir=tmp_path / "knob", execute=execute,
        max_microwave_translation_m=.005, max_microwave_rotation_rad=.02,
        max_knob_angle_error_rad=.02, max_start_error_rad=.02,
        max_gripper_width_error_m=.003,
        check_grasp=lambda scene: CheckResult(
            "pass", "manual_terminal", {"answer": "gripped"}),
        check_rotation=lambda scene: CheckResult(
            "pass", "manual_terminal", {"answer": "turned"}),
        check_last_joint_path=lambda scene, target: CheckResult(
            "pass", "manual_terminal", {"answer": "path-reviewed"}),
    )


@pytest.mark.parametrize("final_microwave_shift_m, rotation_evidence_frame, expected_success", [
    (0.0, None, True),
    (0.02, None, False),
    (0.0, 9, False),
])
def test_knob_skill_executes_same_stages_and_requires_observed_rotation(
    tmp_path, monkeypatch, final_microwave_shift_m, rotation_evidence_frame,
    expected_success,
):
    calls = []
    stages = _setup(monkeypatch, calls)
    rotate = next(stage for stage in stages if stage.name.startswith("rotate_"))
    assert rotate.name == "rotate_06"
    scenes = [
        _scene(1, width=.02), _scene(2),
        _scene(3, q=(.1, .1)), _scene(4, q=(.2, .2)),
        _scene(5, q=(.2, .2), width=0.), _scene(6, q=(.2, .2), width=0.),
        _scene(7, q=(.3, .3), width=0., angle=.1),
        _scene(8, q=(.3, .3), angle=.1),
        _scene(9, q=(.3, .3), angle=.1),
        _scene(10, q=(.4, .4), angle=.1),
    ]
    # 最后一段撤离后不再进入下一阶段；即使旋钮角达标，柜体若被推走
    # 也不能把整个现场 Skill 报为成功。
    scenes[-1]["provenance"]["aligned_object_poses_world"][
        "Microwave/Microwave_0"
    ][0][3] = final_microwave_shift_m
    settings = _settings(tmp_path, scenes, execute=True, calls=calls)
    if rotation_evidence_frame is not None:
        settings["check_rotation"] = lambda _scene: CheckResult(
            "pass", "vision", {"source_frame_number": rotation_evidence_frame},
            "旧画面显示旋钮已转动",
        )
    result = module.run_observed_knob_skill(**settings)
    assert result["status"] == ("executed" if expected_success else "failed")
    assert result["shared_flow"]["success"] is expected_success
    assert result["shared_flow"]["stages"][-1] == "rotation_check"
    assert result["rotation_success"] is expected_success
    if final_microwave_shift_m:
        assert result["first_failure"]["code"] == "microwave_anchor_changed"
    if rotation_evidence_frame is not None:
        assert "现场检查证据与当前观测帧不一致" in result["first_failure"]["message"]
    if expected_success:
        assert result["rotation_postcondition"]["evidence"]["source_frame_number"] == 10
    assert [row["stage"] for row in result["stages"]] == [stage.name for stage in stages]
    assert calls == [
        "execute_open", "select", "plan_pregrasp", "execute_pregrasp",
        "plan_approach", "execute_approach", "execute_close",
        ("wait", .01), "plan_rotate_06", "execute_rotate_06",
        "execute_release", ("wait", .01), "plan_retreat", "execute_retreat",
    ]
    assert scenes == []
    assert (tmp_path / "knob/skill_result.json").is_file()


def test_knob_skill_stops_when_robot_moves_but_knob_does_not(tmp_path, monkeypatch):
    calls = []
    _setup(monkeypatch, calls)
    scenes = [
        _scene(1, width=.02), _scene(2),
        _scene(3, q=(.1, .1)), _scene(4, q=(.2, .2)),
        _scene(5, q=(.2, .2), width=0.), _scene(6, q=(.2, .2), width=0.),
        _scene(7, q=(.3, .3), width=0., angle=0.),
    ]
    result = module.run_observed_knob_skill(
        **_settings(tmp_path, scenes, execute=True, calls=calls),
    )
    assert result["status"] == "failed"
    assert result["first_failure"]["code"] == "knob_angle_not_reached"
    assert "execute_release" not in calls


def test_knob_skill_preview_stops_after_open_without_predicting_later_stages(tmp_path, monkeypatch):
    calls = []
    _setup(monkeypatch, calls)
    scenes = [_scene(1, width=.02)]
    result = module.run_observed_knob_skill(
        **_settings(tmp_path, scenes, execute=False, calls=calls),
    )
    assert not result["shared_flow"]["executed"]
    assert result["shared_flow"]["stages"] == ["open"]
    assert result["status"] == "validated_prefix"
    assert result["rotation_success"] is False
    assert calls == ["execute_open"]
    assert scenes == []


def test_knob_skill_stops_before_rotation_if_grasp_unknown(tmp_path, monkeypatch):
    calls = []
    _setup(monkeypatch, calls)
    scenes = [
        _scene(1, width=.02), _scene(2),
        _scene(3, q=(.1, .1)), _scene(4, q=(.2, .2)),
        _scene(5, q=(.2, .2), width=0.),
    ]
    kwargs = _settings(tmp_path, scenes, execute=True, calls=calls)
    kwargs["check_grasp"] = lambda scene: CheckResult(
        "unknown", "vision", {}, "看不清是否夹住旋钮")
    result = module.run_observed_knob_skill(**kwargs)
    assert result["status"] == "unknown"
    assert result["shared_flow"]["failure_code"] == "postcondition_unknown"
    assert result["shared_flow"]["failure_stage"] == "grasp_check"
    assert result["first_failure"]["code"] == "knob_grasp_not_confirmed"
    assert "plan_rotate_06" not in calls


def test_knob_skill_rejects_stale_observation_after_grasp_settle(tmp_path, monkeypatch):
    calls = []
    _setup(monkeypatch, calls)
    scenes = [
        _scene(1, width=.02), _scene(2),
        _scene(3, q=(.1, .1)), _scene(4, q=(.2, .2)),
        _scene(5, q=(.2, .2), width=0.),
        _scene(5, q=(.2, .2), width=0.),
    ]
    result = module.run_observed_knob_skill(
        **_settings(tmp_path, scenes, execute=True, calls=calls),
    )
    assert result["status"] == "failed"
    assert result["first_failure"]["code"] == "observed_state_discontinuity"
    assert "plan_rotate_06" not in calls


def test_explicit_last_joint_profile_executes_old_command_after_path_check(
    tmp_path, monkeypatch,
):
    calls = []
    _setup(monkeypatch, calls, last_joint=True)
    scenes = [
        _scene(1, width=.02), _scene(2),
        _scene(3, q=(.1, .1)), _scene(4, q=(.2, .2)),
        _scene(5, q=(.2, .2), width=0.), _scene(6, q=(.2, .2), width=0.),
        _scene(7, q=(.2, .3), width=0., angle=.1),
        _scene(8, q=(.2, .3), angle=.1),
        _scene(9, q=(.2, .3), angle=.1),
        _scene(10, q=(.4, .4), angle=.1),
    ]
    result = module.run_observed_knob_skill(
        **_settings(tmp_path, scenes, execute=True, calls=calls),
    )
    assert result["rotation_success"] is True
    assert "execute_rotate_06" in calls
    assert result["last_joint_path_check"]["status"] == "pass"
    rotate = next(row for row in result["stages"] if row["stage"] == "rotate_06")
    assert rotate["execution"]["collision_scope"] == "none_legacy_last_joint"


def test_explicit_last_joint_profile_stops_when_path_not_approved(
    tmp_path, monkeypatch,
):
    calls = []
    _setup(monkeypatch, calls, last_joint=True)
    scenes = [
        _scene(1, width=.02), _scene(2),
        _scene(3, q=(.1, .1)), _scene(4, q=(.2, .2)),
        _scene(5, q=(.2, .2), width=0.), _scene(6, q=(.2, .2), width=0.),
    ]
    settings = _settings(tmp_path, scenes, execute=True, calls=calls)
    settings["check_last_joint_path"] = lambda scene, target: CheckResult(
        "unknown", "not_configured", {}, "旋钮周边接触尚未检查")
    result = module.run_observed_knob_skill(**settings)
    assert result["status"] == "unknown"
    assert result["first_failure"]["code"] == "last_joint_path_not_approved"
    assert "execute_rotate_06" not in calls


def test_knob_failed_command_retains_sdk_endpoint_without_selection(tmp_path, monkeypatch):
    calls = []
    _setup(monkeypatch, calls)
    settings = _settings(tmp_path, [_scene(1, width=.02)], execute=True, calls=calls)
    def command(name, **kwargs):
        calls.append('failed_' + name)
        return {'success': False, 'status': 'failed', 'measured_end': {
            'joint_names': ['j1', 'j2'], 'positions_rad': [.3, .4]}}
    settings['make_runtime'] = lambda scene_hash: SimpleNamespace(run_gripper_stage=command)
    result = module.run_observed_knob_skill(**settings)
    assert calls == ['failed_open']
    assert result['shared_flow']['failure_stage'] == 'open'
    assert result['shared_flow']['end_state']['source'] == 'realworld_sdk'
    assert result['shared_flow']['end_state']['values']['positions_rad'] == [.3, .4]


def test_knob_command_exception_preserves_traceback_and_reobserves_end(tmp_path, monkeypatch):
    calls = []
    _setup(monkeypatch, calls)
    settings = _settings(tmp_path, [_scene(1, width=.02), _scene(2, q=(.3, .4))],
                         execute=True, calls=calls)
    def command(name, **kwargs):
        calls.append('raised_' + name)
        raise RuntimeError('partial knob command')
    settings['make_runtime'] = lambda scene_hash: SimpleNamespace(run_gripper_stage=command)
    result = module.run_observed_knob_skill(**settings)
    assert calls == ['raised_open']
    assert result['shared_flow']['failure_stage'] == 'open'
    assert result['shared_flow']['end_state']['source'] == 'realworld_observation'
    assert result['shared_flow']['end_state']['values']['joint_positions_by_group_rad']['right'] == [.3, .4]
    assert 'partial knob command' in result['first_failure']['traceback']
    assert result['first_failure']['stage'] == 'open'
