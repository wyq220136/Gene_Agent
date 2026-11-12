"""现场放碗子任务按旧阶段顺序运行，逐段从实际状态交接。"""

from types import SimpleNamespace

import numpy as np
import pytest

from gene_agent.realworld.microwave import insert as module
from gene_agent.tasks.checks import CheckResult


def _scene(frame, q, width):
    return {
        "snapshot": {"snapshot_sha256": f"scene-{frame}", "active_group": "right"},
        "provenance": {
            "source_frame_number": frame, "source_timestamp_ms": frame * 1000.,
            "physics_stepped": False, "pose_frame_verified": True,
            "collision_snapshot_sha256": f"scene-{frame}",
            "joint_positions_by_group_rad": {"right": list(q)},
            "gripper_widths_m": {"right": width},
            "aligned_object_poses_world": {
                "Bowl/Bowl_0": np.eye(4).tolist(),
                "Microwave/Microwave_0": np.eye(4).tolist(),
            },
        },
        "robot_metadata": {
            "joint_groups": {"right": ["j1", "j2"]},
            "grippers": {"right": {"width_range_m": [0., .08]}},
        },
    }


def _pick():
    return {"execute": True, "status": "executed", "pick_success": True,
            "post_lift_scene_snapshot_sha256": "scene-1",
            "post_lift_source_frame_number": 1,
            "post_lift_source_timestamp_ms": 1000.,
            "postcondition": {"status": "pass", "source": "manual_terminal",
                              "evidence": {"answer": "y"}},
            "grasp": {"selected_candidate_id": "bowl_ring_00_00"}}


def _settings(tmp_path, scenes, runtime, *, execute, check_hold=None):
    return dict(
        pick_report=_pick(), observe_collision_scene=lambda: scenes.pop(0),
        make_runtime=runtime, planner_pool=object(),
        microwave_frame_alignment={"accepted_for_planning": True},
        output_dir=tmp_path / "insert", execute=execute,
        max_microwave_translation_m=.005, max_microwave_rotation_rad=.02,
        max_hold_translation_m=.005, max_hold_rotation_rad=.02,
        max_start_error_rad=.02, max_gripper_width_error_m=.003,
        check_hold=check_hold,
        check_insert_path=lambda _scene, _plan: CheckResult(
            "pass", "manual_terminal", {"answer": "y"}, "路径已检查"),
        check_release=lambda _scene: CheckResult(
            "pass", "manual_terminal", {"answer": "y"}, "Bowl 已离爪"),
        check_place=lambda _scene: CheckResult(
            "pass", "manual_terminal", {"answer": "y"}, "Bowl 在炉内"),
    )


def _fake_planners(monkeypatch, calls):
    def prefix(scene, pick, *, output_dir, current_hold_check, **kwargs):
        assert scene["snapshot"]["snapshot_sha256"] == "scene-1"
        assert current_hold_check is None
        calls.append("plan_preplace")
        return {"success": True, "targets": {
                    "source_frame_number": 1,
                    "T_world_profile_microwave": np.eye(4).tolist(),
                    "T_tcp_bowl_from_actual_hold": np.eye(4).tolist(),
                    "insert_depth_offset_microwave_y_m": .02,
                    "retreat_rotation_tcp_local_xyz_deg": [0., -10., 0.],
                    "keyposes_world_tcp": {"retreat_1": np.eye(4).tolist(),
                                           "retreat_2": np.eye(4).tolist()},
                },
                "attached_collision_snapshot_sha256": "attached-pre",
                "preplace_stage": SimpleNamespace(name="preplace", to_dict=lambda: {"name": "preplace"})}

    def linear(scene, targets, pick, *, hold_check, output_dir, **kwargs):
        assert scene["snapshot"]["snapshot_sha256"] == "scene-2"
        assert hold_check["status"] == "pass"
        calls.append("plan_insert")
        return {"success": True, "stage_collision_snapshot_sha256": "self-insert",
                "stage": SimpleNamespace(name="insert", to_dict=lambda: {"name": "insert"})}

    def retreat(scene, targets, release, *, stage_name, output_dir, **kwargs):
        assert release["release_success"] is True
        calls.append("plan_" + stage_name)
        return {"success": True, "stage_collision_snapshot_sha256": "self-" + stage_name,
                "stage": SimpleNamespace(name=stage_name,
                                         to_dict=lambda: {"name": stage_name})}

    monkeypatch.setattr(module, "plan_observed_bowl_insert_prefix", prefix)
    monkeypatch.setattr(module, "plan_observed_bowl_insert_linear", linear)
    monkeypatch.setattr(module, "plan_observed_bowl_detached_retreat_stage", retreat)
    monkeypatch.setattr(module.time, "sleep", lambda seconds: calls.append(("settle", seconds)))


@pytest.mark.parametrize("place_evidence_frame, expected_success", [
    (None, True),
    (6, False),
])
def test_insert_executes_original_stage_order_from_fresh_measurements(
    tmp_path, monkeypatch, place_evidence_frame, expected_success,
):
    calls = []
    _fake_planners(monkeypatch, calls)
    scenes = [
        _scene(1, (.1, .2), .007),
        _scene(2, (.3, .4), .007),
        _scene(3, (.5, .6), .007),
        _scene(4, (.5, .6), .08),
        _scene(5, (.5, .6), .08),
        _scene(6, (.7, .8), .08),
        _scene(7, (.9, 1.), .08),
    ]
    expected_end = {"preplace": (.3, .4), "insert": (.5, .6),
                    "retreat_1": (.7, .8), "retreat_2": (.9, 1.)}

    class Runtime:
        def run_motion_stage(self, stage, *, execute):
            assert execute is True
            calls.append("execute_" + stage.name)
            return {"source_format": "realworld_motion_stage", "status": "executed",
                    "success": True, "measured_end": {
                        "joint_names": ["j1", "j2"],
                        "positions_rad": expected_end[stage.name]}}

        def run_gripper_stage(self, stage, *, execute, **kwargs):
            assert stage == "release" and execute is True
            assert kwargs["command_mode"] == "release"
            assert kwargs["planned_start"].positions_rad == (.5, .6)
            calls.append("release")
            return {"source_format": "realworld_gripper_stage", "status": "executed",
                    "success": True, "measured_end_width_m": .08,
                    "measured_joint_start": {"joint_names": ["j1", "j2"],
                                             "positions_rad": [.5, .6]}}

    settings = _settings(
        tmp_path, scenes, lambda _hash: Runtime(), execute=True,
        check_hold=lambda _scene: CheckResult(
            "pass", "manual_terminal", {"answer": "y"}, "仍持碗"),
    )
    if place_evidence_frame is not None:
        settings["check_place"] = lambda _scene: CheckResult(
            "pass", "vision", {"source_frame_number": place_evidence_frame},
            "旧画面显示 Bowl 在炉内",
        )
    report = module.run_observed_bowl_insert(**settings)
    assert report["insert_success"] is expected_success
    assert report["status"] == ("executed" if expected_success else "failed")
    if not expected_success:
        assert "现场检查证据与当前观测帧不一致" in report["first_failure"]["message"]
    assert calls == [
        "plan_preplace", "execute_preplace", "plan_insert", "execute_insert",
        "release", ("settle", .5), "plan_retreat_1", "execute_retreat_1",
        "plan_retreat_2", "execute_retreat_2",
    ]
    assert scenes == []
    if expected_success:
        assert report["postcondition"]["status"] == "pass"
        assert report["postcondition"]["evidence"]["source_frame_number"] == 7
    assert (tmp_path / "insert/insert_result.json").is_file()


def test_insert_preview_stops_after_first_stage_without_predicted_handoff(tmp_path, monkeypatch):
    calls = []
    _fake_planners(monkeypatch, calls)
    scenes = [_scene(1, (.1, .2), .007)]

    class Runtime:
        def run_motion_stage(self, stage, *, execute):
            assert stage.name == "preplace" and execute is False
            calls.append("validate_preplace")
            return {"source_format": "realworld_motion_stage", "status": "validated",
                    "success": True}

    report = module.run_observed_bowl_insert(
        **_settings(tmp_path, scenes, lambda _hash: Runtime(), execute=False)
    )
    assert report["status"] == "validated_prefix"
    assert report["insert_success"] is False
    assert calls == ["plan_preplace", "validate_preplace"]
    assert scenes == []


def test_insert_rechecks_hold_when_new_frame_has_same_collision_hash(tmp_path, monkeypatch):
    def unexpected_plan(*_args, **_kwargs):
        raise AssertionError("没有当前持碗确认时不能开始规划")

    monkeypatch.setattr(module, "plan_observed_bowl_insert_prefix", unexpected_plan)
    later = _scene(2, (.1, .2), .007)
    later["snapshot"]["snapshot_sha256"] = "scene-1"
    later["provenance"]["collision_snapshot_sha256"] = "scene-1"
    called = []

    def check(scene):
        called.append(scene["provenance"]["source_frame_number"])
        return CheckResult("unknown", "vision", {}, "不能确认 Bowl 仍在夹爪中")

    report = module.run_observed_bowl_insert(
        **_settings(tmp_path, [later], lambda _hash: object(), execute=False,
                    check_hold=check)
    )
    assert called == [2]
    assert report["status"] == "unknown"
    assert report["first_failure"]["code"] == "hold_not_confirmed"


def test_insert_stops_when_hold_is_unknown_after_preplace(tmp_path, monkeypatch):
    calls = []
    _fake_planners(monkeypatch, calls)
    scenes = [_scene(1, (.1, .2), .007), _scene(2, (.3, .4), .007)]

    class Runtime:
        def run_motion_stage(self, stage, *, execute):
            calls.append("execute_preplace")
            return {"source_format": "realworld_motion_stage", "status": "executed",
                    "success": True, "measured_end": {
                        "joint_names": ["j1", "j2"], "positions_rad": [.3, .4]}}

    report = module.run_observed_bowl_insert(
        **_settings(tmp_path, scenes, lambda _hash: Runtime(), execute=True,
                    check_hold=lambda _scene: CheckResult(
                        "unknown", "vision", {}, "无法确认仍持碗"))
    )
    assert report["insert_success"] is False
    assert report["status"] == "unknown"
    assert report["first_failure"]["code"] == "hold_not_confirmed"
    assert calls == ["plan_preplace", "execute_preplace"]


def test_manual_prepared_bowl_uses_same_insert_subtask_without_fake_pick(tmp_path, monkeypatch):
    from gene_agent.realworld.microwave.attachment import make_manual_held_bowl_report
    calls = []
    _fake_planners(monkeypatch, calls)
    scene = _scene(1, (.1, .2), .007)
    manual = make_manual_held_bowl_report(
        scene, CheckResult("pass", "manual_terminal", {"answer": "prepared"}),
    )
    assert "grasp" not in manual and "pick_success" not in manual

    class Runtime:
        def run_motion_stage(self, stage, *, execute):
            assert stage.name == "preplace" and execute is False
            calls.append("validate_preplace")
            return {"source_format": "realworld_motion_stage", "status": "validated",
                    "success": True}

    settings = _settings(tmp_path, [scene], lambda _hash: Runtime(), execute=False)
    settings["pick_report"] = manual
    report = module.run_observed_bowl_insert(**settings)
    assert report["status"] == "validated_prefix"
    assert calls == ["plan_preplace", "validate_preplace"]
    assert "initial_hold_check" not in report


def test_insert_rejects_stale_observation_after_release(tmp_path, monkeypatch):
    calls = []
    _fake_planners(monkeypatch, calls)
    scenes = [_scene(1, (.1, .2), .007), _scene(2, (.3, .4), .007),
              _scene(3, (.5, .6), .007), _scene(4, (.5, .6), .08),
              _scene(4, (.5, .6), .08)]

    class Runtime:
        def run_motion_stage(self, stage, *, execute):
            q = {"preplace": [.3, .4], "insert": [.5, .6]}[stage.name]
            return {"source_format": "realworld_motion_stage", "status": "executed",
                    "success": True, "measured_end": {"joint_names": ["j1", "j2"],
                                                  "positions_rad": q}}

        def run_gripper_stage(self, stage, *, execute, **kwargs):
            return {"source_format": "realworld_gripper_stage", "status": "executed",
                    "success": True, "measured_end_width_m": .08,
                    "measured_joint_start": {"joint_names": ["j1", "j2"],
                                             "positions_rad": [.5, .6]}}

    report = module.run_observed_bowl_insert(
        **_settings(tmp_path, scenes, lambda _hash: Runtime(), execute=True,
                    check_hold=lambda _scene: CheckResult(
                        "pass", "manual_terminal", {"answer": "y"})),
    )
    assert report["status"] == "failed"
    assert report["first_failure"]["code"] == "observed_state_discontinuity"
    assert "plan_retreat_1" not in calls


def test_dispatch_exception_records_unknown_end_in_shared_flow(tmp_path, monkeypatch):
    calls = []
    _fake_planners(monkeypatch, calls)
    class Runtime:
        def run_motion_stage(self, stage, *, execute):
            raise RuntimeError('SDK 连接中断，运动终态未知')
    report = module.run_observed_bowl_insert(**_settings(
        tmp_path, [_scene(1, (.1, .2), .007)], lambda _hash: Runtime(), execute=True))
    assert report['insert_success'] is False
    assert report['shared_flow']['end_state']['source'] == 'unavailable'
    assert report['shared_flow']['end_state']['values']['state_unavailable']
    assert report['first_failure']['code'] == 'stage_exception'
    assert calls == ['plan_preplace']


def test_observation_failure_keeps_sdk_actual_end_in_shared_flow(tmp_path, monkeypatch):
    calls = []
    _fake_planners(monkeypatch, calls)
    class Runtime:
        def run_motion_stage(self, stage, *, execute):
            return {'success': True, 'status': 'executed', 'measured_end': {
                'joint_names': ['j1', 'j2'], 'positions_rad': [.3, .4]}}
    report = module.run_observed_bowl_insert(**_settings(
        tmp_path, [_scene(1, (.1, .2), .007)], lambda _hash: Runtime(), execute=True))
    assert report['insert_success'] is False
    assert report['shared_flow']['end_state']['source'] == 'realworld_sdk'
    assert report['shared_flow']['end_state']['values']['positions_rad'] == [.3, .4]
    assert report['shared_flow']['stages'] == ['preplace']


def _successful_insert_runtime():
    class Runtime:
        def run_motion_stage(self, stage, *, execute):
            q = {'preplace': [.3, .4], 'insert': [.5, .6],
                 'retreat_1': [.7, .8], 'retreat_2': [.9, 1.]}[stage.name]
            return {'source_format': 'realworld_motion_stage', 'status': 'executed',
                    'success': True, 'measured_end': {'joint_names': ['j1', 'j2'],
                                                     'positions_rad': q}}
        def run_gripper_stage(self, stage, *, execute, **kwargs):
            return {'source_format': 'realworld_gripper_stage', 'status': 'executed',
                    'success': True, 'measured_end_width_m': .08,
                    'measured_joint_start': {'joint_names': ['j1', 'j2'],
                                             'positions_rad': [.5, .6]}}
    return Runtime()


def test_stale_release_check_cannot_enable_detached_retreat(tmp_path, monkeypatch):
    calls = []
    _fake_planners(monkeypatch, calls)
    scenes = [_scene(1, (.1, .2), .007), _scene(2, (.3, .4), .007),
              _scene(3, (.5, .6), .007), _scene(4, (.5, .6), .08),
              _scene(5, (.5, .6), .08)]
    settings = _settings(tmp_path, scenes, lambda _hash: _successful_insert_runtime(),
                         execute=True, check_hold=lambda scene: CheckResult(
                             'pass', 'manual_terminal', {'answer': 'yes'}))
    settings['check_release'] = lambda scene: CheckResult(
        'pass', 'vision', {'source_frame_number': 1}, '旧图像显示 Bowl 离爪')
    report = module.run_observed_bowl_insert(**settings)
    assert not report['insert_success']
    assert '现场检查证据与当前观测帧不一致' in report['first_failure']['message']
    assert 'plan_retreat_1' not in calls
    assert 'release_report' not in report


def test_settle_observation_failure_marks_current_state_unknown(tmp_path, monkeypatch):
    calls = []
    _fake_planners(monkeypatch, calls)
    scenes = [_scene(1, (.1, .2), .007), _scene(2, (.3, .4), .007),
              _scene(3, (.5, .6), .007), _scene(4, (.5, .6), .08)]
    report = module.run_observed_bowl_insert(**_settings(
        tmp_path, scenes, lambda _hash: _successful_insert_runtime(), execute=True,
        check_hold=lambda scene: CheckResult('pass', 'manual_terminal', {'answer': 'yes'})))
    assert not report['insert_success']
    assert report['shared_flow']['failure_stage'] == 'settle'
    assert report['shared_flow']['end_state']['source'] == 'unavailable'
    assert report['shared_flow']['end_state']['values']['state_unavailable']
    assert 'plan_retreat_1' not in calls
