"""桌面 Place 子任务逐段读取现场状态，松爪确认后才撤离。"""

import numpy as np
import pytest

from gene_agent.realworld.microwave import place as module
from gene_agent.skills.runtime.planning import SkillStage
from gene_agent.tasks.checks import CheckResult


def _scene(frame, width):
    return {"snapshot": {"snapshot_sha256": f"scene-{frame}",
                         "active_group": "right"},
            "provenance": {"source_frame_number": frame,
                           "source_timestamp_ms": float(frame * 100),
                           "gripper_widths_m": {"right": width},
                           "joint_positions_by_group_rad": {"right": (.1, .2)},
                           "aligned_object_poses_world": {
                               "Bowl/Bowl_0": np.eye(4).tolist()}},
            "robot_metadata": {"joint_groups": {"right": ("j1", "j2")},
                               "grippers": {"right": {"width_range_m": (0., .08)}}}}


class _Runtime:
    def __init__(self, actions, execute):
        self.actions = actions
        self.execute = execute

    def run_gripper_stage(self, name, *, target_width_m, execute, **kwargs):
        self.actions.append(name)
        assert execute is self.execute
        if name == "release":
            assert kwargs["command_mode"] == "release"
        return {"source_format": "realworld_gripper_stage", "stage": name,
                "success": True, "status": "executed" if execute else "validated",
                "measured_joint_start": {"joint_names": ["j1", "j2"],
                                         "positions_rad": [.1, .2]},
                "measured_end_width_m": target_width_m if execute else None}

    def run_motion_stage(self, stage, *, execute):
        self.actions.append(stage.name)
        assert execute is self.execute
        return {"source_format": "realworld_motion_stage", "stage": stage.name,
                "success": True, "status": "executed" if execute else "validated",
                "measured_end": {"joint_names": ["j1", "j2"],
                                 "positions_rad": [.1, .2]} if execute else None}


def _kwargs(tmp_path, observe, runtime, execute):
    return dict(
        grasp_report={"execute": True, "status": "executed", "grasp_success": True,
                      "selected_candidate_id": "fixed_keypose_0",
                      "physical_grasp_check": {"status": "pass", "source": "manual_terminal"}},
        observe_collision_scene=observe, make_runtime=runtime,
        planner_pool=object(), table_height_alignment={},
        output_dir=tmp_path / "table", execute=execute,
        max_table_height_delta_m=.01,
        max_table_translation_m=.01, max_table_rotation_rad=.1,
        max_bowl_translation_m=.01, max_bowl_rotation_rad=.1,
        max_hold_translation_m=.01, max_hold_rotation_rad=.1,
        max_start_error_rad=.03, max_gripper_width_error_m=.002,
    )


def test_table_place_plan_only_validates_first_motion_without_release(tmp_path, monkeypatch):
    actions = []
    observations = []

    def observe():
        observations.append(1)
        return _scene(30, .007)

    monkeypatch.setattr(module, "plan_observed_table_place_prefix",
                        lambda scene, grasp, **kwargs: {
                            "success": True, "targets": {},
                            "stage_collision_snapshot_sha256": "planned",
                            "preplace_stage": SkillStage(
                                "preplace", "planned", details={"operation": "linear_tcp"})})
    report = module.run_observed_table_place(
        **_kwargs(tmp_path, observe, lambda hash: _Runtime(actions, False), False),
        check_hold=lambda scene: CheckResult("pass", "manual_terminal", {"answer": "held"}),
    )
    assert report["status"] == "validated_prefix"
    assert report["table_place_success"] is False
    assert actions == ["preplace"]
    assert len(observations) == 1


@pytest.mark.parametrize("final_evidence_frame, expected_success", [
    (None, True),
    (34, False),
])
def test_table_place_executes_release_and_retreat_from_new_scene(
    tmp_path, monkeypatch, final_evidence_frame, expected_success,
):
    actions = []
    widths = (.007, .007, .007, .08, .08, .08)
    observations = []

    def observe():
        index = len(observations)
        scene = _scene(30 + index, widths[index])
        observations.append(scene)
        return scene

    targets = {"source_frame_number": 30,
               "table_pose_world": np.eye(4).tolist(),
               "T_tcp_bowl_from_actual_hold": np.eye(4).tolist(),
               "table_alignment": {"current_surface_top_z_m": -.25},
               "keyposes_world_tcp": {"retreat": np.eye(4).tolist()}}

    def prefix(scene, grasp, *, current_hold_check, **kwargs):
        assert current_hold_check.evidence["source_frame_number"] == 30
        return {"success": True, "targets": targets,
                "stage_collision_snapshot_sha256": "attached-body",
                "preplace_stage": SkillStage(
                    "preplace", "planned", details={"operation": "linear_tcp"})}

    def descent(scene, initial, grasp, *, current_hold_check, **kwargs):
        assert scene["provenance"]["source_frame_number"] == 31
        assert current_hold_check.evidence["source_frame_number"] == 31
        return {"success": True, "stage_collision_snapshot_sha256": "attached-body-2",
                "stage": SkillStage("place", "planned",
                                    details={"operation": "linear_tcp"})}

    def retreat(scene, initial, release, **kwargs):
        assert scene["provenance"]["source_frame_number"] == 34
        assert release["status"] == "executed" and release["release_success"]
        assert release["source_frame_number"] == 33
        return {"success": True, "stage_collision_snapshot_sha256": "without-bowl",
                "stage": SkillStage("retreat", "planned",
                                    details={"operation": "linear_tcp"})}

    monkeypatch.setattr(module, "plan_observed_table_place_prefix", prefix)
    monkeypatch.setattr(module, "plan_observed_table_descent", descent)
    monkeypatch.setattr(module, "plan_observed_table_detached_retreat", retreat)
    monkeypatch.setattr(module.time, "sleep", lambda duration: None)
    passed = lambda scene, *args: CheckResult("pass", "manual_terminal",
                                               {"answer": "yes"})
    def final_check(scene):
        if final_evidence_frame is None:
            return passed(scene)
        # 最终现场为第 35 帧；一个外部检查器若误返回第 34 帧的
        # 通过结果，不能据此报告桌面 Place 成功。
        return CheckResult("pass", "vision", {
            "source_frame_number": final_evidence_frame,
        })
    report = module.run_observed_table_place(
        **_kwargs(tmp_path, observe, lambda hash: _Runtime(actions, True), True),
        check_hold=passed, check_place_path=passed,
        check_release=passed, check_table=final_check,
    )
    assert report["status"] == ("executed" if expected_success else "failed")
    assert report["table_place_success"] is expected_success
    if not expected_success:
        assert "现场检查证据与当前观测帧不一致" in report["first_failure"]["message"]
    assert actions == ["preplace", "place", "release", "retreat"]
    assert len(observations) == len(widths)
    assert report["release_report"]["postcondition"]["status"] == "pass"


def test_table_settle_observation_failure_does_not_reuse_release_frame(tmp_path, monkeypatch):
    actions = []
    frames = [_scene(30, .007), _scene(31, .007), _scene(32, .007), _scene(33, .08)]
    stage = lambda name: SkillStage(name, 'planned', details={'operation': 'linear_tcp'})
    monkeypatch.setattr(module, 'plan_observed_table_place_prefix', lambda *args, **kwargs: {
        'success': True, 'targets': {}, 'stage_collision_snapshot_sha256': 'pre',
        'preplace_stage': stage('preplace')})
    monkeypatch.setattr(module, 'plan_observed_table_descent', lambda *args, **kwargs: {
        'success': True, 'stage_collision_snapshot_sha256': 'descent', 'stage': stage('place')})
    monkeypatch.setattr(module.time, 'sleep', lambda duration: None)
    passed = lambda *args: CheckResult('pass', 'manual_terminal', {'answer': 'yes'})
    report = module.run_observed_table_place(
        **_kwargs(tmp_path, lambda: frames.pop(0), lambda _hash: _Runtime(actions, True), True),
        check_hold=passed, check_place_path=passed, check_release=passed, check_table=passed)
    assert not report['table_place_success']
    assert report['shared_flow']['failure_stage'] == 'settle'
    assert report['shared_flow']['end_state']['source'] == 'unavailable'
    assert report['shared_flow']['end_state']['values']['state_unavailable']
    assert 'retreat' not in actions
