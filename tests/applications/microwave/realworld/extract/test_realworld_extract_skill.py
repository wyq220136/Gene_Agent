"""真机取碗子任务逐段执行同一 Skill；未知夹持不能继续。"""

import numpy as np
import pytest

from gene_agent.realworld.microwave import extract as module
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
            "robot_metadata": {"joint_groups": {"right": ("j1", "j2")}}}


class _Runtime:
    def __init__(self, actions, execute):
        self.actions = actions
        self.execute = execute

    def run_gripper_stage(self, name, *, target_width_m, execute, **kwargs):
        self.actions.append(name)
        assert execute is self.execute
        if name == "close":
            assert kwargs["command_mode"] == "grasp"
            assert kwargs["grasp_force"] == .5
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
        observe_collision_scene=observe, make_runtime=runtime,
        planner_pool=object(), microwave_frame_alignment={},
        output_dir=tmp_path / "extract", execute=execute,
        max_microwave_translation_m=.01, max_microwave_rotation_rad=.1,
        max_bowl_translation_m=.01, max_bowl_rotation_rad=.1,
        max_hold_translation_m=.01, max_hold_rotation_rad=.1,
        max_start_error_rad=.03, max_gripper_width_error_m=.002,
    )


def test_extract_plan_only_stops_after_first_preflight(tmp_path, monkeypatch):
    actions = []
    observed = []

    def observe():
        observed.append(1)
        return _scene(20, .007)

    monkeypatch.setattr(module, "plan_observed_bowl_extract_prefix",
                        lambda *args, **kwargs: (_ for _ in ()).throw(
                            AssertionError("未经执行开爪，不能规划后续阶段")))
    report = module.run_observed_bowl_extract(
        **_kwargs(tmp_path, observe, lambda hash: _Runtime(actions, False), False),
    )
    assert not report["shared_flow"]["executed"]
    assert report["shared_flow"]["stages"] == ["open"]
    assert report["status"] == "validated_prefix"
    assert report["extract_success"] is False
    assert actions == ["open"]
    assert len(observed) == 1


@pytest.mark.parametrize("final_evidence_frame, expected_success", [
    (None, True),
    (27, False),
])
def test_extract_execute_reobserves_each_stage_and_checks_hold(
    tmp_path, monkeypatch, final_evidence_frame, expected_success,
):
    actions = []
    widths = (.007, .041, .041, .041, .041, .007, .007, .007, .007)
    observed = []

    def observe():
        index = len(observed)
        scene = _scene(20 + index, widths[index])
        observed.append(scene)
        return scene

    def pass_check(scene, *args):
        return CheckResult("pass", "manual_terminal", {"answer": "yes"})

    targets = {"source_frame_number": 21,
               "T_world_profile_microwave": np.eye(4).tolist(),
               "bowl_pose_world": np.eye(4).tolist(),
               "keyposes_world_tcp": {name: np.eye(4).tolist() for name in (
                   "pre_extract_1", "pre_extract_2", "grasp",
                   "post_extract_1", "post_extract_2")}}

    def prefix(scene, **kwargs):
        assert scene["provenance"]["source_frame_number"] == 21
        return {"success": True, "targets": targets,
                "pre_extract_1_stage": SkillStage(
                    "pre_extract_1", "planned", details={"operation": "plan"})}

    def motion(scene, initial, *, stage_name, **kwargs):
        assert scene["provenance"]["source_frame_number"] in (22, 23)
        return {"success": True,
                "stage_collision_snapshot_sha256": f"planned-{stage_name}",
                "stage": SkillStage(stage_name, "planned",
                                    details={"operation": "linear_tcp"})}

    def retreat(scene, initial, grasp, *, stage_name, current_hold_check, **kwargs):
        assert current_hold_check.passed
        assert current_hold_check.evidence["collision_snapshot_sha256"] == scene[
            "snapshot"]["snapshot_sha256"]
        assert grasp["grasp_success"] and grasp["T_tcp_bowl_after_close"]
        return {"success": True,
                "stage_collision_snapshot_sha256": f"planned-{stage_name}",
                "stage": SkillStage(stage_name, "planned",
                                    details={"operation": "linear_tcp"})}

    monkeypatch.setattr(module, "plan_observed_bowl_extract_prefix", prefix)
    monkeypatch.setattr(module, "plan_observed_bowl_extract_motion_stage", motion)
    monkeypatch.setattr(module, "plan_observed_bowl_extract_retreat_stage", retreat)
    monkeypatch.setattr(module, "_observed_tcp_pose", lambda scene: np.eye(4))
    def final_check(scene):
        if final_evidence_frame is None:
            return pass_check(scene)
        # 最后撤离后是第 28 帧；第 27 帧的旧通过结果不能作为
        # “已经完整取出”的终点证据。
        return CheckResult("pass", "vision", {
            "source_frame_number": final_evidence_frame,
        })
    report = module.run_observed_bowl_extract(
        **_kwargs(tmp_path, observe, lambda hash: _Runtime(actions, True), True),
        check_grasp=pass_check, check_hold=pass_check,
        check_contact_path=pass_check, check_extracted=final_check,
    )
    assert report["status"] == ("executed" if expected_success else "failed")
    assert report["shared_flow"]["success"] is expected_success
    assert report["shared_flow"]["stages"][-1] == "extract_check"
    assert report["extract_success"] is expected_success
    if not expected_success:
        assert "现场检查证据与当前观测帧不一致" in report["first_failure"]["message"]
    assert actions == ["open", "pre_extract_1", "pre_extract_2", "grasp",
                       "close", "post_extract_1", "post_extract_2"]
    assert len(observed) == len(widths)
    assert report["grasp"]["physical_grasp_check"]["evidence"][
        "source_frame_number"] == 26


def test_extract_unknown_grasp_stops_before_attachment(tmp_path, monkeypatch):
    actions = []
    widths = (.007, .041, .041, .041, .041, .007, .007)
    index = 0

    def observe():
        nonlocal index
        scene = _scene(20 + index, widths[index])
        index += 1
        return scene

    targets = {"source_frame_number": 21}
    monkeypatch.setattr(module, "plan_observed_bowl_extract_prefix",
                        lambda *args, **kwargs: {
                            "success": True, "targets": targets,
                            "pre_extract_1_stage": SkillStage(
                                "pre_extract_1", "planned",
                                details={"operation": "plan"})})
    monkeypatch.setattr(module, "plan_observed_bowl_extract_motion_stage",
                        lambda scene, initial, *, stage_name, **kwargs: {
                            "success": True,
                            "stage_collision_snapshot_sha256": f"planned-{stage_name}",
                            "stage": SkillStage(stage_name, "planned",
                                                details={"operation": "linear_tcp"})})
    monkeypatch.setattr(module, "plan_observed_bowl_extract_retreat_stage",
                        lambda *args, **kwargs: (_ for _ in ()).throw(
                            AssertionError("夹持未知不能规划携物撤离")))
    report = module.run_observed_bowl_extract(
        **_kwargs(tmp_path, observe, lambda hash: _Runtime(actions, True), True),
        check_contact_path=lambda scene, plan: CheckResult(
            "pass", "manual_terminal", {"answer": "yes"}),
    )
    assert report["status"] == "unknown"
    assert report["first_failure"]["code"] == "grasp_not_confirmed"
    assert actions[-1] == "close"


def test_extract_failed_command_retains_sdk_endpoint_without_next_motion(tmp_path):
    from types import SimpleNamespace
    actions = []
    def command(name, **kwargs):
        actions.append(name)
        return {'success': False, 'status': 'failed', 'measured_end': {
            'joint_names': ['j1', 'j2'], 'positions_rad': [.3, .4]}}
    result = module.run_observed_bowl_extract(**_kwargs(
        tmp_path, lambda: _scene(20, .007),
        lambda scene_hash: SimpleNamespace(run_gripper_stage=command), True))
    assert actions == ['open']
    assert not result['shared_flow']['success']
    assert result['shared_flow']['end_state']['source'] == 'realworld_sdk'
    assert result['shared_flow']['end_state']['values']['positions_rad'] == [.3, .4]


def test_extract_command_exception_preserves_traceback_and_reobserves_end(tmp_path):
    from types import SimpleNamespace
    scenes = iter([_scene(20, .007), _scene(21, .025)])
    def command(name, **kwargs):
        raise RuntimeError('partial extract command')
    result = module.run_observed_bowl_extract(**_kwargs(
        tmp_path, lambda: next(scenes),
        lambda scene_hash: SimpleNamespace(run_gripper_stage=command), True))
    assert result['shared_flow']['failure_stage'] == 'open'
    assert result['shared_flow']['end_state']['source'] == 'realworld_observation'
    assert result['shared_flow']['end_state']['values']['gripper_widths_m']['right'] == .025
    assert 'partial extract command' in result['first_failure']['traceback']
