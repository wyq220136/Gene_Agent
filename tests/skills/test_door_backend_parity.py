"""共享门决策：实测角决定下一阶段，未知证据不能当成到达。"""
import math

import numpy as np
import pytest

from gene_agent.skills.library.door.sequence import (
    door_angle_check, door_observation_freshness, door_skill_angle_range,
    door_skill_sequence,
)
from gene_agent.tasks.policies.microwave_task.policy import default_tcp_transition


def test_updated_angle_changes_push_start_without_changing_pull_target():
    config = {"pull_angle_deg": 25., "push_angle_deg": 90.}
    assert door_skill_angle_range("handle_pull", config, 0.) == (0., 25.)
    assert door_skill_angle_range("push_open", config, math.radians(31.)) == (31., 90.)
    assert door_skill_angle_range("push_close", config, math.radians(86.)) == (86., 0.)
    with pytest.raises(ValueError, match="关闭"):
        door_skill_angle_range("handle_pull", config, .2)


@pytest.mark.parametrize("angle", [None, float("nan"), float("inf")])
def test_push_cannot_plan_from_unknown_angle(angle):
    with pytest.raises(ValueError, match="门角"):
        door_skill_angle_range("push_open", {"push_angle_deg": 90.}, angle)


def test_subtask_does_not_insert_task_transition():
    assert door_skill_sequence("open") == ("handle_pull", "push_open")
    assert door_skill_sequence("close") == ("push_close",)


@pytest.mark.parametrize("source", ["manual_input", "mujoco_gt"])
def test_same_angle_produces_same_check_with_explicit_source(source):
    check = door_angle_check(.43, target_rad=.44, tolerance_rad=.02, source=source)
    assert check.status == "pass"
    assert check.source == source
    assert check.evidence["error_rad"] == pytest.approx(.01)


@pytest.mark.parametrize("angle,predicted,status", [
    (None, False, "unknown"), (float("nan"), False, "unknown"),
    (.44, True, "unknown"), (.8, False, "fail"),
])
def test_unknown_prediction_and_failure_are_distinct(angle, predicted, status):
    result = door_angle_check(angle, target_rad=.44, tolerance_rad=.02,
                              source="observation", predicted=predicted)
    assert result.status == status
    assert result.passed is False


@pytest.mark.parametrize("new_id,new_time,want", [
    (12, 101., "pass"), (11, 101., "unknown"),
    (12, 100., "unknown"), (None, 101., "unknown"),
])
def test_stale_frame_cannot_be_used_to_reestimate_angle(new_id, new_time, want):
    check = door_observation_freshness(
        previous_frame=11, previous_time=100., current_frame=new_id,
        current_time=new_time, source="camera",
    )
    assert check.status == want


def test_default_tcp_transition_keeps_home_fk_in_world_and_copies_pose():
    home_fk = np.eye(4)
    home_fk[:3, 3] = [.4, -.2, .7]
    phase = default_tcp_transition(home_fk, door_angle_rad=.6, gripper_width_m=.06)
    np.testing.assert_array_equal(phase.tcp_pose_world[:3, 3], [.4, -.2, .7])
    home_fk[0, 3] = 5.
    assert phase.tcp_pose_world[0, 3] == .4
    assert phase.motion_kind == "linear_tcp"
    assert phase.contact_policy == "door_contact"
    before_reopen = default_tcp_transition(
        np.eye(4), door_angle_rad=0., gripper_width_m=.06,
        phase_id="task/before_reopen/return_default_tcp",
    )
    assert before_reopen.motion_kind == "plan"
    assert before_reopen.contact_policy == "avoid_door"
    before_knob = default_tcp_transition(
        np.eye(4), door_angle_rad=0., gripper_width_m=.06,
        phase_id="task/before_knob/normalize_right_arm",
    )
    assert before_knob.motion_kind == "plan"
    assert before_knob.contact_policy == "avoid_door"


@pytest.mark.parametrize("after_frame,after_angle", [(10, .44), (11, None)])
def test_realworld_skill_reports_unknown_for_stale_or_missing_angle(
    tmp_path, monkeypatch, after_frame, after_angle,
):
    from gene_agent.realworld.microwave import door as module
    from gene_agent.skills.library.door.sequence import DoorPhase

    def scene(frame, angle):
        return {
            "snapshot": {"snapshot_sha256": f"snapshot-{frame}", "active_group": "right"},
            "provenance": {
                "joint_positions_by_group_rad": {"right": [0.]},
                "gripper_widths_m": {"right": .08}, "door_angle_rad": angle,
                "source_frame_number": frame, "source_timestamp_ms": float(frame),
            }, "robot_metadata": {"joint_groups": {"right": ["q"]}},
        }

    phase = DoorPhase("handle_pull/00_release", "handle_pull", "release",
                      np.eye(4), .44, .08, "gripper", "avoid_door")
    targets = {"phases": (phase,)}
    monkeypatch.setattr(module, "build_observed_door_skill_targets", lambda *a, **kw: targets)
    monkeypatch.setattr(module, "rebase_observed_door_skill_targets",
                        lambda *a, **kw: {**targets, "target_rebase": {}})
    monkeypatch.setattr(module, "select_observed_door_phase_candidate",
                        lambda *a, **kw: (targets, None))

    class Runtime:
        def run_gripper_stage(self, *a, **kw):
            return {"source_format": "realworld_gripper_stage", "success": True,
                    "status": "executed", "measured_end_width_m": .08}

    observations = iter((scene(10, .44), scene(after_frame, after_angle)))
    report = module.run_observed_door_skill(
        skill_id="handle_pull", observe_collision_scene=lambda: next(observations),
        make_runtime=lambda _hash: Runtime(), planner_pool=object(),
        output_dir=tmp_path, config_path="unused", execute=True,
        max_microwave_translation_m=.01, max_microwave_rotation_rad=.02,
        max_start_error_rad=.02, max_gripper_width_error_m=.005,
        max_door_angle_error_rad=.03,
    )
    assert report["status"] == "unknown"
    assert report["success"] is False
    assert report["first_failure"]["code"] in {"door_observation_stale", "door_angle_unknown"}
    assert report["stages"][0]["execution"]["status"] == "executed"


def test_missing_initial_angle_stops_unknown_before_planning_or_devices(tmp_path):
    from gene_agent.realworld.microwave import door as module

    def forbidden(_hash):
        raise AssertionError("未知门角不能创建设备")

    report = module.run_observed_door_skill(
        skill_id="push_open", observe_collision_scene=lambda: {"provenance": {}},
        make_runtime=forbidden, planner_pool=object(),
        output_dir=tmp_path, config_path="unused", execute=True,
        max_microwave_translation_m=.01, max_microwave_rotation_rad=.02,
        max_start_error_rad=.02, max_gripper_width_error_m=.005,
        max_door_angle_error_rad=.03,
    )
    assert report["status"] == "unknown"
    assert report["first_failure"]["code"] == "door_angle_unknown"
    assert report["stages"] == []


def test_invalid_observation_metadata_is_unknown_and_can_be_saved():
    import json

    check = door_observation_freshness(
        previous_frame=10, previous_time=100., current_frame=11,
        current_time=float("nan"), source="camera",
    )
    assert check.status == "unknown"
    json.dumps(check.to_dict(), allow_nan=False)
