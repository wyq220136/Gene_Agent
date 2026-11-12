"""同步视觉运行的状态传递必须与轨迹阶段、夹爪及 attach 事件一致。"""

import numpy as np
import pytest

from gene_agent.visualization.skill_inspector.planned_timeline import PlannedSegment
from gene_agent.visualization.viser_sim.model import VisualRun, VisualStage
from gene_agent.visualization.viser_sim.playback import VisualPlayback, VisualState


def _pose(x: float) -> np.ndarray:
    result = np.eye(4)
    result[0, 3] = x
    return result


def _tcp(_names: tuple[str, ...], joints: np.ndarray) -> np.ndarray:
    return _pose(float(joints[0]))


def _motion(stage_id: str, begin: float, end: float, *, attached: bool) -> VisualStage:
    segment = PlannedSegment(stage_id, "bowl", ("arm_joint",),
                             np.array([[begin], [end]]), 1.0, {})
    carry = ("bowl",) if attached else ()
    return VisualStage(stage_id, "bowl", "plan", True, trajectory=segment,
                       attach_before=carry, attach_after=carry)


def test_attached_bowl_follows_tcp_then_stays_at_release_pose():
    stages = (
        _motion("move_to_bowl", 0.0, 1.0, attached=False),
        VisualStage("close", "bowl", "gripper", True, gripper_command="close",
                    gripper_width_m=.007,
                    attach_before=(), attach_after=()),
        VisualStage("attach", "bowl", "attachment", True,
                    attach_before=(), attach_after=("bowl",)),
        _motion("carry", 1.0, 2.0, attached=True),
        VisualStage("release", "bowl", "gripper", True, gripper_command="release",
                    gripper_width_m=.08,
                    attach_before=("bowl",), attach_after=("bowl",)),
        VisualStage("detach", "bowl", "attachment", True,
                    attach_before=("bowl",), attach_after=()),
        _motion("retreat", 2.0, 2.5, attached=False),
    )
    initial = VisualState(("arm_joint",), np.array([0.0]), gripper_width_m=.08,
                          object_poses_world={"bowl": _pose(.2)})
    playback = VisualPlayback(VisualRun("insert_bowl", 0, stages), initial, _tcp)

    before_attach = playback.sample(playback.stage_start_s("close"))
    carrying = playback.sample(playback.stage_start_s("carry") + .5)
    after_release = playback.sample(playback.duration_s)
    back_to_carry = playback.sample(playback.stage_start_s("carry") + .5)

    assert before_attach.state.object_poses_world["bowl"][0, 3] == .2
    assert np.isclose(carrying.state.object_poses_world["bowl"][0, 3], .7)
    assert carrying.state.attachments == ("bowl",)
    assert carrying.state.gripper_width_m == .007
    assert np.isclose(after_release.state.object_poses_world["bowl"][0, 3], 1.2)
    assert after_release.state.attachments == ()
    assert after_release.state.gripper_width_m == .08
    assert np.isclose(back_to_carry.state.object_poses_world["bowl"][0, 3], .7)
    assert np.isclose(initial.object_poses_world["bowl"][0, 3], .2)


def test_door_angle_changes_only_when_waypoint_motion_completes():
    stage = VisualStage("push_to_45", "door", "linear_tcp", True,
                        trajectory=PlannedSegment("push_to_45", "door", ("arm_joint",),
                                                  np.array([[0.], [1.]]), 1., {}),
                        door_angle_rad=np.pi / 4)
    initial = VisualState(("arm_joint",), np.array([0.]), gripper_width_m=.08,
                          object_poses_world={}, door_angle_rad=0.)
    playback = VisualPlayback(VisualRun("push_open", 0, (stage,)), initial, _tcp)

    assert playback.sample(.5).state.door_angle_rad == 0.
    assert np.isclose(playback.sample(1.).state.door_angle_rad, np.pi / 4)


def test_place_event_updates_only_named_object_pose():
    stage = VisualStage("place", "place_bowl", "gripper", True,
                        gripper_command="release", gripper_width_m=.08,
                        object_instance_id="bowl", object_pose_world=_pose(3.))
    initial = VisualState(("arm_joint",), np.array([0.]), gripper_width_m=.007,
                          object_poses_world={"bowl": _pose(.2), "plate": _pose(.4)})
    state = VisualPlayback(VisualRun("place_bowl", 0, (stage,)), initial, _tcp).sample(0.).state

    assert state.object_poses_world["bowl"][0, 3] == 3.
    assert state.object_poses_world["plate"][0, 3] == .4


def test_first_planned_motion_cannot_silently_reset_robot_from_scene_initial_state():
    stage = _motion("pregrasp", 0., 1., attached=False)
    initial = VisualState(("arm_joint",), np.array([.3]), gripper_width_m=.08)

    with pytest.raises(ValueError, match="初始关节状态不连续"):
        VisualPlayback(VisualRun("pick_bowl", 0, (stage,)), initial, _tcp)


def test_door_and_gripper_follow_saved_trajectory_channels_during_motion():
    segment = PlannedSegment(
        "pull", "handle_pull", ("arm_joint",),
        np.array([[0.], [.5], [1.]]), .5,
        {"door_angle_rad": np.array([0., .2, .4]),
         "gripper_width_m": np.array([.08, .04, .01])},
    )
    stage = VisualStage("pull", "handle_pull", "linear_tcp", True,
                        trajectory=segment, door_angle_rad=.4)
    initial = VisualState(("arm_joint",), np.array([0.]), .08, door_angle_rad=0.)
    playback = VisualPlayback(VisualRun("handle_pull", 0, (stage,)), initial, _tcp)

    middle = playback.sample(.5).state
    assert np.isclose(middle.door_angle_rad, .2)
    assert np.isclose(middle.gripper_width_m, .04)
    assert np.isclose(middle.joint_positions_rad[0], .5)


def test_state_only_gripper_stage_preserves_saved_duration_and_interpolates():
    stage = VisualStage("close", "handle_pull", "gripper", True,
                        gripper_command="close", gripper_width_m=.01,
                        state_duration_s=.25)
    initial = VisualState(("arm_joint",), np.array([0.]), .08)
    playback = VisualPlayback(VisualRun("handle_pull", 0, (stage,)), initial, _tcp)

    assert np.isclose(playback.duration_s, .25)
    assert np.isclose(playback.sample(.125).state.gripper_width_m, .045)
    assert np.isclose(playback.sample(.25).state.gripper_width_m, .01)


def test_failed_motion_does_not_apply_unreached_target_door_or_object_state():
    failed = VisualStage("push", "push_open", "linear_tcp", False,
                         door_angle_rad=.8,
                         object_instance_id="bowl", object_pose_world=_pose(3.),
                         failure={"code": "goal_ik_no_solution"})
    initial = VisualState(("arm_joint",), np.array([0.]), .08,
                          {"bowl": _pose(.2)}, door_angle_rad=.1)
    playback = VisualPlayback(VisualRun("push_open", 0, (failed,)), initial, _tcp)

    state = playback.sample(playback.duration_s).state
    assert state.door_angle_rad == .1
    assert state.object_poses_world["bowl"][0, 3] == .2
    assert state.joint_positions_rad[0] == 0.
