"""发 SDK 命令前的重观测失败不得继续执行任何真机动作。"""

import numpy as np

from gene_agent.motion.model import (
    JointState, JointTrajectory, MotionOperation, MotionRequest,
    MotionResult, PlannerMode, PlanningSceneRef, PoseTarget,
)
from gene_agent.realworld.runtime import RealWorldRuntime
from gene_agent.skills.runtime.planning import SkillStage
from gene_agent.tasks.checks import CheckResult


NAMES = tuple(f"right_joint_{i}" for i in range(1, 8))
START = JointState(NAMES, (0.,) * 7)


class Robot:
    group = "right"

    def __init__(self):
        self.commands = []

    def read_joint_state(self):
        return START

    def preflight_trajectory(self, _trajectory, **_kwargs):
        return None

    def execute_trajectory(self, _trajectory, **_kwargs):
        self.commands.append("motion")
        return START

    def rotate_last_joint(self, _angle, **_kwargs):
        self.commands.append("last_joint")
        return START


class Gripper:
    max_width_m = .08

    def __init__(self):
        self.commands = []

    def read_width_m(self):
        return .08

    def command_width_m(self, width):
        self.commands.append(width)


def _stage():
    request = MotionRequest(
        request_id="motion-1", operation=MotionOperation.PLAN,
        group="right", start=START, target=PoseTarget("world_tcp", np.eye(4)),
        scene=PlanningSceneRef(1, "scene-A", "full"),
    )
    trajectory = JointTrajectory(NAMES, (0., .01), (START.positions_rad,
                                                   tuple([.1] * 7)))
    result = MotionResult(
        request_id="motion-1", operation=MotionOperation.PLAN,
        success=True, backend="fake", planner_mode=PlannerMode.DEVELOPMENT,
        duration_s=.01, trajectory=trajectory,
    )
    return SkillStage("pregrasp", "planned", request=request, result=result)


def _runtime(robot, gripper, check):
    return RealWorldRuntime(
        robot, scene_snapshot_sha256="scene-A", max_start_error_rad=.02,
        tcp_height_fn=lambda _joints: .2, min_tcp_z_m=.08,
        gripper=gripper, pre_execute_scene_check=check,
    )


def _changed_scene():
    return CheckResult("fail", "realworld_reobserved_scene",
                       {"object_translation_m": {"Bowl/Bowl_0": .04}},
                       "Bowl/Bowl_0 pose 已偏离规划值")


def test_motion_guard_rejects_moved_object_before_sdk_command():
    robot, gripper = Robot(), Gripper()
    runtime = _runtime(robot, gripper, _changed_scene)
    result = runtime.run_motion_stage(_stage(), execute=True)
    assert result["failure"]["code"] == "scene_recheck_failed"
    assert result["failure"]["details"]["status"] == "fail"
    assert robot.commands == []


def test_last_joint_and_gripper_guards_stop_commands():
    robot, gripper = Robot(), Gripper()
    runtime = _runtime(robot, gripper, _changed_scene)
    last = runtime.run_last_joint_stage(
        "rotate", group="right", joint_delta_rad=.1, planned_start=START,
        planned_scene_snapshot_sha256="scene-A", execute=True,
    )
    fingers = runtime.run_gripper_stage(
        "close", group="right", target_width_m=.04, planned_start=START,
        planned_start_width_m=.08, max_width_error_m=.005,
        planned_scene_snapshot_sha256="scene-A", execute=True,
    )
    assert last["failure"]["code"] == "scene_recheck_failed"
    assert fingers["failure"]["code"] == "scene_recheck_failed"
    assert robot.commands == [] and gripper.commands == []


def test_preview_does_not_take_extra_camera_frame():
    robot, gripper = Robot(), Gripper()
    runtime = _runtime(robot, gripper,
                       lambda: (_ for _ in ()).throw(AssertionError("预览不能重拍")))
    result = runtime.run_motion_stage(_stage(), execute=False)
    assert result["status"] == "validated"


def test_passing_live_recheck_is_recorded_before_motion():
    robot, gripper = Robot(), Gripper()
    runtime = _runtime(robot, gripper, lambda: CheckResult(
        "pass", "realworld_reobserved_scene", {"current_frame_number": 74},
    ))
    result = runtime.run_motion_stage(_stage(), execute=True)
    assert result["status"] == "executed"
    assert result["scene_recheck"]["evidence"]["current_frame_number"] == 74
    assert robot.commands == ["motion"]


def test_unknown_or_broken_recheck_cannot_authorize_motion():
    for check, code in (
        (lambda: CheckResult("unknown", "camera", {}, "读数不足"),
         "scene_recheck_failed"),
        (lambda: (_ for _ in ()).throw(RuntimeError("相机离线")),
         "scene_recheck_error"),
        (lambda: {"status": "pass"}, "scene_recheck_error"),
    ):
        robot, gripper = Robot(), Gripper()
        result = _runtime(robot, gripper, check).run_motion_stage(
            _stage(), execute=True,
        )
        assert result["failure"]["code"] == code
        assert robot.commands == [] and gripper.commands == []


def test_sdk_failure_still_records_successful_pre_execute_recheck():
    class FailingRobot(Robot):
        def execute_trajectory(self, _trajectory, **_kwargs):
            raise RuntimeError("SDK 断开")

    robot, gripper = FailingRobot(), Gripper()
    runtime = _runtime(robot, gripper, lambda: CheckResult(
        "pass", "realworld_reobserved_scene", {"current_frame_number": 75},
    ))
    result = runtime.run_motion_stage(_stage(), execute=True)
    assert result["failure"]["code"] == "execution_failed"
    assert result["scene_recheck"]["evidence"]["current_frame_number"] == 75


def test_motion_stage_passes_explicit_legacy_speed_profile_to_robot():
    """旧 Task1 的 0.1 比例和 CANFD/TCP 上限必须穿过 Runtime 到 SDK 适配。"""
    class CapturingRobot(Robot):
        def execute_trajectory(self, _trajectory, **kwargs):
            self.options = kwargs
            return START

    robot = CapturingRobot()
    runtime = RealWorldRuntime(
        robot, scene_snapshot_sha256="scene-A", max_start_error_rad=.02,
        tcp_height_fn=lambda _q: .2, min_tcp_z_m=.08,
        motion_speed_scale=.2, max_tcp_speed_mps=.1,
        canfd_frequency_hz=100.,
    )

    report = runtime.run_motion_stage(_stage(), execute=True)

    assert report["status"] == "executed"
    assert robot.options["speed_scale"] == .2
    assert robot.options["max_tcp_speed_mps"] == .1
    assert robot.options["canfd_frequency_hz"] == 100.
    assert report["execution_speed_profile"]["speed_scale"] == .2
