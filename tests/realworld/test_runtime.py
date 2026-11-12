import numpy as np

from gene_agent.motion.model import (
    JointState, JointTrajectory, MotionFailure, MotionFailureKind,
    MotionOperation, MotionRequest, MotionResult, PlannerMode,
    PlanningSceneRef, PoseTarget,
)
from gene_agent.skills.runtime.planning import SkillStage
from gene_agent.realworld.runtime import RealWorldRuntime


NAMES = tuple(f"right_joint_{index}" for index in range(1, 8))
START = JointState(NAMES, tuple([0.0] * 7))


class FakeRobot:
    group = "right"

    def __init__(self):
        self.state = START
        self.calls = 0

    def read_joint_state(self):
        return self.state

    def preflight_trajectory(self, trajectory, **kwargs):
        if trajectory.joint_names != self.state.joint_names:
            raise ValueError("关节顺序不一致")

    def execute_trajectory(self, trajectory, **kwargs):
        self.calls += 1
        self.state = JointState(NAMES, trajectory.positions_rad[-1])
        return self.state


def _stage(*, scene_hash="current", success=True):
    request = MotionRequest(
        request_id="approach-1", operation=MotionOperation.LINEAR_TCP,
        group="right", start=START,
        target=PoseTarget("world_tcp", np.eye(4)),
        scene=PlanningSceneRef(1, scene_hash, "full"),
    )
    trajectory = JointTrajectory(
        NAMES, (0.0, 0.001),
        (tuple([0.0] * 7), tuple([0.1] * 7)),
    )
    result = MotionResult(
        request_id=request.request_id, operation=request.operation,
        success=success, backend="fake", planner_mode=PlannerMode.DEVELOPMENT,
        duration_s=0.01, trajectory=trajectory if success else None,
        failure=None if success else MotionFailure(
            MotionFailureKind.GOAL_IK_NO_SOLUTION, "没有 IK 解"
        ),
    )
    return SkillStage("approach", "planned" if success else "failed",
                      request=request, result=result)


def _runtime(robot, scene_hash="current"):
    return RealWorldRuntime(
        robot, scene_snapshot_sha256=scene_hash,
        max_start_error_rad=0.02,
        tcp_height_fn=lambda joints: 0.2,
        min_tcp_z_m=0.08,
    )


def test_runtime_preview_validates_without_sdk_motion():
    robot = FakeRobot()
    outcome = _runtime(robot).run_motion_stage(_stage())
    assert outcome["status"] == "validated"
    assert outcome["trajectory_points"] == 2
    assert robot.calls == 0


def test_runtime_rejects_stale_scene_and_failed_plan_before_motion():
    robot = FakeRobot()
    stale = _runtime(robot).run_motion_stage(_stage(scene_hash="old"), execute=True)
    assert stale["failure"]["code"] == "scene_mismatch"
    failed = _runtime(robot).run_motion_stage(_stage(success=False), execute=True)
    assert failed["failure"]["code"] == "planner_failed"
    assert robot.calls == 0


def test_runtime_executes_only_explicitly_and_returns_measured_end():
    robot = FakeRobot()
    outcome = _runtime(robot).run_motion_stage(_stage(), execute=True)
    assert outcome["status"] == "executed"
    assert outcome["measured_end"]["positions_rad"][-1] == 0.1
    assert robot.calls == 1


def test_runtime_rejects_live_state_reset_before_execution():
    robot = FakeRobot()
    robot.state = JointState(NAMES, tuple([0.1] * 7))
    outcome = _runtime(robot).run_motion_stage(_stage(), execute=True)
    assert outcome["failure"]["code"] == "start_state_mismatch"
    assert robot.calls == 0
