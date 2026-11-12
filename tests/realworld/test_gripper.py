"""RealMan 夹爪与旧版已验证 SDK 命令的单位和失败边界。"""

import pytest

from gene_agent.realworld.devices.grippers.realman.adapter import RealmanGripperAdapter


class FakeGripperSdk:
    def __init__(self):
        self.position = 1000
        self.calls = []
        self.next_code = 0

    def rm_set_gripper_position(self, position, blocking, mode):
        self.calls.append(("position", position, blocking, mode))
        if self.next_code == 0:
            self.position = position
        return self.next_code

    def rm_get_gripper_state(self):
        return 0, {"actpos": self.position, "status": 0}

    def rm_set_gripper_pick_on(self, speed, force, blocking, mode):
        self.calls.append(("pick_on", speed, force, blocking, mode))
        return self.next_code

    def rm_set_gripper_release(self, speed, blocking, mode):
        self.calls.append(("release", speed, blocking, mode))
        return self.next_code


def test_realman_gripper_maps_meter_width_to_existing_normalized_sdk_command():
    """错误比例会让原有 8 cm 最大开口的 Skill 在真机明显失真。"""
    sdk = FakeGripperSdk()
    gripper = RealmanGripperAdapter(sdk, max_width_m=0.08)

    gripper.command_width_m(0.04)

    assert sdk.calls == [("position", 500, True, 3)]
    assert gripper.read_width_m() == pytest.approx(0.04)


def test_realman_gripper_rejects_out_of_range_or_sdk_failure():
    """非法指令和 SDK 错误都不能当作夹爪已经执行。"""
    sdk = FakeGripperSdk()
    gripper = RealmanGripperAdapter(sdk, max_width_m=0.08)

    with pytest.raises(ValueError, match="开口"):
        gripper.command_width_m(0.09)
    assert not sdk.calls
    sdk.next_code = 17
    with pytest.raises(RuntimeError, match="17"):
        gripper.command_width_m(0.04)
    assert gripper.read_width_m() == pytest.approx(0.08)


def test_realman_gripper_preserves_old_continuous_pick_and_release_calls():
    """抓取和松开仍使用旧项目的 pick_on/release SDK 入口。"""
    sdk = FakeGripperSdk()
    gripper = RealmanGripperAdapter(sdk, max_width_m=0.08)

    gripper.grasp(speed=0.6, force=0.5)
    gripper.release()

    assert sdk.calls == [
        ("pick_on", 600, 500, True, 3),
        ("release", 500, True, 3),
    ]


def test_runtime_gripper_preview_checks_live_start_without_sending_command():
    """只规划模式需核对场景、关节与夹爪实际起点，但不能发 SDK 命令。"""
    from gene_agent.motion.model import JointState
    from gene_agent.realworld.runtime import RealWorldRuntime

    names = tuple(f"right_joint_{i}" for i in range(1, 8))
    start = JointState(names, tuple([0.0] * 7))

    class Robot:
        group = "right"

        def read_joint_state(self):
            return start

    sdk = FakeGripperSdk()
    runtime = RealWorldRuntime(
        Robot(), scene_snapshot_sha256="scene-A", max_start_error_rad=0.02,
        tcp_height_fn=lambda _joints: 0.2, min_tcp_z_m=0.08,
        gripper=RealmanGripperAdapter(sdk, max_width_m=0.08),
    )

    result = runtime.run_gripper_stage(
        "open", group="right", target_width_m=0.04,
        planned_start=start, planned_start_width_m=0.08,
        max_width_error_m=0.005, planned_scene_snapshot_sha256="scene-A",
    )

    assert result["status"] == "validated"
    assert result["measured_start_width_m"] == pytest.approx(0.08)
    assert result["measured_end_width_m"] is None
    assert not sdk.calls


def test_runtime_gripper_executes_only_with_matching_snapshot_and_state():
    """现场已移动或场景版本变化时，不能向夹爪发过时命令。"""
    from gene_agent.motion.model import JointState
    from gene_agent.realworld.runtime import RealWorldRuntime

    names = tuple(f"right_joint_{i}" for i in range(1, 8))
    start = JointState(names, tuple([0.0] * 7))

    class Robot:
        group = "right"

        def __init__(self):
            self.state = start

        def read_joint_state(self):
            return self.state

    robot = Robot()
    sdk = FakeGripperSdk()
    runtime = RealWorldRuntime(
        robot, scene_snapshot_sha256="scene-A", max_start_error_rad=0.02,
        tcp_height_fn=lambda _joints: 0.2, min_tcp_z_m=0.08,
        gripper=RealmanGripperAdapter(sdk, max_width_m=0.08),
    )
    kwargs = dict(
        group="right", target_width_m=0.04, planned_start=start,
        planned_start_width_m=0.08, max_width_error_m=0.005,
        planned_scene_snapshot_sha256="scene-A", execute=True,
    )
    stale = runtime.run_gripper_stage("open", **{**kwargs,
                                                   "planned_scene_snapshot_sha256": "scene-old"})
    assert stale["failure"]["code"] == "scene_mismatch"
    assert not sdk.calls

    robot.state = JointState(names, tuple([0.1] * 7))
    drifted = runtime.run_gripper_stage("open", **kwargs)
    assert drifted["failure"]["code"] == "start_state_mismatch"
    assert not sdk.calls

    robot.state = start
    result = runtime.run_gripper_stage("open", **kwargs)
    assert result["status"] == "executed"
    assert result["measured_end_width_m"] == pytest.approx(0.04)
    assert sdk.calls == [("position", 500, True, 3)]


def test_runtime_gripper_uses_legacy_pick_on_and_release_for_contact_actions():
    """抓住 Bowl/把手时不能把几何闭合宽度误当作 SDK 位置控制。"""
    from gene_agent.motion.model import JointState
    from gene_agent.realworld.runtime import RealWorldRuntime

    names = tuple(f"right_joint_{i}" for i in range(1, 8))
    start = JointState(names, (0.,) * 7)

    class Robot:
        group = "right"

        def read_joint_state(self):
            return start

    sdk = FakeGripperSdk()
    runtime = RealWorldRuntime(
        Robot(), scene_snapshot_sha256="scene-A", max_start_error_rad=.02,
        tcp_height_fn=lambda _q: .2, min_tcp_z_m=.08,
        gripper=RealmanGripperAdapter(sdk, max_width_m=.08),
    )
    common = dict(
        group="right", target_width_m=0., planned_start=start,
        planned_start_width_m=.08, max_width_error_m=.005,
        planned_scene_snapshot_sha256="scene-A", execute=True,
    )
    grasped = runtime.run_gripper_stage(
        "close", command_mode="grasp", grasp_force=.5, **common,
    )
    released = runtime.run_gripper_stage(
        "release", command_mode="release", **common,
    )
    assert grasped["status"] == released["status"] == "executed"
    assert grasped["hardware_command"]["mode"] == "grasp"
    assert sdk.calls == [
        ("pick_on", 1000, 500, True, 3),
        ("release", 500, True, 3),
    ]


def test_motion_stage_failure_reports_measured_partial_stop_state():
    """SDK 运动中途失败后，记录必须给出实测位置以便安全恢复。"""
    import numpy as np

    from gene_agent.motion.model import (
        JointState, JointTrajectory, MotionOperation, MotionRequest, MotionResult,
        PlannerMode, PlanningSceneRef, PoseTarget,
    )
    from gene_agent.realworld.runtime import RealWorldRuntime
    from gene_agent.skills.runtime.planning import SkillStage

    names = tuple(f"right_joint_{i}" for i in range(1, 8))
    start = JointState(names, tuple([0.0] * 7))
    stopped = JointState(names, tuple([0.05] * 7))

    class PartialRobot:
        group = "right"

        def __init__(self):
            self.state = start

        def read_joint_state(self):
            return self.state

        def preflight_trajectory(self, _trajectory, **_kwargs):
            return None

        def execute_trajectory(self, _trajectory, **_kwargs):
            self.state = stopped
            raise RuntimeError("轨迹点 3 SDK 码 17")

    request = MotionRequest(
        request_id="stage-1", operation=MotionOperation.PLAN, group="right",
        start=start, target=PoseTarget("world_tcp", np.eye(4)),
        scene=PlanningSceneRef(1, "scene-A", "full"),
    )
    trajectory = JointTrajectory(
        names, (0.0, 0.01), (tuple([0.0] * 7), tuple([0.1] * 7)),
    )
    result = MotionResult(
        request_id="stage-1", operation=MotionOperation.PLAN,
        success=True, backend="fake", planner_mode=PlannerMode.DEVELOPMENT,
        duration_s=0.01, trajectory=trajectory,
    )
    stage = SkillStage("pregrasp", "planned", request=request, result=result)
    runtime = RealWorldRuntime(
        PartialRobot(), scene_snapshot_sha256="scene-A", max_start_error_rad=0.02,
        tcp_height_fn=lambda _joints: 0.2, min_tcp_z_m=0.08,
    )

    outcome = runtime.run_motion_stage(stage, execute=True)

    assert outcome["failure"]["code"] == "execution_failed"
    assert outcome["measured_end"]["positions_rad"] == pytest.approx([0.05] * 7)
    assert outcome["measured_start"]["positions_rad"] == pytest.approx([0.0] * 7)


def test_realman_stream_stop_on_keyboard_interrupt_after_first_command():
    """操作员中断流式关节轨迹后，应立即停止机械臂并清空命令队列。"""
    from gene_agent.motion.model import JointState, JointTrajectory
    from gene_agent.realworld.devices.arms.realman.adapter import RealmanAdapter
    import numpy as np

    names = tuple(f"right_joint_{i}" for i in range(1, 8))

    class InterruptedSdk:
        def __init__(self):
            self.sent = 0
            self.stopped = 0
            self.cleared = 0

        def rm_get_joint_degree(self):
            return 0, [0.0] * 7

        def rm_algo_forward_kinematics(self, _degrees, flag):
            assert flag == 1
            return [0., 0., .2, 0., 0., 0.]

        def rm_algo_pos2matrix(self, _pose):
            matrix = np.eye(4)
            matrix[2, 3] = .2
            return type("SdkMatrix", (), {"data": matrix.reshape(-1).tolist()})()

        def rm_movej_canfd(self, _degrees, *_options):
            self.sent += 1
            raise KeyboardInterrupt()

        def rm_set_arm_stop(self):
            self.stopped += 1
            return 0

        def rm_set_arm_delete_trajectory(self):
            self.cleared += 1
            return 0

    sdk = InterruptedSdk()
    robot = RealmanAdapter(
        sdk, group="right", joint_names=names, T_flange_tcp=np.eye(4),
    )
    trajectory = JointTrajectory(
        names, (0.0, 0.01), (tuple([0.0] * 7), tuple([0.1] * 7)),
    )

    with pytest.raises(KeyboardInterrupt):
        robot.execute_trajectory(
            trajectory, measured_start=JointState(names, tuple([0.0] * 7)),
            max_start_error_rad=0.02,
            tcp_height_fn=lambda _joints: 0.2, min_tcp_z_m=0.08,
        )

    assert sdk.sent == 1
    assert sdk.stopped == 1
    assert sdk.cleared == 1


def test_realman_connection_uses_old_sdk_handshake_and_owns_disconnect():
    """真机连接需先握手、设置 0～1000 夹爪行程，再显式释放连接。"""
    import numpy as np
    from gene_agent.realworld.devices.arms.realman.connection import connect_realman

    class ConnectionSdk(FakeGripperSdk):
        def __init__(self):
            super().__init__()
            self.created = []
            self.routes = []
            self.deleted = 0

        def rm_create_robot_arm(self, ip, port, level):
            self.created.append((ip, port, level))
            return type("Handle", (), {"id": 7})()

        def rm_set_gripper_route(self, *, min_route, max_route):
            self.routes.append((min_route, max_route))
            return 0

        def rm_delete_robot_arm(self):
            self.deleted += 1
            return 0

    sdk = ConnectionSdk()
    connection = connect_realman(
        "169.254.128.19", group="right", sdk=sdk,
        T_flange_tcp=np.eye(4), max_width_m=0.08,
    )

    assert sdk.created == [("169.254.128.19", 8080, 3)]
    assert sdk.routes == [(0, 1000)]
    assert connection.robot.group == "right"
    assert connection.gripper.max_width_m == pytest.approx(0.08)
    assert not sdk.calls
    connection.close()
    connection.close()
    assert sdk.deleted == 1
