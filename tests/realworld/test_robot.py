import numpy as np
import pytest

from gene_agent.motion.model import JointState, JointTrajectory
from gene_agent.realworld.devices.arms.realman.adapter import RealmanAdapter, load_standard_gripper_flange_tcp


NAMES = tuple(f"right_joint_{index}" for index in range(1, 8))


class FakeSdk:
    def __init__(self):
        self.degrees = [0.0] * 7
        self.commands = []
        self.stops = 0
        self.fail_command_index = None

    def rm_get_joint_degree(self):
        return 0, list(self.degrees)

    def rm_algo_forward_kinematics(self, _degrees, flag):
        assert flag == 1
        return [0., 0., .2, 0., 0., 0.]

    def rm_algo_pos2matrix(self, _pose):
        matrix = np.eye(4)
        matrix[2, 3] = .2
        return type("SdkMatrix", (), {"data": matrix.reshape(-1).tolist()})()

    def rm_movej_canfd(self, degrees, *options):
        self.commands.append((list(degrees), options))
        if len(self.commands) == self.fail_command_index:
            return 17
        self.degrees = list(degrees)
        return 0

    def rm_set_arm_stop(self):
        self.stops += 1
        return 0


def _adapter(sdk):
    T_flange_tcp = np.eye(4)
    T_flange_tcp[2, 3] = 0.15
    return RealmanAdapter(sdk, group="right", joint_names=NAMES,
                          T_flange_tcp=T_flange_tcp)


def _trajectory(first=None, second=None):
    start = [0.0] * 7 if first is None else first
    end = [0.1] * 7 if second is None else second
    return JointTrajectory(NAMES, (0.0, 0.00001), (tuple(start), tuple(end)))


def test_realman_reads_sdk_degrees_as_named_radians_and_composes_tcp():
    sdk = FakeSdk()
    sdk.degrees[0] = 90.0
    adapter = _adapter(sdk)
    state = adapter.read_joint_state()
    assert state.joint_names == NAMES
    assert state.positions_rad[0] == pytest.approx(np.pi / 2)
    flange = np.eye(4)
    flange[0, 3] = 0.3
    assert adapter.tcp_from_flange_pose(flange)[2, 3] == pytest.approx(0.15)


def test_current_gripper_flange_tcp_matches_frozen_robot_config():
    transform = load_standard_gripper_flange_tcp()
    np.testing.assert_allclose(transform[:3, 3], [0.0, 0.0, 0.15])
    np.testing.assert_allclose(transform[:3, :3],
                               np.diag([-1.0, -1.0, 1.0]), atol=1e-12)


def test_tcp_height_uses_sdk_fk_and_loaded_flange_tcp():
    class FkSdk(FakeSdk):
        def rm_algo_forward_kinematics(self, degrees, flag):
            assert flag == 1
            assert degrees[0] == pytest.approx(90.0)
            return [0.0, 0.0, 0.2, 0.0, 0.0, 0.0]

        def rm_algo_pos2matrix(self, pose):
            matrix = np.eye(4)
            matrix[2, 3] = pose[2]
            return type("SdkMatrix", (), {"data": matrix.reshape(-1).tolist()})()

    adapter = RealmanAdapter(FkSdk(), group="right", joint_names=NAMES,
                             T_flange_tcp=load_standard_gripper_flange_tcp())
    assert adapter.tcp_height_from_sdk_fk([np.pi / 2, *([0.0] * 6)]) == pytest.approx(0.35)


def test_realman_rejects_joint_order_and_start_mismatch_before_any_command():
    sdk = FakeSdk()
    adapter = _adapter(sdk)
    wrong = JointTrajectory(tuple(reversed(NAMES)), (0.0,), (tuple([0.0] * 7),))
    with pytest.raises(ValueError, match="关节顺序"):
        adapter.preflight_trajectory(wrong, measured_start=adapter.read_joint_state(),
                                     max_start_error_rad=0.02)
    with pytest.raises(ValueError, match="起点"):
        adapter.preflight_trajectory(_trajectory(first=[0.1] * 7),
                                     measured_start=adapter.read_joint_state(),
                                     max_start_error_rad=0.02)
    assert sdk.commands == []


def test_realman_rejects_entire_path_if_later_waypoint_violates_height():
    sdk = FakeSdk()
    adapter = _adapter(sdk)
    with pytest.raises(ValueError, match="第 1 个点.*高度"):
        adapter.execute_trajectory(
            _trajectory(), measured_start=adapter.read_joint_state(),
            max_start_error_rad=0.02,
            tcp_height_fn=lambda joints: 0.12 if joints[0] == 0 else 0.04,
            min_tcp_z_m=0.08,
        )
    assert sdk.commands == []


def test_realman_streams_degrees_only_after_full_preflight():
    sdk = FakeSdk()
    adapter = _adapter(sdk)
    result = adapter.execute_trajectory(
        _trajectory(), measured_start=adapter.read_joint_state(),
        max_start_error_rad=0.02,
        tcp_height_fn=lambda joints: 0.2, min_tcp_z_m=0.08,
    )
    assert result.positions_rad == pytest.approx(tuple([0.1] * 7))
    assert len(sdk.commands) == 2
    assert sdk.commands[-1][0][0] == pytest.approx(np.rad2deg(0.1))


def test_realman_rejects_missing_height_checker_for_execution():
    sdk = FakeSdk()
    adapter = _adapter(sdk)
    with pytest.raises(ValueError, match="高度"):
        adapter.execute_trajectory(_trajectory(),
                                   measured_start=adapter.read_joint_state(),
                                   max_start_error_rad=0.02)
    assert sdk.commands == []


def test_realman_stops_after_first_failed_sdk_waypoint():
    sdk = FakeSdk()
    sdk.fail_command_index = 1
    adapter = _adapter(sdk)
    with pytest.raises(RuntimeError, match="SDK 码 17"):
        adapter.execute_trajectory(
            _trajectory(), measured_start=adapter.read_joint_state(),
            max_start_error_rad=0.02,
            tcp_height_fn=lambda joints: 0.2, min_tcp_z_m=0.08,
        )
    assert len(sdk.commands) == 1
    assert sdk.stops == 1


def test_execution_guard_failure_mid_trajectory_stops_arm(monkeypatch):
    sdk=FakeSdk();adapter=_adapter(sdk)
    monkeypatch.setattr(adapter,'prepare_trajectory_for_execution',lambda trajectory,**kw:trajectory)
    counter=0
    def guard():
        nonlocal counter
        counter+=1
        if counter==2:raise RuntimeError('Pika state stale')
    with pytest.raises(RuntimeError,match='Pika'):
        adapter.execute_trajectory(_trajectory(),measured_start=adapter.read_joint_state(),
            max_start_error_rad=.02,tcp_height_fn=lambda q:.3,min_tcp_z_m=.1,execution_guard=guard)
    assert len(sdk.commands)==1
    assert sdk.stops==1
