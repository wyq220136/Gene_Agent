"""真机 CANFD 轨迹必须按旧 Task1 的速度与采样语义准备。"""

from types import SimpleNamespace

import numpy as np
import pytest

from gene_agent.motion.model import JointState, JointTrajectory
from gene_agent.realworld.devices.arms.realman.adapter import RealmanAdapter


NAMES = tuple(f"right_joint_{index}" for index in range(1, 8))


class FakeKinematicsSdk:
    """只模拟 SDK FK；第一轴弧度值对应同等米数的 TCP X。"""

    def rm_algo_forward_kinematics(self, joint_degrees, flag):
        assert flag == 1
        return [float(np.deg2rad(joint_degrees[0])), 0., .2, 0., 0., 0.]

    def rm_algo_pos2matrix(self, flange_euler):
        matrix = np.eye(4)
        matrix[:3, 3] = flange_euler[:3]
        return SimpleNamespace(data=matrix.reshape(-1).tolist())


def _robot():
    return RealmanAdapter(
        FakeKinematicsSdk(), group="right", joint_names=NAMES,
        T_flange_tcp=np.eye(4),
    )


def test_prepare_canfd_trajectory_caps_tcp_speed_and_resamples():
    """漏掉 TCP 速度限制或按原规划时戳发送，都会让 .01 m 在 .01 s 走完。"""
    start = (0.,) * 7
    end = (.01,) + (0.,) * 6
    planned = JointTrajectory(NAMES, (0., .01), (start, end))

    sent = _robot().prepare_trajectory_for_execution(
        planned, speed_scale=.5, max_tcp_speed_mps=.1,
        canfd_frequency_hz=10.,
    )

    # 旧算法先缩放为 .02 s，再限制 TCP ≤ .05 m/s，因此总时长 .2 s。
    assert sent.times_s == pytest.approx((0., .1, .2))
    assert [row[0] for row in sent.positions_rad] == pytest.approx((0., .005, .01))
    assert sent.joint_names == NAMES


def test_prepare_canfd_trajectory_scales_rotation_even_without_tcp_translation():
    """纯姿态运动也必须受全局 speed_scale 控制。"""
    start = (0.,) * 7
    end = (0., .1) + (0.,) * 5
    planned = JointTrajectory(NAMES, (0., .1), (start, end))

    sent = _robot().prepare_trajectory_for_execution(
        planned, speed_scale=.1, max_tcp_speed_mps=.1,
        canfd_frequency_hz=10.,
    )

    assert sent.times_s[-1] == pytest.approx(1.)
    assert len(sent.times_s) == 11
    assert sent.positions_rad[-1] == pytest.approx(end)


def test_prepare_rejects_invalid_speed_before_any_motion():
    planned = JointTrajectory(NAMES, (0., .1), ((0.,) * 7, (.01,) + (0.,) * 6))
    with pytest.raises(ValueError, match="speed_scale"):
        _robot().prepare_trajectory_for_execution(
            planned, speed_scale=0., max_tcp_speed_mps=.1,
            canfd_frequency_hz=100.,
        )
