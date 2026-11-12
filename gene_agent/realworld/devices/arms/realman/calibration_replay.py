"""标定姿态的 SDK MoveL 回放；直线失败不降级为关节运动。"""

from __future__ import annotations

import time

import numpy as np

from gene_agent.knowledge.geometry.transforms import validate_pose
from .connection import RealmanStateConnection
from .adapter import RealmanStateReadError


def _angle_deg(first, second):
    relative = first[:3, :3].T @ second[:3, :3]
    return float(np.rad2deg(np.arccos(np.clip((np.trace(relative) - 1) / 2, -1, 1))))


class ControllerReadError(RuntimeError):
    """显式区分发令前读取失败和运动后校验失败，避免错误地重发运动。"""

    def __init__(self, message: str, *, motion_sent: bool):
        super().__init__(message)
        self.motion_sent = motion_sent


class ControllerFrameReadError(ControllerReadError):
    """工具或工作坐标系读取失败。"""


class RealmanPoseReplayConnection(RealmanStateConnection):
    """不改变控制器工具/工作系；将记录法兰目标转换为当前工作系 TCP 目标。"""

    def _frame(self, getter, label, *, motion_sent=False):
        # SDK -1/-2/-3 为通信失败。只重试读取，不推断未读到的坐标系为单位阵。
        for attempt in range(1, 4):
            code, frame = getter()
            if code == 0 or code not in (-1, -2, -3):
                break
            if attempt < 3:
                print(f"读取当前{label}坐标系通信失败（SDK {code}），{attempt}/3，0.3秒后重试。", flush=True)
                time.sleep(0.3)
        if code != 0:
            stage = "运动已发出，拒绝到位结果并停止" if motion_sent else "未发送运动命令"
            raise ControllerFrameReadError(
                f"无法读取当前{label}坐标系，尝试 {attempt} 次，SDK 码 {code}；{stage}",
                motion_sent=motion_sent,
            )
        raw = np.asarray(frame.get("pose"), dtype=float)
        if raw.shape != (6,) or not np.isfinite(raw).all():
            raise ValueError(f"当前{label}坐标系缺少有效米/弧度位姿")
        matrix = self.sdk.rm_algo_pos2matrix(raw.tolist())
        transform = validate_pose(np.asarray(matrix.data, dtype=float).reshape(4, 4))
        return transform, frame

    def move_to_flange(self, T_sdk_flange, *, speed_percent: int = 5,
                       timeout_s: float = 60.0, position_tolerance_mm: float = 2.0,
                       rotation_tolerance_deg: float = 0.5, expected_start_joints_rad=None) -> dict:
        target = validate_pose(np.asarray(T_sdk_flange, dtype=float), field_name="T_sdk_flange_goal")
        limits = [timeout_s, position_tolerance_mm, rotation_tolerance_deg]
        if (not isinstance(speed_percent, int) or not 1 <= speed_percent <= 10
                or not np.isfinite(limits).all() or min(limits) <= 0):
            raise ValueError("MoveL 速度须为整数1–10%，超时与到位阈值须为有限正数")
        tool, tool_record = self._frame(self.sdk.rm_get_current_tool_frame, "工具")
        work, work_record = self._frame(self.sdk.rm_get_current_work_frame, "工作")
        # MoveL 的目标是当前工作系中的当前工具 TCP；不能直接发送 SDK base 下的法兰 pose。
        tcp_target = target @ tool
        work_tcp = validate_pose(np.linalg.inv(work) @ tcp_target)
        matrix = self.sdk.rm_algo_pos2matrix([0.0] * 6)
        for index, value in enumerate(work_tcp.reshape(-1)):
            matrix.data[index] = float(value)
        command = np.asarray(self.sdk.rm_algo_matrix2pos(matrix, flag=1), dtype=float)
        if command.shape != (6,) or not np.isfinite(command).all():
            raise ValueError("SDK 无法将 MoveL 目标转换为有效米/弧度位姿")
        try:
            before = np.asarray(self.robot.read_joint_state().positions_rad, dtype=float)
        except RealmanStateReadError as exc:
            raise ControllerReadError(f"{exc}；未发送运动命令", motion_sent=False) from exc
        if expected_start_joints_rad is not None:
            expected = np.asarray(expected_start_joints_rad, dtype=float)
            if expected.shape != (7,) or not np.isfinite(expected).all():
                raise ValueError("确认前起点必须是七个有限弧度值")
            if np.rad2deg(np.max(np.abs(before - expected))) > 0.5:
                raise ValueError("确认后机械臂起点已变化，未发送 MoveL 命令")
        sent = False
        try:
            deadline = time.monotonic() + timeout_s
            sent = True
            code = self.sdk.rm_movel(command.tolist(), speed_percent, 0, 0, block=0)
            uncertain = code in (-1, -2, -3)
            if code != 0 and not uncertain:
                raise RuntimeError(f"MoveL 直线失败，SDK 码 {code}；不重发、不降级为 MoveJ")
            if uncertain:
                print(f"MoveL 回执通信异常（SDK {code}），命令可能已执行；保持原连接检查实测到位，不重发。", flush=True)
            settled = 0
            while True:
                try:
                    actual_joints = np.asarray(self.robot.read_joint_state().positions_rad, dtype=float)
                except RealmanStateReadError as exc:
                    raise ControllerReadError(f"{exc}；运动已经发出，停止并拒绝判定到位", motion_sent=True) from exc
                actual_flange = self.robot.flange_pose_from_sdk_fk(actual_joints)
                actual_tcp = actual_flange @ tool
                position_mm = float(np.linalg.norm(actual_tcp[:3, 3] - tcp_target[:3, 3]) * 1000)
                angle_deg = _angle_deg(actual_tcp, tcp_target)
                if time.monotonic() >= deadline:
                    raise RuntimeError(f"MoveL 到位超时：TCP 误差 {position_mm:.2f} mm / {angle_deg:.2f}°；不降级")
                settled = settled + 1 if position_mm <= position_tolerance_mm and angle_deg <= rotation_tolerance_deg else 0
                if settled >= 3:
                    current_tool, _ = self._frame(self.sdk.rm_get_current_tool_frame, "工具", motion_sent=True)
                    current_work, _ = self._frame(self.sdk.rm_get_current_work_frame, "工作", motion_sent=True)
                    if not np.allclose(tool, current_tool, atol=1e-6, rtol=0) or not np.allclose(work, current_work, atol=1e-6, rtol=0):
                        raise RuntimeError("MoveL 期间控制器工具或工作坐标系变化，拒绝到位结果")
                    return {
                        "backend": "realman_movel", "path_type": "linear_current_tool_tcp",
                        "sdk_command_return_code": code, "acknowledgement_uncertain": uncertain,
                        "endpoint_verified": True,
                        "T_sdk_flange_goal": target.tolist(), "command_work_tcp_m_rad": command.tolist(),
                        "controller_tool_frame": tool_record, "controller_work_frame": work_record,
                        "measured_start_joints_rad": before.tolist(), "measured_joints_rad": actual_joints.tolist(),
                        "measured_T_sdk_flange": actual_flange.tolist(), "max_position_error_mm": position_mm,
                        "rotation_error_deg": angle_deg, "sdk_speed_percent": speed_percent,
                        "collision_checked": False, "fallback_used": False,
                    }
                time.sleep(0.05)
        except BaseException as error:
            if sent:
                try:
                    self.robot.stop()
                except Exception as stop_error:
                    raise RuntimeError(f"MoveL 失败且停止失败：{stop_error}") from error
            raise
