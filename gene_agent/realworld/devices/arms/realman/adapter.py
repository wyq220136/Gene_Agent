"""RealMan SDK 的同步薄适配；运动前检查整条轨迹。

此类接收已经连接的 SDK 对象，不在 import 或构造时连接机械臂。
使用者必须显式调用 ``execute_trajectory`` 才会下发关节命令。
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import time
from typing import Callable, Sequence

import numpy as np

from gene_agent.knowledge.geometry import validate_pose
from gene_agent.motion.model import JointState, JointTrajectory


def load_standard_gripper_flange_tcp(path: str | Path | None = None) -> np.ndarray:
    """从仿真已使用的配置读取 ``T_flange_tcp``，避免真机另设数值。

    这里只读取标准夹爪 profile；换 gripper 时必须提供对应安装标定，
    不能把本值当成所有机械臂/夹爪的公共默认值。
    """
    source = (Path(__file__).resolve().parents[5]
              / "config/robots/standard_gripper.json") if path is None else Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    theta = math.radians(float(payload["flange_to_tcp_rotation_z_deg"]))
    height = float(payload["flange_to_tcp_z_m"])
    if not math.isfinite(theta) or not math.isfinite(height):
        raise ValueError("法兰到 TCP 配置必须为有限数")
    transform = np.eye(4)
    transform[:3, :3] = [[math.cos(theta), -math.sin(theta), 0],
                         [math.sin(theta), math.cos(theta), 0],
                         [0, 0, 1]]
    transform[2, 3] = height
    return validate_pose(transform, field_name="T_flange_tcp")


class RealmanStateReadError(RuntimeError):
    """保留 SDK 错误码，供调用方区分通信丢失与发令阶段。"""

    def __init__(self, code: int, attempts: int, *, detail: str = ""):
        super().__init__(f"RealMan 关节状态读取失败，SDK 码 {code}，已尝试 {attempts} 次" + (f"；{detail}" if detail else ""))
        self.code = code
        self.attempts = attempts


class RealmanAdapter:
    """一个 SDK arm group；其他型号机器人可提供相同上层行为。"""

    def __init__(self, sdk, *, group: str, joint_names: Sequence[str],
                 T_flange_tcp: np.ndarray, joint_state_reader=None):
        if group not in {"left", "right"}:
            raise ValueError("RealMan group 必须是 left 或 right")
        names = tuple(joint_names)
        expected = tuple(f"{group}_joint_{index}" for index in range(1, 8))
        if names != expected:
            raise ValueError(f"RealMan 关节顺序必须与 SDK 七轴一致: {expected}")
        self.sdk = sdk
        self.group = group
        self.joint_names = names
        self.T_flange_tcp = validate_pose(T_flange_tcp, field_name="T_flange_tcp")
        self.joint_state_reader = joint_state_reader

    def read_joint_state(self) -> JointState:
        """SDK 顺序1～7、单位度；通信失败最多读3次，不复用旧读数。"""
        if self.joint_state_reader is not None:
            values = np.asarray(self.joint_state_reader(), dtype=float)
            if values.shape != (7,) or not np.isfinite(values).all():
                raise ValueError("实时状态未提供七个有限关节弧度值")
            return JointState(self.joint_names, tuple(values))
        for attempt in range(1, 4):
            status, degrees = self.sdk.rm_get_joint_degree()
            if status == 0 or status not in (-1, -2, -3):
                break
            if attempt < 3:
                print(f"关节状态通信读取失败（SDK {status}），{attempt}/3，0.1秒后重读。", flush=True)
                time.sleep(0.1)
        if status != 0:
            raise RealmanStateReadError(status, attempt)
        values = np.asarray(degrees, dtype=float)
        if values.shape != (7,) or not np.isfinite(values).all():
            raise ValueError("RealMan 返回的关节状态不是七个有限角度")
        return JointState(self.joint_names, tuple(np.deg2rad(values)))

    def tcp_from_flange_pose(self, T_sdk_flange: np.ndarray) -> np.ndarray:
        """显式采用 ``T_sdk_tcp = T_sdk_flange @ T_flange_tcp``。"""
        flange = validate_pose(T_sdk_flange, field_name="T_sdk_flange")
        return validate_pose(flange @ self.T_flange_tcp, field_name="T_sdk_tcp")

    def tcp_pose_from_sdk_fk(self, joints_rad: Sequence[float]) -> np.ndarray:
        """用厂家 FK 和已校准安装变换得到 SDK base 下的 TCP pose。"""
        return self.tcp_from_flange_pose(self.flange_pose_from_sdk_fk(joints_rad))

    def flange_pose_from_sdk_fk(self, joints_rad: Sequence[float]) -> np.ndarray:
        """厂家 FK 的原始法兰位姿 T_sdk_flange，不叠加夹爪 TCP 外参。"""
        joints = np.asarray(joints_rad, dtype=float)
        if joints.shape != (7,) or not np.isfinite(joints).all():
            raise ValueError("SDK FK 需要七个有限关节角，单位 rad")
        flange_euler = self.sdk.rm_algo_forward_kinematics(
            np.rad2deg(joints).tolist(), flag=1,
        )
        raw = np.asarray(flange_euler, dtype=float)
        if raw.shape != (6,) or not np.isfinite(raw).all():
            raise ValueError("RealMan SDK FK 未返回有效 6D flange pose")
        matrix = self.sdk.rm_algo_pos2matrix(raw.tolist())
        values = np.asarray(matrix.data, dtype=float)
        if values.size != 16:
            raise ValueError("RealMan SDK FK 矩阵必须有 16 个元素")
        return validate_pose(values.reshape(4, 4), field_name="T_sdk_flange")

    def tcp_height_from_sdk_fk(self, joints_rad: Sequence[float]) -> float:
        """用厂家 FK 计算 TCP 高度，避免混用 MuJoCo frame。"""
        return float(self.tcp_pose_from_sdk_fk(joints_rad)[2, 3])

    def prepare_trajectory_for_execution(
        self, trajectory: JointTrajectory, *, speed_scale: float = 0.1,
        max_tcp_speed_mps: float = 0.1, canfd_frequency_hz: float = 100.0,
        tcp_speed_mps: float | None = None,
    ) -> JointTrajectory:
        """沿用旧 Task1 的缩时、TCP 速度上限与固定 CANFD 点频率。

        此处只改变关节轨迹的时间采样，不改变几何路径。默认值来自旧
        `configs/curobo/realman_execution.yaml` 与 Task1 的 0.1 速度比例；
        现场可以覆盖，但必须在发送任何点之前完成验证和整段预检。
        """
        scale = float(speed_scale)
        maximum = float(max_tcp_speed_mps)
        frequency = float(canfd_frequency_hz)
        if not math.isfinite(scale) or not 0.0 < scale <= 1.0:
            raise ValueError("speed_scale 必须位于 (0, 1] 区间")
        if not math.isfinite(maximum) or maximum <= 0.0:
            raise ValueError("max_tcp_speed_mps 必须为正有限数")
        if not math.isfinite(frequency) or frequency <= 0.0:
            raise ValueError("canfd_frequency_hz 必须为正有限数")
        requested = maximum * scale if tcp_speed_mps is None else float(tcp_speed_mps)
        if not math.isfinite(requested) or not 0.0 < requested <= maximum:
            raise ValueError("tcp_speed_mps 必须位于 (0, max_tcp_speed_mps] 区间")
        if trajectory.joint_names != self.joint_names:
            raise ValueError("轨迹关节顺序与 RealMan SDK 顺序不一致")

        # 旧版先把原轨迹时间按 speed_scale 拉长，再用相邻 TCP 位移
        # 限制平移速度；纯旋转时也会保留第一步减速。
        scaled_times = np.asarray(trajectory.times_s, dtype=float) / scale
        tcp_positions = np.asarray([
            self.tcp_pose_from_sdk_fk(joints)[:3, 3]
            for joints in trajectory.positions_rad
        ], dtype=float)
        if len(scaled_times) < 2:
            return JointTrajectory(trajectory.joint_names,
                                   tuple(scaled_times), trajectory.positions_rad)
        nominal = np.diff(scaled_times)
        distance = np.linalg.norm(np.diff(tcp_positions, axis=0), axis=1)
        durations = np.maximum(nominal, distance / requested)
        retimed = np.concatenate((scaled_times[:1],
                                  scaled_times[0] + np.cumsum(durations)))
        period = 1.0 / frequency
        count = int(np.floor((retimed[-1] - retimed[0]) / period))
        sample_times = retimed[0] + np.arange(count + 1, dtype=float) * period
        if sample_times[-1] < retimed[-1] - 1e-10:
            sample_times = np.append(sample_times, retimed[-1])
        else:
            sample_times[-1] = retimed[-1]
        positions = np.column_stack([
            np.interp(sample_times, retimed, np.asarray(trajectory.positions_rad)[:, index])
            for index in range(len(self.joint_names))
        ])
        return JointTrajectory(
            self.joint_names, tuple(sample_times),
            tuple(tuple(row) for row in positions),
        )

    def preflight_trajectory(
        self, trajectory: JointTrajectory, *, measured_start: JointState,
        max_start_error_rad: float,
        tcp_height_fn: Callable[[Sequence[float]], float] | None = None,
        min_tcp_z_m: float | None = None,
    ) -> None:
        """在发令之前一次性检查所有点；绝不截断轨迹后继续任务。"""
        if trajectory.joint_names != self.joint_names:
            raise ValueError("轨迹关节顺序与 RealMan SDK 顺序不一致")
        if measured_start.joint_names != self.joint_names:
            raise ValueError("实测关节顺序与 RealMan SDK 顺序不一致")
        tolerance = float(max_start_error_rad)
        if not math.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("起点误差阈值必须为正有限数")
        start_error = float(np.max(np.abs(
            np.asarray(trajectory.positions_rad[0])
            - np.asarray(measured_start.positions_rad)
        )))
        if start_error > tolerance:
            raise ValueError(
                f"轨迹起点与实测关节不一致：最大误差 {start_error:.5f} rad，"
                f"阈值 {tolerance:.5f} rad"
            )
        if (tcp_height_fn is None) != (min_tcp_z_m is None):
            raise ValueError("TCP 高度检查函数与最低高度必须同时提供")
        if tcp_height_fn is None:
            return
        minimum = float(min_tcp_z_m)
        if not math.isfinite(minimum):
            raise ValueError("最低 TCP 高度必须为有限数")
        for index, joints in enumerate(trajectory.positions_rad):
            height = float(tcp_height_fn(joints))
            if not math.isfinite(height) or height < minimum:
                raise ValueError(
                    f"轨迹第 {index} 个点违反 TCP 高度限制："
                    f"{height:.4f} m < {minimum:.4f} m"
                )

    def rotate_last_joint(
        self, angle_rad: float, *, measured_start: JointState,
        speed_scale: float, max_start_error_rad: float,
        endpoint_timeout_s: float = 5.0,
    ) -> JointState:
        """沿旧真机 `rm_movej` 路径，只改变最后一个关节的目标。

        命令前重读七轴，以免检查后起点被其他控制源改动。旧版的
        `joint_speed=20` 与 `speed_scale` 换算沿用，不把该命令伪装成
        已经由 cuRobo 检查过碰撞的轨迹。高度预检由 Runtime 完成。
        """
        delta, speed = float(angle_rad), float(speed_scale)
        if not math.isfinite(delta) or not math.isfinite(speed) or speed <= 0:
            raise ValueError("末关节增量和速度必须为有限正数")
        if measured_start.joint_names != self.joint_names:
            raise ValueError("末关节命令的关节顺序与 SDK 不一致")
        current = self.read_joint_state()
        if float(np.max(np.abs(
            np.asarray(current.positions_rad) - measured_start.positions_rad
        ))) > max_start_error_rad:
            raise ValueError("末关节发令前实测起点已变化")
        target = np.asarray(current.positions_rad, dtype=float).copy()
        target[-1] += delta
        sdk_speed = max(1, min(100, int(round(20 * speed))))
        sent = False
        try:
            # 旧版向 SDK 只发送一次七轴关节目标，前六轴保持当前读数。
            sent = True
            code = self.sdk.rm_movej(np.rad2deg(target).tolist(), sdk_speed,
                                     0, 0, block=0)
            if code != 0:
                raise RuntimeError(f"RealMan 末关节命令失败，SDK 码 {code}")
            deadline = time.monotonic() + endpoint_timeout_s
            while True:
                actual = self.read_joint_state()
                error = float(np.max(np.abs(
                    np.asarray(actual.positions_rad) - target
                )))
                if error <= max_start_error_rad:
                    return actual
                if time.monotonic() >= deadline:
                    raise RuntimeError(f"末关节目标未到达，最大关节误差 {error:.5f} rad")
                time.sleep(0.02)
        except BaseException as error:
            if sent:
                try:
                    self.stop()
                except Exception as stop_error:
                    raise RuntimeError(f"末关节异常后急停失败：{stop_error}") from error
            raise

    def stop(self) -> None:
        """停止运动并清掉 SDK 队列，失败时不自动回 home。"""
        result = self.sdk.rm_set_arm_stop()
        if result not in (None, 0):
            raise RuntimeError(f"RealMan 停止失败，SDK 码 {result}")
        clear = getattr(self.sdk, "rm_set_arm_delete_trajectory", None)
        if callable(clear):
            result = clear()
            if result not in (None, 0):
                raise RuntimeError(f"RealMan 清空轨迹失败，SDK 码 {result}")

    def execute_trajectory(
        self, trajectory: JointTrajectory, *, measured_start: JointState,
        max_start_error_rad: float,
        tcp_height_fn: Callable[[Sequence[float]], float] | None = None,
        min_tcp_z_m: float | None = None,
        endpoint_tolerance_rad: float = 0.02,
        endpoint_timeout_s: float = 5.0,
        speed_scale: float = 0.1,
        max_tcp_speed_mps: float = 0.1,
        canfd_frequency_hz: float = 100.0,
        tcp_speed_mps: float | None = None,
        execution_guard: Callable[[], None] | None = None,
    ) -> JointState:
        """显式下发已完整预检的轨迹，并读取实际终点。

        高度阈值和对应 FK 函数是执行必填项。旧版遇到中途低点会
        截断并继续后续 Task，这里将整段拒绝以免状态交接失真。
        """
        if tcp_height_fn is None or min_tcp_z_m is None:
            raise ValueError("真机执行必须提供整段 TCP 高度检查")
        trajectory = self.prepare_trajectory_for_execution(
            trajectory, speed_scale=speed_scale,
            max_tcp_speed_mps=max_tcp_speed_mps,
            canfd_frequency_hz=canfd_frequency_hz,
            tcp_speed_mps=tcp_speed_mps,
        )
        self.preflight_trajectory(
            trajectory, measured_start=measured_start,
            max_start_error_rad=max_start_error_rad,
            tcp_height_fn=tcp_height_fn, min_tcp_z_m=min_tcp_z_m,
        )
        current = self.read_joint_state()
        if float(np.max(np.abs(np.asarray(current.positions_rad)
                               - measured_start.positions_rad))) > max_start_error_rad:
            raise ValueError("发令前实测起点已变化，必须重新规划")
        if not math.isfinite(endpoint_tolerance_rad) or endpoint_tolerance_rad <= 0:
            raise ValueError("终点误差阈值必须为正有限数")
        if not math.isfinite(endpoint_timeout_s) or endpoint_timeout_s <= 0:
            raise ValueError("终点等待超时必须为正有限数")
        started = time.monotonic()
        sent = False
        try:
            for joints, target_time in zip(trajectory.positions_rad, trajectory.times_s):
                delay = target_time - (time.monotonic() - started)
                if delay > 0:
                    time.sleep(delay)
                # SDK 流式接口接受角度；选项与旧版已验证调用一致。
                if execution_guard is not None:
                    execution_guard()
                sent = True
                code = self.sdk.rm_movej_canfd(np.rad2deg(joints).tolist(), True, 0, 1, 0)
                if code != 0:
                    raise RuntimeError(f"RealMan 轨迹点下发失败，SDK 码 {code}")
            deadline = time.monotonic() + endpoint_timeout_s
            while True:
                if execution_guard is not None:
                    execution_guard()
                actual = self.read_joint_state()
                error = float(np.max(np.abs(
                    np.asarray(actual.positions_rad) - trajectory.positions_rad[-1]
                )))
                if error <= endpoint_tolerance_rad:
                    return actual
                if time.monotonic() >= deadline:
                    raise RuntimeError(f"RealMan 实测终点未到达，最大关节误差 {error:.4f} rad")
                time.sleep(0.02)
        except BaseException as error:
            # Ctrl+C 同样必须走 SDK 急停；恢复仍由操作者从实测状态决定。
            if sent:
                try:
                    self.stop()
                except Exception as stop_error:
                    raise RuntimeError(f"运动异常后急停失败：{stop_error}") from error
            raise
