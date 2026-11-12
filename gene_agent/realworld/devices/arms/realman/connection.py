"""RealMan 真机连接的显式生命周期；公共 Task 不依赖厂家 SDK。"""

from __future__ import annotations

from dataclasses import dataclass, field
import ipaddress
import json
from pathlib import Path
import time
from urllib.parse import urlsplit

import numpy as np

from gene_agent.realworld.devices.grippers.realman.adapter import RealmanGripperAdapter
from .adapter import RealmanAdapter, load_standard_gripper_flange_tcp


@dataclass
class RealmanConnection:
    """只拥有本次创建的 SDK 连接，关闭时不会发送额外运动。"""

    sdk: object
    robot: RealmanAdapter
    gripper: RealmanGripperAdapter
    _closed: bool = field(default=False, init=False, repr=False)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        code = self.sdk.rm_delete_robot_arm()
        if code not in (None, 0):
            raise RuntimeError(f"RealMan 断开失败，SDK 码 {code}")

    def __enter__(self) -> "RealmanConnection":
        return self

    def __exit__(self, _kind, _error, _traceback) -> None:
        self.close()


@dataclass
class RealmanStateConnection:
    """关节与 flange FK 连接生命周期；不配置夹爪。"""

    sdk: object
    robot: RealmanAdapter
    _closed: bool = field(default=False, init=False, repr=False)
    _state_stream: object = field(default=None, init=False, repr=False)
    communication_info: dict = field(default_factory=lambda: {"state_source": "tcp"}, init=False)

    def read_flange_pose(self) -> tuple[np.ndarray, np.ndarray]:
        if self._closed:
            raise RuntimeError("RealMan 连接已经关闭，不能读取状态")
        state = self.robot.read_joint_state()
        joints = np.asarray(state.positions_rad, dtype=float)
        return self.robot.flange_pose_from_sdk_fk(joints), joints

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            if self._state_stream is not None:
                self.robot.joint_state_reader = None
                self._state_stream.close()
            code = self.sdk.rm_delete_robot_arm()
            if code not in (None, 0):
                raise RuntimeError(f"RealMan 断开失败，SDK 码 {code}")

    def __enter__(self) -> "RealmanStateConnection":
        return self

    def __exit__(self, _kind, _error, _traceback) -> None:
        self.close()

    def enable_udp_state(self, ip: str, *, target_ip=None, port=8888, timeout_s=0.5):
        from .realtime_state import RealmanUDPState
        try:
            stream = RealmanUDPState(ip, target_ip=target_ip, port=port, read_timeout_s=timeout_s)
            stream.start(self.sdk)
        except BaseException:
            self.close()
            raise
        self._state_stream = stream
        self.robot.joint_state_reader = stream.read_joints
        self.communication_info.update(stream.describe())


class RealmanReadOnlyConnection(RealmanStateConnection):
    """供标定读取关节和 flange FK；不会下发运动。"""


def normalize_realman_ip(value: str) -> str:
    """SDK 只接收纯 IPv4；兼容用户从浏览器复制的 http(s)://IP/。"""
    raw = str(value).strip()
    if raw.startswith(("http://", "https://")):
        parsed = urlsplit(raw)
        if (parsed.path not in ("", "/") or parsed.port is not None
                or parsed.query or parsed.fragment or parsed.username or parsed.password):
            raise ValueError("RealMan 地址只允许 http(s)://IPv4/；端口请用 --port 指定")
        raw = parsed.hostname or ""
    try:
        return str(ipaddress.IPv4Address(raw))
    except ipaddress.AddressValueError as exc:
        raise ValueError("RealMan --ip 必须是纯 IPv4，或 http(s)://IPv4/ 形式") from exc


def connect_realman_read_only(
    ip: str, *, group: str, port: int = 8080, level: int = 3,
    mode: int = 2, sdk=None, sdk_factory=None,
    connect_retries: int = 30, connect_retry_interval_s: float = 0.5,
    state_source: str = "tcp", sdk_timeout_ms: int = 2000,
    udp_target_ip=None, udp_port: int = 8888, udp_timeout_s: float = 0.5,
) -> RealmanReadOnlyConnection:
    """只读连接；失败后销毁旧 SDK 实例并重建，0 次上限表示持续重试。"""
    if state_source not in ("udp", "tcp"):
        raise ValueError("state_source 只能为 udp 或 tcp")
    connection = _connect_realman_state(
        ip, group=group, port=port, level=level, mode=mode, sdk=sdk, sdk_factory=sdk_factory,
        connect_retries=connect_retries, connect_retry_interval_s=connect_retry_interval_s,
        connection_type=RealmanReadOnlyConnection, label="只读", sdk_timeout_ms=sdk_timeout_ms,
    )
    if state_source == "udp":
        connection.enable_udp_state(normalize_realman_ip(ip), target_ip=udp_target_ip,
                                    port=udp_port, timeout_s=udp_timeout_s)
    return connection


def connect_realman_pose_replay(ip: str, *, group: str, port: int = 8080,
                                connect_retries: int = 30,
                                connect_retry_interval_s: float = 0.5,
                                state_source: str = "udp", udp_target_ip=None, udp_port: int = 8888,
                                udp_timeout_s: float = 0.5, sdk_timeout_ms: int = 2000):
    """显式启用标定位姿 MoveL 能力；连接本身不发令，不配置夹爪。"""
    from .calibration_replay import RealmanPoseReplayConnection
    if state_source not in ("udp", "tcp"):
        raise ValueError("state_source 只能为 udp 或 tcp")
    connection = _connect_realman_state(
        ip, group=group, port=port, connect_retries=connect_retries,
        connect_retry_interval_s=connect_retry_interval_s,
        connection_type=RealmanPoseReplayConnection, label="标定姿态回放", sdk_timeout_ms=sdk_timeout_ms,
    )
    if state_source == "udp":
        connection.enable_udp_state(normalize_realman_ip(ip), target_ip=udp_target_ip,
                                    port=udp_port, timeout_s=udp_timeout_s)
    return connection


def _connect_realman_state(
    ip: str, *, group: str, connection_type, label: str,
    port: int = 8080, level: int = 3, mode: int = 2, sdk=None, sdk_factory=None,
    connect_retries: int = 30, connect_retry_interval_s: float = 0.5,
    sdk_timeout_ms: int = 2000,
) -> RealmanStateConnection:
    """共享 SDK 创建/失败清理/重试；运动能力由调用方显式选择。"""
    ip = normalize_realman_ip(ip)
    if group not in {"left", "right"}:
        raise ValueError("必须提供 left/right 机器人组")
    if connect_retries < 0 or connect_retry_interval_s < 0:
        raise ValueError("重试次数和间隔不能为负")
    if not isinstance(sdk_timeout_ms, int) or sdk_timeout_ms < 100:
        raise ValueError("SDK 接收超时必须为至少100的整数毫秒值")
    if sdk is not None and sdk_factory is not None:
        raise ValueError("sdk 与 sdk_factory 只能提供一个")
    if sdk_factory is None and sdk is None:
        from Robotic_Arm.rm_robot_interface import RoboticArm, rm_thread_mode_e
        def sdk_factory():
            return RoboticArm(rm_thread_mode_e(mode))
    elif sdk_factory is None:
        def sdk_factory():
            return sdk
        connect_retries = 1  # 注入单个 SDK 对象时不在释放后复用它。
    attempt = 0
    last_error = "未知"
    while connect_retries == 0 or attempt < connect_retries:
        attempt += 1
        current_sdk = None
        try:
            current_sdk = sdk_factory()
            timeout_setter = getattr(current_sdk, "rm_set_timeout", None)
            if callable(timeout_setter):
                timeout_setter(sdk_timeout_ms)
            handle = current_sdk.rm_create_robot_arm(ip, int(port), int(level))
            if getattr(handle, "id", -1) == -1:
                last_error = f"handle_id={getattr(handle, 'id', None)}"
            else:
                names = tuple(f"{group}_joint_{index}" for index in range(1, 8))
                robot = RealmanAdapter(
                    current_sdk, group=group, joint_names=names, T_flange_tcp=np.eye(4),
                )
                print(f"RealMan {label}连接成功：{ip}:{port}，handle_id={handle.id}", flush=True)
                connection = connection_type(current_sdk, robot)
                connection.communication_info["sdk_timeout_ms"] = sdk_timeout_ms if callable(timeout_setter) else None
                return connection
        except Exception as exc:
            last_error = repr(exc)
        # SDK 失败连接也可能占用内部槽位；下一次必须重新创建对象。
        if current_sdk is not None:
            try:
                current_sdk.rm_delete_robot_arm()
            except Exception:
                pass
        if connect_retries == 0 or attempt < connect_retries:
            attempt_label = "持续重试" if connect_retries == 0 else f"{attempt}/{connect_retries}"
            print(f"RealMan {label}连接失败，{attempt_label}：{ip}:{port}，{last_error}", flush=True)
            time.sleep(connect_retry_interval_s)
    raise RuntimeError(
        f"无法连接 RealMan：{ip}:{port}；尝试 {attempt} 次，最后错误 {last_error}。"
        "请核对纯 IP、机器人电源及网络连接。"
    )


def connect_realman(
    ip: str, *, group: str, port: int = 8080, level: int = 3,
    mode: int = 2, sdk=None, T_flange_tcp=None,
    max_width_m: float | None = None,
) -> RealmanConnection:
    """按旧真机入口连接并配置夹爪行程；调用时才导入厂家 SDK。

    `sdk` 注入仅用于离线验证。真实使用必须明确调用本函数；导入模块
    或创建 Task 配置不会自动连接、复位或移动机械臂。
    """
    if not ip or group not in {"left", "right"}:
        raise ValueError("必须提供 IP 和 left/right 机器人组")
    if sdk is None:
        from Robotic_Arm.rm_robot_interface import RoboticArm, rm_thread_mode_e
        sdk = RoboticArm(rm_thread_mode_e(mode))
    if T_flange_tcp is None:
        T_flange_tcp = load_standard_gripper_flange_tcp()
    if max_width_m is None:
        path = Path(__file__).resolve().parents[5] / "config/robots/standard_gripper.json"
        max_width_m = float(json.loads(path.read_text(encoding="utf-8"))[
            "max_opening_width_m"])
    handle = sdk.rm_create_robot_arm(ip, int(port), int(level))
    if getattr(handle, "id", -1) == -1:
        raise RuntimeError(f"无法连接 RealMan：{ip}:{port}")
    try:
        code = sdk.rm_set_gripper_route(min_route=0, max_route=1000)
        if code not in (None, 0):
            raise RuntimeError(f"RealMan 夹爪行程设置失败，SDK 码 {code}")
        names = tuple(f"{group}_joint_{index}" for index in range(1, 8))
        robot = RealmanAdapter(
            sdk, group=group, joint_names=names, T_flange_tcp=T_flange_tcp,
        )
        gripper = RealmanGripperAdapter(sdk, max_width_m=max_width_m)
        return RealmanConnection(sdk, robot, gripper)
    except Exception:
        sdk.rm_delete_robot_arm()
        raise


def read_realman_tcp_pose(
    ip: str, *, group: str, port: int = 8080, level: int = 3,
    mode: int = 2, sdk=None, T_flange_tcp=None,
) -> tuple[np.ndarray, np.ndarray]:
    """只读当前七轴并用 RealMan SDK FK 求 T_sdk_tcp；不配置夹爪或运动。"""
    if not ip or group not in {"left", "right"}:
        raise ValueError("必须提供 IP 和 left/right 机器人组")
    if sdk is None:
        from Robotic_Arm.rm_robot_interface import RoboticArm, rm_thread_mode_e
        sdk = RoboticArm(rm_thread_mode_e(mode))
    if T_flange_tcp is None:
        T_flange_tcp = load_standard_gripper_flange_tcp()
    handle = sdk.rm_create_robot_arm(ip, int(port), int(level))
    if getattr(handle, "id", -1) == -1:
        raise RuntimeError(f"无法连接 RealMan：{ip}:{port}")
    try:
        names = tuple(f"{group}_joint_{index}" for index in range(1, 8))
        adapter = RealmanAdapter(
            sdk, group=group, joint_names=names, T_flange_tcp=T_flange_tcp,
        )
        state = adapter.read_joint_state()
        joints = np.asarray(state.positions_rad, dtype=float)
        return adapter.tcp_pose_from_sdk_fk(joints), joints
    finally:
        code = sdk.rm_delete_robot_arm()
        if code not in (None, 0):
            raise RuntimeError(f"RealMan 断开失败，SDK 码 {code}")
