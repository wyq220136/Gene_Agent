"""参考 UMI 的 RealMan UDP 状态接收；控制仍由同一个 SDK 连接发送。"""

from __future__ import annotations

import socket
import threading
import time

import numpy as np

from .adapter import RealmanStateReadError

_active_stream = None
_active_lock = threading.Lock()
_noop_callback = None
_retired_callbacks = []  # 极少数注销失败时保留引用，避免 C 指针悬空。


def infer_local_ip(robot_ip: str) -> str:
    """UDP connect 只做路由选择，不向机器人发包。"""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        probe.connect((robot_ip, 9))
        return probe.getsockname()[0]


class RealmanUDPState:
    def __init__(self, robot_ip: str, *, target_ip: str | None = None, port: int = 8888,
                 cycle: int = 2, max_age_s: float = 0.2, read_timeout_s: float = 0.5):
        if not 1 <= port <= 65535 or cycle < 1 or not np.isfinite([max_age_s, read_timeout_s]).all() or min(max_age_s, read_timeout_s) <= 0:
            raise ValueError("UDP 端口、周期和新鲜度/读取超时参数无效")
        self.robot_ip = robot_ip
        self.target_ip = target_ip or infer_local_ip(robot_ip)
        self.port, self.cycle = port, cycle
        self.max_age_s, self.read_timeout_s = max_age_s, read_timeout_s
        self._condition = threading.Condition()
        self._joints = None
        self._received_at = -float("inf")
        self._sequence = self._consumed_sequence = 0
        self._closed = False
        self._sdk = self._callback = None
        self._last_packet_error = "尚未收到有效数据包"

    def accept_packet(self, state):
        """C 回调入口内只复制数据，不进行 SDK 查询或运动。"""
        try:
            ip = state.arm_ip.decode("ascii").rstrip("\x00") if isinstance(state.arm_ip, bytes) else str(state.arm_ip)
            if ip != self.robot_ip:
                raise ValueError(f"忽略其他机器人 {ip} 的状态")
            if state.errCode != 0:
                raise ValueError(f"UDP 数据解析错误 {state.errCode}")
            degrees = np.asarray(state.joint_status.joint_position, dtype=float)
            if degrees.shape != (7,) or not np.isfinite(degrees).all():
                raise ValueError("UDP 未提供七个有限关节角")
            joints = np.deg2rad(degrees)
        except (AttributeError, TypeError, ValueError, UnicodeError) as exc:
            with self._condition:
                self._last_packet_error = str(exc)
            return
        with self._condition:
            if self._closed:
                return
            self._joints = joints.copy()
            self._received_at = time.monotonic()
            self._sequence += 1
            self._condition.notify_all()

    def _wait(self, timeout_s: float, *, consume: bool):
        deadline = time.monotonic() + timeout_s
        with self._condition:
            while not self._closed:
                now = time.monotonic()
                fresh = self._joints is not None and now - self._received_at <= self.max_age_s
                if fresh and (not consume or self._sequence > self._consumed_sequence):
                    if consume:
                        self._consumed_sequence = self._sequence
                    return self._joints.copy()
                remaining = deadline - now
                if remaining <= 0:
                    break
                self._condition.wait(min(remaining, 0.05))
            detail = (f"UDP {self.robot_ip} → {self.target_ip}:{self.port} 未提供新鲜的新帧，"
                      f"最大帧龄 {self.max_age_s:.2f}s；{self._last_packet_error}")
            raise RealmanStateReadError(-2, 1, detail=detail)

    def read_joints(self):
        return self._wait(self.read_timeout_s, consume=True)

    def start(self, sdk, *, startup_timeout_s: float = 3.0):
        global _active_stream, _noop_callback
        with _active_lock:
            if _active_stream is not None:
                raise RuntimeError("同一进程只能启用一条 RealMan UDP 状态流，避免 SDK 全局回调被覆盖")
            _active_stream = self
        try:
            from ctypes import CFUNCTYPE
            from Robotic_Arm.rm_robot_interface import rm_realtime_arm_joint_state_t, rm_realtime_push_config_t
            callback_type = CFUNCTYPE(None, rm_realtime_arm_joint_state_t)
            if _noop_callback is None:
                _noop_callback = callback_type(lambda _state: None)
            self._sdk = sdk
            self._callback = callback_type(self.accept_packet)
            sdk.rm_realtime_arm_state_call_back(self._callback)
            print(f"启用 UDP 状态：{self.robot_ip} → {self.target_ip}:{self.port}，周期 {self.cycle * 5}ms。", flush=True)
            last_code = None
            for force_coordinate in (-1, 0):
                config = rm_realtime_push_config_t(cycle=self.cycle, enable=True, port=self.port,
                                                   force_coordinate=force_coordinate, ip=self.target_ip)
                last_code = sdk.rm_set_realtime_push(config)
                if last_code == 0:
                    break
                read_code, actual = sdk.rm_get_realtime_push()
                if (read_code == 0 and actual.get("enable") and actual.get("ip") == self.target_ip
                        and actual.get("port") == self.port and actual.get("cycle") == self.cycle):
                    print(f"UDP 配置回执 {last_code}，回读确认配置已经生效。", flush=True)
                    break
            else:
                raise RuntimeError(f"无法启用 UDP 推送，SDK {last_code}，目标 {self.target_ip}:{self.port}")
            self._wait(startup_timeout_s, consume=False)
            print("UDP 已收到匹配机器人 IP 的有效新鲜关节状态。", flush=True)
        except BaseException:
            self.close()
            raise

    def describe(self):
        return {"state_source": "udp", "robot_ip": self.robot_ip, "target_ip": self.target_ip,
                "port": self.port, "cycle_ms": self.cycle * 5,
                "max_age_s": self.max_age_s, "read_timeout_s": self.read_timeout_s}

    def close(self):
        global _active_stream
        with self._condition:
            if self._closed:
                return
            self._closed = True
            self._condition.notify_all()
        with _active_lock:
            if _active_stream is self:
                if self._sdk is not None and self._callback is not None:
                    try:
                        self._sdk.rm_realtime_arm_state_call_back(_noop_callback)
                    except Exception:
                        _retired_callbacks.append(self._callback)
                _active_stream = None
