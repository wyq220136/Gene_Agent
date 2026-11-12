"""RealMan 夹爪 SDK 的米制薄适配，保留旧真机命令含义。"""

from __future__ import annotations

import math


class RealmanGripperAdapter:
    """SDK 注入后才可发命令；构造本身不会连接或移动夹爪。

    旧版 `RealmanGripper` 使用 0～1000 的归一化行程。本类用任务配置的
    最大开口把米制 Skill 目标映射到该行程；这是命令值映射，实测夹持
    是否成功仍由操作后置检查判定。
    """

    def __init__(self, sdk, *, max_width_m: float) -> None:
        width = float(max_width_m)
        if not math.isfinite(width) or width <= 0:
            raise ValueError("最大夹爪开口必须为正有限米数")
        self.sdk = sdk
        self.max_width_m = width

    @staticmethod
    def _scaled(value: float, *, name: str) -> int:
        number = float(value)
        if not math.isfinite(number) or not 0.0 <= number <= 1.0:
            raise ValueError(f"{name} 必须在 0～1 范围内")
        return int(round(number * 1000.0))

    @staticmethod
    def _check(code: int, *, action: str) -> None:
        if code != 0:
            raise RuntimeError(f"RealMan {action}失败，SDK 码 {code}")

    def command_width_m(self, width_m: float, *, blocking: bool = True) -> None:
        """按旧 `set_opening` 路径发送 0～1000 的位置命令。"""
        width = float(width_m)
        if not math.isfinite(width) or not 0.0 <= width <= self.max_width_m:
            raise ValueError(f"夹爪开口必须在 [0, {self.max_width_m}] m 范围内")
        position = self._scaled(width / self.max_width_m, name="opening")
        self._check(self.sdk.rm_set_gripper_position(position, blocking, 3),
                    action="位置控制")

    def read_width_m(self) -> float:
        """读取 SDK `actpos`，按同一最大开口换算成米。"""
        value = self.sdk.rm_get_gripper_state()
        if isinstance(value, tuple):
            code, state = value
            self._check(code, action="状态读取")
        else:
            state = value
        position = float(state["actpos"])
        if not math.isfinite(position) or not 0.0 <= position <= 1000.0:
            raise ValueError("RealMan 返回的夹爪位置无效")
        return position / 1000.0 * self.max_width_m

    def grasp(self, *, speed: float = 1.0, force: float = 0.5,
              continuous: bool = True, blocking: bool = True) -> None:
        """沿用旧版连续夹取命令；速度与力保持原 0～1 归一化含义。"""
        sdk_speed = self._scaled(speed, name="speed")
        sdk_force = self._scaled(force, name="force")
        if continuous:
            code = self.sdk.rm_set_gripper_pick_on(sdk_speed, sdk_force, blocking, 3)
        else:
            code = self.sdk.rm_set_gripper_pick(sdk_speed, sdk_force, blocking, 3)
        self._check(code, action="抓取")

    def release(self, *, blocking: bool = True) -> None:
        """沿用旧版固定速度 500 的松爪命令。"""
        self._check(self.sdk.rm_set_gripper_release(500, blocking, 3), action="松开")
