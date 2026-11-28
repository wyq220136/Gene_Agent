"""RealMan 夹爪 SDK 的米制薄适配，保留旧真机命令含义。"""

from __future__ import annotations

import math
import time

from gene_agent.realworld.gripper_state import invalidate_endpoint


MAX_OPEN_POSITION = 990


def bounded_opening_width(width_m: float, maximum_m: float) -> float:
    """保留1000满量程换算，仅给真机开爪位置留1%端点余量。"""
    width, maximum = float(width_m), float(maximum_m)
    if (not math.isfinite(maximum) or maximum <= 0 or not math.isfinite(width)
            or not 0 <= width <= maximum):
        raise ValueError("夹爪开口必须在配置的有限行程范围内")
    return min(width, maximum * MAX_OPEN_POSITION / 1000.0)


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
        # 反馈容许范围与可下发位置范围分离；不改变原1000满量程换算。
        self.max_feedback_width_m = 1.5 * width
        self.last_reading = None
        self.confirmed_endpoint_width_m = None
        self.supports_motion_completion_feedback = True

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

    def position_command_width_m(self, width_m: float) -> float:
        return bounded_opening_width(width_m, self.max_width_m)

    def command_width_m(self, width_m: float, *, blocking: bool = True) -> None:
        """米制位置命令仍按1000换算；最大命令位置为990。"""
        width = self.position_command_width_m(width_m)
        position = min(MAX_OPEN_POSITION, self._scaled(width / self.max_width_m, name="opening"))
        invalidate_endpoint(self)
        self._check(self.sdk.rm_set_gripper_position(position, blocking, 3),
                    action="位置控制")

    def read_width_m(self) -> float:
        """接收-10～1500反馈；负端反馈映射零宽度，控制仍限0～1000。"""
        value = self.sdk.rm_get_gripper_state()
        if isinstance(value, tuple):
            code, state = value
            self._check(code, action="状态读取")
        else:
            state = value
        position = float(state["actpos"])
        if not math.isfinite(position) or not -10.0 <= position <= 1500.0:
            raise ValueError(f"RealMan 返回的夹爪位置无效: actpos={position!r}, "
                             f"反馈要求有限且在 [-10, 1500]; raw={value!r}")
        width = max(0.0, position) / 1000.0 * self.max_width_m
        self.last_reading = {
            "raw": value, "actpos": position, "width_m": width,
            "sampled_at_unix_ms": time.time() * 1000.,
            "negative_endpoint_clamped": position < 0,
            # mode=4/5 仍在运动；status 仅表示在线，不是到位信号。
            "motion_complete": state.get("mode") in (1, 2, 3, 6),
        }
        return width

    def grasp(self, *, speed: float = 1.0, force: float = 0.5,
              continuous: bool = True, blocking: bool = True) -> None:
        """沿用旧版连续夹取命令；速度与力保持原 0～1 归一化含义。"""
        sdk_speed = self._scaled(speed, name="speed")
        sdk_force = self._scaled(force, name="force")
        invalidate_endpoint(self)
        if continuous:
            code = self.sdk.rm_set_gripper_pick_on(sdk_speed, sdk_force, blocking, 3)
        else:
            code = self.sdk.rm_set_gripper_pick(sdk_speed, sdk_force, blocking, 3)
        self._check(code, action="抓取")

    def release(self, *, blocking: bool = True) -> None:
        """释放夹持后开到990，避免无位置目标的release命令直达硬限位。"""
        self.command_width_m(self.position_command_width_m(self.max_width_m), blocking=blocking)
