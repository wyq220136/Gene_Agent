"""Motion backend 必须实现的最小同步协议。"""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

import numpy as np

from .model import JointState, MotionRequest, MotionResult


@runtime_checkable
class MotionPlanner(Protocol):
    """cuRobo 之外的 backend 也可以实现该协议。"""

    def fk(self, state: JointState) -> np.ndarray:
        """返回 ``T_world_tcp``。"""

    def solve_ik(self, request: MotionRequest) -> MotionResult:
        """只判断目标 pose，不规划到达路径。"""

    def evaluate_candidates(self, requests: Sequence[MotionRequest]) -> tuple[MotionResult, ...]:
        """同一冻结场景、起点和规划配置下批量检查 IK，保持输入顺序。"""

    def plan(self, request: MotionRequest) -> MotionResult:
        """处理PLAN/LINEAR_TCP及后端支持的PIVOT_TCP；不暗换目标/路径。

        不支持的operation须明确拒绝。返回的是预测轨迹与终点，执行器负责
        实际运动和反馈。Skill简便调用入口见motion/README.md的StageMotion。
        """


__all__ = ["MotionPlanner"]
