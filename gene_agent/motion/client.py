"""Skill的阶段调用入口：绑定冻结上下文，沿用原请求/结果和后端检查。"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable, Mapping, Sequence
import time
from uuid import uuid4

import numpy as np

from .model import (
    CollisionPolicy, JointState, MotionOperation, MotionRequest, MotionResult,
    PivotLimits, PlannerMode, PlanningSceneRef, PoseTarget,
)
from .planner import MotionPlanner


@dataclass(frozen=True)
class PlanningContext:
    group: str
    start: JointState
    scene: PlanningSceneRef
    seed: int = 123
    planner_mode: PlannerMode = PlannerMode.CUDA_GRAPH_BENCHMARK
    collision_policy: CollisionPolicy = field(default_factory=CollisionPolicy)

    def __post_init__(self):
        if not isinstance(self.group, str) or not self.group.strip():
            raise ValueError("group不能为空")
        for value, kind in ((self.start, JointState), (self.scene, PlanningSceneRef),
                            (self.collision_policy, CollisionPolicy)):
            if not isinstance(value, kind):
                raise TypeError(f"阶段上下文需要{kind.__name__}")
        object.__setattr__(self, "seed", int(self.seed))
        object.__setattr__(self, "planner_mode", PlannerMode(self.planner_mode))


class _BoundPlanner:
    """给旧Skill使用的协议视图；保留阶段生命周期和冻结起点检查。"""

    def __init__(self, owner: StageMotion):
        self._owner = owner

    def _check(self, request):
        self._owner._ensure_active()
        context = self._owner.context
        if (request.start != context.start or request.scene != context.scene
                or request.group != context.group or request.seed != context.seed
                or request.planner_mode != context.planner_mode
                or request.collision_policy != context.collision_policy):
            raise ValueError("请求与冻结阶段上下文不一致；下一起点/场景须建立新stage")

    def fk(self, state):
        self._owner._ensure_active()
        if state != self._owner.context.start:
            raise ValueError("FK状态不属于当前冻结stage")
        return self._owner._backend.fk(state)

    def solve_ik(self, request):
        self._check(request)
        return self._owner._invoke(self._owner._backend.solve_ik, request)

    def plan(self, request):
        self._check(request)
        return self._owner._invoke(self._owner._backend.plan, request)

    def evaluate_candidates(self, requests):
        self._owner._ensure_active()
        requests = tuple(requests)
        for request in requests:
            self._check(request)
        if not requests:
            return ()
        return self._owner._invoke(self._owner._backend.evaluate_candidates, requests, batch=True)


class StageMotion:
    """一个冻结起点上的规划/候选评估；不会执行或自动推进机器人状态。"""

    def __init__(self, planner: MotionPlanner, context: PlanningContext, *,
                 ensure_active: Callable[[], None] | None = None,
                 cache_evidence: Mapping | None = None):
        if not isinstance(context, PlanningContext):
            raise TypeError("context必须是PlanningContext")
        self._backend = planner
        self._context = context
        self._guard = ensure_active or (lambda: None)
        self._cache = None if cache_evidence is None else dict(cache_evidence)
        self._planner = _BoundPlanner(self)

    def _ensure_active(self):
        self._guard()

    @property
    def context(self):
        self._ensure_active()
        return self._context

    @property
    def planner(self) -> MotionPlanner:
        self._ensure_active()
        return self._planner

    @property
    def cache_evidence(self):
        return None if self._cache is None else dict(self._cache)

    def _invoke(self, method, request, *, batch=False):
        """一次实际调用一个身份；批次逐项保留结果，但共享墙钟时间不相加。"""
        call_id = uuid4().hex
        started = time.perf_counter()
        raw = method(request)
        elapsed = time.perf_counter() - started
        rows = tuple(raw) if batch else (raw,)
        results = tuple(self._with_evidence(replace(row, metrics={
            **row.metrics, "planning_timing": {
                "call_id": call_id, "call_wall_s": elapsed,
                "batch_size": len(rows), "batch_index": index,
            },
        })) for index, row in enumerate(rows))
        return results if batch else results[0]

    def _with_evidence(self, result):
        if self._cache is None:
            return result
        return replace(result, metrics={**result.metrics, "planner_cache": dict(self._cache)})

    def request(self, operation, target: PoseTarget, *, request_id: str,
                approximate_ik_budget=None, pivot_limits=None) -> MotionRequest:
        c = self.context
        return MotionRequest(request_id, operation, c.group, c.start, target,
                             c.scene, c.collision_policy, seed=c.seed,
                             planner_mode=c.planner_mode,
                             approximate_ik_budget=approximate_ik_budget,
                             pivot_limits=pivot_limits)

    def fk(self) -> np.ndarray:
        return self.planner.fk(self.context.start)

    def solve_ik(self, target: PoseTarget, *, request_id: str) -> MotionResult:
        return self.planner.solve_ik(self.request(MotionOperation.IK, target, request_id=request_id))

    def evaluate_ik(self, targets: Sequence[PoseTarget], *, request_ids: Sequence[str]):
        self._ensure_active()
        targets, identifiers = tuple(targets), tuple(request_ids)
        if len(targets) != len(identifiers) or len(set(identifiers)) != len(identifiers):
            raise ValueError("批量IK的ID必须与目标数量一致且唯一")
        requests = tuple(self.request(MotionOperation.IK, target, request_id=identifier)
                         for target, identifier in zip(targets, identifiers))
        return self.planner.evaluate_candidates(requests)

    def plan_to(self, target: PoseTarget, *, request_id: str, approximate_ik_budget=None):
        return self.planner.plan(self.request(MotionOperation.PLAN, target, request_id=request_id,
                                             approximate_ik_budget=approximate_ik_budget))

    def plan_linear(self, target: PoseTarget, *, request_id: str, approximate_ik_budget=None):
        return self.planner.plan(self.request(MotionOperation.LINEAR_TCP, target, request_id=request_id,
                                             approximate_ik_budget=approximate_ik_budget))

    def plan_pivot(self, target: PoseTarget, *, request_id: str, limits: PivotLimits):
        return self.planner.plan(self.request(MotionOperation.PIVOT_TCP, target,
                                             request_id=request_id, pivot_limits=limits))


__all__ = ["PlanningContext", "StageMotion"]
