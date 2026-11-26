"""单次抓取筛选内复用完全相同的 pregrasp 请求；不跨现场缓存。"""
from __future__ import annotations

from dataclasses import replace
import json

from gene_agent.motion.model import MotionRequest, MotionResult
from gene_agent.motion.planner import MotionPlanner


class PregraspRequestCache:
    """只绑定一个严格场景 planner；成功和失败都保留原始证据。"""

    def __init__(self, planner: MotionPlanner):
        self.planner = planner
        self.results: dict[str, MotionResult] = {}

    @staticmethod
    def key(request: MotionRequest) -> str:
        payload = request.to_dict()
        # ID 仅用于追踪；其余字段都参与精确比较，尤其包括起点、场景和种子。
        payload.pop("request_id")
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)

    @staticmethod
    def validate(request: MotionRequest, result: MotionResult) -> None:
        if result.request_id != request.request_id or result.operation is not request.operation:
            raise RuntimeError("planner 的 pregrasp 结果顺序或 operation 不匹配")

    @staticmethod
    def reuse(request: MotionRequest, source: MotionResult) -> MotionResult:
        return replace(source, request_id=request.request_id, duration_s=0.0,
                       metrics={**source.metrics, "request_reuse": {
                           "source_request_id": source.request_id,
                           "source_duration_s": source.duration_s,
                           "backend_evaluated": False,
                           "scope": "single_grasp_plan",
                       }})

    def evaluate_candidates(self, requests: tuple[MotionRequest, ...]) -> tuple[MotionResult, ...]:
        keys = tuple(self.key(request) for request in requests)
        pending: dict[str, MotionRequest] = {}
        for key, request in zip(keys, requests):
            if key not in self.results:
                pending.setdefault(key, request)
        # 批内重复也先筛掉；整批命中时完全不进入后端。
        if pending:
            fresh = self.planner.evaluate_candidates(tuple(pending.values()))
            if len(fresh) != len(pending):
                raise RuntimeError("planner 返回的 pregrasp IK 数量与候选不一致")
            for request, result in zip(pending.values(), fresh):
                self.validate(request, result)
            self.results.update(zip(pending, fresh))
        results = []
        for key, request in zip(keys, requests):
            source = self.results[key]
            results.append(source if source.request_id == request.request_id
                           else self.reuse(request, source))
        return tuple(results)

    def plan(self, request: MotionRequest) -> MotionResult:
        key = self.key(request)
        if key in self.results:
            return self.reuse(request, self.results[key])
        result = self.planner.plan(request)
        self.validate(request, result)
        self.results[key] = result
        return result
