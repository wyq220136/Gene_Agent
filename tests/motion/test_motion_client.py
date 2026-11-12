"""Skill调用面只绑定请求；不启动GPU或复制后端规划算法。"""
from dataclasses import replace

import numpy as np
import pytest

from gene_agent.motion import (
    CollisionPolicy, JointState, MotionFailure, MotionFailureKind,
    MotionOperation, MotionResult, PlannerMode, PlanningSceneRef, PoseTarget,
    PlanningContext, StageMotion, PivotLimits,
)
from gene_agent.motion.model import motion_request_from_dict


class RecordingPlanner:
    def __init__(self):
        self.requests = []

    def fk(self, state):
        return np.eye(4)

    def plan(self, request):
        self.requests.append(request)
        return MotionResult(request.request_id, request.operation, False, "recording",
                            request.planner_mode, .01,
                            failure=MotionFailure(MotionFailureKind.PATH_NOT_FOUND, "无路径"))

    solve_ik = plan

    def evaluate_candidates(self, requests):
        return tuple(self.plan(r) for r in requests)


def context():
    return PlanningContext("right", JointState(("j1",), (.2,)),
                           PlanningSceneRef(3, "hash", "scope"), seed=17,
                           planner_mode=PlannerMode.DEVELOPMENT)


@pytest.mark.parametrize("method,operation", [
    ("solve_ik", MotionOperation.IK), ("plan_to", MotionOperation.PLAN),
    ("plan_linear", MotionOperation.LINEAR_TCP),
])
def test_methods_bind_same_context_and_preserve_failure(method, operation):
    backend = RecordingPlanner()
    c = context()
    client = StageMotion(backend, c)
    target = PoseTarget("world_tcp", np.eye(4))
    result = getattr(client, method)(target, request_id="stage")
    request = backend.requests[0]
    assert (request.operation, request.start, request.group, request.seed) == (operation, c.start, "right", 17)
    assert request.scene == c.scene and request.planner_mode == c.planner_mode
    assert result.failure.kind is MotionFailureKind.PATH_NOT_FOUND
    np.testing.assert_array_equal(client.fk(), np.eye(4))


def test_batch_order_ids_and_empty_are_explicit():
    backend = RecordingPlanner()
    client = StageMotion(backend, context())
    targets = [PoseTarget("world_tcp", np.eye(4))] * 2
    assert [r.request_id for r in client.evaluate_ik(targets, request_ids=["b", "a"])] == ["b", "a"]
    assert client.evaluate_ik([], request_ids=[]) == ()
    for ids in (["one"], ["same", "same"]):
        with pytest.raises(ValueError):
            client.evaluate_ik(targets, request_ids=ids)
    assert len(backend.requests) == 2


def test_budget_and_pivot_roundtrip_without_changing_legacy_json():
    backend = RecordingPlanner()
    client = StageMotion(backend, context())
    target = PoseTarget("world_tcp", np.eye(4))
    budget = {"max_position_error_m": .01, "max_rotation_error_rad": .04}
    client.plan_to(target, request_id="approx", approximate_ik_budget=budget)
    assert backend.requests[-1].approximate_ik_budget == budget
    limits = PivotLimits(.015, .03)
    client.plan_pivot(target, request_id="pivot", limits=limits)
    request = backend.requests[-1]
    assert request.operation is MotionOperation.PIVOT_TCP and request.pivot_limits == limits
    assert motion_request_from_dict(request.to_dict()).pivot_limits == limits
    with pytest.raises(ValueError):
        replace(request, approximate_ik_budget=budget)
    with pytest.raises(ValueError):
        replace(request, operation=MotionOperation.PLAN)
    with pytest.raises(ValueError):
        replace(request, pivot_limits=None)
    legacy = backend.requests[0].to_dict()
    legacy.pop("pivot_limits", None)
    assert motion_request_from_dict(legacy).pivot_limits is None
    for values in ((0, .1), (.1, float("nan"))):
        with pytest.raises(ValueError):
            PivotLimits(*values)


def test_bound_planner_rejects_start_outside_frozen_stage():
    backend = RecordingPlanner()
    client = StageMotion(backend, context())
    request = client.request(MotionOperation.PLAN, PoseTarget("world_tcp", np.eye(4)), request_id="r")
    with pytest.raises(ValueError):
        client.planner.plan(replace(request, start=JointState(("j1",), (.3,))))
    assert backend.requests == []


def test_stage_timing_identifies_repeated_calls_without_changing_request_ids():
    from gene_agent.motion import StageMotion,PlanningContext,JointState,PlanningSceneRef,PoseTarget
    from gene_agent.motion.model import MotionResult,MotionFailure,MotionFailureKind
    import numpy as np
    class Backend:
        def plan(self,request):return MotionResult(request.request_id,request.operation,False,'test',request.planner_mode,.1,failure=MotionFailure(MotionFailureKind.PATH_NOT_FOUND,'test'))
    motion=StageMotion(Backend(),PlanningContext('right',JointState(('q',),(0.,)),PlanningSceneRef(0,'s','world')))
    a=motion.plan_to(PoseTarget('world_tcp',np.eye(4)),request_id='same')
    b=motion.plan_to(PoseTarget('world_tcp',np.eye(4)),request_id='same')
    assert a.metrics['planning_timing']['call_id']!=b.metrics['planning_timing']['call_id']
    assert a.metrics['planning_timing']['call_wall_s']>=0
    assert a.request_id==b.request_id=='same'
