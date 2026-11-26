"""按需抓取筛选的调用边界；CPU 替身，不连接真机或仿真。"""
from dataclasses import replace

import numpy as np
import pytest

from gene_agent.knowledge.grasp.model import GraspProposal
from gene_agent.motion.model import (
    CollisionPolicy, JointState, JointTrajectory, MotionFailure, MotionFailureKind,
    MotionOperation, MotionRequest, MotionResult, PlannerMode, PlanningSceneRef, PoseTarget,
)
from gene_agent.skills import GraspSettings, GraspSkill, SkillContext
from gene_agent.skills.library.pick.pregrasp_search import PregraspRequestCache


def proposal(name, x=.3):
    pose = np.eye(4)
    pose[:3, 3] = [x, .1, .25]
    return GraspProposal(name, "ring", "rim", pose, .007, {})


def context():
    return SkillContext("right", JointState(("j1",), (0.,)),
                        PlanningSceneRef(0, "scene_hash", "scene"), "bowl")


def settings(**kwargs):
    return GraspSettings(candidate_selection="legacy_ik", pregrasp_search="first_valid_prefix",
                         max_pregrasp_joint_travel_rad=2.,
                         max_approach_joint_travel_rad=1.5, **kwargs)


class Planner:
    def __init__(self, *, failure=None):
        self.calls = []
        self.failure = failure

    def result(self, request):
        name = request.request_id.split(":")[0]
        kind = ("final" if request.request_id.endswith(":final_ik") else
                "ik" if request.operation is MotionOperation.IK else
                "pre" if request.operation is MotionOperation.PLAN else "approach")
        if name in {"a", "b"} and self.failure == kind:
            return MotionResult(request.request_id, request.operation, False, "fake",
                                PlannerMode.DEVELOPMENT, .01,
                                failure=MotionFailure(MotionFailureKind.PATH_NOT_FOUND, "无解"))
        endpoint = float(request.target.matrix[0, 3])
        if name == "a" and self.failure == kind + "_posture":
            endpoint = 3.
        goal = JointState(("j1",), (endpoint,))
        path = None if request.operation is MotionOperation.IK else JointTrajectory(
            ("j1",), (0., 1.), (request.start.positions_rad, goal.positions_rad))
        return MotionResult(request.request_id, request.operation, True, "fake",
                            PlannerMode.DEVELOPMENT, .01, goal_state=goal, trajectory=path)

    def evaluate_candidates(self, requests):
        self.calls.append(("batch", tuple(requests)))
        return tuple(self.result(r) for r in requests)

    def plan(self, request):
        self.calls.append(("path", (request,)))
        return self.result(request)


def solve(planner, proposals):
    return GraspSkill().plan(planner, proposals, context(), settings())


def requests(planner, operation):
    return [r for _, batch in planner.calls for r in batch if r.operation is operation]


def test_first_complete_prefix_stops_before_other_pregrasp_candidates():
    planner = Planner()
    plan = solve(planner, (proposal("a"), proposal("b", .4)))
    assert plan.selected_candidate_id == "a"
    assert plan.stage("approach").status == "planned"
    batches = [batch for kind, batch in planner.calls if kind == "batch"]
    assert [len(b) for b in batches] == [2, 23]  # 最终 IK 批筛，随后仅首个 grasp。
    assert all(r.request_id.startswith("a:") for r in batches[1])
    assert len(requests(planner, MotionOperation.PLAN)) == 1
    assert len(requests(planner, MotionOperation.LINEAR_TCP)) == 1


@pytest.mark.parametrize("failure", ["final", "ik", "pre", "approach",
                                    "ik_posture", "pre_posture", "approach_posture"])
def test_failed_or_unsafe_prefix_continues_search(failure):
    planner = Planner(failure=failure)
    plan = solve(planner, (proposal("a"), proposal("b", .4), proposal("c", .5)))
    expected = "b" if failure.endswith("_posture") else "c"
    assert plan.selected_candidate_id == expected
    assert all(plan.stage(s).status == "planned" for s in ("pregrasp", "approach", "close"))
    if failure.endswith("_posture"):
        stage = "pregrasp_posture_check" if failure != "approach_posture" else "approach_posture_check"
        assert any(a["stage"] == stage and not a["result"]["success"] for a in plan.attempts)


@pytest.mark.parametrize("failure,operation", [("ik", MotionOperation.IK),
                                               ("pre", MotionOperation.PLAN)])
def test_failed_duplicate_requests_reused_before_backend(failure, operation):
    planner = Planner(failure=failure)
    plan = solve(planner, (proposal("a"), proposal("b"), proposal("c", .5)))
    assert plan.selected_candidate_id == "c"
    duplicates = [a for a in plan.attempts if a["candidate_id"] == "b"
                  and a["stage"] == ("pregrasp_ik" if failure == "ik" else "pregrasp")]
    assert len(duplicates) == 23
    assert all(not a["result"]["success"] and a["result"]["duration_s"] == 0 for a in duplicates)
    assert all(a["result"]["metrics"]["request_reuse"]["source_request_id"].startswith("a:")
               for a in duplicates)
    assert not any(r.request_id.startswith("b:") and ":final_ik" not in r.request_id
                   for r in requests(planner, operation))


def test_successful_duplicate_ik_and_path_reused_but_approach_checked_for_new_grasp():
    planner = Planner()
    original = planner.result
    def result(request):
        if request.operation is MotionOperation.LINEAR_TCP and request.request_id.startswith("a:"):
            return replace(original(request), success=False, trajectory=None,
                           failure=MotionFailure(MotionFailureKind.PATH_NOT_FOUND, "接近失败"))
        return original(request)
    planner.result = result
    plan = solve(planner, (proposal("a"), proposal("b"), proposal("c", .5)))
    assert plan.selected_candidate_id == "b"
    assert len(requests(planner, MotionOperation.PLAN)) == 23
    pre = plan.stage("pregrasp").result
    assert pre.success and pre.request_id == "b:pregrasp" and pre.duration_s == 0
    assert pre.metrics["request_reuse"]["source_request_id"] == "a:pregrasp"
    assert plan.stage("approach").request.request_id == "b:approach"
    assert not any(r.request_id.startswith("c:") and ":final_ik" not in r.request_id
                   for _, batch in planner.calls for r in batch)


def test_batch_duplicates_are_filtered_and_cache_does_not_cross_plan_instances():
    planner = Planner()
    ctx = context()
    request = MotionRequest("a:ik", MotionOperation.IK, ctx.group, ctx.start,
                            PoseTarget("world_tcp", np.eye(4)), ctx.scene)
    cache = PregraspRequestCache(planner)
    first, second = cache.evaluate_candidates((request, replace(request, request_id="b:ik")))
    assert len(planner.calls[0][1]) == 1
    assert first.success and second.success and second.request_id == "b:ik"
    assert second.duration_s == 0 and not second.metrics["request_reuse"]["backend_evaluated"]
    cache.evaluate_candidates((replace(request, request_id="c:ik"),))
    assert len(planner.calls) == 1
    PregraspRequestCache(planner).evaluate_candidates((request,))
    assert len(planner.calls) == 2


@pytest.mark.parametrize("change", ["start", "scene", "seed", "group", "policy", "target", "matrix", "mode"])
def test_nonidentical_context_is_never_reused(change):
    planner = Planner()
    ctx = context()
    request = MotionRequest("a:ik", MotionOperation.IK, ctx.group, ctx.start,
                            PoseTarget("world_tcp", np.eye(4)), ctx.scene)
    shifted = np.eye(4)
    shifted[0, 3] = np.nextafter(0., 1.)  # 即使极小差异也不当成完全相同。
    values = {
        "start": {"start": JointState(("j1",), (.01,))},
        "scene": {"scene": PlanningSceneRef(1, "new_hash", "scene")},
        "seed": {"seed": 1}, "group": {"group": "left"},
        "policy": {"collision_policy": CollisionPolicy("other_policy")},
        "target": {"target": PoseTarget("other_frame", np.eye(4))},
        "matrix": {"target": PoseTarget("world_tcp", shifted)},
        "mode": {"planner_mode": PlannerMode.DEVELOPMENT},
    }
    cache = PregraspRequestCache(planner)
    cache.evaluate_candidates((request,))
    cache.evaluate_candidates((replace(request, request_id="b:ik", **values[change]),))
    assert len(planner.calls) == 2


def test_exhausted_search_reports_failure_and_cache_does_not_invent_path():
    plan = solve(Planner(failure="approach"), (proposal("a"), proposal("b")))
    assert plan.selected_candidate_id is None
    assert all(plan.stage(s).status == "not_executed" for s in ("open", "approach", "close"))


@pytest.mark.parametrize("failure", [None, "ik", "pre"])
def test_batch_all_dedup_keeps_global_ranking_and_reuses_success_or_failure(failure):
    proposals = (proposal("a"), proposal("b"), proposal("c", .5))
    base = replace(settings(), pregrasp_search="batch_all")
    baseline_planner = Planner(failure=failure)
    baseline = GraspSkill().plan(baseline_planner, proposals, context(), base)
    planner = Planner(failure=failure)
    optimized = GraspSkill().plan(planner, proposals, context(),
                                  replace(base, deduplicate_pregrasp=True))
    assert optimized.selected_candidate_id == baseline.selected_candidate_id
    assert optimized.stage("pregrasp").request.joint_target == baseline.stage("pregrasp").request.joint_target
    pre_ik = [r for r in requests(planner, MotionOperation.IK) if ":pregrasp_ik_" in r.request_id]
    old_pre_ik = [r for r in requests(baseline_planner, MotionOperation.IK) if ":pregrasp_ik_" in r.request_id]
    assert len(pre_ik) < len(old_pre_ik)
    reused = [row for row in optimized.attempts if row.get("result", {}).get("metrics", {}).get("request_reuse")]
    assert reused and all(row["result"]["duration_s"] == 0 for row in reused)
    if failure in {"ik", "pre"}:
        assert any(not row["result"]["success"] for row in reused)


def test_shared_cache_reuses_same_id_on_fallback_with_zero_duration():
    planner = Planner(failure="ik")
    ctx = context()
    req = MotionRequest("a:pregrasp_ik_00", MotionOperation.IK, ctx.group, ctx.start,
                        PoseTarget("world_tcp", np.eye(4)), ctx.scene)
    cache = PregraspRequestCache(planner)
    first = cache.evaluate_candidates((req,))[0]
    second = cache.evaluate_candidates((req,))[0]
    assert not first.success and not second.success
    assert len(planner.calls) == 1
    assert second.duration_s == 0 and second.metrics["request_reuse"]["source_request_id"] == req.request_id
