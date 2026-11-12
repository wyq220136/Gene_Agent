"""共享求解器不能改变原先两个独立候选筛选器的采样起点。"""
from types import SimpleNamespace

import numpy as np
import pytest

from gene_agent.tasks.policies.microwave_task.planning.knob import _select_legacy_grasp
from gene_agent.motion.model import JointState, PlanningSceneRef
from gene_agent.skills.library.knob.planning import KnobSkillSettings


@pytest.mark.parametrize("shared", [True, False])
def test_selection_preserves_independent_sampling_and_chunk_continuity(shared):
    class Solver:
        def __init__(self):
            self.cursor = 0

        def reset_seed(self):
            self.cursor = 0

    class Adapter:
        def __init__(self, planner, final=False):
            self.planner, self.final = planner, final
            self.observed = []

        def evaluate_candidates(self, requests):
            self.observed.append(self.planner.cursor)
            self.planner.cursor += 1
            return tuple(SimpleNamespace(
                request_id=request.request_id, operation=request.operation,
                success=self.final, to_dict=lambda: {},
            ) for request in requests)

    solver = Solver()
    grasp = Adapter(solver, final=True)
    pregrasp = Adapter(solver if shared else Solver())
    proposals = tuple(SimpleNamespace(candidate_id=str(i), tcp_pose_world=np.eye(4))
                      for i in range(30))  # 150 pregrasp：两块间必须保持采样连续。
    scene = PlanningSceneRef(0, "hash", "scene")
    _select_legacy_grasp(
        proposals, KnobSkillSettings(.05, 1., .1, .08, .02, .08), group="right",
        start=JointState(("j1",), (0.0,)), grasp_adapter=grasp,
        full_adapter=pregrasp, grasp_scene=scene, full_scene=scene, seed=123,
    )
    assert grasp.observed == [0]
    assert pregrasp.observed == [0, 1]


def test_selection_rejects_ik_only_candidate_without_full_path():
    class Result:
        def __init__(self, request, success, trajectory=None):
            self.request_id = request.request_id
            self.operation = request.operation
            self.success = success
            self.trajectory = trajectory

        def to_dict(self):
            return {"success": self.success}

    class Adapter:
        planner = SimpleNamespace(reset_seed=lambda: None)

        def evaluate_candidates(self, requests):
            return tuple(Result(request, True) for request in requests)

        def plan(self, request):
            feasible = request.request_id.endswith("pregrasp_path_01")
            return Result(request, feasible, SimpleNamespace() if feasible else None)

    proposal = SimpleNamespace(candidate_id="candidate", tcp_pose_world=np.eye(4))
    scene = PlanningSceneRef(0, "hash", "scene")
    candidate, pregrasp, attempts, reachable = _select_legacy_grasp(
        (proposal,), KnobSkillSettings(.05, 1., .1, .08, .02, .08),
        group="right", start=JointState(("j1",), (0.0,)),
        grasp_adapter=Adapter(), full_adapter=Adapter(),
        grasp_scene=scene, full_scene=scene, seed=123,
        screen_pregrasp_path=True,
    )
    assert candidate is proposal
    assert pregrasp is not None
    assert reachable == 1
    path_attempts = [row for row in attempts if row["stage"] == "pregrasp_full_path"]
    assert [row["result"]["success"] for row in path_attempts] == [False, True]
