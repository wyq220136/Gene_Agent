"""批量候选筛选只为真正无解的批次保留昂贵的碰撞归因。"""

import numpy as np

from gene_agent.motion.curobo.core import CuroboPlannerAdapter
from gene_agent.motion.model import (
    JointState, MotionOperation, MotionRequest, PlannerMode,
    PlanningSceneRef, PoseTarget,
)


def test_reachable_batch_avoids_diagnosing_each_rejected_candidate():
    class Planner:
        joint_names = ("joint",)
        scene_snapshot_sha256 = "scene"
        scene_scope = "test scene"
        planner_mode = PlannerMode.CUDA_GRAPH_BENCHMARK
        random_seed = 123

        def solve_ik_batch(self, names, positions, targets, *, target_frame,
                           diagnose_failures):
            assert diagnose_failures == "when_no_success"
            assert len(targets) == 2
            return (
                {"success": False, "status": "solver_failed", "duration_s": .01,
                 "failure_category": "ik_pose_not_converged"},
                {"success": True, "status": "success", "duration_s": .01,
                 "positions_rad": [[.2]]},
            )

    planner = CuroboPlannerAdapter(Planner(), group="right", scene_revision=0)
    scene = PlanningSceneRef(0, "scene", "test scene")
    requests = tuple(MotionRequest(
        str(index), MotionOperation.IK, "right", JointState(("joint",), (0.,)),
        PoseTarget("world_tcp", np.eye(4)), scene, seed=123,
    ) for index in range(2))

    results = planner.evaluate_candidates(requests)

    assert [result.success for result in results] == [False, True]
