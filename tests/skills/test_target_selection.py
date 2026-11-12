import numpy as np

from gene_agent.skills.runtime import target_selection


class _BatchPlanner:
    def __init__(self, successes):
        self.successes = tuple(successes)
        self.batch_targets = None
        self.diagnose_failures = None

    def solve_ik_batch(
        self,
        joint_names,
        start_positions,
        targets,
        *,
        target_frame,
        diagnose_failures=True,
    ):
        self.batch_targets = tuple(np.asarray(target, dtype=float) for target in targets)
        self.diagnose_failures = bool(diagnose_failures)
        assert len(self.batch_targets) == len(self.successes)
        return tuple(
            {
                "success": success,
                "failure_category": None if success else "ik_constraint_rejected",
            }
            for success in self.successes
        )


def test_local_rotation_search_batches_nominal_and_candidates_before_planning(monkeypatch):
    """复用旧版快路径：一次 batch IK 筛选后只规划选中的目标。"""
    planner = _BatchPlanner((False, True, True))
    planned_targets = []

    def fake_plan(_planner, _joint_names, _start_positions, target, **_kwargs):
        planned_targets.append(np.asarray(target, dtype=float))
        return {"success": True, "positions_rad": [[0.0], [0.1]]}

    monkeypatch.setattr(target_selection, "plan_with_approximate_ik", fake_plan)
    requested = np.eye(4)
    result = target_selection.plan_with_local_rotation_search(
        planner,
        ("joint_1",),
        (0.0,),
        requested,
        budget=None,
        x_degrees=(0.0, -5.0, 5.0),
        y_degrees=(0.0,),
    )

    assert len(planner.batch_targets) == 3
    assert planner.diagnose_failures is False
    np.testing.assert_allclose(planner.batch_targets[0], requested)
    assert len(planned_targets) == 1
    np.testing.assert_allclose(planned_targets[0], planner.batch_targets[1])
    assert result["target_selection"]["kind"] == "local_rotation_candidate"
    assert result["target_selection_attempts"]["ik_success_count"] == 2


def test_local_rotation_search_runs_detailed_nominal_failure_only_when_batch_has_no_solution(
    monkeypatch,
):
    """全部候选失败后才运行带详细诊断和近似 IK 的原目标规划。"""
    planner = _BatchPlanner((False, False, False))
    planned_targets = []

    def fake_plan(_planner, _joint_names, _start_positions, target, **_kwargs):
        planned_targets.append(np.asarray(target, dtype=float))
        return {"success": False, "failure_category": "goal_ik_no_solution"}

    monkeypatch.setattr(target_selection, "plan_with_approximate_ik", fake_plan)
    requested = np.eye(4)
    result = target_selection.plan_with_local_rotation_search(
        planner,
        ("joint_1",),
        (0.0,),
        requested,
        budget=None,
        x_degrees=(0.0, -5.0, 5.0),
        y_degrees=(0.0,),
    )

    assert len(planned_targets) == 1
    np.testing.assert_allclose(planned_targets[0], requested)
    assert result["success"] is False
    assert result["target_selection_attempts"]["ik_success_count"] == 0
