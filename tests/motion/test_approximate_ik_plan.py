"""近似 IK 只能在有限误差和原碰撞策略下触发二次规划。"""
import numpy as np

from gene_agent.motion.curobo.approximate import ApproximateIKBudget, plan_with_approximate_ik


class Planner:
    joint_names = ("j1",)

    def __init__(self, *, position_error=.012, feasible=True, collision=False, second_success=True):
        self.position_error = position_error
        self.feasible = feasible
        self.collision = collision
        self.second_success = second_success
        self.targets = []

    def fk(self, names, q):
        pose = np.eye(4)
        pose[0, 3] = float(q[0])
        return pose

    def plan(self, names, q, target, *, target_frame="world_tcp"):
        self.targets.append(np.array(target, copy=True))
        if len(self.targets) == 1:
            return {"success": False, "failure_category": "goal_ik_no_solution",
                    "ik_diagnostic": {
                        "selected_joint_positions_rad": [.988],
                        "selected_seed_index": 0,
                        "seeds": [{"constraint_feasible": self.feasible,
                                   "position_error_m": self.position_error,
                                   "rotation_error_rad": 0.}],
                        "collision_checked": True,
                        "collision_check": {"joint_limit_violation": False,
                                            "self_collision": self.collision,
                                            "scene_collision": False},
                    }}
        return {"success": self.second_success, "failure_category": None if self.second_success else "path_not_found",
                "positions_rad": [[.2], [.988]], "interpolation_dt_s": .025}


def test_near_ik_replans_to_actual_fk_pose_and_preserves_original_target():
    planner = Planner()
    requested = np.eye(4); requested[0, 3] = 1.
    result = plan_with_approximate_ik(planner, ("j1",), (.2,), requested,
                                      budget=ApproximateIKBudget(.02, np.deg2rad(10)))
    assert result["success"] and result["approximate_ik"]["approximate"]
    assert np.isclose(result["approximate_ik"]["position_error_m"], .012)
    np.testing.assert_allclose(planner.targets[0], requested)
    np.testing.assert_allclose(planner.targets[1], planner.fk(("j1",), (.988,)))
    np.testing.assert_allclose(result["approximate_ik"]["requested_tcp_pose_world"], requested)


def test_near_ik_never_converts_unsafe_or_out_of_budget_candidate_to_success():
    requested = np.eye(4); requested[0, 3] = 1.
    for options in ({"position_error": .03}, {"feasible": False}, {"collision": True},
                    {"second_success": False}):
        planner = Planner(**options)
        result = plan_with_approximate_ik(planner, ("j1",), (.2,), requested,
                                          budget=ApproximateIKBudget(.02, np.deg2rad(10)))
        assert not result["success"]
        assert len(planner.targets) == (2 if options == {"second_success": False} else 1)


def test_near_ik_disabled_preserves_strict_failure():
    planner = Planner()
    result = plan_with_approximate_ik(planner, ("j1",), (.2,), np.eye(4), budget=None)
    assert not result["success"] and len(planner.targets) == 1


def test_near_ik_tries_another_feasible_seed_within_budget():
    class MultiSeedPlanner(Planner):
        def plan(self, names, q, target, *, target_frame="world_tcp"):
            result = super().plan(names, q, target, target_frame=target_frame)
            if len(self.targets) == 1:
                diagnostic = result["ik_diagnostic"]
                diagnostic["seeds"][0]["position_error_m"] = .04
                diagnostic["seeds"][0]["joint_positions_rad"] = [.96]
                diagnostic["seeds"].append({"constraint_feasible": True,
                                             "position_error_m": .012,
                                             "rotation_error_rad": 0.,
                                             "joint_positions_rad": [.988]})
            return result
        def _collision_diagnostic(self, q):
            return {"joint_limit_violation": False, "self_collision": False,
                    "scene_collision": False}
    planner = MultiSeedPlanner()
    requested = np.eye(4); requested[0, 3] = 1.
    result = plan_with_approximate_ik(planner, ("j1",), (.2,), requested,
                                      budget=ApproximateIKBudget(.02, .1))
    assert result["success"]
    assert result["approximate_ik"]["selected_seed_index"] == 1


def test_unconverged_seed_can_be_revalidated_before_replanning():
    class RecheckedPlanner(Planner):
        def _collision_diagnostic(self, q):
            return {"joint_limit_violation": False, "self_collision": False,
                    "scene_collision": False}

    planner = RecheckedPlanner(feasible=False)
    requested = np.eye(4); requested[0, 3] = 1.
    result = plan_with_approximate_ik(planner, ("j1",), (.2,), requested,
                                      budget=ApproximateIKBudget(.02, .1))
    assert result["success"]
    assert result["approximate_ik"]["source_constraint_feasible"] is False
    assert result["approximate_ik"]["independent_constraint_check"] is True
    assert len(planner.targets) == 2


def test_unconverged_seed_with_rechecked_collision_stays_rejected():
    class RecheckedPlanner(Planner):
        def _collision_diagnostic(self, q):
            return {"joint_limit_violation": False, "self_collision": True,
                    "scene_collision": False}

    planner = RecheckedPlanner(feasible=False)
    requested = np.eye(4); requested[0, 3] = 1.
    result = plan_with_approximate_ik(planner, ("j1",), (.2,), requested,
                                      budget=ApproximateIKBudget(.02, .1))
    assert not result["success"]
    assert len(planner.targets) == 1


def test_approximate_duration_includes_failed_strict_solve_and_candidate_checks(monkeypatch):
    from gene_agent.motion.curobo import approximate
    clock=iter([10.,12.5])
    monkeypatch.setattr(approximate.time,'perf_counter',lambda:next(clock))
    requested=np.eye(4);requested[0,3]=1.
    result=plan_with_approximate_ik(Planner(),('j1',),(.2,),requested,budget=ApproximateIKBudget(.02,.1))
    assert result['success'] and result['duration_s']==2.5
