"""同一目标偶发约束失败时，保留每次规划事实并使用成功的那条轨迹。"""

from gene_agent.tasks.policies.microwave_task.planning.push_door import _plan_with_retries


def test_retry_uses_first_successful_plan_and_records_failures():
    results = iter([
        {"success": False, "failure_category": "constraint_violation", "duration_s": .5},
        {"success": False, "failure_category": "constraint_violation", "duration_s": .4},
        {"success": True, "positions_rad": [[0], [1]], "duration_s": .3},
    ])
    selected, attempts = _plan_with_retries(lambda: next(results), max_attempts=4)
    assert selected["positions_rad"] == [[0], [1]]
    assert [row["success"] for row in attempts] == [False, False, True]
    assert [row["failure_category"] for row in attempts] == [
        "constraint_violation", "constraint_violation", None,
    ]


def test_retry_stops_after_bound_and_keeps_last_failure():
    calls = []

    def fail():
        calls.append(1)
        return {"success": False, "failure_category": "goal_ik_no_solution"}

    selected, attempts = _plan_with_retries(fail, max_attempts=2)
    assert selected["failure_category"] == "goal_ik_no_solution"
    assert len(calls) == len(attempts) == 2


def test_start_collision_does_not_repeat_the_same_invalid_state():
    calls = []

    def collide():
        calls.append(1)
        return {"success": False, "failure_category": "start_collision"}

    selected, attempts = _plan_with_retries(collide, max_attempts=4)
    assert selected["failure_category"] == "start_collision"
    assert len(calls) == len(attempts) == 1
