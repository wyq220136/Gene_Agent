"""显式回位目标只影响单次同步规划，不污染后续普通位姿规划。"""

import numpy as np
import pytest

from gene_agent.motion.curobo.core import SynchronousPlanner


@pytest.mark.parametrize("target", ([0.], [0., float("nan")]))
def test_invalid_joint_transition_target_never_calls_solver(target):
    planner = SynchronousPlanner.__new__(SynchronousPlanner)
    planner.joint_names = ("j1", "j2")
    planner._solve = lambda *a, **kw: pytest.fail("invalid target reached backend")
    with pytest.raises(ValueError, match="关节目标"):
        planner.plan_joint_target(planner.joint_names, [0., 0.], target, np.eye(4))


def test_failed_joint_transition_does_not_leak_goal_into_next_pose_request():
    planner = SynchronousPlanner.__new__(SynchronousPlanner)
    planner.joint_names = ("j1", "j2")
    def failed_solve(*args, **kwargs):
        raise RuntimeError("backend failed")
    planner._solve = failed_solve
    with pytest.raises(RuntimeError, match="backend failed"):
        planner.plan_joint_target(planner.joint_names, [0., 0.], [.1, .2], np.eye(4))
    assert getattr(planner, "_cspace_goal_positions", None) is None

