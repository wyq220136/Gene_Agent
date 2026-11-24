"""复现旧 cuRobo 的 Tensor/state 契约错误；CPU 替身不声称完成 GPU 规划。"""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from gene_agent.motion.curobo import SynchronousPlanner


class StateCollisionCost:
    def __init__(self, scores):
        self.scores = scores
        self.num_spheres = 0
        self.config = SimpleNamespace(update_num_spheres=self.update_num_spheres)

    def update_num_spheres(self, count):
        self.num_spheres = count

    def setup_batch_tensors(self, batch, horizon):
        self.buffer_shape = (batch, horizon, self.num_spheres, 4)

    def forward(self, state, idxs_env_query=None):
        spheres = state.robot_spheres  # 真机 traceback 中最先抛错的位置。
        assert spheres.shape == self.buffer_shape
        assert spheres.dtype is torch.float32
        assert idxs_env_query is None
        return self.scores


class MixedAPIChecker:
    def __init__(self, bound, self_cost, world):
        self.bound = bound
        self.self_scores = self_cost
        self.self_collision_cost = object() if self_cost is not None else None
        self.collision_constraint = StateCollisionCost(world) if world is not None else None
        self.num_spheres = 2

    def setup_batch_tensors(self, batch, horizon):
        self.batch_shape = (batch, horizon)

    def get_kinematics(self, q):
        assert q.shape[:2] == self.batch_shape
        return SimpleNamespace(robot_spheres=torch.zeros(
            *q.shape[:2], self.num_spheres, 4, dtype=torch.float32,
        ))

    def get_bound(self, q):
        return self.bound

    def get_self_collision(self, spheres):
        assert spheres.shape[2] == self.num_spheres
        return self.self_scores

    def validate(self, q):
        self.setup_batch_tensors(*q.shape[:2])
        state = self.get_kinematics(q)
        # 还原现场版本 validate -> get_collision_constraint -> forward 的错误。
        return self.collision_constraint.forward(state.robot_spheres)


def _planner(checker):
    planner = object.__new__(SynchronousPlanner)
    # 只替换设备分配，正式代码仍使用 float32 CUDA Tensor。
    planner.torch = SimpleNamespace(
        as_tensor=lambda value, **kw: torch.as_tensor(value, dtype=kw["dtype"]),
        float32=torch.float32, cat=torch.cat,
    )
    planner.joint_names = ["a", "b"]
    planner._collision_checker = checker
    planner._validator = None
    return planner


def test_mixed_api_exception_is_reproduced_and_full_state_fixes_it():
    checker = MixedAPIChecker(torch.zeros(1, 1, 2), torch.zeros(1, 1, 1),
                              torch.zeros(1, 1, 1))
    with pytest.raises(AttributeError, match="robot_spheres"):
        checker.validate(torch.zeros(1, 1, 2))
    planner = _planner(checker)
    assert planner._configuration_validity([[.1, .2]]) == [True]
    assert checker.collision_constraint.num_spheres == 2
    checker.num_spheres = 3
    assert planner._configuration_validity([[.1, .2]]) == [True]
    assert checker.collision_constraint.buffer_shape == (1, 1, 3, 4)


@pytest.mark.parametrize("term", ["bound", "self", "world"])
def test_each_constraint_rejects_its_waypoint_without_masking_other_samples(term):
    scores = {name: torch.zeros(1, 4, 1) for name in ("bound", "self", "world")}
    scores[term][0, 2, 0] = .01
    checker = MixedAPIChecker(scores["bound"], scores["self"], scores["world"])
    assert _planner(checker)._configuration_validity(np.zeros((4, 2))) == [
        True, True, False, True,
    ]


@pytest.mark.parametrize("self_enabled,world_enabled", [(False, True), (True, False), (False, False)])
def test_absent_optional_costs_do_not_remove_joint_bounds(self_enabled, world_enabled):
    bound = torch.tensor([[[0.], [1.]]])
    checker = MixedAPIChecker(bound, torch.zeros(1, 2, 1) if self_enabled else None,
                              torch.zeros(1, 2, 1) if world_enabled else None)
    assert _planner(checker)._configuration_validity([[0., 0.], [1., 1.]]) == [True, False]


def test_unrelated_cost_exception_is_not_swallowed():
    checker = MixedAPIChecker(torch.zeros(1, 1, 1), None, torch.zeros(1, 1, 1))
    def fail(*args, **kwargs):
        raise RuntimeError("CUDA collision failure")
    checker.collision_constraint.forward = fail
    with pytest.raises(RuntimeError, match="CUDA collision failure"):
        _planner(checker)._configuration_validity([[0., 0.]])


def test_mounting_sphere_mask_is_scoped_and_restored_on_cost_exception():
    checker = MixedAPIChecker(torch.zeros(1, 1, 1), None, torch.zeros(1, 1, 1))
    class Params:
        link_spheres = torch.tensor([[0., 0., 0., .02]])
        def disable_link_spheres(self, name):
            assert name == "base"
            self.link_spheres[:, 3] = -100.
    params = Params()
    checker.kinematics = SimpleNamespace(config=SimpleNamespace(kinematics_config=params))
    planner = _planner(checker)
    planner._validator = object()
    planner.mounting_contact_policy = {"robot_links": ["base"]}
    original = params.link_spheres.clone()
    def fail(*args, **kwargs):
        assert params.link_spheres[0, 3] == -100.
        raise RuntimeError("collision cost failed")
    checker.collision_constraint.forward = fail
    with pytest.raises(RuntimeError, match="collision cost failed"):
        planner._configuration_validity([[0., 0.]])
    torch.testing.assert_close(params.link_spheres, original)
