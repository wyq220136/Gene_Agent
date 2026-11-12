"""真实pool加CPU后端替身，验证阶段生命周期与配置隔离。"""
from dataclasses import replace

import numpy as np
import pytest

from gene_agent.motion import JointState, PoseTarget, motion_configuration
from gene_agent.motion.curobo import CuroboPlanningSession
from gene_agent.motion.curobo import session as implementation


@pytest.fixture
def setup(monkeypatch):
    made = []

    class Backend:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            self.joint_names = ("j1",)
            self.closed = False
            self.resets = 0
            self.calls = 0
            made.append(self)
        def close(self):
            self.closed = True
        def reset_seed(self):
            self.resets += 1
        def fk(self, names, positions):
            self.calls += 1
            return np.eye(4)
        def plan(self, *a, **k):
            self.calls += 1
            return {"success": True, "positions_rad": [[.1], [.2]],
                    "interpolation_dt_s": .025, "duration_s": .01}

    monkeypatch.setattr(implementation, "SynchronousPlanner", Backend)
    monkeypatch.setattr(implementation, "load_realman_motion_config", lambda *a, **k: {})
    robot = {"joint_groups": {"right": ["j1"]}, "world_frame": "base"}
    snapshot = {"scene_model": {}, "scope": "test", "snapshot_sha256": "hash",
                "active_group": "right", "world_frame": "base", "collision_mode": "full"}
    return robot, snapshot, JointState(("j1",), (.1,)), made


def test_stage_invalidates_escaped_handles_and_reuses_solver(setup):
    robot, snapshot, start, made = setup
    with CuroboPlanningSession(robot, seed=17) as session:
        with session.stage(snapshot=snapshot, start=start, group="right", scene_revision=2) as first:
            escaped = first.planner
            context = first.context
            assert context.seed == 17
            first.fk()
            with pytest.raises(RuntimeError):
                with session.stage(snapshot=snapshot, start=start, group="right", scene_revision=2):
                    pass
        for call in (first.fk, lambda: escaped.fk(start),
                     lambda: first.evaluate_ik([], request_ids=[])):
            with pytest.raises(RuntimeError):
                call()
        with session.stage(snapshot=snapshot, start=start, group="right", scene_revision=2,
                           restart_sampling=True) as second:
            result = second.plan_to(PoseTarget("world_tcp", np.eye(4)), request_id="next")
            assert result.success and result.metrics["planner_cache"]["cache_hit"]
            assert made[0].resets == 1
        assert first.cache_evidence["cache_hit"] is False
    assert len(made) == 1 and made[0].closed and made[0].calls == 2
    with pytest.raises(RuntimeError):
        with session.stage(snapshot=snapshot, start=start, group="right", scene_revision=2):
            pass


def test_exception_and_failed_entry_leave_no_active_stage(setup):
    robot, snapshot, start, made = setup
    session = CuroboPlanningSession(robot)
    with pytest.raises(ValueError):
        with session:
            with session.stage(snapshot={**snapshot, "active_group": "left"}, start=start,
                               group="right", scene_revision=0):
                pass
    assert made == []
    with pytest.raises(LookupError):
        with CuroboPlanningSession(robot) as session:
            with session.stage(snapshot=snapshot, start=start, group="right", scene_revision=0):
                raise LookupError("fail")
    assert made[0].closed


@pytest.mark.parametrize("bad", ["group", "joints", "frame", "mode", "gripper"])
def test_context_mismatch_fails_before_acquire(setup, bad):
    robot, snapshot, start, made = setup
    group = "right"
    if bad == "group": group = "left"
    elif bad == "joints": start = JointState(("other",), (.1,))
    elif bad == "frame": snapshot = {**snapshot, "world_frame": "other"}
    elif bad == "mode": snapshot = {**snapshot, "collision_mode": "robot_only"}
    elif bad == "gripper": snapshot = {**snapshot, "gripper_snapshot": {"group": "left"}}
    with CuroboPlanningSession(robot) as session:
        with pytest.raises(ValueError):
            with session.stage(snapshot=snapshot, start=start, group=group, scene_revision=0):
                pass
    assert made == []


def test_attachment_identity_and_explicit_mode_conflict(setup):
    robot, snapshot, start, made = setup
    with CuroboPlanningSession(robot) as session:
        for sphere in (None, [[0, 0, 0, .01]], None):
            with session.stage(snapshot=snapshot, start=start, group="right", scene_revision=0,
                               attachment_spheres_tool=sphere):
                pass
    assert len(made) == 2 and all(x.closed for x in made)
    with motion_configuration({"collision_mode": "robot_only"}):
        with pytest.raises(ValueError):
            CuroboPlanningSession(robot, collision_mode="full")
        with CuroboPlanningSession(robot) as inherited:
            assert inherited.collision_mode.value == "robot_only"


def test_shared_pool_cannot_change_world_during_active_stage(setup):
    robot, snapshot, start, made = setup
    with implementation.CuroboPlannerPool(planner_factory=implementation.SynchronousPlanner) as pool:
        with CuroboPlanningSession(robot, planner_pool=pool) as first, \
             CuroboPlanningSession(robot, planner_pool=pool) as second:
            with first.stage(snapshot=snapshot, start=start, group="right", scene_revision=0) as motion:
                with pytest.raises(RuntimeError):
                    with second.stage(snapshot=snapshot, start=start, group="right", scene_revision=0):
                        pass
                pool.clear()
                with pytest.raises(RuntimeError):
                    motion.fk()
    assert len(made) == 1 and made[0].closed


@pytest.mark.parametrize("sphere", [None, [[0., 0., 0., .01]]])
def test_legacy_and_session_share_effectively_identical_cache_entry(setup, sphere):
    robot, snapshot, start, made = setup
    with implementation.CuroboPlannerPool(planner_factory=implementation.SynchronousPlanner) as pool:
        pool.acquire({}, snapshot, attachment_spheres_tool=sphere,
                     planner_mode=implementation.PlannerMode.CUDA_GRAPH_BENCHMARK,
                     random_seed=123)
        with CuroboPlanningSession(robot, planner_pool=pool) as session:
            with session.stage(snapshot=snapshot, start=start, group="right", scene_revision=0,
                               attachment_spheres_tool=sphere) as motion:
                assert motion.cache_evidence["cache_hit"]
        assert len(made) == 1
