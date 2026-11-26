"""可选复用模式的生命周期与候选边界，不需要CUDA。"""

import pytest


def test_default_and_nested_scope_restore():
    from gene_agent.motion.curobo.optimization import (
        optimization_scope,
        current_optimization,
    )

    assert current_optimization() is None
    with optimization_scope("single_reuse") as owner:
        assert current_optimization() is owner
        with optimization_scope("baseline"):
            assert current_optimization() is None
        assert current_optimization() is owner
    assert current_optimization() is None
    assert owner.closed


def test_handle_rejects_after_scope_and_update_failure(monkeypatch):
    from gene_agent.motion.curobo.optimization import optimization_scope, create_planner

    class Engine:
        def __init__(self, settings):
            self.closed = False

        def reset_seed(self):
            pass

        def activate(self, settings):
            raise ValueError("bad geometry")

        def close(self):
            self.closed = True

        def plan(self):
            return {"success": True}

    with optimization_scope("single_reuse") as owner:
        monkeypatch.setattr(owner, "_engine_factory", Engine)
        handle = create_planner(x=1)
        assert handle.plan()["success"]
        handle.update_scene(x=2)
        with pytest.raises(ValueError, match="bad geometry"):
            handle.plan()
        with pytest.raises(RuntimeError):
            handle.plan()
    with pytest.raises(RuntimeError):
        handle.plan()


def test_unknown_mode_rejected():
    from gene_agent.motion.curobo.optimization import optimization_scope

    with pytest.raises(ValueError):
        with optimization_scope("typo"):
            pass


def test_unset_handle_diagnostics_keep_baseline_attribute_semantics():
    from types import SimpleNamespace
    from gene_agent.motion.curobo.adapter import CuroboPlannerAdapter
    from gene_agent.motion.curobo.optimization import optimization_scope, create_planner
    from gene_agent.motion.recording import motion_recording
    with optimization_scope('single_reuse') as owner:
        handle = create_planner()
        assert not hasattr(handle, 'diagnostic_options')
        assert getattr(handle, 'diagnostic_snapshot', {}) == {}
        assert owner.engine is None
        with motion_recording(SimpleNamespace(options={'mode': 'on_failure'}, request_output_dir=None)):
            CuroboPlannerAdapter(handle, group='right', scene_revision=0)
        assert handle.diagnostic_options.mode == 'on_failure'
        assert owner.engine is None
        configured = object()
        handle.diagnostic_options = configured
        assert handle.diagnostic_options is configured


def test_ik_bucket_never_discards_targets():
    from gene_agent.motion.curobo.optimization import ik_bucket_size

    assert [ik_bucket_size(n) for n in [1, 16, 25, 40, 119, 120]] == [
        25,
        25,
        25,
        119,
        119,
        120,
    ]
    with pytest.raises(ValueError):
        ik_bucket_size(0)


def test_batch_world_rejects_overflow_before_mutation():
    from types import SimpleNamespace
    from gene_agent.motion.curobo.batch_world import load_batch

    calls = []
    data = SimpleNamespace(
        cuboids=SimpleNamespace(max_n=1),
        meshes=None,
        num_envs=1,
        clear=lambda index: calls.append(index),
    )
    scene = SimpleNamespace(
        voxel=None,
        cuboid=[SimpleNamespace(name="a"), SimpleNamespace(name="b")],
        mesh=[],
    )
    with pytest.raises(ValueError, match="容量"):
        load_batch(data, scene)
    assert calls == []


def test_batch_pose_rejects_missing_world_before_writing():
    from types import SimpleNamespace
    from gene_agent.motion.curobo.batch_world import apply_batch

    world = SimpleNamespace(
        get_obstacle_names=lambda: ["a"], num_envs=1, data=SimpleNamespace(voxels=None)
    )
    with pytest.raises(ValueError, match="缺少"):
        apply_batch([(world, {"b"})], {})


def test_unsupported_world_is_rejected_before_gpu_allocation():
    from gene_agent.motion.curobo.reusable import validate_settings

    with pytest.raises(ValueError, match="cuboid/mesh"):
        validate_settings({"scene_model": {"voxel": {"v": {}}}})


def test_mesh_cache_tracks_geometry_and_returns_independent_objects():
    import numpy as np
    from types import SimpleNamespace

    mujoco = pytest.importorskip("mujoco")
    from gene_agent.motion.curobo.scene import _build_mesh_for_geom
    from gene_agent.motion.curobo.scene_mesh_cache import cached_mesh

    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><geom type="sphere" size=".1"/></worldbody></mujoco>'
    )
    owner = SimpleNamespace(mesh_cache={}, mesh_hits=0, mesh_misses=0)
    a, _ = cached_mesh(owner, model, 0, 0.0, _build_mesh_for_geom)
    expected = a.vertices.copy()
    a.vertices[:] = 999
    b, _ = cached_mesh(owner, model, 0, 0.0, _build_mesh_for_geom)
    assert np.array_equal(b.vertices, expected)
    model.geom_size[0, 0] = 0.2
    c, _ = cached_mesh(owner, model, 0, 0.0, _build_mesh_for_geom)
    assert not np.array_equal(c.vertices, expected)
    assert owner.mesh_hits == 1 and owner.mesh_misses == 2


@pytest.mark.parametrize(
    "attachment", [[], [[0.0, 0.0], [0.0, 0.2]], [0.0, 0.0, 0.0, 0.2]]
)
def test_invalid_attachment_rejected_during_activation(attachment):
    import numpy as np
    from gene_agent.motion.curobo.reusable import padded_config

    settings = dict(
        robot_config={"kinematics": {"collision_spheres": {"tool": []}}},
        tool_frame="tool",
        link_from_tcp=np.eye(4),
        attachment_spheres_tool=attachment,
    )
    with pytest.raises(ValueError):
        padded_config(settings)


def test_diagnostics_do_not_leak_and_close_releases_engine(monkeypatch):
    from gene_agent.motion.curobo.optimization import optimization_scope, create_planner

    class Engine:
        def __init__(self, settings):
            pass

        def activate(self, settings):
            pass

        def reset_seed(self):
            pass

        def close(self):
            pass

        def plan(self):
            return (
                self.diagnostic_snapshot,
                self.diagnostic_options,
                self.capture_solver_repr,
            )

    with optimization_scope("single_reuse") as owner:
        monkeypatch.setattr(owner, "_engine_factory", Engine)
        a = create_planner()
        a.diagnostic_snapshot = {"name": "A"}
        a.capture_solver_repr = True
        assert a.plan()[0] == {"name": "A"}
        b = create_planner()
        assert b.plan() == ({}, None, False)
    assert owner.engine is None


def test_cross_thread_update_and_close_are_rejected():
    import threading
    from gene_agent.motion.curobo.optimization import optimization_scope, create_planner

    with optimization_scope("single_reuse") as owner:
        handle = create_planner()
        errors = []

        def other():
            for operation in (lambda: handle.update_scene(x=2), owner.close):
                try:
                    operation()
                except RuntimeError:
                    errors.append(True)

        thread = threading.Thread(target=other)
        thread.start()
        thread.join()
        assert len(errors) == 2 and handle.revision == 0 and not owner.closed


def test_pool_cannot_reuse_entry_from_another_scope(monkeypatch):
    from gene_agent.motion.curobo.optimization import optimization_scope, create_planner
    from gene_agent.motion.curobo.session import CuroboPlannerPool
    from gene_agent.motion.curobo import core, reusable

    class Fake:
        def __init__(self, *a, **kw):
            pass

        def close(self):
            pass

        def reset_seed(self):
            pass

        def activate(self, s):
            pass

        def plan(self):
            return True

    monkeypatch.setattr(core, "SynchronousPlanner", Fake)
    monkeypatch.setattr(reusable, "ReusablePlanner", Fake)
    snapshot = {"scene_model": {}, "scope": "test", "snapshot_sha256": "hash"}
    with CuroboPlannerPool(planner_factory=create_planner) as pool:
        for mode in ["baseline", "single_reuse", "single_reuse", "baseline"]:
            with optimization_scope(mode):
                lease = pool.acquire({}, snapshot)
                assert not lease.cache_hit
                assert lease.planner.plan()


def test_exact_geometry_identity_and_optional_scene_hash():
    from gene_agent.motion.curobo.reusable import exact_digest, scene_hash

    assert exact_digest({"x": 0.001}) != exact_digest({"x": 0.0010000004})
    assert scene_hash({"scene_model": {}}) == scene_hash(
        {"scene_model": {}, "scene_snapshot_sha256": None}
    )
    assert scene_hash({"scene_model": {}, "scene_snapshot_sha256": "given"}) == "given"


def test_optimized_replay_selects_same_configuration(monkeypatch):
    from gene_agent.motion.curobo.diagnostics import _replay_backend
    from gene_agent.motion.curobo import reusable

    seen = []
    monkeypatch.setattr(
        reusable,
        "ReusablePlanner",
        lambda settings: seen.append(settings) or "optimized",
    )
    factory, _, settings = _replay_backend(
        {
            "replay_config": {
                "planner_variant": "v2",
                "optimization_mode": "single_reuse",
                "x": 1,
            }
        }
    )
    assert factory(**settings) == "optimized" and seen == [{"x": 1}]


def test_optimized_pool_keeps_subnanometer_geometry_changes():
    import copy
    from gene_agent.motion.curobo.optimization import optimization_scope, create_planner
    from gene_agent.motion.curobo.session import CuroboPlannerPool

    snap = {
        "scene_model": {
            "cuboid": {
                "a": {
                    "dims": [0.1, 0.1, 0.1],
                    "pose": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                }
            }
        },
        "scope": "test",
        "snapshot_sha256": "a",
    }
    with optimization_scope("single_reuse"):
        with CuroboPlannerPool(planner_factory=create_planner) as pool:
            a = pool.acquire(
                {}, snap, attachment_spheres_tool=[[0.001, 0.0, 0.0, 0.02]]
            )
            updated = copy.deepcopy(snap)
            updated["scene_model"]["cuboid"]["a"]["pose"][0] = 4e-10
            updated["snapshot_sha256"] = "b"
            b = pool.acquire(
                {}, updated, attachment_spheres_tool=[[0.0010000004, 0.0, 0.0, 0.02]]
            )
            assert b.planner.settings["attachment_spheres_tool"][0][0] == 0.0010000004
            assert b.planner.settings["scene_model"]["cuboid"]["a"]["pose"][0] == 4e-10


def test_copied_context_cannot_reenter_closed_or_foreign_owner():
    import contextvars, threading
    from gene_agent.motion.curobo.optimization import (
        optimization_scope,
        current_optimization,
    )

    with optimization_scope("single_reuse"):
        ctx = contextvars.copy_context()
        errors = []

        def other():
            try:
                ctx.run(current_optimization)
            except RuntimeError:
                errors.append(True)

        thread = threading.Thread(target=other)
        thread.start()
        thread.join()
        assert errors == [True]
    with pytest.raises(RuntimeError):
        ctx.run(current_optimization)
