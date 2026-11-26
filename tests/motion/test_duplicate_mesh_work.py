"""重复导出/哈希/共享 world 更新的 CPU 检查；不创建 cuRobo solver。"""
from pathlib import Path
from types import SimpleNamespace
import os

import numpy as np
import pytest

from gene_agent.motion.curobo.world_update import MeshFileHashCache, mesh_content_hashes, mesh_file_cache_scope


@pytest.fixture
def monitored(monkeypatch):
    # 性能快路径用确定性的事件源测试；主机配额耗尽的真实回退另测。
    from gene_agent.motion.curobo import mesh_file_watch
    class Watch:
        fd = 1
        enabled_ever = True
        def __init__(self): self.events = {}
        def token(self, path): return (0, self.events.get(str(path), 0))
        def emit(self, path): self.events[str(path)] = self.events.get(str(path), 0) + 1
        def close(self): self.fd = -1
    monkeypatch.setattr(mesh_file_watch, "MeshFileWatch", Watch)
    return Watch


def scene(path):
    return {"mesh": {"target": {"file_path": str(path), "pose": [0, 0, 0, 1, 0, 0, 0]}}}


def test_scoped_hash_cache_avoids_reread_but_invalidates_same_size_restored_mtime(tmp_path, monitored):
    path = tmp_path / "mesh.obj"
    path.write_bytes(b"first")
    with mesh_file_cache_scope() as cache:
        old = mesh_content_hashes(scene(path))
        assert mesh_content_hashes(scene(path)) == old
        assert cache.bytes_read == 5 and cache.hits == 1
        before = path.stat()
        path.write_bytes(b"other")
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
        cache.watch.emit(path)
        assert mesh_content_hashes(scene(path)) != old
        assert cache.bytes_read == 10
    with mesh_file_cache_scope() as fresh:
        mesh_content_hashes(scene(path))
        assert fresh.bytes_read == 5 and fresh.hits == 0


def test_replaced_deleted_or_external_reference_mesh_never_uses_stale_digest(tmp_path):
    path = tmp_path / "mesh.obj"
    path.write_bytes(b"first")
    with mesh_file_cache_scope():
        old = mesh_content_hashes(scene(path))
        replacement = tmp_path / "replacement.obj"
        replacement.write_bytes(b"other")
        replacement.replace(path)
        assert mesh_content_hashes(scene(path)) != old
        path.unlink()
        assert mesh_content_hashes(scene(path)) == {"target": None}
        gltf = tmp_path / "mesh.gltf"
        gltf.write_text("{}")
        assert mesh_content_hashes(scene(gltf)) == {"target": None}


def test_file_cache_is_bounded(tmp_path, monitored):
    cache = MeshFileHashCache(capacity=2)
    for index in range(4):
        path = tmp_path / f"{index}.obj"
        path.write_text(str(index))
        cache.digest(path)
    assert len(cache.entries) == 2


def test_mesh_asset_export_reuses_content_path_and_repairs_missing_or_modified_asset(tmp_path, monitored):
    from gene_agent.motion.curobo.scene_mesh_cache import cached_mesh_asset
    # 只使用枚举/数组和静态假 mesh，不运行 MuJoCo 状态或物理。
    model = SimpleNamespace(geom_type=np.array([2]), geom_size=np.array([[.1, .1, .1]]))
    owner = SimpleNamespace(mesh_cache={}, mesh_hits=0, mesh_misses=0, mesh_export_cache={},
                            mesh_asset_dir=None, mesh_export_hits=0, mesh_export_misses=0)
    counts = {"build": 0, "export": 0}
    class Mesh:
        def __init__(self, payload): self.payload = payload
        def copy(self): return Mesh(self.payload)
        def export(self, *, file_type):
            assert file_type == "obj"
            counts["export"] += 1
            return self.payload
    def build(model, geom_id, margin):
        counts["build"] += 1
        return Mesh(f"v {model.geom_size[0, 0] + margin} 0 0\n"), "fixture"
    with mesh_file_cache_scope() as files:
        first = cached_mesh_asset(owner, model, 0, 0., build, tmp_path / "strict")
        second = cached_mesh_asset(owner, model, 0, 0., build, tmp_path / "contact")
        assert first == second and counts == {"build": 1, "export": 1}
        assert mesh_content_hashes(scene(first[0])) == {"target": first[1]}
        assert files.bytes_read == 0  # 写入者已计算 digest，加载时只核对文件身份。
        original = first[0].read_bytes()
        first[0].write_text("corrupted")
        repaired = cached_mesh_asset(owner, model, 0, 0., build, tmp_path / "later")
        assert repaired[0].read_bytes() == original
        repaired[0].unlink()
        restored = cached_mesh_asset(owner, model, 0, 0., build, tmp_path / "again")
        assert restored[0].read_bytes() == original
        model.geom_size[0, 0] = .2
        changed = cached_mesh_asset(owner, model, 0, 0., build, tmp_path / "changed")
        assert changed[1] != first[1] and counts["build"] == 2


@pytest.mark.parametrize("shared_graph", [False, True])
def test_shared_world_data_loaded_once_and_each_graph_reset(tmp_path, monkeypatch, shared_graph):
    from gene_agent.motion.curobo import batch_world
    loaded = []
    monkeypatch.setattr(batch_world, "load_batch", lambda data, cfg: loaded.append(data))
    data = object()
    one = SimpleNamespace(data=data, scene_model=None)
    two = SimpleNamespace(data=data, scene_model=None)
    class Graph:
        def __init__(self): self.calls = 0
        def reset_buffer(self): self.calls += 1
    ga, gb = Graph(), Graph()
    if shared_graph: gb = ga
    solvers = (SimpleNamespace(scene_collision_checker=one, graph_planner=ga),
               SimpleNamespace(scene_collision_checker=two, graph_planner=gb))
    cfg = object()
    stats = batch_world.update_solver_worlds(solvers, cfg)
    assert loaded == [data] and one.scene_model is cfg and two.scene_model is cfg
    assert ga.calls == gb.calls == 1
    assert stats["solver_count"] == 2 and stats["unique_world_data"] == 1


def test_independent_world_data_still_both_receive_update(monkeypatch):
    from gene_agent.motion.curobo import batch_world
    loaded = []
    monkeypatch.setattr(batch_world, "load_batch", lambda data, cfg: loaded.append(data))
    worlds = [SimpleNamespace(data=object(), scene_model=None) for _ in range(2)]
    batch_world.update_solver_worlds([SimpleNamespace(scene_collision_checker=w) for w in worlds], object())
    assert len(loaded) == 2


@pytest.mark.parametrize("changed", ["mesh", "cuboid"])
def test_partial_world_sync_preserves_unchanged_storage_and_does_not_clear_world(changed):
    from gene_agent.motion.curobo.batch_world import load_batch
    class Storage:
        max_n = 10
        def __init__(self, names): self.names = list(names); self.calls = 0
        def load_batch(self, objects, env_idx):
            self.calls += 1
            self.names = [value.name for value in objects]
    cubes, meshes = Storage(["table"]), Storage(["bowl"])
    cleared = []
    data = SimpleNamespace(num_envs=1, cuboids=cubes, meshes=meshes,
                           clear=lambda env_idx: cleared.append(env_idx))
    cfg = SimpleNamespace(voxel=None, cuboid=[SimpleNamespace(name="table")], mesh=[])
    load_batch(data, cfg, changed_kinds={changed})
    assert not cleared
    if changed == "mesh":
        assert cubes.names == ["table"] and cubes.calls == 0
        assert meshes.names == [] and meshes.calls == 1
    else:
        assert meshes.names == ["bowl"] and meshes.calls == 0
        assert cubes.names == ["table"] and cubes.calls == 1
    assert data.scene_model is cfg


def test_baseline_scope_has_asset_cache_and_releases_it_on_exit():
    from gene_agent.motion.curobo.optimization import optimization_scope
    from gene_agent.motion.curobo.scene_mesh_cache import current_mesh_asset_cache
    assert current_mesh_asset_cache() is None
    with optimization_scope("baseline"):
        cache = current_mesh_asset_cache()
        assert cache is not None
        cache.mesh_export_cache["fixture"] = "entry"
    assert current_mesh_asset_cache() is None and not cache.mesh_export_cache


def test_monitor_unavailable_falls_back_to_fresh_reads(tmp_path, monkeypatch):
    from gene_agent.motion.curobo import mesh_file_watch
    class Unavailable:
        fd = -1
        enabled_ever = False
        def token(self, path): return None
        def close(self): pass
    monkeypatch.setattr(mesh_file_watch, "MeshFileWatch", Unavailable)
    path = tmp_path / "mesh.obj"
    path.write_bytes(b"first")
    with mesh_file_cache_scope() as cache:
        old = mesh_content_hashes(scene(path))
        assert mesh_content_hashes(scene(path)) == old
        assert cache.bytes_read == 10 and cache.hits == 0
        before = path.stat()
        path.write_bytes(b"other")
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
        assert mesh_content_hashes(scene(path)) != old
        assert cache.bytes_read == 15
        assert not cache.entries


def test_asset_reuses_export_even_when_hash_must_be_reread(tmp_path, monitored):
    from gene_agent.motion.curobo.scene_mesh_cache import cached_mesh_asset
    monitored.token = lambda self, path: None
    monitored.enabled_ever = False
    model = SimpleNamespace(geom_type=np.array([2]), geom_size=np.array([[.1, .1, .1]]))
    owner = SimpleNamespace(mesh_cache={}, mesh_hits=0, mesh_misses=0, mesh_export_cache={},
                            mesh_asset_dir=None, mesh_export_hits=0, mesh_export_misses=0)
    calls = []
    class Mesh:
        def copy(self): return self
        def export(self, **kwargs):
            calls.append("export")
            return "v 0 0 0\n"
    with mesh_file_cache_scope() as files:
        a = cached_mesh_asset(owner, model, 0, 0., lambda *args: (Mesh(), "fixture"), tmp_path / "one")
        b = cached_mesh_asset(owner, model, 0, 0., lambda *args: (Mesh(), "fixture"), tmp_path / "two")
        assert a == b and calls == ["export"]
        assert files.bytes_read > 0 and files.hits == 0
