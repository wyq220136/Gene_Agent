"""同几何世界的位姿同步；只处理已加载的障碍，不改变接触策略。"""
from __future__ import annotations

import hashlib
import os
import stat
from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

import numpy as np


_SELF_CONTAINED_MESH_SUFFIXES = {'.obj', '.ply', '.stl'}


def _mesh_format(path):
    try:
        suffix = Path(path).suffix.lower()
    except (TypeError, ValueError):
        return None
    return suffix if suffix in _SELF_CONTAINED_MESH_SUFFIXES else None



_FILE_HASH_CACHE = ContextVar("curobo_mesh_file_hash_cache", default=None)


def file_signature(path):
    value = os.stat(path)
    if not stat.S_ISREG(value.st_mode):
        raise OSError("mesh 必须是普通文件")
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


class MeshFileHashCache:
    """任务内有界内容身份缓存；ctime 也参与，覆盖同大小改写并恢复 mtime。"""
    def __init__(self, capacity=256):
        self.capacity = capacity
        self.entries = OrderedDict()
        self.hits = self.misses = self.bytes_read = 0
        from .mesh_file_watch import MeshFileWatch
        self.watch = MeshFileWatch()

    def token(self, path):
        signature = file_signature(path)
        generation = self.watch.token(path)
        return None if generation is None else (signature, generation)

    def remember(self, path, digest, *, token=None):
        token = self.token(path) if token is None else token
        if token is not None:
            self.entries[token] = digest
            self.entries.move_to_end(token)
            while len(self.entries) > self.capacity:
                self.entries.popitem(last=False)
        return token

    def digest(self, path):
        token = self.token(path)
        if token is not None and token in self.entries and self.token(path) == token:
            self.hits += 1
            self.entries.move_to_end(token)
            return self.entries[token]
        self.misses += 1
        digest = hashlib.sha256()
        with open(path, "rb") as stream:
            before = os.fstat(stream.fileno())
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                self.bytes_read += len(chunk)
                digest.update(chunk)
            after = os.fstat(stream.fileno())
        identity = lambda value: (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)
        final_token = self.token(path)
        if (identity(before) != identity(after) or identity(after) != file_signature(path)
                or (token is not None and token != final_token)):
            raise OSError("mesh 读取期间发生变化，不能证明几何身份")
        value = digest.hexdigest()
        if final_token is not None:
            self.remember(path, value, token=final_token)
        return value

    def close(self):
        self.entries.clear()
        self.watch.close()

    def stats(self):
        return dict(hits=self.hits, misses=self.misses, bytes_read=self.bytes_read,
                    change_monitor_available=self.watch.enabled_ever)


@contextmanager
def mesh_file_cache_scope():
    existing = _FILE_HASH_CACHE.get()
    if existing is not None:
        yield existing
        return
    cache = MeshFileHashCache()
    token = _FILE_HASH_CACHE.set(cache)
    try:
        yield cache
    finally:
        cache.close()
        _FILE_HASH_CACHE.reset(token)


def mesh_file_token(path):
    cache = _FILE_HASH_CACHE.get()
    return None if cache is None else cache.token(path)


def remember_mesh_digest(path, digest):
    cache = _FILE_HASH_CACHE.get()
    if cache is not None:
        cache.remember(path, digest)


def mesh_content_hashes(scene: dict) -> dict[str, str | None]:
    """复用任务内未变文件；修改/替换重读，外部引用格式不作几何同一性证明。"""
    from gene_agent.feedback.runtime_timing import measure_runtime
    meshes = scene.get("mesh", {})
    cache = _FILE_HASH_CACHE.get()
    if cache is None:
        with mesh_file_cache_scope():
            return mesh_content_hashes(scene)
    with measure_runtime("mesh_file_read", "hash_mesh_files", metadata={"mesh_count": len(meshes)}):
        result = {}
        for name, row in meshes.items():
            try:
                path = row["file_path"]
                result[name] = cache.digest(path) if _mesh_format(path) is not None else None
            except (OSError, KeyError, TypeError, ValueError):
                result[name] = None
        return result


def pose_updates(previous: dict, current: dict, *, previous_mesh_hashes=None,
                 current_mesh_hashes=None) -> dict | None:
    """返回改变的world pose（米、wxyz）；无法证明已加载几何相同时完整重载。

    mesh必须对照加载时保存的hash和本次新读的hash；重读旧路径不能代替加载身份。
    glTF等文件可引用外部几何，因此即使主文件相同也不复用其GPU几何。
    """
    if previous.keys() != current.keys():
        return None
    updates = {}
    for kind, objects in current.items():
        if objects.keys() != previous[kind].keys():
            return None
        for name, row in objects.items():
            old = previous[kind][name]
            geometry = {k: v for k, v in row.items() if k != 'pose'}
            previous_geometry = {k: v for k, v in old.items() if k != 'pose'}
            if kind == 'mesh':
                mesh_format = _mesh_format(row.get('file_path'))
                if mesh_format is None or mesh_format != _mesh_format(old.get('file_path')):
                    return None
                if previous_mesh_hashes is None or current_mesh_hashes is None:
                    return None
                loaded_hash = previous_mesh_hashes.get(name)
                if loaded_hash is None or loaded_hash != current_mesh_hashes.get(name):
                    return None
                geometry['file_path'] = previous_geometry['file_path']
            if geometry != previous_geometry:
                return None
            pose = np.asarray(row.get('pose'), dtype=float)
            if pose.shape != (7,) or not np.isfinite(pose).all() or not np.isclose(np.linalg.norm(pose[3:]), 1., atol=1e-5):
                raise ValueError(f'障碍 {name} pose 必须是有限 xyz + 单位 wxyz')
            if row['pose'] != old.get('pose'):
                updates[name] = row['pose']
    return updates


def apply_pose_updates(worlds, updates: dict, make_pose) -> None:
    """每个 world 配明确障碍集合；先核对完整集合，再更新，避免静默漏碰撞。"""
    unique = {}
    for world, expected_names in worlds:
        names = set(world.get_obstacle_names())
        if missing := set(expected_names) - names:
            raise ValueError(f'碰撞 world 缺少障碍: {sorted(missing)}')
        unique[id(world)] = (world, names)
    poses = {name: make_pose(pose) for name, pose in updates.items()}
    for world, names in unique.values():
        for name, pose in poses.items():
            if name in names:
                world.update_obstacle_pose(name, pose)
