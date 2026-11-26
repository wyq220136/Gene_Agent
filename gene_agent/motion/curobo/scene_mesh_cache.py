"""作用域内有界局部shape缓存；不缓存世界pose、启用策略或可达结论。"""

import hashlib
from contextlib import contextmanager
from contextvars import ContextVar
from types import SimpleNamespace
from pathlib import Path
from uuid import uuid4
import numpy as np


def mesh_geometry_key(model, geom_id, margin):
    import mujoco

    kind = int(model.geom_type[geom_id])
    parts = [
        np.asarray([kind, margin], dtype=np.float64).tobytes(),
        np.asarray(model.geom_size[geom_id]).tobytes(),
    ]
    if kind == int(mujoco.mjtGeom.mjGEOM_MESH):
        mid = int(model.geom_dataid[geom_id])
        start, n = int(model.mesh_vertadr[mid]), int(model.mesh_vertnum[mid])
        parts.append(np.asarray(model.mesh_vert[start : start + n]).tobytes())
        if hasattr(model, "mesh_faceadr") and hasattr(model, "mesh_facenum"):
            face_start, face_count = int(model.mesh_faceadr[mid]), int(model.mesh_facenum[mid])
            parts.append(np.asarray(model.mesh_face[face_start:face_start + face_count]).tobytes())
    return hashlib.sha256(b"".join(parts)).digest()


def cached_mesh(owner, model, geom_id, margin, build, *, key=None):
    key = mesh_geometry_key(model, geom_id, margin) if key is None else key
    if key not in owner.mesh_cache:
        if len(owner.mesh_cache) >= 128:
            del owner.mesh_cache[next(iter(owner.mesh_cache))]
        owner.mesh_cache[key] = build(model, geom_id, margin)
        owner.mesh_misses += 1
    else:
        owner.mesh_hits += 1
    geometry, description = owner.mesh_cache[key]
    # 独立返回，防止后续transform/export改坏共享局部几何。
    return geometry.copy(), description


def cached_mesh_asset(owner, model, geom_id, margin, build, output_dir):
    """相同局部几何只序列化/写入一次；世界位姿仍由各快照独立保存。"""
    from .world_update import mesh_content_hashes, mesh_file_token, remember_mesh_digest
    key = mesh_geometry_key(model, geom_id, margin)
    entry = owner.mesh_export_cache.get(key)
    if entry is not None:
        path, digest, approximation, signature = entry
        try:
            token = mesh_file_token(path)
            unchanged = token is not None and token == signature
            if not unchanged:
                # 不能监测或文件身份变化时安全重读；内容仍一致就避免重新序列化。
                unchanged = mesh_content_hashes({"mesh": {"asset": {"file_path": str(path)}}}).get("asset") == digest
                if unchanged:
                    owner.mesh_export_cache[key] = (path, digest, approximation, mesh_file_token(path))
        except OSError:
            unchanged = False
        if unchanged:
            owner.mesh_hits += 1
            owner.mesh_export_hits += 1
            return path, digest, approximation
        owner.mesh_export_cache.pop(key, None)
    owner.mesh_export_misses += 1
    mesh, approximation = cached_mesh(owner, model, geom_id, margin, build, key=key)
    payload = mesh.export(file_type="obj")
    content = payload.encode("utf-8") if isinstance(payload, str) else bytes(payload)
    digest = hashlib.sha256(content).hexdigest()
    if owner.mesh_asset_dir is None:
        owner.mesh_asset_dir = Path(output_dir).resolve() / ".mesh_assets"
    owner.mesh_asset_dir.mkdir(parents=True, exist_ok=True)
    path = owner.mesh_asset_dir / (digest + ".obj")
    temporary = path.with_name(path.name + "." + uuid4().hex + ".pending")
    try:
        temporary.write_bytes(content)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    remember_mesh_digest(path, digest)
    signature = mesh_file_token(path)
    if len(owner.mesh_export_cache) >= 128:
        owner.mesh_export_cache.pop(next(iter(owner.mesh_export_cache)))
    owner.mesh_export_cache[key] = (path, digest, approximation, signature)
    return path, digest, approximation


_ASSET_CACHE = ContextVar("curobo_collision_mesh_assets", default=None)


def current_mesh_asset_cache():
    return _ASSET_CACHE.get()


@contextmanager
def mesh_asset_cache_scope(owner=None):
    """baseline 和复用引擎都只在本任务内共享静态资产。"""
    cache = owner or SimpleNamespace(mesh_cache={}, mesh_hits=0, mesh_misses=0,
        mesh_export_cache={}, mesh_asset_dir=None, mesh_export_hits=0, mesh_export_misses=0)
    token = _ASSET_CACHE.set(cache)
    try:
        yield cache
    finally:
        cache.mesh_cache.clear()
        cache.mesh_export_cache.clear()
        _ASSET_CACHE.reset(token)
