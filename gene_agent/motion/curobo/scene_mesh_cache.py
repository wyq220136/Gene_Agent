"""作用域内有界局部shape缓存；不缓存世界pose、启用策略或可达结论。"""

import hashlib
import numpy as np


def cached_mesh(owner, model, geom_id, margin, build):
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
    key = hashlib.sha256(b"".join(parts)).digest()
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
