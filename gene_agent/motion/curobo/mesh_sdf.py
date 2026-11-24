"""旧版 cuRobo mesh SDF 查询范围兼容；不改变几何或允许接触规则。"""
from __future__ import annotations

import inspect
import sys
from threading import RLock

_LOCK = RLock()
_KERNEL_MODULES = (
    "curobo._src.geom.collision.wp_collision_kernel",
    "curobo._src.geom.collision.wp_sweep_collision_kernel",
)


def ensure_mesh_sdf_compatibility() -> dict:
    """在创建 solver/CUDA Graph 前修复旧四参数网格距离查询。

    旧核未把 sphere radius + activation distance 传给 mesh 查询；查询
    未命中时返回 mesh 包围盒半对角线，远处大球因此被误报成穿透。
    新五参数核已有上游半径修复，保持原样。旧核改为真实最近表面查询，
    查询界由真实网格上的一个顶点推导，球覆盖小网格时也不会漏碰撞。
    """
    with _LOCK:
        from curobo._src.geom.data import data_mesh

        function = data_mesh.compute_local_sdf_with_grad
        mode = getattr(function, "_gene_agent_mesh_sdf_mode", None)
        if mode is None:
            parameters = tuple(inspect.signature(function).parameters)
            if parameters == ("obs_set", "env_idx", "local_idx", "local_pt", "query_distance"):
                mode = "native_query_radius"
            elif parameters == ("obs_set", "env_idx", "local_idx", "local_pt"):
                import warp as wp
                from ._legacy_mesh_sdf import compute_local_sdf_with_grad

                # 两种核共享同一 mesh overload；已有导入也须更新，不能只修
                # 诊断而让 IK/连续轨迹仍调用旧函数。wp.func 会替换同签名重载。
                for name in _KERNEL_MODULES:
                    kernel = sys.modules.get(name)
                    if kernel is not None:
                        _register_mesh_overload(wp, kernel, compute_local_sdf_with_grad)
                mode = "legacy_exact_surface_query"
                compute_local_sdf_with_grad._gene_agent_mesh_sdf_mode = mode
                data_mesh.compute_local_sdf_with_grad = compute_local_sdf_with_grad
            else:
                raise RuntimeError(f"不支持的 cuRobo mesh SDF 签名 {parameters}；不能保证碰撞查询正确")
        return {"mode": mode, "source": str(data_mesh.__file__)}


def _register_mesh_overload(wp, kernel, function):
    # Warp 通过同名局部变量找到原 overload 集合，保留 cuboid/voxel 重载。
    compute_local_sdf_with_grad = kernel.compute_local_sdf_with_grad
    kernel.compute_local_sdf_with_grad = wp.func(function, module=kernel.__name__)
