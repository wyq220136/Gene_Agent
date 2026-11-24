"""仅旧四参数 cuRobo 使用的真实网格距离函数；由 mesh_sdf 延迟导入。"""
import warp as wp

from curobo._src.geom.data.data_mesh import MeshDataWarp, get_obs_idx


def compute_local_sdf_with_grad(
    obs_set: MeshDataWarp,
    env_idx: wp.int32,
    local_idx: wp.int32,
    local_pt: wp.vec3,
) -> wp.vec4:
    flat_idx = get_obs_idx(env_idx, local_idx, obs_set.max_n)
    mesh_id = obs_set.mesh_ids[flat_idx]
    # triangle 0 的第一个顶点位于真实表面，故最近距离必不大于到该点的
    # 距离。小余量避免查询恰好落在 Warp 搜索边界；不依赖 mesh 是否居中。
    surface_point = wp.mesh_eval_position(mesh_id, 0, 1.0, 0.0)
    vertex_distance = wp.length(surface_point - local_pt)
    max_distance = vertex_distance + 1.0e-5 * wp.max(1.0, vertex_distance)
    result = wp.mesh_query_point(mesh_id, local_pt, max_distance)
    if not result.result:
        return wp.vec4(max_distance, 0.0, 0.0, 0.0)
    closest = wp.mesh_eval_position(mesh_id, result.face, result.u, result.v)
    delta = closest - local_pt
    distance = wp.length(delta)
    gradient = wp.vec3(0.0, 0.0, 0.0)
    if distance > 1.0e-8:
        # 与新 cuRobo 的 -grad(SDF) 约定相同，内部和外部都指向障碍内部。
        gradient = result.sign * delta / distance
    return wp.vec4(distance * result.sign, gradient[0], gradient[1], gradient[2])
