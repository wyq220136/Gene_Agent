"""从 MuJoCo 编译碰撞几何获取刚体的局部保守边界；不做 IK 或动力学。"""
from __future__ import annotations

import mujoco
import numpy as np


def collision_geom_bounds(model, data, geom_id: int, body_id: int) -> np.ndarray:
    body = data.body(body_id)
    rotation = body.xmat.reshape(3,3).T @ data.geom_xmat[geom_id].reshape(3,3)
    center = body.xmat.reshape(3,3).T @ (data.geom_xpos[geom_id]-body.xpos)
    kind = int(model.geom_type[geom_id])
    if kind == mujoco.mjtGeom.mjGEOM_MESH:
        mesh = model.geom_dataid[geom_id]
        start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        points = model.mesh_vert[start:start+count] @ rotation.T + center
        return np.vstack((points.min(axis=0), points.max(axis=0)))
    if kind == mujoco.mjtGeom.mjGEOM_BOX:
        extent = np.abs(rotation) @ model.geom_size[geom_id,:3]
    elif kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
        radius, half = model.geom_size[geom_id,:2]
        extent = radius*np.linalg.norm(rotation[:,:2],axis=1)+half*np.abs(rotation[:,2])
    else:
        raise ValueError(f"刚体碰撞边界不支持 geom 类型 {kind}")
    return np.vstack((center-extent,center+extent))


def collision_geoms(model, body_name: str):
    root = model.body(body_name).id
    for geom in range(model.ngeom):
        body = int(model.geom_bodyid[geom])
        while body and body != root:
            body = int(model.body_parentid[body])
        if body == root and (model.geom_contype[geom] or model.geom_conaffinity[geom]):
            yield geom


def collision_vertices_object(model, data, body_name: str) -> np.ndarray:
    body_id = model.body(body_name).id
    vertices = []
    for geom in collision_geoms(model, body_name):
        bounds = collision_geom_bounds(model,data,geom,body_id)
        vertices.extend([[x,y,z] for x in bounds[:,0] for y in bounds[:,1] for z in bounds[:,2]])
    if not vertices:
        raise ValueError(f"{body_name} 没有启用的碰撞几何")
    return np.asarray(vertices)


def collision_spheres_object(model, data, body_name: str, *, max_cell_m: float=.04) -> np.ndarray:
    from gene_agent.motion.curobo.gripper import cover_bounds
    root = model.body(body_name).id
    rows = []
    for geom in collision_geoms(model, body_name):
        rows.extend([*sphere["center"],sphere["radius"]] for sphere in cover_bounds(
            collision_geom_bounds(model,data,geom,root),max_cell_m=max_cell_m))
    if not rows:
        raise ValueError(f"{body_name} 没有启用的碰撞几何")
    return np.asarray(rows)


def visual_mesh_object(model, data, body_name: str):
    """导出 body frame 下的真实尺度 visual；编译 mesh 和 geom 变换各应用一次。"""
    import trimesh
    root = model.body(body_name).id
    body = data.body(root)
    pieces = []
    for geom in range(model.ngeom):
        ancestor = int(model.geom_bodyid[geom])
        while ancestor and ancestor != root:
            ancestor = int(model.body_parentid[ancestor])
        if ancestor != root or model.geom_group[geom] == 3 or model.geom_rgba[geom, 3] <= 0:
            continue
        # 源资产同时存在 visual 与接触 proxy，只导出可见 visual。
        if model.geom_contype[geom] or model.geom_conaffinity[geom]:
            continue
        kind = int(model.geom_type[geom])
        if kind == mujoco.mjtGeom.mjGEOM_MESH:
            mesh = int(model.geom_dataid[geom])
            va, vn = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
            fa, fn = int(model.mesh_faceadr[mesh]), int(model.mesh_facenum[mesh])
            piece = trimesh.Trimesh(vertices=model.mesh_vert[va:va+vn].copy(),
                                    faces=model.mesh_face[fa:fa+fn].copy(), process=False)
        elif kind == mujoco.mjtGeom.mjGEOM_BOX:
            piece = trimesh.creation.box(extents=2*model.geom_size[geom, :3])
        elif kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
            piece = trimesh.creation.cylinder(radius=model.geom_size[geom, 0],
                                               height=2*model.geom_size[geom, 1])
        else:
            raise ValueError(f'{body_name} 不支持导出 visual 类型 {kind}')
        rotation = body.xmat.reshape(3, 3).T @ data.geom_xmat[geom].reshape(3, 3)
        center = body.xmat.reshape(3, 3).T @ (data.geom_xpos[geom]-body.xpos)
        piece.vertices = piece.vertices @ rotation.T + center
        pieces.append(piece)
    if not pieces:
        raise ValueError(f'{body_name} 没有可导出的 visual mesh')
    return trimesh.util.concatenate(pieces)
