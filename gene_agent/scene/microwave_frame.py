"""旧微波炉操作 frame 与当前校准 visual 的临时叠图对齐。

旧放入/取出 keypose 不能直接作用在新 URDF 的 root 上。这里用同一门板
及把手 visual 建立一个可复查的刚体叠图初值；它只供检查，不代表炉腔、
旋钮和所有接触位置均已标定，更不能自动宣布规划可用。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh

from gene_agent.knowledge.geometry import validate_pose


# 旧网格的门宽/高度/深度分别沿 X/Y/Z；当前校准 URDF 沿 X/Z/Y。
# 旧门的把手在门板 X 小端，当前 URDF 中把手在 X 大端，因此宽度轴反向。
_CALIBRATED_FROM_LEGACY_AXES = np.array([
    [-1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0],
    [0.0, 1.0, 0.0],
])


@dataclass(frozen=True)
class LegacyFrameAlignment:
    T_calibrated_legacy: np.ndarray
    handle_center_residual_m: float
    door_size_max_residual_m: float
    accepted_for_planning: bool = False

    def project_for_preview(self, T_legacy_target: np.ndarray) -> np.ndarray:
        """仅生成 Viser 叠图目标；Skill 规划不得把此结果当成标定事实。"""
        local = validate_pose(T_legacy_target, field_name="T_legacy_target")
        return validate_pose(
            self.T_calibrated_legacy @ local,
            field_name="T_calibrated_target_preview",
        )


def _compiled_geom_vertices_in_root(model, data, name: str, T_world_root: np.ndarray):
    import mujoco
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
    if geom_id < 0:
        raise ValueError(f"当前场景缺少 visual geom：{name}")
    mesh_id = int(model.geom_dataid[geom_id])
    if mesh_id < 0:
        raise ValueError(f"{name} 不是 mesh geom")
    start = int(model.mesh_vertadr[mesh_id])
    count = int(model.mesh_vertnum[mesh_id])
    face_start = int(model.mesh_faceadr[mesh_id])
    face_count = int(model.mesh_facenum[mesh_id])
    vertices = np.asarray(model.mesh_vert[start : start + count], dtype=float)
    faces = np.asarray(model.mesh_face[face_start : face_start + face_count], dtype=int)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    T_world_geom = np.eye(4)
    T_world_geom[:3, :3] = data.geom_xmat[geom_id].reshape(3, 3)
    T_world_geom[:3, 3] = data.geom_xpos[geom_id]
    T_root_geom = np.linalg.inv(T_world_root) @ T_world_geom
    points = (T_root_geom[:3, :3] @ vertices.T).T + T_root_geom[:3, 3]
    center = T_root_geom[:3, :3] @ mesh.centroid + T_root_geom[:3, 3]
    return points, center


def estimate_legacy_microwave_frame(
    model, data, task_dir: str | Path,
) -> LegacyFrameAlignment:
    """从旧门/把手网格与当前 MuJoCo 编译 visual 求叠图初值。

    场景的门应当关闭；若门打开，门板相对 root 转动后无法和旧静态网格
    直接求同一变换。微波炉整体可移动，因为测量都先转回它的 root。
    """
    import mujoco
    joint_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "microwave_door_joint"
    )
    if joint_id < 0 or abs(float(data.qpos[model.jnt_qposadr[joint_id]])) > 1e-6:
        raise ValueError("资产 frame 对齐要求微波炉门处于关闭角 0")
    body = data.body("microwave")
    T_world_root = np.eye(4)
    T_world_root[:3, :3] = body.xmat.reshape(3, 3)
    T_world_root[:3, 3] = body.xpos
    door_vertices, _ = _compiled_geom_vertices_in_root(
        model, data, "microwave_door_door_frame_visual", T_world_root
    )
    _, new_handle_center = _compiled_geom_vertices_in_root(
        model, data, "microwave_door_handle_visual", T_world_root
    )
    return estimate_legacy_microwave_frame_from_meshes(task_dir, door_vertices, new_handle_center)


def estimate_legacy_microwave_frame_from_meshes(task_dir, door_vertices, new_handle_center):
    """门关闭时在校准 root frame 中的 visual 顶点/把手质心；无需物理引擎。"""
    mesh_dir = Path(task_dir) / "objs/Microwave/Microwave_0/meshes"
    old_door = trimesh.load_mesh(mesh_dir / "handle_door_mesh.obj", process=False)
    old_handle = trimesh.load_mesh(mesh_dir / "handle_mesh.obj", process=False)
    if not isinstance(old_door, trimesh.Trimesh) or not isinstance(old_handle, trimesh.Trimesh):
        raise ValueError("旧微波炉门和把手必须是单个 mesh")
    door_vertices = np.asarray(door_vertices, dtype=float)
    new_handle_center = np.asarray(new_handle_center, dtype=float)
    rotation = _CALIBRATED_FROM_LEGACY_AXES
    old_door_center = old_door.bounds.mean(axis=0)
    new_door_bounds = np.stack([door_vertices.min(axis=0), door_vertices.max(axis=0)])
    new_door_center = new_door_bounds.mean(axis=0)
    translation = new_door_center - rotation @ old_door_center
    T_calibrated_legacy = np.eye(4)
    T_calibrated_legacy[:3, :3] = rotation
    T_calibrated_legacy[:3, 3] = translation
    old_door_size_rotated = np.ptp((rotation @ old_door.vertices.T).T, axis=0)
    new_door_size = np.ptp(door_vertices, axis=0)
    return LegacyFrameAlignment(
        T_calibrated_legacy=validate_pose(
            T_calibrated_legacy, field_name="T_calibrated_legacy"
        ),
        handle_center_residual_m=float(np.linalg.norm(
            rotation @ old_handle.centroid + translation - new_handle_center
        )),
        door_size_max_residual_m=float(np.max(np.abs(
            old_door_size_rotated - new_door_size
        ))),
    )


__all__ = ["LegacyFrameAlignment", "estimate_legacy_microwave_frame", "estimate_legacy_microwave_frame_from_meshes"]
