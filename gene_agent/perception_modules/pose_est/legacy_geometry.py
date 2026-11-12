"""旧感知接口所需的 Concept 形状视图；几何真值来自新 Knowledge。"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from gene_agent.knowledge.geometry.loader import load_concept
from gene_agent.knowledge.geometry.transforms import validate_pose as _validate_pose


def validate_pose(value, *, name: str = "pose") -> np.ndarray:
    """旧感知调用方使用 name 关键字；保持该签名并复用统一 SO(3) 校验。"""
    return _validate_pose(value, field_name=name)


def camera_object_to_task_object(task_from_camera, camera_from_object) -> np.ndarray:
    """T_task_object = T_task_camera @ T_camera_object，平移单位米。"""
    return validate_pose(
        validate_pose(task_from_camera, name="T_task_camera")
        @ validate_pose(camera_from_object, name="T_camera_object"),
        name="T_task_object",
    )


def load_camera_object_pose(path: str | Path, *, camera_id: str = "0") -> np.ndarray:
    source = Path(path).expanduser()
    payload = json.loads(source.read_text(encoding="utf-8"))
    key = f"camera_parameters_{camera_id}"
    values = payload.get(key)
    if not isinstance(values, list) or not values:
        raise ValueError(f"pose 文件缺少 {key}: {source}")
    return validate_pose(values[0], name="T_camera_object")


@dataclass(frozen=True)
class _LegacyMeshView:
    pose: np.ndarray
    local_vertices: np.ndarray
    faces: np.ndarray
    size_xyz: np.ndarray | None = None


def _box_mesh(size_xyz_m, pose_in_part=None) -> tuple[np.ndarray, np.ndarray]:
    half = np.asarray(size_xyz_m, dtype=float) / 2.0
    vertices = np.array(
        [[x, y, z] for x in (-half[0], half[0])
         for y in (-half[1], half[1]) for z in (-half[2], half[2])],
        dtype=float,
    )
    if pose_in_part is not None:
        pose = validate_pose(pose_in_part, name="T_part_segment")
        vertices = (pose[:3, :3] @ vertices.T).T + pose[:3, 3]
    faces = np.array(
        [[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5],
         [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6],
         [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]], dtype=np.int32,
    )
    return vertices, faces


def load_microwave_concepts(path: str | Path) -> tuple[_LegacyMeshView, ...]:
    """返回旧门板平面修正需要的 door、handle、cavity 局部几何。"""
    door = load_concept(path, semantic="door").concept
    handle = load_concept(path, semantic="handle").concept
    cavity = load_concept(path, semantic="cavity").concept
    door_vertices, door_faces = _box_mesh(door.size_xyz_m)
    handle_vertices: list[np.ndarray] = []
    handle_faces: list[np.ndarray] = []
    for segment in (handle.grip_segment, handle.positive_connector, handle.negative_connector):
        vertices, faces = _box_mesh(segment.size_xyz_m, segment.pose_in_trifold)
        handle_faces.append(faces + sum(len(item) for item in handle_vertices))
        handle_vertices.append(vertices)
    return (
        _LegacyMeshView(door.pose_in_object, door_vertices, door_faces, door.size_xyz_m),
        _LegacyMeshView(handle.pose_in_object, np.concatenate(handle_vertices), np.concatenate(handle_faces)),
        _LegacyMeshView(cavity.pose_in_object, np.empty((0, 3)), np.empty((0, 3), dtype=np.int32)),
    )
