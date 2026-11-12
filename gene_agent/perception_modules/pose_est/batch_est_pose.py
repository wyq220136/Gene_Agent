"""FoundationPose 离线输入、Microwave 修正和 pose.json 保存工具。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .legacy_geometry import validate_pose


@dataclass(frozen=True)
class PoseInputs:
    """一次首帧位姿注册所需的完整输入。"""

    rgb: np.ndarray
    depth: np.ndarray
    mask: np.ndarray
    mesh: object
    intrinsic: np.ndarray
    camera_parameters: dict


def load_rgb_image(path: str | Path) -> np.ndarray:
    """读取 RGB PNG，并返回连续的 HxWx3 uint8 数组。"""
    with Image.open(path) as image:
        return np.ascontiguousarray(np.asarray(image.convert("RGB"), dtype=np.uint8))


def load_depth_image(
    path: str | Path,
    *,
    depth_unit: str = "millimeter",
) -> np.ndarray:
    """读取深度图，并统一转换为米制 float32。"""
    with Image.open(path) as image:
        depth = np.asarray(image)
    if depth.ndim != 2:
        raise ValueError(f"深度图必须是单通道: {path}")
    depth = depth.astype(np.float32)
    if depth_unit == "millimeter":
        depth /= 1000.0
    elif depth_unit != "meter":
        raise ValueError(f"不支持的深度单位: {depth_unit}")
    depth[~np.isfinite(depth) | (depth < 0.001)] = 0.0
    return np.ascontiguousarray(depth)


def load_mask_image(path: str | Path) -> np.ndarray:
    """读取二值 mask，并返回 HxW uint8 数组。"""
    with Image.open(path) as image:
        mask = np.asarray(image.convert("L")) > 0
    if not mask.any():
        raise ValueError(f"mask 为空: {path}")
    return np.ascontiguousarray(mask.astype(np.uint8))


def _load_extrinsic(raw: dict, key: str, path: Path) -> np.ndarray:
    """读取可选的相机外参，并检查其齐次矩阵格式。"""
    if key not in raw:
        return None
    try:
        return validate_pose(raw[key], name=key)
    except ValueError as exc:
        raise ValueError(f"{path} 中的 {key} 无效") from exc


def load_camera_parameters(path: str | Path) -> dict:
    """读取 camera_parameters_<id>.json 的新格式。"""
    camera_path = Path(path)
    with camera_path.open("r", encoding="utf-8") as file:
        raw = json.load(file)
    if "intrinsic_matrix" not in raw:
        raise ValueError(f"相机参数缺少 intrinsic_matrix: {camera_path}")
    intrinsic = np.asarray(raw["intrinsic_matrix"], dtype=np.float64)
    if intrinsic.shape != (3, 3):
        raise ValueError("intrinsic_matrix 必须是 3x3 矩阵")
    if not np.all(np.isfinite(intrinsic)):
        raise ValueError("intrinsic_matrix 必须只包含有限数")

    camera = dict(raw)
    camera["intrinsic_matrix"] = intrinsic
    for key in ("left_extrinsic_matrix", "right_extrinsic_matrix"):
        value = _load_extrinsic(raw, key, camera_path)
        if value is not None:
            camera[key] = value
    return camera


def load_mesh(path: str | Path):
    """惰性导入 trimesh 并读取单个物体 mesh。"""
    import trimesh

    mesh = trimesh.load(path, force="mesh")
    if mesh.is_empty:
        raise ValueError(f"mesh 为空: {path}")
    return mesh


def load_pose_inputs(
    *,
    rgb_path: str | Path,
    depth_path: str | Path,
    mask_path: str | Path,
    mesh_path: str | Path,
    camera_path: str | Path,
) -> PoseInputs:
    """按显式路径加载首帧注册数据。"""
    camera = load_camera_parameters(camera_path)
    rgb = load_rgb_image(rgb_path)
    depth = load_depth_image(
        depth_path,
        depth_unit=str(camera.get("depth_unit", "millimeter")),
    )
    mask = load_mask_image(mask_path)
    if rgb.shape[:2] != depth.shape or depth.shape != mask.shape:
        raise ValueError(
            f"RGB、depth、mask 尺寸不一致: {rgb.shape[:2]}, {depth.shape}, {mask.shape}"
        )
    return PoseInputs(
        rgb=rgb,
        depth=depth,
        mask=mask,
        mesh=load_mesh(mesh_path),
        intrinsic=camera["intrinsic_matrix"],
        camera_parameters=camera,
    )


def correct_pose_with_external_knowledge(
    category: str,
    pose_camera: np.ndarray,
    camera_parameters: dict,
    *,
    side: str = "left",
) -> tuple[np.ndarray, dict]:
    """只对 Microwave 在当前机械臂 SDK 坐标系中修正局部 y 轴朝向。

    对应外参的约定是 ``T_Sside_C0``，因此先将 ``T_C0_object`` 转为当前
    ``S_side`` 下的物体 pose，修正后再变回相机坐标系。
    """
    pose_camera = validate_pose(pose_camera, name="pose_camera")
    if category.lower() != "microwave":
        return pose_camera, {"applied": False, "reason": "该类别没有外界知识修正规则"}

    if side not in {"left", "right"}:
        raise ValueError("side 必须是 'left' 或 'right'")
    side_from_camera = camera_parameters.get(f"{side}_extrinsic_matrix")
    if side_from_camera is None:
        return pose_camera, {
            "applied": False,
            "reason": f"camera_parameters 缺少 {side}_extrinsic_matrix",
        }
    side_from_camera = validate_pose(
        side_from_camera,
        name=f"{side}_extrinsic_matrix",
    )
    pose_side = side_from_camera @ pose_camera
    if pose_side[2, 1] >= 0:
        return pose_camera, {"applied": False, "reason": "Microwave 局部 y 轴已经朝上"}

    pose_side = pose_side.copy()
    pose_side[:3, 0] *= -1
    pose_side[:3, 1] *= -1
    corrected = np.linalg.inv(side_from_camera) @ pose_side
    return validate_pose(corrected, name="corrected_pose_camera"), {
        "applied": True,
        "reason": f"Microwave 局部 y 轴已按{side}臂坐标系朝上修正",
    }


def save_pose_json(
    path: str | Path,
    *,
    camera_id: str = "0",
    pose: np.ndarray,
) -> None:
    """按多相机格式保存物体的相机坐标系 pose。"""
    pose_value = validate_pose(pose, name="pose")
    camera_key = f"camera_parameters_{camera_id}"
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {}
    if output_path.is_file():
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        if isinstance(existing, dict):
            payload.update(existing)
    payload[camera_key] = [pose_value.tolist()]
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )


__all__ = [
    "PoseInputs",
    "correct_pose_with_external_knowledge",
    "load_camera_parameters",
    "load_depth_image",
    "load_mask_image",
    "load_mesh",
    "load_pose_inputs",
    "load_rgb_image",
    "save_pose_json",
]
