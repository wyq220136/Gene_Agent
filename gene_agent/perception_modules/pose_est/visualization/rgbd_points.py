"""感知调试图使用的 RGB-D 点云读取。"""

from pathlib import Path

import numpy as np

from ..batch_est_pose import load_camera_parameters, load_depth_image, load_rgb_image
from ..legacy_geometry import validate_pose


def load_rgbd_scene_points(
    *, rgb_path: str | Path, depth_path: str | Path, camera_path: str | Path,
    side: str, max_depth: float = 3.0, max_points: int = 50000,
) -> tuple[np.ndarray, np.ndarray]:
    camera = load_camera_parameters(camera_path)
    key = f"{side}_extrinsic_matrix"
    if key not in camera:
        raise ValueError(f"camera 参数缺少 {key}: {camera_path}")
    rgb = load_rgb_image(rgb_path)
    depth = load_depth_image(depth_path, depth_unit=str(camera.get("depth_unit", "millimeter")))
    if rgb.shape[:2] != depth.shape:
        raise ValueError("RGB 和 depth 尺寸不一致")
    rows, cols = np.nonzero(np.isfinite(depth) & (depth > 0) & (depth <= max_depth))
    if len(rows) > max_points:
        selected = np.linspace(0, len(rows) - 1, max_points, dtype=np.int64)
        rows, cols = rows[selected], cols[selected]
    if not len(rows):
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)
    intrinsic = np.asarray(camera["intrinsic_matrix"], dtype=float)
    z = depth[rows, cols]
    points_camera = np.column_stack(((cols - intrinsic[0, 2]) * z / intrinsic[0, 0],
                                      (rows - intrinsic[1, 2]) * z / intrinsic[1, 1], z))
    pose = validate_pose(camera[key], name=key)
    points = (pose[:3, :3] @ points_camera.T).T + pose[:3, 3]
    return points.astype(np.float32), np.ascontiguousarray(rgb[rows, cols])
