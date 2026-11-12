"""使用门板附近点云拟合平面，并预览 Microwave pose 的 yaw 修正。"""

from __future__ import annotations

from gene_agent.perception_modules.paths import DEFAULT_TEST_DIR, PROJECT_ROOT

import argparse
from pathlib import Path
import sys

import numpy as np

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gene_agent.perception_modules.pose_est.legacy_geometry import (
    camera_object_to_task_object,
    load_camera_object_pose,
    load_microwave_concepts,
    validate_pose,
)
from gene_agent.perception_modules.pose_est.microwave_pose_postprocess import (
    correct_pose_yaw_from_plane,
    fit_plane_ransac,
    select_door_region_points,
    _unit,
)


def _make_concept_mesh(o3d, primitive, root_pose: np.ndarray, color):
    """把一个 concept primitive 转成 Open3D mesh。"""
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(primitive.local_vertices, dtype=float)),
        o3d.utility.Vector3iVector(np.asarray(primitive.faces, dtype=np.int32)),
    )
    mesh.transform(validate_pose(root_pose, name="root_pose") @ primitive.pose)
    mesh.paint_uniform_color(color)
    mesh.compute_vertex_normals()
    return mesh


def _make_plane_patch(
    o3d,
    center: np.ndarray,
    normal: np.ndarray,
    door_pose_task: np.ndarray,
    door_size: np.ndarray,
):
    """创建与拟合平面对齐的 door 尺寸矩形。"""
    n = _unit(normal, name="plane_normal")
    x_axis = door_pose_task[:3, 0].copy()
    x_axis -= float(x_axis @ n) * n
    x_axis = _unit(x_axis, name="plane_patch_x_axis")
    y_axis = _unit(np.cross(n, x_axis), name="plane_patch_y_axis")
    if float(y_axis @ door_pose_task[:3, 1]) < 0.0:
        y_axis = -y_axis
    half_x = float(door_size[0]) / 2.0
    half_y = float(door_size[1]) / 2.0
    vertices = np.asarray(
        [
            center - x_axis * half_x - y_axis * half_y,
            center + x_axis * half_x - y_axis * half_y,
            center + x_axis * half_x + y_axis * half_y,
            center - x_axis * half_x + y_axis * half_y,
        ],
        dtype=float,
    )
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices),
        o3d.utility.Vector3iVector(np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)),
    )
    mesh.paint_uniform_color([0.1, 0.8, 0.9])
    return mesh


def _region_box_vertices(
    region_bounds: np.ndarray,
    *,
    plane_band_m: float,
) -> np.ndarray:
    """返回筛选区域在 door-local 坐标系中的 8 个盒体顶点。"""
    bounds = np.asarray(region_bounds, dtype=float)
    if bounds.shape != (4,) or not np.all(np.isfinite(bounds)):
        raise ValueError("region_bounds 必须是有限的 [x_min, x_max, y_min, y_max]")
    x_min, x_max, y_min, y_max = bounds
    band = float(plane_band_m)
    if not np.isfinite(band) or band <= 0.0:
        raise ValueError("plane_band_m 必须是正有限数")
    if x_min >= x_max or y_min >= y_max:
        raise ValueError("region_bounds 必须描述有效矩形")
    return np.asarray(
        [
            [x_min, y_min, -band],
            [x_max, y_min, -band],
            [x_max, y_max, -band],
            [x_min, y_max, -band],
            [x_min, y_min, band],
            [x_max, y_min, band],
            [x_max, y_max, band],
            [x_min, y_max, band],
        ],
        dtype=float,
    )


def _make_region_box(
    o3d,
    door_pose_task: np.ndarray,
    region_bounds: np.ndarray,
    *,
    plane_band_m: float,
):
    """创建显示初始点云筛选区域的 door-local 3D 线框盒。"""
    local_vertices = _region_box_vertices(
        region_bounds,
        plane_band_m=plane_band_m,
    )
    root_pose = validate_pose(door_pose_task, name="door_pose_task")
    vertices = (root_pose[:3, :3] @ local_vertices.T).T + root_pose[:3, 3]
    lines = np.asarray(
        [
            [0, 1], [1, 2], [2, 3], [3, 0],
            [4, 5], [5, 6], [6, 7], [7, 4],
            [0, 4], [1, 5], [2, 6], [3, 7],
        ],
        dtype=np.int32,
    )
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(vertices),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(
        np.tile(np.array([[1.0, 0.0, 1.0]]), (len(lines), 1))
    )
    return line_set


def _load_microwave_pose(task_dir: Path, side: str) -> tuple[np.ndarray, np.ndarray, Path]:
    """读取 pose.json，并把 Microwave pose 转到当前 SDK base。"""
    from gene_agent.perception_modules.pose_est.batch_est_pose import load_camera_parameters

    camera_path = task_dir / "camera/camera_parameters_0.json"
    pose_path = task_dir / "objs/Microwave/Microwave_0/pose.json"
    camera = load_camera_parameters(camera_path)
    task_from_camera = camera.get(f"{side}_extrinsic_matrix")
    if task_from_camera is None:
        raise ValueError(f"{camera_path} 缺少 {side}_extrinsic_matrix")
    pose_camera = load_camera_object_pose(pose_path)
    return camera_object_to_task_object(task_from_camera, pose_camera), task_from_camera, pose_path


def visualize_microwave_pose_correction(
    task_dir: str | Path,
    *,
    side: str = "right",
    region_margin_m: float = 0.04,
    plane_band_m: float = 0.04,
    distance_threshold_m: float = 0.006,
    ransac_iterations: int = 1500,
    max_depth_m: float = 3.0,
    max_points: int = 80000,
    show_solid_mesh: bool = False,
) -> tuple[np.ndarray, np.ndarray, float, Path, np.ndarray]:
    """执行点云平面修正并打开 Open3D 预览。"""
    if side not in {"left", "right"}:
        raise ValueError("side 必须是 left 或 right")
    task_path = Path(task_dir).expanduser().resolve()
    concept_path = task_path / "objs/Microwave/Microwave_0/conceptualization.json"
    body_mesh_path = task_path / "objs/Microwave/Microwave_0/meshes/obj_mesh.obj"
    microwave_pose, task_from_camera, pose_path = _load_microwave_pose(task_path, side)
    door, handle, _cavity = load_microwave_concepts(concept_path)

    from .rgbd_points import load_rgbd_scene_points

    scene_points, scene_colors = load_rgbd_scene_points(
        rgb_path=task_path / "rgbs/rgb_0.png",
        depth_path=task_path / "depths/depth_0.png",
        camera_path=task_path / "camera/camera_parameters_0.json",
        side=side,
        max_depth=max_depth_m,
        max_points=max_points,
    )
    selected_mask, region_bounds = select_door_region_points(
        scene_points,
        microwave_pose,
        door,
        handle,
        region_margin_m=region_margin_m,
        plane_band_m=plane_band_m,
    )
    selected_points = np.asarray(scene_points[selected_mask], dtype=float)
    if len(selected_points) < 20:
        raise RuntimeError(
            f"handle 附近只筛到 {len(selected_points)} 个点，"
            "请增大 --region-margin-m 或 --plane-band-m"
        )
    plane_normal, plane_offset, inlier_mask = fit_plane_ransac(
        selected_points,
        distance_threshold_m=distance_threshold_m,
        iterations=ransac_iterations,
    )
    corrected_pose, yaw_deg = correct_pose_yaw_from_plane(
        microwave_pose,
        door.pose,
        plane_normal,
    )
    print("[visualization] 颜色说明:", flush=True)
    print("  原始 RGB 点云：保留 RGB 相机颜色", flush=True)
    print("  绿色点：进入 handle 附近局部矩形区域的全部点", flush=True)
    print("  红色点：RANSAC 最终平面内点（最终 SVD 拟合使用）", flush=True)
    print("  紫色线框：初始点云筛选区域，door-local XY 矩形 + Z 厚度范围", flush=True)
    if show_solid_mesh:
        print("  橙色/蓝色实体门板和机身：原始/修正后的 Microwave pose", flush=True)
    else:
        print("  实体门板/机身 mesh：默认隐藏，避免遮挡点云；使用 --show-solid-mesh 显示", flush=True)
    print("  青色平面：根据红色内点拟合出的门板平面", flush=True)
    print("  坐标系：红/绿/蓝轴分别表示 X/Y/Z", flush=True)
    print(f"[plane] 区域点数: {len(selected_points)}；平面内点: {int(inlier_mask.sum())}", flush=True)
    print(
        f"[plane] 内点比例: {int(inlier_mask.sum()) / len(selected_points):.1%}",
        flush=True,
    )
    print(f"[plane] 矩形范围 door-local: {region_bounds}", flush=True)
    print(f"[plane] 法向: {plane_normal}；偏置: {plane_offset:.6f}", flush=True)
    print(f"[pose] 已施加全局 Z yaw 修正: {yaw_deg:.3f}°", flush=True)
    print(f"[pose] 原始 pose:\n{microwave_pose}", flush=True)
    print(f"[pose] 修正 pose:\n{corrected_pose}", flush=True)

    import open3d as o3d

    geometries = []
    scene_cloud = o3d.geometry.PointCloud()
    scene_cloud.points = o3d.utility.Vector3dVector(scene_points)
    scene_cloud.colors = o3d.utility.Vector3dVector(
        np.asarray(scene_colors, dtype=float) / 255.0
    )
    geometries.append(scene_cloud)

    selected_cloud = o3d.geometry.PointCloud()
    selected_cloud.points = o3d.utility.Vector3dVector(selected_points)
    selected_cloud.colors = o3d.utility.Vector3dVector(
        np.tile(np.array([[0.1, 0.85, 0.2]]), (len(selected_points), 1))
    )
    geometries.append(selected_cloud)

    inlier_cloud = o3d.geometry.PointCloud()
    inlier_cloud.points = o3d.utility.Vector3dVector(selected_points[inlier_mask])
    inlier_cloud.colors = o3d.utility.Vector3dVector(
        np.tile(np.array([[0.95, 0.1, 0.1]]), (int(inlier_mask.sum()), 1))
    )
    geometries.append(inlier_cloud)

    door_pose_old = microwave_pose @ door.pose
    door_pose_new = corrected_pose @ door.pose
    geometries.append(
        _make_region_box(
            o3d,
            door_pose_old,
            region_bounds,
            plane_band_m=plane_band_m,
        )
    )
    if show_solid_mesh:
        geometries.extend(
            [
                _make_concept_mesh(o3d, door, microwave_pose, [1.0, 0.45, 0.05]),
                _make_concept_mesh(o3d, door, corrected_pose, [0.05, 0.35, 1.0]),
                _make_concept_mesh(o3d, handle, microwave_pose, [1.0, 0.45, 0.05]),
                _make_concept_mesh(o3d, handle, corrected_pose, [0.05, 0.35, 1.0]),
            ]
        )
    plane_center = selected_points[inlier_mask].mean(axis=0)
    geometries.append(
        _make_plane_patch(
            o3d,
            plane_center,
            plane_normal,
            door_pose_new,
            np.asarray(door.size_xyz, dtype=float),
        )
    )
    old_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.10)
    old_frame.transform(microwave_pose)
    new_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.10)
    new_frame.transform(corrected_pose)
    old_door_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.08)
    old_door_frame.transform(door_pose_old)
    new_door_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.08)
    new_door_frame.transform(door_pose_new)
    geometries.extend([old_frame, new_frame, old_door_frame, new_door_frame])
    if show_solid_mesh and body_mesh_path.is_file():
        body_old = o3d.io.read_triangle_mesh(str(body_mesh_path))
        body_new = o3d.io.read_triangle_mesh(str(body_mesh_path))
        if not body_old.is_empty() and not body_new.is_empty():
            body_old.transform(microwave_pose)
            body_new.transform(corrected_pose)
            body_old.paint_uniform_color([0.75, 0.35, 0.05])
            body_new.paint_uniform_color([0.05, 0.25, 0.80])
            body_old.compute_vertex_normals()
            body_new.compute_vertex_normals()
            geometries.extend([body_old, body_new])
    o3d.visualization.draw_geometries(
        geometries,
        window_name=f"Microwave pose plane correction ({side})",
    )
    return corrected_pose, task_from_camera, yaw_deg, pose_path, microwave_pose


def main() -> None:
    parser = argparse.ArgumentParser(description="用 Microwave door 点云平面修正 pose yaw")
    parser.add_argument("--task-dir", type=Path, default=PROJECT_ROOT / DEFAULT_TEST_DIR)
    parser.add_argument("--side", choices=("left", "right"), default="right")
    parser.add_argument("--region-margin-m", type=float, default=0.04)
    parser.add_argument("--plane-band-m", type=float, default=0.04)
    parser.add_argument("--distance-threshold-m", type=float, default=0.006)
    parser.add_argument("--ransac-iterations", type=int, default=1500)
    parser.add_argument("--max-depth-m", type=float, default=3.0)
    parser.add_argument("--max-points", type=int, default=80000)
    parser.add_argument(
        "--show-solid-mesh",
        action="store_true",
        help="显示实体门板、把手和机身 mesh；默认隐藏以便观察点云内点",
    )
    parser.add_argument(
        "--save-pose",
        action="store_true",
        help="关闭 Open3D 后，将只修正 rotation 的结果写回 pose.json",
    )
    args = parser.parse_args()

    corrected_pose, task_from_camera, _yaw_deg, pose_path, _old_pose = (
        visualize_microwave_pose_correction(
            args.task_dir,
            side=args.side,
            region_margin_m=args.region_margin_m,
            plane_band_m=args.plane_band_m,
            distance_threshold_m=args.distance_threshold_m,
            ransac_iterations=args.ransac_iterations,
            max_depth_m=args.max_depth_m,
            max_points=args.max_points,
            show_solid_mesh=args.show_solid_mesh,
        )
    )
    if args.save_pose:
        from gene_agent.perception_modules.pose_est.batch_est_pose import save_pose_json

        corrected_camera_pose = np.linalg.inv(task_from_camera) @ corrected_pose
        save_pose_json(pose_path, camera_id="0", pose=corrected_camera_pose)
        print(f"[pose] 已保存修正后的 pose.json: {pose_path}", flush=True)


if __name__ == "__main__":
    main()


__all__ = [
    "correct_pose_yaw_from_plane",
    "fit_plane_ransac",
    "select_door_region_points",
    "visualize_microwave_pose_correction",
]
