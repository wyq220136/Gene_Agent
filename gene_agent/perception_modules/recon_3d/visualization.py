"""项目通用的 RGB-D、mesh 和 pose 可视化工具。"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


def _load_intrinsics(cam_path: Path) -> np.ndarray:
    """读取项目相机 JSON 中的内参矩阵。"""
    with cam_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    for key in ("intrinsic", "intrinsic_matrix", "camera_matrix"):
        if key in data:
            matrix = np.asarray(data[key], dtype=np.float32)
            if matrix.size == 9:
                return matrix.reshape(3, 3)
    raise ValueError(f"相机参数中没有有效内参矩阵: {cam_path}")


def _load_depth(path: Path) -> np.ndarray:
    """读取 PNG 或 NPY 深度图。"""
    if path.suffix.lower() == ".npy":
        depth = np.load(path)
    else:
        import cv2

        depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise FileNotFoundError(f"无法读取深度图: {path}")
    depth = np.asarray(depth)
    if depth.ndim == 3:
        depth = depth.squeeze()
    if depth.ndim != 2:
        raise ValueError(f"深度图必须是二维数组: {path}")
    return depth


def _load_rgb(path: Path) -> np.ndarray:
    """读取 RGB 图像并转换为 RGB 排列。"""
    import cv2

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"无法读取 RGB 图像: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def load_rgbd_to_pcd(
    instance_dir: str | Path | None = None,
    cam_path: str | Path | None = None,
    max_depth: float = 3.0,
    in_world_frame: bool = False,
    *,
    rgb_path: str | Path | None = None,
    depth_path: str | Path | None = None,
    intrinsics_path: str | Path | None = None,
    depth_scale: float = 0.001,
) :
    """根据项目 RGB-D 文件生成相机坐标系场景点云。

    新接口直接传入 ``rgb_path``、``depth_path`` 和 ``intrinsics_path``。
    保留 ``instance_dir`` 和 ``cam_path`` 参数以兼容旧的调用方式。
    """
    import open3d as o3d

    if rgb_path is None or depth_path is None or intrinsics_path is None:
        if instance_dir is None or cam_path is None:
            raise ValueError("必须提供 RGB、depth、相机内参路径")
        instance_dir = Path(instance_dir)
        rgb_path = instance_dir / "rgb.png"
        depth_path = instance_dir / "depth.npy"
        intrinsics_path = Path(cam_path)

    rgb_path = Path(rgb_path)
    depth_path = Path(depth_path)
    intrinsics_path = Path(intrinsics_path)
    for path in (rgb_path, depth_path, intrinsics_path):
        if not path.is_file():
            raise FileNotFoundError(f"可视化输入文件不存在: {path}")

    rgb = _load_rgb(rgb_path)
    depth = _load_depth(depth_path).astype(np.float32) * float(depth_scale)
    if rgb.shape[:2] != depth.shape:
        raise ValueError("RGB 和 depth 的尺寸不一致")

    height, width = depth.shape
    K = _load_intrinsics(intrinsics_path)
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width, height, float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
    )

    extrinsic = np.eye(4)
    if in_world_frame:
        with intrinsics_path.open("r", encoding="utf-8") as handle:
            camera_data = json.load(handle)
        if "extrinsic_matrix" in camera_data:
            extrinsic = np.linalg.inv(np.asarray(camera_data["extrinsic_matrix"]).reshape(4, 4))

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(np.ascontiguousarray(rgb)),
        o3d.geometry.Image(np.ascontiguousarray(depth)),
        depth_scale=1.0,
        depth_trunc=max_depth,
        convert_rgb_to_intensity=False,
    )
    return o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic, extrinsic=extrinsic)


def sam3d_pose_to_opencv_matrix(
    rotation,
    translation,
    scale,
    local_x_rotation_degrees: float = 90.0,
) -> np.ndarray:
    """将 SAM3D pose 转为相机坐标系，并补偿 mesh 的局部 X 轴朝向。"""
    w, x, y, z = np.asarray(rotation, dtype=float).reshape(-1)[:4]
    quaternion_norm = np.linalg.norm([w, x, y, z])
    if quaternion_norm <= 1e-12:
        raise ValueError("SAM3D rotation 四元数不能为零")
    w, x, y, z = np.asarray([w, x, y, z]) / quaternion_norm
    rotation_matrix = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    angle = math.radians(float(local_x_rotation_degrees))
    local_x_rotation = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, math.cos(angle), -math.sin(angle)],
            [0.0, math.sin(angle), math.cos(angle)],
        ]
    )
    matrix = np.eye(4)
    matrix[:3, :3] = (
        rotation_matrix
        @ local_x_rotation
        @ np.diag(np.asarray(scale, dtype=float).reshape(-1)[:3])
    )
    matrix[:3, 3] = np.asarray(translation, dtype=float).reshape(-1)[:3]
    # SAM3D 自带 renderer 从 PyTorch3D 到 OpenCV 会翻转 X、Y 轴。
    return np.diag([-1.0, -1.0, 1.0, 1.0]) @ matrix


def create_pose_axes(pose: np.ndarray, size: float = 0.08):
    """创建表示 mesh 局部坐标系的 Open3D 线段。"""
    import open3d as o3d

    pose = np.asarray(pose, dtype=float)
    origin = pose[:3, 3]
    rotation = pose[:3, :3]
    rotation = rotation / np.maximum(np.linalg.norm(rotation, axis=0, keepdims=True), 1e-12)
    points = np.vstack(
        [origin, origin + rotation[:, 0] * size, origin + rotation[:, 1] * size, origin + rotation[:, 2] * size]
    )
    axes = o3d.geometry.LineSet()
    axes.points = o3d.utility.Vector3dVector(points)
    axes.lines = o3d.utility.Vector2iVector([[0, 1], [0, 2], [0, 3]])
    axes.colors = o3d.utility.Vector3dVector([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    return axes


def visualize_reconstruction(
    *,
    rgb_path: str | Path,
    depth_path: str | Path,
    intrinsics_path: str | Path,
    mesh_path: str | Path,
    pose: dict[str, np.ndarray] | np.ndarray,
    output_dir: str | Path,
    title: str = "SAM3D RGB-D Reconstruction",
    depth_scale: float = 0.001,
) -> dict[str, str]:
    """显示场景点云、按 SAM3D pose 放置的 mesh 和两个坐标系。"""
    import open3d as o3d

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if isinstance(pose, dict):
        pose_matrix = sam3d_pose_to_opencv_matrix(
            pose["rotation"], pose["translation"], pose["scale"]
        )
    else:
        pose_matrix = np.asarray(pose, dtype=float)

    point_cloud = load_rgbd_to_pcd(
        rgb_path=rgb_path,
        depth_path=depth_path,
        intrinsics_path=intrinsics_path,
        depth_scale=depth_scale,
    )
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if mesh.is_empty():
        raise RuntimeError(f"无法读取 mesh: {mesh_path}")
    mesh.transform(pose_matrix)
    mesh.paint_uniform_color([0.9, 0.2, 0.2])
    mesh.compute_vertex_normals()

    camera_axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
    mesh_axes = create_pose_axes(pose_matrix)
    scene_path = output_dir / "scene_rgbd.ply"
    mesh_path_out = output_dir / "sam3d_mesh_camera.obj"
    axes_path = output_dir / "sam3d_axes.ply"
    o3d.io.write_point_cloud(str(scene_path), point_cloud)
    o3d.io.write_triangle_mesh(str(mesh_path_out), mesh)
    o3d.io.write_line_set(str(axes_path), mesh_axes)

    print("[recon_3d] 正在打开 Open3D 可视化窗口", flush=True)
    o3d.visualization.draw_geometries(
        [point_cloud, mesh, camera_axes, mesh_axes],
        window_name=title,
    )
    return {
        "scene_point_cloud": str(scene_path),
        "mesh": str(mesh_path_out),
        "axes": str(axes_path),
    }


def visualize_pcd_and_mesh(
    pcd=None,
    mesh=None,
    pcd_color=None,
    mesh_color=None,
    title="Open3D Visualizer",
    vis_coord_frame=True,
):
    """兼容旧调用的通用点云和 mesh 可视化函数。"""
    import open3d as o3d
    import trimesh

    geometries = []
    if isinstance(mesh, trimesh.Trimesh):
        mesh = trimesh_to_o3dmesh(mesh, with_texture=False)
    if isinstance(pcd, (str, Path)):
        pcd = o3d.io.read_point_cloud(str(pcd))
    if pcd is not None:
        if pcd_color is not None:
            pcd.paint_uniform_color(pcd_color)
        geometries.append(pcd)
    if isinstance(mesh, (str, Path)):
        mesh = o3d.io.read_triangle_mesh(str(mesh))
    if mesh is not None:
        if not mesh.has_vertex_normals():
            mesh.compute_vertex_normals()
        if mesh_color is not None:
            mesh.paint_uniform_color(mesh_color)
        geometries.append(mesh)
    if vis_coord_frame:
        geometries.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2))
    if not geometries:
        raise ValueError("未提供任何几何体")
    print("[visualization] 正在打开 Open3D 可视化窗口", flush=True)
    o3d.visualization.draw_geometries(geometries, window_name=title)


def trimesh_to_o3dmesh(tm, with_texture=False):
    """将 trimesh 转为 Open3D TriangleMesh。"""
    import open3d as o3d

    mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(tm.vertices),
        triangles=o3d.utility.Vector3iVector(tm.faces),
    )
    if not with_texture and hasattr(tm, "visual") and tm.visual is not None:
        visual = tm.visual
        if hasattr(visual, "vertex_colors") and visual.vertex_colors is not None:
            colors = np.asarray(visual.vertex_colors[:, :3]) / 255.0
            mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    mesh.compute_vertex_normals()
    return mesh
