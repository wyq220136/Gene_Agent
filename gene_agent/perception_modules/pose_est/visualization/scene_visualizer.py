"""可视化任务初始 RGB-D 场景和物体估计 pose。"""

from __future__ import annotations

from gene_agent.perception_modules.paths import DEFAULT_TEST_DIR

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

_VIS_DIR = Path(__file__).resolve().parent
_POSE_DIR = _VIS_DIR.parent
if str(_POSE_DIR) not in sys.path:
    # 追加而不是插到首位，避免本目录下的 visualization 包遮蔽项目根目录
    # 的 visualization.task 等模块。
    sys.path.append(str(_POSE_DIR))

from ..batch_est_pose import load_camera_parameters, load_depth_image, load_rgb_image
from ..task_pose_est import discover_task_objects


def depth_to_points(
    depth: np.ndarray,
    intrinsic: np.ndarray,
    *,
    max_depth: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """把米制深度图投影为相机坐标系点，并返回展平后的有效掩码。"""
    depth = np.asarray(depth, dtype=np.float32)
    intrinsic = np.asarray(intrinsic, dtype=np.float64).reshape(3, 3)
    height, width = depth.shape
    v, u = np.indices((height, width))
    valid = np.isfinite(depth) & (depth > 0) & (depth <= max_depth)
    z = depth[valid]
    x = (u[valid] - intrinsic[0, 2]) * z / intrinsic[0, 0]
    y = (v[valid] - intrinsic[1, 2]) * z / intrinsic[1, 1]
    points = np.stack([x, y, z], axis=1)
    return points, valid.reshape(-1)


def transform_vertices(vertices: np.ndarray, pose: np.ndarray) -> np.ndarray:
    """将父坐标系中的顶点通过齐次 pose 变换到目标坐标系。"""
    vertices = np.asarray(vertices, dtype=np.float64)
    pose = np.asarray(pose, dtype=np.float64).reshape(4, 4)
    homogeneous = np.concatenate(
        [vertices, np.ones((len(vertices), 1), dtype=np.float64)],
        axis=1,
    )
    return (pose @ homogeneous.T).T[:, :3]


def load_cylinder_concept_meshes(
    concept_path: str | Path,
    *,
    semantic: str | None = None,
) -> tuple[tuple[str, np.ndarray, np.ndarray], ...]:
    """加载 Concept 中的圆柱，并返回已经应用局部 pose 的物体坐标系 mesh。"""
    path = Path(concept_path).expanduser()
    if not path.is_file():
        return ()
    payload = json.loads(path.read_text(encoding="utf-8"))
    meshes: list[tuple[str, np.ndarray, np.ndarray]] = []
    for index, item in enumerate(payload.get("conceptualization", [])):
        if not isinstance(item, dict) or item.get("template") != "Cylinder":
            continue
        if semantic is not None and item.get("semantic") != semantic:
            continue
        parameters = item.get("parameters")
        if not isinstance(parameters, dict):
            continue
        import trimesh

        radius_x, radius_y, height = map(float, parameters["size"])
        cylinder = trimesh.creation.cylinder(radius=1.0, height=height, sections=32)
        cylinder.vertices[:, 0] *= radius_x
        cylinder.vertices[:, 1] *= radius_y
        local_pose = np.asarray(parameters.get("pose_matrix", np.eye(4)), dtype=float)
        cylinder.apply_transform(local_pose)
        name = str(
            item.get("component")
            or item.get("semantic")
            or f"Cylinder_{index}"
        )
        meshes.append(
            (
                name,
                np.asarray(cylinder.vertices, dtype=np.float64),
                np.asarray(cylinder.faces, dtype=np.int32),
            )
        )
    return tuple(meshes)


def visualize_task_scene(
    task_dir: str | Path,
    *,
    categories: str | Iterable[str] = "all",
    max_depth: float = 3.0,
) -> None:
    """显示初始点云和所有成功估计的 mesh pose。"""
    import open3d as o3d
    import trimesh

    task_path = Path(task_dir).expanduser().resolve()
    camera = load_camera_parameters(task_path / "camera" / "camera_parameters_0.json")
    rgb = load_rgb_image(task_path / "rgbs" / "rgb_0.png")
    depth = load_depth_image(
        task_path / "depths" / "depth_0.png",
        depth_unit=str(camera.get("depth_unit", "millimeter")),
    )
    points, valid = depth_to_points(
        depth,
        camera["intrinsic_matrix"],
        max_depth=max_depth,
    )

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(
        rgb.reshape(-1, 3)[valid].astype(np.float64) / 255.0
    )
    geometries = [point_cloud, o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)]

    for _category, object_dir in discover_task_objects(task_path, categories):
        pose_path = object_dir / "pose.json"
        mesh_path = object_dir / "meshes" / "obj_mesh.obj"
        if not pose_path.is_file() or not mesh_path.is_file():
            print(f"跳过未完成 pose 的物体: {object_dir}")
            continue
        pose_data = json.loads(pose_path.read_text(encoding="utf-8"))
        pose_values = pose_data.get("camera_parameters_0")
        if not isinstance(pose_values, list) or not pose_values:
            print(f"跳过没有 camera_parameters_0 pose 的物体: {object_dir}")
            continue
        pose = np.asarray(pose_values[0], dtype=np.float64)
        mesh = trimesh.load(mesh_path, force="mesh")
        display_mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(np.asarray(mesh.vertices)),
            o3d.utility.Vector3iVector(np.asarray(mesh.faces)),
        )
        display_mesh.compute_vertex_normals()
        display_mesh.paint_uniform_color([0.2, 0.8, 0.2])
        display_mesh.transform(pose)
        geometries.append(display_mesh)

        concept_path = object_dir / "conceptualization.json"
        knob_colors = ([0.95, 0.25, 0.15], [0.20, 0.55, 0.95])
        for index, (name, vertices, faces) in enumerate(
            load_cylinder_concept_meshes(concept_path, semantic="knob")
        ):
            knob_mesh = o3d.geometry.TriangleMesh(
                o3d.utility.Vector3dVector(transform_vertices(vertices, pose)),
                o3d.utility.Vector3iVector(faces),
            )
            knob_mesh.compute_vertex_normals()
            knob_mesh.paint_uniform_color(knob_colors[index % len(knob_colors)])
            geometries.append(knob_mesh)
            print(f"显示 Concept: {object_dir.name}/{name}")

    o3d.visualization.draw_geometries(geometries)


def main() -> None:
    parser = argparse.ArgumentParser(description="显示任务场景和估计 mesh pose")
    parser.add_argument("--task-dir", default=DEFAULT_TEST_DIR)
    parser.add_argument("--categories", nargs="+", default=["all"])
    parser.add_argument("--max-depth", type=float, default=3.0)
    args = parser.parse_args()
    visualize_task_scene(
        args.task_dir,
        categories=args.categories,
        max_depth=args.max_depth,
    )


if __name__ == "__main__":
    main()
