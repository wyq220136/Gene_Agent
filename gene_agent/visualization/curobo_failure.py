"""Export auditable 3D evidence for failed cuRobo planning calls."""
from __future__ import annotations

import json
from pathlib import Path
import re
import textwrap
from typing import Any, Sequence

import numpy as np
from scipy.spatial.transform import Rotation


TRACE_COLORS = (
    (230, 57, 70), (255, 159, 28), (255, 202, 58), (138, 201, 38),
    (25, 130, 196), (106, 76, 147), (247, 37, 133), (67, 170, 139),
)

FORK_PROGRESS_COLORS = (
    ("red", (230, 57, 70)),
    ("orange", (255, 145, 0)),
    ("yellow", (255, 210, 0)),
    ("green", (50, 180, 70)),
    ("cyan", (0, 190, 190)),
    ("blue", (35, 100, 220)),
    ("purple", (135, 65, 190)),
)
MAX_TRAJECTORY_FORKS = len(FORK_PROGRESS_COLORS)
MIN_FORK_PATH_SPACING_M = 0.055
MAX_PATH_LINE_POINTS = 96
FORK_RADIUS_M = 0.003
COLLISION_MARKER_RADIUS_M = 0.016
COLLISION_MARKER_ARM_M = 0.035
TRAJECTORY_VIEW_ANGLES = (
    ("task", 25.0, -24.0),
    ("front", 0.0, -18.0),
    ("front_high", 0.0, -42.0),
    ("oblique_left", 45.0, -24.0),
    ("oblique_left_high", 45.0, -42.0),
    ("left", 75.0, -18.0),
    ("top", 90.0, -89.0),
)
PLY_PATH_RADIUS_M = 0.0015
PLY_TARGET_RADIUS_M = 0.010
PLY_OBJECT_COLORS = {
    "table": np.asarray([145, 125, 100], dtype=np.uint8),
    "microwave": np.asarray([185, 190, 195], dtype=np.uint8),
    "bowl": np.asarray([82, 156, 235], dtype=np.uint8),
}
PLY_MAX_EDGE_M = {
    "table": 0.08,
    "microwave": 0.06,
    "bowl": 0.012,
}
FORK_LOCAL_SEGMENTS_M = (
    # 单个两叉符号：手柄沿 -Z 指向腕部/末端关节，两根叉齿沿 +Z 表示指爪。
    # TCP 的 +X 是开合方向，+Z 是 approach 方向。
    ((0.0, 0.0, -0.052), (0.0, 0.0, -0.012)),
    ((-0.028, 0.0, -0.012), (0.028, 0.0, -0.012)),
    ((-0.028, 0.0, -0.012), (-0.028, 0.0, 0.046)),
    ((0.028, 0.0, -0.012), (0.028, 0.0, 0.046)),
)


def _fork_progress_colors(count: int) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """按时间顺序从红到紫抽取清晰、离散的夹爪颜色。"""
    if count < 0:
        raise ValueError("fork color count must be non-negative")
    if count == 0:
        return []
    palette_indices = np.rint(np.linspace(
        0, len(FORK_PROGRESS_COLORS) - 1, count,
    )).astype(int)
    colors = []
    for palette_index in palette_indices:
        name, values = FORK_PROGRESS_COLORS[int(palette_index)]
        rgb = np.asarray(values, dtype=np.uint8)
        rgba = np.concatenate((rgb.astype(np.float32) / 255.0, [0.92])).astype(
            np.float32,
        )
        colors.append((name, rgb, rgba))
    return colors


def _trajectory_view_path(path: Path, label: str) -> Path:
    """Return the standalone PNG path for one named trajectory view."""
    slug = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
    if not slug:
        raise ValueError("trajectory view label must contain an ASCII name")
    return path.with_name(f"{path.stem}_{slug}{path.suffix}")


def _path_sample_indices(points: np.ndarray, *, maximum: int,
                         minimum_spacing_m: float = 0.0) -> list[int]:
    """按弧长均匀取样，始终保留首尾点。"""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
        raise ValueError("trajectory points must be finite [N,3]")
    count = len(points)
    if count <= 1:
        return list(range(count))
    distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(distances)))
    length = float(cumulative[-1])
    if length <= 1e-12:
        return [0, count - 1]
    desired = maximum
    if minimum_spacing_m > 0.0:
        desired = min(desired, max(2, int(np.ceil(length / minimum_spacing_m)) + 1))
    targets = np.linspace(0.0, length, min(desired, count))
    indices = [min(int(np.searchsorted(cumulative, value, side="left")), count - 1)
               for value in targets]
    return sorted(set((0, count - 1, *indices)))


def _fork_segments(tcp_pose_world: np.ndarray) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """将音叉夹爪的局部线段变换到 world。"""
    pose = np.asarray(tcp_pose_world, dtype=float)
    if (pose.shape != (4, 4) or not np.all(np.isfinite(pose))
            or not np.allclose(pose[3], [0.0, 0.0, 0.0, 1.0])):
        raise ValueError("tcp pose must be a finite 4x4 transform")
    rotation, origin = pose[:3, :3], pose[:3, 3]
    return tuple(
        (rotation @ np.asarray(start, dtype=float) + origin,
         rotation @ np.asarray(end, dtype=float) + origin)
        for start, end in FORK_LOCAL_SEGMENTS_M
    )


def _pose_matrix(pose: Sequence[float]) -> np.ndarray:
    values = np.asarray(pose, dtype=float)
    if values.shape != (7,) or not np.all(np.isfinite(values)):
        raise ValueError("scene obstacle pose must be [x,y,z,qw,qx,qy,qz]")
    transform = np.eye(4)
    transform[:3, 3] = values[:3]
    transform[:3, :3] = Rotation.from_quat(values[[4, 5, 6, 3]]).as_matrix()
    return transform


def sample_scene_points(scene_model: dict, *, points_per_obstacle: int = 96,
                        max_points: int = 40000,
                        world_from_scene: np.ndarray | None = None,
                        ) -> tuple[np.ndarray, list[str]]:
    """Sample the exact cuboid/mesh world used by cuRobo into a compact cloud."""
    import trimesh

    transform = (np.eye(4) if world_from_scene is None
                 else np.asarray(world_from_scene, dtype=float))
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError("world_from_scene must be a finite 4x4 transform")
    points: list[np.ndarray] = []
    names: list[str] = []
    for name, record in (scene_model.get("cuboid") or {}).items():
        mesh = trimesh.creation.box(extents=np.asarray(record["dims"], dtype=float))
        mesh.apply_transform(transform @ _pose_matrix(record["pose"]))
        sampled, _ = trimesh.sample.sample_surface(mesh, points_per_obstacle, seed=0)
        points.append(np.asarray(sampled, dtype=float))
        names.extend([str(name)] * len(sampled))
    for name, record in (scene_model.get("mesh") or {}).items():
        source = Path(record["file_path"])
        if not source.is_file():
            continue
        loaded = trimesh.load_mesh(source, process=False)
        if isinstance(loaded, trimesh.Scene):
            geometries = tuple(loaded.geometry.values())
            if not geometries:
                continue
            loaded = trimesh.util.concatenate(geometries)
        if not isinstance(loaded, trimesh.Trimesh) or loaded.is_empty:
            continue
        loaded = loaded.copy()
        loaded.apply_transform(transform @ _pose_matrix(record["pose"]))
        sampled, _ = trimesh.sample.sample_surface(loaded, points_per_obstacle, seed=0)
        points.append(np.asarray(sampled, dtype=float))
        names.extend([str(name)] * len(sampled))
    if not points:
        return np.empty((0, 3), dtype=float), []
    cloud = np.concatenate(points, axis=0)
    if len(cloud) > max_points:
        indices = np.linspace(0, len(cloud) - 1, max_points, dtype=int)
        cloud = cloud[indices]
        names = [names[index] for index in indices]
    return cloud, names


def write_colored_ply(path: Path, points: np.ndarray, colors: np.ndarray,
                      faces: np.ndarray | None = None) -> None:
    points = np.asarray(points, dtype=float)
    colors = np.asarray(colors, dtype=np.uint8)
    if points.ndim != 2 or points.shape[1] != 3 or colors.shape != points.shape:
        raise ValueError("PLY points and colors must both be [N,3]")
    if faces is None:
        faces = np.empty((0, 3), dtype=np.int64)
    else:
        faces = np.asarray(faces, dtype=np.int64)
    if (faces.ndim != 2 or faces.shape[1] != 3
            or (len(faces) and (faces.min() < 0 or faces.max() >= len(points)))):
        raise ValueError("PLY faces must contain valid triangle vertex indices")
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "ply\nformat ascii 1.0\n"
        f"element vertex {len(points)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        f"element face {len(faces)}\n"
        "property list uchar int vertex_indices\n"
        "end_header\n"
    )
    with path.open("w", encoding="ascii") as stream:
        stream.write(header)
        for point, color in zip(points, colors):
            stream.write(
                f"{point[0]:.8g} {point[1]:.8g} {point[2]:.8g} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )
        for face in faces:
            stream.write(f"3 {int(face[0])} {int(face[1])} {int(face[2])}\n")


def read_colored_ply(path: Path) -> tuple[np.ndarray, np.ndarray]:
    lines = path.read_text(encoding="ascii").splitlines()
    vertex_count = None
    header_end = None
    for index, line in enumerate(lines):
        if line.startswith("element vertex "):
            vertex_count = int(line.rsplit(" ", 1)[-1])
        if line == "end_header":
            header_end = index + 1
            break
    if vertex_count is None or header_end is None:
        raise ValueError(f"unsupported PLY header: {path}")
    values = np.asarray(
        [[float(item) for item in line.split()[:6]]
         for line in lines[header_end:header_end + vertex_count]],
        dtype=float,
    )
    return values[:, :3], values[:, 3:6].astype(np.uint8)


def _find_run_dir(path: Path) -> Path | None:
    for parent in (Path(path), *Path(path).parents):
        if parent.name.startswith("run_") and (
            parent / "scene" / "source_scene" / "scene.xml"
        ).is_file():
            return parent
    return None


class _MujocoTrajectoryRenderer:
    """Render failed paths with sparse tuning-fork TCP poses in the saved scene."""

    def __init__(self, bundle_dir: Path, *, scene_model: dict | None = None,
                 world_from_scene: np.ndarray | None = None):
        import mujoco

        self.mujoco = mujoco
        self.bundle_dir = Path(bundle_dir)
        self.scene_model = scene_model or {"cuboid": {}, "mesh": {}}
        self.world_from_scene = (np.eye(4) if world_from_scene is None
                                 else np.asarray(world_from_scene, dtype=float))
        if (self.world_from_scene.shape != (4, 4)
                or not np.all(np.isfinite(self.world_from_scene))):
            raise ValueError("world_from_scene must be a finite 4x4 transform")
        self.run_dir = _find_run_dir(self.bundle_dir)
        if self.run_dir is None:
            raise FileNotFoundError("owning run directory with scene.xml was not found")
        self.scene_path = self.run_dir / "scene" / "source_scene" / "scene.xml"
        self.model = mujoco.MjModel.from_xml_path(str(self.scene_path))
        self.base_qpos = np.asarray(mujoco.MjData(self.model).qpos, dtype=float).copy()
        self.base_state_source = "scene_xml_default"
        if "physical_insert" in self.bundle_dir.parts:
            pick = self.run_dir / "pick_insert" / "result.json"
            try:
                state = json.loads(pick.read_text(encoding="utf-8"))
                qpos = np.asarray(state["actual_final"]["values"]["qpos"], dtype=float)
                if qpos.shape == (self.model.nq,):
                    self.base_qpos = qpos
                    self.base_state_source = "pick_insert/result.json actual_final qpos"
            except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError):
                pass
        self.joint_qpos_addresses = np.asarray([
            int(self.model.joint(f"right_joint_{index}").qposadr[0])
            for index in range(1, 8)
        ], dtype=int)
        self.tcp_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "right_tcp",
        )
        if self.tcp_site_id < 0:
            raise RuntimeError("right_tcp site was not found in MuJoCo model")
        robot_roots = {"left_mount", "right_mount", "robot_body"}
        self.robot_geom_ids = set()
        for geom_id in range(self.model.ngeom):
            body_id = int(self.model.geom_bodyid[geom_id])
            while body_id > 0:
                if (self.model.body(body_id).name or "") in robot_roots:
                    self.robot_geom_ids.add(geom_id)
                    break
                body_id = int(self.model.body_parentid[body_id])

    def _data(self, joints: np.ndarray | None = None):
        data = self.mujoco.MjData(self.model)
        data.qpos[:] = self.base_qpos
        if joints is not None:
            data.qpos[self.joint_qpos_addresses] = joints
        self.mujoco.mj_forward(self.model, data)
        return data

    def _selected_waypoints(self, trace: dict) -> list[int]:
        count = len(trace["joint_positions_rad"])
        if count <= 0:
            return []
        points = np.asarray(trace.get("tcp_path_world_m") or [], dtype=float)
        invalid = trace.get("first_invalid_waypoint")
        anchors = {0, count - 1}
        if isinstance(invalid, int) and 0 <= invalid < count:
            anchors.add(invalid)
        selected = set(anchors)

        if points.shape == (count, 3) and np.all(np.isfinite(points)):
            distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
            progress = np.concatenate(([0.0], np.cumsum(distances)))
            enforce_spacing = True
        else:
            progress = np.arange(count, dtype=float)
            enforce_spacing = False

        while len(selected) < min(MAX_TRAJECTORY_FORKS, count):
            candidates = [index for index in range(count) if index not in selected]
            if not candidates:
                break
            best = max(
                candidates,
                key=lambda index: min(abs(progress[index] - progress[item])
                                      for item in selected),
            )
            nearest = min(abs(progress[best] - progress[item]) for item in selected)
            if enforce_spacing and nearest < MIN_FORK_PATH_SPACING_M:
                break
            selected.add(best)
        return sorted(selected)

    def _ghosts(self, trace: dict) -> tuple[list[dict], np.ndarray, np.ndarray]:
        joints = np.asarray(trace["joint_positions_rad"], dtype=float)
        if joints.ndim != 2 or joints.shape[1] != 7:
            raise ValueError("joint_positions_rad must be [N,7] for MuJoCo gripper rendering")
        invalid = trace.get("first_invalid_waypoint")
        selected = self._selected_waypoints(trace)
        render_order = sorted(selected)
        progress_colors = _fork_progress_colors(len(render_order))
        ghosts: list[dict] = []
        cloud_points: list[np.ndarray] = []
        cloud_colors: list[np.ndarray] = []
        for waypoint, (color_name, rgb, rgba) in zip(render_order, progress_colors):
            data = self._data(joints[waypoint])
            site = data.site(self.tcp_site_id)
            tcp_pose = np.eye(4)
            tcp_pose[:3, :3] = np.asarray(site.xmat, dtype=float).reshape(3, 3)
            tcp_pose[:3, 3] = np.asarray(site.xpos, dtype=float)
            failed = waypoint == invalid
            segments = _fork_segments(tcp_pose)
            for start, end in segments:
                samples = np.linspace(start, end, 24)
                cloud_points.append(samples)
                cloud_colors.append(np.repeat(rgb[None], len(samples), axis=0))
            ghosts.append({
                "waypoint": waypoint, "failed": failed,
                "tcp_pose_world": tcp_pose, "segments": segments,
                "rgba": rgba, "rgb": rgb, "progress_color": color_name,
            })
        points = np.concatenate(cloud_points, axis=0) if cloud_points else np.empty((0, 3))
        colors = np.concatenate(cloud_colors, axis=0) if cloud_colors else np.empty((0, 3), dtype=np.uint8)
        return ghosts, points, colors

    def _scene_object_kind(self, geom_id: int) -> str | None:
        name = self.model.geom(geom_id).name or ""
        body_id = int(self.model.geom_bodyid[geom_id])
        body_names = []
        while body_id > 0:
            body_names.append(self.model.body(body_id).name or "")
            body_id = int(self.model.body_parentid[body_id])
        context = " ".join((name, *body_names)).lower()
        if name == "scene_table":
            return "table"
        if name == "bowl_visual":
            return "bowl"
        if "microwave" in context and name.endswith("_visual"):
            return "microwave"
        return None

    def _geom_world_mesh(self, data, geom_id: int):
        import trimesh

        geom_type = int(self.model.geom_type[geom_id])
        if geom_type == int(self.mujoco.mjtGeom.mjGEOM_BOX):
            mesh = trimesh.creation.box(
                extents=2.0 * np.asarray(self.model.geom_size[geom_id], dtype=float),
            )
        elif geom_type == int(self.mujoco.mjtGeom.mjGEOM_MESH):
            mesh_id = int(self.model.geom_dataid[geom_id])
            if mesh_id < 0:
                return None
            vertex_start = int(self.model.mesh_vertadr[mesh_id])
            vertex_count = int(self.model.mesh_vertnum[mesh_id])
            face_start = int(self.model.mesh_faceadr[mesh_id])
            face_count = int(self.model.mesh_facenum[mesh_id])
            mesh = trimesh.Trimesh(
                vertices=np.asarray(
                    self.model.mesh_vert[vertex_start:vertex_start + vertex_count],
                    dtype=float,
                ).copy(),
                faces=np.asarray(
                    self.model.mesh_face[face_start:face_start + face_count],
                    dtype=np.int64,
                ).copy(),
                process=False,
            )
        else:
            return None
        transform = np.eye(4)
        transform[:3, :3] = np.asarray(data.geom_xmat[geom_id], dtype=float).reshape(3, 3)
        transform[:3, 3] = np.asarray(data.geom_xpos[geom_id], dtype=float)
        mesh.apply_transform(transform)
        return mesh

    @staticmethod
    def _merge_colored_meshes(records: Sequence[tuple[Any, np.ndarray]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        vertices = []
        colors = []
        faces = []
        offset = 0
        for mesh, color in records:
            mesh_vertices = np.asarray(mesh.vertices, dtype=float)
            mesh_faces = np.asarray(mesh.faces, dtype=np.int64)
            if not len(mesh_vertices) or not len(mesh_faces):
                continue
            vertices.append(mesh_vertices)
            colors.append(np.repeat(np.asarray(color, dtype=np.uint8)[None], len(mesh_vertices), axis=0))
            faces.append(mesh_faces + offset)
            offset += len(mesh_vertices)
        if not vertices:
            return (
                np.empty((0, 3), dtype=float),
                np.empty((0, 3), dtype=np.uint8),
                np.empty((0, 3), dtype=np.int64),
            )
        return np.concatenate(vertices), np.concatenate(colors), np.concatenate(faces)

    def _ply_scene_mesh(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        import trimesh

        if hasattr(self, "_cached_ply_scene_mesh"):
            return self._cached_ply_scene_mesh
        data = self._data()
        records = []
        counts = {"table": 0, "microwave": 0, "bowl": 0}
        for geom_id in range(self.model.ngeom):
            kind = self._scene_object_kind(geom_id)
            if kind is None:
                continue
            mesh = self._geom_world_mesh(data, geom_id)
            if mesh is None or mesh.is_empty:
                continue
            vertices, faces = trimesh.remesh.subdivide_to_size(
                np.asarray(mesh.vertices, dtype=float),
                np.asarray(mesh.faces, dtype=np.int64),
                max_edge=PLY_MAX_EDGE_M[kind],
                max_iter=8,
            )
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
            records.append((mesh, PLY_OBJECT_COLORS[kind]))
            counts[kind] += 1
        vertices, colors, faces = self._merge_colored_meshes(records)
        metadata = {
            "objects": counts,
            "vertex_count": len(vertices),
            "face_count": len(faces),
            "robot_geometry_count": 0,
            "source": "saved_mujoco_visual_geometry",
        }
        self._cached_ply_scene_mesh = vertices, colors, faces, metadata
        return self._cached_ply_scene_mesh

    @staticmethod
    def _cylinder_mesh(start: np.ndarray, end: np.ndarray, radius: float,
                       *, sections: int = 12):
        import trimesh

        start = np.asarray(start, dtype=float)
        end = np.asarray(end, dtype=float)
        if float(np.linalg.norm(end - start)) <= 1e-9:
            return None
        return trimesh.creation.cylinder(
            radius=radius, segment=np.vstack((start, end)), sections=sections,
        )

    def _trajectory_mesh(self, trace: dict, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        import trimesh

        records = []
        path = np.asarray(trace.get("tcp_path_world_m") or [], dtype=float).reshape(-1, 3)
        path_indices = _path_sample_indices(path, maximum=MAX_PATH_LINE_POINTS) if len(path) else []
        for start, end in zip(path[path_indices][:-1], path[path_indices][1:]):
            mesh = self._cylinder_mesh(start, end, PLY_PATH_RADIUS_M, sections=8)
            if mesh is not None:
                records.append((mesh, np.asarray([0, 205, 255], dtype=np.uint8)))
        target_mesh = trimesh.creation.icosphere(subdivisions=2, radius=PLY_TARGET_RADIUS_M)
        target_mesh.apply_translation(np.asarray(target, dtype=float))
        records.append((target_mesh, np.asarray([255, 198, 5], dtype=np.uint8)))

        ghosts, _, _ = self._ghosts(trace)
        gripper_vertex_count = 0
        gripper_face_count = 0
        for ghost in ghosts:
            color = np.asarray(ghost["rgb"], dtype=np.uint8)
            for start, end in ghost["segments"]:
                mesh = self._cylinder_mesh(start, end, FORK_RADIUS_M, sections=12)
                if mesh is not None:
                    gripper_vertex_count += len(mesh.vertices)
                    gripper_face_count += len(mesh.faces)
                    records.append((mesh, color))
        vertices, colors, faces = self._merge_colored_meshes(records)
        return vertices, colors, faces, {
            "path_segment_count": max(0, len(path_indices) - 1),
            "gripper_pose_count": len(ghosts),
            "gripper_progress_colors": [ghost["progress_color"] for ghost in ghosts],
            "gripper_vertex_count": gripper_vertex_count,
            "gripper_face_count": gripper_face_count,
            "target_mesh": "icosphere",
        }

    def write_trajectory_ply(self, path: Path, trace: dict, target: np.ndarray) -> dict:
        scene_vertices, scene_colors, scene_faces, scene_meta = self._ply_scene_mesh()
        trajectory_vertices, trajectory_colors, trajectory_faces, trajectory_meta = (
            self._trajectory_mesh(trace, target)
        )
        faces = np.concatenate((
            scene_faces,
            trajectory_faces + len(scene_vertices),
        ), axis=0)
        vertices = np.concatenate((scene_vertices, trajectory_vertices), axis=0)
        colors = np.concatenate((scene_colors, trajectory_colors), axis=0)
        write_colored_ply(path, vertices, colors, faces)
        return {
            "format": "colored_triangle_mesh_ply",
            "scene": scene_meta,
            "trajectory": trajectory_meta,
            "vertex_count": len(vertices),
            "face_count": len(faces),
            "robot_geometry_count": 0,
        }

    def _append_ghosts(self, renderer, ghosts: list[dict], target: np.ndarray) -> None:
        mujoco = self.mujoco
        for ghost in ghosts:
            for start, end in ghost["segments"]:
                if renderer.scene.ngeom >= renderer.scene.maxgeom:
                    raise RuntimeError("MuJoCo renderer scene has no room for tuning-fork grippers")
                scene_geom = renderer.scene.geoms[renderer.scene.ngeom]
                mujoco.mjv_initGeom(
                    scene_geom, mujoco.mjtGeom.mjGEOM_CAPSULE,
                    np.zeros(3), np.zeros(3), np.eye(3).reshape(-1), ghost["rgba"],
                )
                mujoco.mjv_connector(
                    scene_geom, mujoco.mjtGeom.mjGEOM_CAPSULE,
                    FORK_RADIUS_M, start, end,
                )
                scene_geom.category = mujoco.mjtCatBit.mjCAT_DECOR
                renderer.scene.ngeom += 1
        marker = renderer.scene.geoms[renderer.scene.ngeom]
        mujoco.mjv_initGeom(
            marker, mujoco.mjtGeom.mjGEOM_SPHERE,
            np.asarray([0.012, 0.012, 0.012], dtype=float),
            np.asarray(target, dtype=float), np.eye(3).reshape(-1),
            np.asarray([1.0, 0.78, 0.02, 0.95], dtype=np.float32),
        )
        marker.category = mujoco.mjtCatBit.mjCAT_DECOR
        renderer.scene.ngeom += 1

    def _append_collision_waypoint_marker(self, renderer, trace: dict) -> dict | None:
        """用红色球和白色三轴十字标出首个碰撞 TCP waypoint。"""
        invalid = trace.get("first_invalid_waypoint")
        points = np.asarray(trace.get("tcp_path_world_m") or [], dtype=float).reshape(-1, 3)
        if not isinstance(invalid, int) or not 0 <= invalid < len(points):
            return None
        center = points[invalid]
        if not np.all(np.isfinite(center)):
            return None

        required_geoms = 4
        if renderer.scene.ngeom + required_geoms > renderer.scene.maxgeom:
            raise RuntimeError("MuJoCo renderer scene has no room for collision waypoint marker")
        sphere = renderer.scene.geoms[renderer.scene.ngeom]
        self.mujoco.mjv_initGeom(
            sphere, self.mujoco.mjtGeom.mjGEOM_SPHERE,
            np.repeat(COLLISION_MARKER_RADIUS_M, 3), center,
            np.eye(3).reshape(-1),
            np.asarray([1.0, 0.02, 0.02, 0.98], dtype=np.float32),
        )
        sphere.category = self.mujoco.mjtCatBit.mjCAT_DECOR
        renderer.scene.ngeom += 1
        for direction in np.eye(3):
            cross = renderer.scene.geoms[renderer.scene.ngeom]
            self.mujoco.mjv_initGeom(
                cross, self.mujoco.mjtGeom.mjGEOM_CAPSULE,
                np.zeros(3), np.zeros(3), np.eye(3).reshape(-1),
                np.asarray([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            )
            self.mujoco.mjv_connector(
                cross, self.mujoco.mjtGeom.mjGEOM_CAPSULE, 0.004,
                center - COLLISION_MARKER_ARM_M * direction,
                center + COLLISION_MARKER_ARM_M * direction,
            )
            cross.category = self.mujoco.mjtCatBit.mjCAT_DECOR
            renderer.scene.ngeom += 1
        return {
            "waypoint": invalid,
            "tcp_position_world_m": center.tolist(),
            "semantics": "first invalid TCP waypoint; not an exact surface contact point",
        }

    def _hide_robot(self, renderer) -> None:
        """trajectory.png 只保留环境；机械臂和实体夹爪由音叉符号替代。"""
        def copy_geom(target, source):
            self.mujoco.mjv_initGeom(
                target, int(source.type), np.asarray(source.size, dtype=float),
                np.asarray(source.pos, dtype=float),
                np.asarray(source.mat, dtype=float).reshape(-1),
                np.asarray(source.rgba, dtype=np.float32),
            )
            for field in (
                "dataid", "matid", "objtype", "objid", "category", "texid",
                "texuniform", "texcoord", "segid", "transparent",
            ):
                setattr(target, field, getattr(source, field))
            for field in (
                "emission", "specular", "shininess", "reflectance",
                "modelrbound", "camdist",
            ):
                setattr(target, field, float(getattr(source, field)))
            target.texrepeat[:] = source.texrepeat

        write = 0
        for read in range(renderer.scene.ngeom):
            source = renderer.scene.geoms[read]
            is_robot = (
                int(source.objtype) == int(self.mujoco.mjtObj.mjOBJ_GEOM)
                and int(source.objid) in self.robot_geom_ids
            )
            if is_robot:
                continue
            if write != read:
                copy_geom(renderer.scene.geoms[write], source)
            write += 1
        renderer.scene.ngeom = write

    def _append_path_line(self, renderer, trace: dict) -> list[int]:
        points = np.asarray(trace.get("tcp_path_world_m") or [], dtype=float)
        if len(points) < 2:
            return list(range(len(points)))
        indices = _path_sample_indices(points, maximum=MAX_PATH_LINE_POINTS)
        sampled = points[indices]
        for start, end in zip(sampled[:-1], sampled[1:]):
            if renderer.scene.ngeom >= renderer.scene.maxgeom:
                raise RuntimeError("MuJoCo renderer scene has no room for the TCP path")
            geom = renderer.scene.geoms[renderer.scene.ngeom]
            self.mujoco.mjv_initGeom(
                geom, self.mujoco.mjtGeom.mjGEOM_LINE,
                np.zeros(3), np.zeros(3), np.eye(3).reshape(-1),
                np.asarray([0.02, 0.78, 1.0, 0.88], dtype=np.float32),
            )
            self.mujoco.mjv_connector(
                geom, self.mujoco.mjtGeom.mjGEOM_LINE, 2.5, start, end,
            )
            geom.category = self.mujoco.mjtCatBit.mjCAT_DECOR
            renderer.scene.ngeom += 1
        return indices

    @staticmethod
    def _sphere_color(record: dict) -> tuple[np.ndarray, np.ndarray]:
        types = set(record.get("collision_types") or [])
        if types == {"self_collision"}:
            return (np.asarray([0.95, 0.05, 1.0, 0.82], dtype=np.float32),
                    np.asarray([242, 13, 255], dtype=np.uint8))
        if "scene_collision" in types:
            return (np.asarray([1.0, 0.02, 0.02, 0.84], dtype=np.float32),
                    np.asarray([255, 5, 5], dtype=np.uint8))
        return (np.asarray([1.0, 0.58, 0.02, 0.18], dtype=np.float32),
                np.asarray([255, 148, 5], dtype=np.uint8))

    @staticmethod
    def _sample_sphere(center: np.ndarray, radius: float, count: int = 64) -> np.ndarray:
        indices = np.arange(count, dtype=float) + 0.5
        z = 1.0 - 2.0 * indices / count
        ring = np.sqrt(np.maximum(0.0, 1.0 - z * z))
        angle = np.pi * (3.0 - np.sqrt(5.0)) * indices
        unit = np.column_stack((ring * np.cos(angle), ring * np.sin(angle), z))
        return np.asarray(center, dtype=float) + float(radius) * unit

    def _append_collision_spheres(self, renderer, trace: dict) -> tuple[np.ndarray, np.ndarray, dict]:
        records = (trace.get("collision_check") or {}).get(
            "robot_collision_spheres_world", [],
        )
        points = []
        colors = []
        scene_count = 0
        self_count = 0
        for record in records:
            center = np.asarray(record["center_world_m"], dtype=float)
            radius = float(record["radius_m"])
            if center.shape != (3,) or not np.all(np.isfinite(center)) or radius <= 0.0:
                continue
            if renderer.scene.ngeom >= renderer.scene.maxgeom:
                raise RuntimeError("MuJoCo renderer scene has no room for cuRobo collision spheres")
            rgba, rgb = self._sphere_color(record)
            geom = renderer.scene.geoms[renderer.scene.ngeom]
            self.mujoco.mjv_initGeom(
                geom, self.mujoco.mjtGeom.mjGEOM_SPHERE,
                np.asarray([radius, radius, radius], dtype=float),
                center, np.eye(3).reshape(-1), rgba,
            )
            geom.category = self.mujoco.mjtCatBit.mjCAT_DECOR
            renderer.scene.ngeom += 1
            points.append(self._sample_sphere(center, radius))
            colors.append(np.repeat(rgb[None], 64, axis=0))
            types = set(record.get("collision_types") or [])
            scene_count += "scene_collision" in types
            self_count += "self_collision" in types
        return (
            np.concatenate(points, axis=0) if points else np.empty((0, 3), dtype=float),
            np.concatenate(colors, axis=0) if colors else np.empty((0, 3), dtype=np.uint8),
            {
                "active_count": len(records),
                "scene_colliding_count": scene_count,
                "self_colliding_count": self_count,
            },
        )

    def _append_collision_meshes(self, renderer, trace: dict) -> dict:
        """Overlay the exact cuRobo obstacle records implicated by diagnostics.

        Cuboids use cuRobo's exported dimensions.  Mesh records reuse the source
        MuJoCo collision asset when its exported obstacle name retains the geom
        id; pose comes from the cuRobo snapshot transformed back to world.
        """
        check = trace.get("collision_check") or {}
        names = set(check.get("colliding_obstacles") or [])
        for sphere in check.get("robot_collision_spheres_world") or []:
            names.update(sphere.get("colliding_obstacles") or [])
        rendered = []
        unavailable = []
        for name in sorted(names):
            cuboid = (self.scene_model.get("cuboid") or {}).get(name)
            mesh = (self.scene_model.get("mesh") or {}).get(name)
            record = cuboid or mesh
            if record is None:
                unavailable.append(name)
                continue
            if renderer.scene.ngeom >= renderer.scene.maxgeom:
                raise RuntimeError("MuJoCo renderer scene has no room for cuRobo collision meshes")
            world_pose = self.world_from_scene @ _pose_matrix(record["pose"])
            geom = renderer.scene.geoms[renderer.scene.ngeom]
            rgba = np.asarray([1.0, 0.03, 0.03, 0.46], dtype=np.float32)
            if cuboid is not None:
                size = np.asarray(cuboid["dims"], dtype=float) / 2.0
                self.mujoco.mjv_initGeom(
                    geom, self.mujoco.mjtGeom.mjGEOM_BOX, size,
                    world_pose[:3, 3], world_pose[:3, :3].reshape(-1), rgba,
                )
                source = "curobo_exported_cuboid"
            else:
                match = re.match(r"mj_geom_(\d+)_", str(name))
                geom_id = int(match.group(1)) if match else -1
                if not 0 <= geom_id < self.model.ngeom:
                    unavailable.append(name)
                    continue
                geom_type = int(self.model.geom_type[geom_id])
                data_id = int(self.model.geom_dataid[geom_id])
                if geom_type != int(self.mujoco.mjtGeom.mjGEOM_MESH) or data_id < 0:
                    unavailable.append(name)
                    continue
                self.mujoco.mjv_initGeom(
                    geom, geom_type, self.model.geom_size[geom_id],
                    world_pose[:3, 3], world_pose[:3, :3].reshape(-1), rgba,
                )
                geom.dataid = data_id
                source = "curobo_exported_mesh_pose_with_mujoco_collision_asset"
            geom.matid = -1
            geom.category = self.mujoco.mjtCatBit.mjCAT_DECOR
            renderer.scene.ngeom += 1
            rendered.append({"name": name, "source": source})
        return {"rendered": rendered, "unavailable": unavailable}

    def render(self, path: Path, trace: dict, target: np.ndarray, *, title: str,
               collision_only: bool = False) -> tuple[np.ndarray, np.ndarray, dict]:
        from PIL import Image, ImageDraw

        if collision_only:
            ghosts = []
            ghost_points = np.empty((0, 3), dtype=float)
            ghost_colors = np.empty((0, 3), dtype=np.uint8)
        else:
            ghosts, ghost_points, ghost_colors = self._ghosts(trace)
        invalid = trace.get("first_invalid_waypoint")
        joints = np.asarray(trace["joint_positions_rad"], dtype=float)
        base = self._data(joints[invalid]) if isinstance(invalid, int) and 0 <= invalid < len(joints) else self._data()
        collision_records = [
            record for record in (trace.get("collision_check") or {}).get(
                "robot_collision_spheres_world", [],
            ) if record.get("collision_types")
        ]
        if collision_only and collision_records:
            focus = np.mean([
                np.asarray(record["center_world_m"], dtype=float)
                for record in collision_records
            ], axis=0)
            views = (
                ("collision task", -45.0, -20.0, 0.58, focus),
                ("collision top", 90.0, -89.0, 0.70, focus),
                ("collision side", 90.0, -20.0, 0.58, focus),
            )
        else:
            path_points = np.asarray(trace.get("tcp_path_world_m") or [], dtype=float).reshape(-1, 3)
            focus_points = np.vstack((path_points, np.asarray(target, dtype=float).reshape(1, 3)))
            lower, upper = focus_points.min(axis=0), focus_points.max(axis=0)
            focus = (lower + upper) / 2.0
            span = float(np.max(upper - lower))
            distance = min(0.85, max(0.30, 0.16 + 2.5 * span))
            views = tuple(
                (label, azimuth, elevation, distance, focus)
                for label, azimuth, elevation in TRAJECTORY_VIEW_ANGLES
            )
        panels: list[tuple[str, Image.Image]] = []
        sphere_points = np.empty((0, 3), dtype=float)
        sphere_colors = np.empty((0, 3), dtype=np.uint8)
        sphere_summary = {"active_count": 0, "scene_colliding_count": 0,
                          "self_colliding_count": 0}
        mesh_summary = {"rendered": [], "unavailable": []}
        collision_waypoint_marker = None
        path_line_indices = []
        for label, azimuth, elevation, distance, lookat in views:
            camera = self.mujoco.MjvCamera()
            camera.type = self.mujoco.mjtCamera.mjCAMERA_FREE
            camera.azimuth = azimuth
            camera.elevation = elevation
            camera.distance = distance
            camera.lookat[:] = lookat
            renderer = self.mujoco.Renderer(self.model, height=480, width=640)
            renderer.update_scene(base, camera=camera)
            if not collision_only:
                self._hide_robot(renderer)
                path_line_indices = self._append_path_line(renderer, trace)
            self._append_ghosts(renderer, ghosts, target)
            if not collision_only:
                panel_collision_marker = self._append_collision_waypoint_marker(
                    renderer, trace,
                )
            else:
                panel_collision_marker = None
            if collision_only:
                panel_points, panel_colors, panel_summary = self._append_collision_spheres(
                    renderer, trace,
                )
            else:
                panel_points = np.empty((0, 3), dtype=float)
                panel_colors = np.empty((0, 3), dtype=np.uint8)
                panel_summary = sphere_summary
            if collision_only:
                panel_mesh_summary = self._append_collision_meshes(renderer, trace)
            else:
                # Keep trajectory.png focused on the gripper path. Exact cuRobo
                # collision proxies remain available in collision_frame.png.
                panel_mesh_summary = {"rendered": [], "unavailable": []}
            if not panels:
                sphere_points, sphere_colors = panel_points, panel_colors
                sphere_summary = panel_summary
                mesh_summary = panel_mesh_summary
                collision_waypoint_marker = panel_collision_marker
            panel = Image.fromarray(renderer.render())
            renderer.close()
            draw = ImageDraw.Draw(panel)
            draw.rectangle((0, 0, panel.width, 28), fill=(0, 0, 0))
            draw.text((10, 8), label, fill=(255, 255, 255))
            panels.append((label, panel))
        reasons = trace.get("collision_check") or {}
        collision_names = reasons.get("colliding_obstacles") or []
        colliding_spheres = [
            record for record in reasons.get("robot_collision_spheres_world", [])
            if record.get("collision_types")
        ]
        sphere_evidence = ", ".join(
            f"#{record.get('sphere_index')} {record.get('link')} "
            f"r={1000.0 * float(record.get('radius_m', 0.0)):.1f}mm"
            for record in colliding_spheres[:4]
        )
        if collision_only:
            subtitle = (
                "collision frame only; yellow = requested TCP; spheres orange=active, "
                "red=scene, magenta=self; red translucent geometry = cuRobo collision mesh; "
                f"hit = {sphere_evidence or 'not individually attributed'}; "
                f"obstacle = {', '.join(collision_names) or 'not confirmed'}"
            )
        else:
            subtitle = (
                "fork order red>orange>yellow>green>cyan>blue>purple; "
                f"red/white cross = first collision waypoint {invalid}; "
                "cyan line = TCP path; yellow sphere = requested TCP"
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        view_images = {}
        if collision_only:
            canvas = Image.new("RGB", (640 * len(panels), 552), (0, 0, 0))
            for index, (_, panel) in enumerate(panels):
                canvas.paste(panel, (640 * index, 72))
            draw = ImageDraw.Draw(canvas)
            draw.text((12, 12), title[:220], fill=(255, 255, 255))
            draw.text((12, 42), subtitle[:260], fill=(255, 220, 80))
            canvas.save(path)
        else:
            header_height = 104
            title_lines = textwrap.wrap(title, width=94)[:2] or [title]
            subtitle_lines = textwrap.wrap(subtitle, width=94)[:2] or [subtitle]
            for label, panel in panels:
                canvas = Image.new(
                    "RGB", (panel.width, panel.height + header_height), (0, 0, 0),
                )
                canvas.paste(panel, (0, header_height))
                draw = ImageDraw.Draw(canvas)
                y = 10
                for line in title_lines:
                    draw.text((12, y), line, fill=(255, 255, 255))
                    y += 16
                y += 4
                for line in subtitle_lines:
                    draw.text((12, y), line, fill=(255, 220, 80))
                    y += 16
                view_path = _trajectory_view_path(path, label)
                canvas.save(view_path)
                view_images[label] = view_path.name
                if label == "task":
                    canvas.save(path)
        overlay_points = np.concatenate((ghost_points, sphere_points), axis=0)
        overlay_colors = np.concatenate((ghost_colors, sphere_colors), axis=0)
        return overlay_points, overlay_colors, {
            "renderer": "mujoco_saved_scene_with_tuning_fork_grippers",
            "scene_ref": str(self.scene_path),
            "base_state_source": self.base_state_source,
            "ghost_waypoints": [item["waypoint"] for item in ghosts],
            "ghost_progress_colors": [item["progress_color"] for item in ghosts],
            "gripper_representation": "tuning_fork_tcp_symbol",
            "fork_handle_direction": "TCP -Z toward wrist/end joint",
            "fork_segment_count": len(FORK_LOCAL_SEGMENTS_M),
            "path_line_waypoints": path_line_indices,
            "robot_hidden": not collision_only,
            "views": [item[0] for item in views],
            "view_images": view_images,
            "image_layout": "montage" if collision_only else "standalone_per_view",
            "primary_image": path.name,
            "mode": "collision_frame" if collision_only else "trajectory",
            "rendered_configuration": (
                f"first_invalid_waypoint_{invalid}" if isinstance(invalid, int)
                else "saved_simulation_state"
            ),
            "collision_spheres": sphere_summary,
            "collision_meshes": mesh_summary,
            "collision_waypoint_marker": collision_waypoint_marker,
            "colliding_sphere_evidence": colliding_spheres,
            "collision_sphere_color_key": {
                "orange": "active cuRobo sphere without per-sphere collision attribution",
                "red": "sphere colliding with a scene obstacle",
                "magenta": "sphere participating in self-collision",
            },
        }


def _trace_reason(trace: dict) -> tuple[list[str], list[str], str]:
    collision_check = trace.get("collision_check") or {}
    collision_names = collision_check.get("colliding_obstacles", [])
    invalid_reasons = []
    if collision_check.get("joint_limit_violation"):
        invalid_reasons.append("joint_limit")
    if collision_check.get("self_collision"):
        invalid_reasons.append("self_collision")
    if collision_check.get("scene_collision"):
        invalid_reasons.append("scene_collision")
    invalid_reasons.extend(collision_names)
    reason = (
        f"first invalid={trace.get('first_invalid_waypoint')}; "
        f"reason={', '.join(invalid_reasons) or 'not confirmed'}"
    )
    return collision_names, invalid_reasons, reason


def _write_index(output: Path, metadata: dict, rows: Sequence[dict],
                 rendering: dict | None = None) -> None:
    lines = [
        "# cuRobo no-solution 3D evidence", "",
        f"- Failure category: `{metadata.get('failure_category')}`",
        f"- Trace semantics: {metadata.get('trace_semantics')}",
        f"- Exact solver trajectories: {metadata.get('exact_solver_trajectory_count', 0)}",
        f"- Diagnostic IK interpolations: {metadata.get('synthetic_trace_count', 0)}",
    ]
    if rendering:
        lines.extend([
            "- Visualization: saved MuJoCo environment with the robot hidden and sparse tuning-fork TCP poses",
            f"- Views: `{', '.join(rendering.get('views') or [])}`",
            "- Image layout: every trajectory view is a standalone PNG; trajectory.png is the task-view compatibility image",
            "- Fork order: red -> orange -> yellow -> green -> cyan -> blue -> purple follows trajectory time",
            "- Fork geometry: one handle points toward the wrist/end joint; two tines represent the two fingers",
            "- Collision waypoint: red sphere with a white 3-axis cross marks the first invalid TCP waypoint, not an exact surface contact point",
            "- Other colors: cyan line = TCP path; yellow sphere = requested TCP",
            "- Collision spheres: orange = active cuRobo sphere; red = scene collision; magenta = self-collision",
            "- Collision mesh: translucent red in PNG; matching obstacle points are red in PLY",
        ])
    lines.extend([
        "",
        "| Trace | Source | Exact solver path | Invalid waypoint | Invalid reason | Trajectory | Collision frame | Point cloud |",
        "|---|---|---:|---:|---|---|---|---|",
    ])
    for row in rows:
        reasons = ", ".join(row["invalid_reasons"]) or "-"
        view_refs = row.get("view_image_refs") or {}
        if view_refs:
            trajectory_cell = "<br>".join(
                f"[{label}]({reference})" for label, reference in view_refs.items()
            )
        else:
            trajectory_cell = f"![]({row['image_ref']})"
        lines.append(
            f"| `{row['trace_id']}` | `{row['source']}` | "
            f"{'no' if row['synthetic'] else 'yes'} | "
            f"{row['first_invalid_waypoint'] if row['first_invalid_waypoint'] is not None else '-'} | "
            f"{reasons} | {trajectory_cell} | "
            f"{('![](' + row['collision_frame_ref'] + ')') if row.get('collision_frame_ref') else '-'} | "
            f"`{row['point_cloud_ref']}` |"
        )
    (output / "index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _set_equal_axes(ax, points: np.ndarray) -> None:
    if len(points) == 0:
        return
    lower = points.min(axis=0)
    upper = points.max(axis=0)
    center = (lower + upper) / 2.0
    radius = max(float(np.max(upper - lower)) / 2.0, 0.1)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def _plot_bundle(path: Path, scene_points: np.ndarray, traces: Sequence[dict],
                 target_position: np.ndarray, *, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(18, 6), dpi=150)
    all_points = [scene_points, target_position.reshape(1, 3)]
    all_points.extend(np.asarray(trace["tcp_path_world_m"], dtype=float) for trace in traces)
    all_points.extend(
        np.asarray(trace.get("robot_sphere_centers_world_m") or [], dtype=float).reshape(-1, 3)
        for trace in traces
    )
    bounds = np.concatenate([item for item in all_points if len(item)], axis=0)
    ax3d = figure.add_subplot(1, 3, 1, projection="3d")
    ax_top = figure.add_subplot(1, 3, 2)
    ax_side = figure.add_subplot(1, 3, 3)
    if len(scene_points):
        ax3d.scatter(scene_points[:, 0], scene_points[:, 1], scene_points[:, 2],
                     s=0.45, c="#a0a0a0", alpha=0.14, depthshade=False)
        ax_top.scatter(scene_points[:, 0], scene_points[:, 1],
                       s=0.35, c="#a0a0a0", alpha=0.12)
        ax_side.scatter(scene_points[:, 1], scene_points[:, 2],
                        s=0.35, c="#a0a0a0", alpha=0.12)
    for index, trace in enumerate(traces):
        points = np.asarray(trace["tcp_path_world_m"], dtype=float)
        color = np.asarray(TRACE_COLORS[index % len(TRACE_COLORS)]) / 255.0
        linestyle = "--" if trace.get("synthetic") else "-"
        ax3d.plot(points[:, 0], points[:, 1], points[:, 2], color=color,
                  linewidth=1.5, linestyle=linestyle, alpha=0.95)
        ax_top.plot(points[:, 0], points[:, 1], color=color, linewidth=1.5,
                    linestyle=linestyle, alpha=0.95)
        ax_side.plot(points[:, 1], points[:, 2], color=color, linewidth=1.5,
                     linestyle=linestyle, alpha=0.95)
        ax3d.scatter(*points[-1], color=color, s=18, depthshade=False)
        ax_top.scatter(points[-1, 0], points[-1, 1], color=color, s=20)
        ax_side.scatter(points[-1, 1], points[-1, 2], color=color, s=20)
        spheres = np.asarray(
            trace.get("robot_sphere_centers_world_m") or [], dtype=float,
        ).reshape(-1, 3)
        if len(spheres):
            ax3d.scatter(spheres[:, 0], spheres[:, 1], spheres[:, 2],
                         s=0.7, color="#466eff", alpha=0.12, depthshade=False)
            ax_top.scatter(spheres[:, 0], spheres[:, 1], s=0.6,
                           color="#466eff", alpha=0.10)
            ax_side.scatter(spheres[:, 1], spheres[:, 2], s=0.6,
                            color="#466eff", alpha=0.10)
        invalid = trace.get("first_invalid_waypoint")
        if invalid is not None and 0 <= int(invalid) < len(points):
            failed = points[int(invalid)]
            ax3d.scatter(*failed, color="#ff0000", marker="x", s=55,
                         linewidths=2.0, depthshade=False)
            ax_top.scatter(failed[0], failed[1], color="#ff0000", marker="x",
                           s=55, linewidths=2.0)
            ax_side.scatter(failed[1], failed[2], color="#ff0000", marker="x",
                            s=55, linewidths=2.0)
    ax3d.scatter(*target_position, marker="*", s=110, color="#00d4ff",
                 edgecolor="#00333d", linewidth=0.7, depthshade=False)
    ax_top.scatter(target_position[0], target_position[1], marker="*", s=110,
                   color="#00d4ff", edgecolor="#00333d", linewidth=0.7)
    ax_side.scatter(target_position[1], target_position[2], marker="*", s=110,
                    color="#00d4ff", edgecolor="#00333d", linewidth=0.7)
    ax3d.set(xlabel="world X (m)", ylabel="world Y (m)", zlabel="world Z (m)",
             title="3D")
    ax3d.view_init(elev=25, azim=-55)
    _set_equal_axes(ax3d, bounds)
    ax_top.set(xlabel="world X (m)", ylabel="world Y (m)", title="top / XY")
    ax_side.set(xlabel="world Y (m)", ylabel="world Z (m)", title="side / YZ")
    ax_top.set_aspect("equal", adjustable="box")
    ax_side.set_aspect("equal", adjustable="box")
    ax_top.grid(alpha=0.25)
    ax_side.grid(alpha=0.25)
    figure.suptitle(title)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def export_failure_bundle(output_dir: Path, *, scene_model: dict,
                          target_pose_world: np.ndarray, traces: Sequence[dict],
                          metadata: dict[str, Any], optimizer_history=None,
                          world_from_scene: np.ndarray | None = None) -> dict:
    """Write a PNG/PLY/JSON/Markdown bundle for every available failed trace."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    traces = tuple(traces)
    if not traces:
        raise ValueError("failure bundle requires at least one trace")
    scene_transform = (np.eye(4) if world_from_scene is None
                       else np.asarray(world_from_scene, dtype=float))
    scene_points, obstacle_names = sample_scene_points(
        scene_model, world_from_scene=scene_transform,
    )
    target = np.asarray(target_pose_world, dtype=float)
    if target.shape != (4, 4):
        raise ValueError("target_pose_world must be 4x4")

    simulation_renderer = None
    renderer_error = None
    try:
        simulation_renderer = _MujocoTrajectoryRenderer(
            output, scene_model=scene_model, world_from_scene=scene_transform,
        )
    except Exception as exc:
        renderer_error = f"{type(exc).__name__}: {exc}"

    rows = []
    render_metadata = None
    ply_render_metadata = None
    for index, trace in enumerate(traces):
        trace_id = str(trace.get("trace_id") or f"trace_{index:03d}")
        points = np.asarray(trace["tcp_path_world_m"], dtype=float)
        color = np.asarray(TRACE_COLORS[index % len(TRACE_COLORS)], dtype=np.uint8)
        per_dir = output / "trajectories" / trace_id
        per_dir.mkdir(parents=True, exist_ok=True)
        path_colors = np.repeat(color[None], len(points), axis=0)
        target_points = np.repeat(target[None, :3, 3], 32, axis=0)
        target_colors = np.repeat(
            np.array([[0, 212, 255]], dtype=np.uint8), len(target_points), axis=0,
        )
        collision_names, invalid_reasons, reason = _trace_reason(trace)
        scene_colors = np.repeat(
            np.array([[150, 150, 150]], dtype=np.uint8), len(scene_points), axis=0,
        )
        if collision_names:
            collision_set = set(collision_names)
            for point_index, obstacle_name in enumerate(obstacle_names):
                if obstacle_name in collision_set:
                    scene_colors[point_index] = [255, 8, 8]
        ghost_points = np.empty((0, 3), dtype=float)
        ghost_colors = np.empty((0, 3), dtype=np.uint8)
        trace_rendering = None
        ply_rendering = None
        collision_frame_rendering = None
        collision_frame_ref = None
        if simulation_renderer is not None:
            try:
                ghost_points, ghost_colors, trace_rendering = simulation_renderer.render(
                    per_dir / "trajectory.png", trace, target[:3, 3],
                    title=f"{trace_id}: {trace.get('source', 'unknown')} | {reason}",
                )
                render_metadata = trace_rendering
                if (trace.get("collision_check") or {}).get(
                    "robot_collision_spheres_world", [],
                ):
                    _, _, collision_frame_rendering = simulation_renderer.render(
                        per_dir / "collision_frame.png", trace, target[:3, 3],
                        title=f"{trace_id}: collision frame at waypoint {trace.get('first_invalid_waypoint')}",
                        collision_only=True,
                    )
                    collision_frame_ref = f"trajectories/{trace_id}/collision_frame.png"
            except Exception as exc:
                renderer_error = f"{type(exc).__name__}: {exc}"
        if trace_rendering is None:
            _plot_bundle(
                per_dir / "trajectory.png", scene_points, (trace,), target[:3, 3],
                title=f"{trace_id}: {trace.get('source', 'unknown')}\n{reason}",
            )
        if simulation_renderer is not None:
            ply_rendering = simulation_renderer.write_trajectory_ply(
                per_dir / "scene_and_trajectory.ply", trace, target[:3, 3],
            )
        else:
            # Fallback stays robot-free even when MuJoCo cannot be loaded.
            per_points = np.concatenate(
                (scene_points, points, target_points), axis=0,
            )
            per_colors = np.concatenate(
                (scene_colors, path_colors, target_colors), axis=0,
            )
            write_colored_ply(per_dir / "scene_and_trajectory.ply", per_points, per_colors)
            ply_rendering = {
                "format": "colored_point_cloud_ply",
                "fallback": True,
                "robot_geometry_count": 0,
                "vertex_count": len(per_points),
                "face_count": 0,
            }
        if ply_render_metadata is None:
            ply_render_metadata = ply_rendering
        (per_dir / "trajectory.json").write_text(
            json.dumps(trace, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        rows.append({
            "trace_id": trace_id,
            "source": trace.get("source"),
            "synthetic": bool(trace.get("synthetic")),
            "waypoint_count": len(points),
            "first_invalid_waypoint": trace.get("first_invalid_waypoint"),
            "invalid_reasons": invalid_reasons,
            "colliding_obstacles": collision_names,
            "image_ref": f"trajectories/{trace_id}/trajectory.png",
            "view_image_refs": {
                label: f"trajectories/{trace_id}/{filename}"
                for label, filename in (
                    (trace_rendering or {}).get("view_images") or {}
                ).items()
            },
            "point_cloud_ref": f"trajectories/{trace_id}/scene_and_trajectory.ply",
            "ply_rendering": ply_rendering,
            "rendering": trace_rendering or {"renderer": "matplotlib_geometry_fallback"},
            "collision_frame_ref": collision_frame_ref,
            "collision_frame_rendering": collision_frame_rendering,
            "ghost_gripper_point_count": 0,
            "gripper_mesh_vertex_count": int(
                (ply_rendering or {}).get("trajectory", {}).get("gripper_vertex_count", 0)
            ),
            "gripper_mesh_face_count": int(
                (ply_rendering or {}).get("trajectory", {}).get("gripper_face_count", 0)
            ),
            "collision_sphere_point_count": 0,
        })

    history_ref = None
    if optimizer_history is not None:
        history_ref = "optimizer_iteration_joint_trajectories.npz"
        if isinstance(optimizer_history, dict):
            np.savez_compressed(
                output / history_ref,
                **{str(key): np.asarray(value, dtype=np.float32)
                   for key, value in optimizer_history.items()},
            )
        else:
            history = np.asarray(optimizer_history, dtype=np.float32)
            np.savez_compressed(output / history_ref, joint_positions_rad=history)

    manifest = {
        "schema_version": 1,
        **metadata,
        "trace_count": len(traces),
        "scene_point_count": len(scene_points),
        "scene_obstacle_count": len(set(obstacle_names)),
        "target_pose_world": target.tolist(),
        "world_from_scene": scene_transform.tolist(),
        "scene_model": scene_model,
        "optimizer_history_ref": history_ref,
        "rendering": render_metadata or {
            "renderer": "matplotlib_geometry_fallback",
            "fallback_reason": renderer_error,
        },
        "ply_rendering": ply_render_metadata,
        "traces": rows,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_index(output, metadata, rows, render_metadata)
    return {
        "output_dir": str(output),
        "manifest_ref": str(output / "manifest.json"),
        "index_ref": str(output / "index.md"),
        "trace_count": len(traces),
    }


def rerender_failure_bundle(bundle_dir: Path) -> dict:
    """Upgrade an existing failure bundle to MuJoCo-background tuning-fork renders."""
    output = Path(bundle_dir)
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    target = np.asarray(manifest["target_pose_world"], dtype=float)[:3, 3]
    renderer = _MujocoTrajectoryRenderer(
        output, scene_model=manifest.get("scene_model") or {},
        world_from_scene=np.asarray(manifest.get("world_from_scene", np.eye(4)), dtype=float),
    )
    updated = 0
    ply_render_metadata = None
    for row in manifest.get("traces", []):
        per_dir = output / "trajectories" / row["trace_id"]
        trace = json.loads((per_dir / "trajectory.json").read_text(encoding="utf-8"))
        _, _, reason = _trace_reason(trace)
        _, _, rendering = renderer.render(
            per_dir / "trajectory.png", trace, target,
            title=f"{row['trace_id']}: {trace.get('source', 'unknown')} | {reason}",
        )
        collision_frame_rendering = None
        collision_frame_ref = None
        if (trace.get("collision_check") or {}).get(
            "robot_collision_spheres_world", [],
        ):
            _, _, collision_frame_rendering = renderer.render(
                per_dir / "collision_frame.png", trace, target,
                title=f"{row['trace_id']}: collision frame at waypoint {trace.get('first_invalid_waypoint')}",
                collision_only=True,
            )
            collision_frame_ref = f"trajectories/{row['trace_id']}/collision_frame.png"
        ply_rendering = renderer.write_trajectory_ply(
            per_dir / "scene_and_trajectory.ply", trace, target,
        )
        if ply_render_metadata is None:
            ply_render_metadata = ply_rendering
        row["rendering"] = rendering
        row["view_image_refs"] = {
            label: f"trajectories/{row['trace_id']}/{filename}"
            for label, filename in (rendering.get("view_images") or {}).items()
        }
        row["ply_rendering"] = ply_rendering
        row["collision_frame_ref"] = collision_frame_ref
        row["collision_frame_rendering"] = collision_frame_rendering
        row["ghost_gripper_point_count"] = 0
        row["gripper_mesh_vertex_count"] = int(
            ply_rendering["trajectory"]["gripper_vertex_count"]
        )
        row["gripper_mesh_face_count"] = int(
            ply_rendering["trajectory"]["gripper_face_count"]
        )
        row["collision_sphere_point_count"] = 0
        updated += 1
    manifest["rendering"] = {
        "renderer": "mujoco_saved_scene_with_tuning_fork_grippers",
        "scene_ref": str(renderer.scene_path),
        "base_state_source": renderer.base_state_source,
        "views": [item[0] for item in TRAJECTORY_VIEW_ANGLES],
        "image_layout": "standalone_per_view",
        "color_key": {
            "cyan_line": "TCP path",
            "fork_progression": "red, orange, yellow, green, cyan, blue, purple in trajectory order",
            "red_white_cross": "first invalid TCP waypoint; not exact surface contact",
            "yellow": "requested TCP",
            "orange_spheres": "active cuRobo collision spheres",
            "red_spheres": "scene-colliding cuRobo spheres",
            "magenta_spheres": "self-colliding cuRobo spheres",
        },
    }
    manifest["ply_rendering"] = ply_render_metadata
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_index(output, manifest, manifest.get("traces", []), manifest["rendering"])
    return {
        "output_dir": str(output),
        "updated_trace_count": updated,
        "scene_ref": str(renderer.scene_path),
    }


__all__ = [
    "export_failure_bundle", "rerender_failure_bundle", "sample_scene_points",
    "write_colored_ply", "read_colored_ply",
]
