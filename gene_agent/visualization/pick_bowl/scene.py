"""显示当前 MuJoCo visual 和实际导出的 cuRobo 碰撞快照。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import trimesh


VISUAL_ROLE_COLORS = {
    "robot": (77, 112, 153),
    "gripper": (220, 111, 57),
    "bowl": (54, 163, 166),
    "microwave": (116, 132, 150),
    "table": (180, 155, 126),
}


def _visual_color(geom_name: str, body_name: str, source_rgb: np.ndarray) -> tuple[int, int, int]:
    """角色色只调整显示，不改原 mesh、MuJoCo 接触或 cuRobo 碰撞文件。"""
    label = f"{body_name}/{geom_name}".lower()
    if any(word in label for word in ("gripper", "finger", "jaw")):
        role = "gripper"
    elif "bowl" in label:
        role = "bowl"
    elif "microwave" in label:
        role = "microwave"
    elif "table" in label:
        role = "table"
    elif any(word in label for word in ("right_", "left_", "robot", "arm")):
        role = "robot"
    else:
        return tuple(int(round(float(x) * 255)) for x in source_rgb)
    base = np.asarray(VISUAL_ROLE_COLORS[role], dtype=float)
    # 保留源材料约四分之一亮度差，让微波炉按钮/外壳仍有层次。
    tint = np.clip(.78 * base + .22 * np.asarray(source_rgb) * 255., 0, 255)
    return tuple(int(round(x)) for x in tint)


def _pose(matrix: np.ndarray) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """将 MuJoCo 的旋转矩阵转成 Viser 要求的 wxyz 四元数。"""
    q = Rotation.from_matrix(matrix[:3, :3]).as_quat()
    return tuple(float(x) for x in q[[3, 0, 1, 2]]), tuple(float(x) for x in matrix[:3, 3])


def _geom_pose(data: Any, geom_id: int) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, :3] = np.asarray(data.geom_xmat[geom_id]).reshape(3, 3)
    transform[:3, 3] = data.geom_xpos[geom_id]
    return transform


def _compiled_mesh(model: Any, geom_id: int) -> trimesh.Trimesh:
    """读取 MuJoCo 编译后的局部顶点；mesh scale 与质心修正已包含在编译结果中。"""
    mesh_id = int(model.geom_dataid[geom_id])
    va = int(model.mesh_vertadr[mesh_id])
    vn = int(model.mesh_vertnum[mesh_id])
    fa = int(model.mesh_faceadr[mesh_id])
    fn = int(model.mesh_facenum[mesh_id])
    mesh = trimesh.Trimesh(
        vertices=np.asarray(model.mesh_vert[va : va + vn]).copy(),
        faces=np.asarray(model.mesh_face[fa : fa + fn]).copy(),
        process=False,
    )
    return mesh


def _collision_group(record: dict, frozen_groups: set[str]) -> str:
    """按物体/部件归并规划代理，不让用户逐个翻看几十个细小 geom。"""
    body = str(record.get("body_name") or "").lower()
    geom = str(record.get("geom_name") or record.get("obstacle_name") or "").lower()
    if "table" in body or (body in ("", "world") and "table" in geom):
        return "桌面"
    if "bowl" in body or "bowl" in geom:
        return "碗"
    if "microwave_support" in body or "microwave_support" in geom:
        return "微波炉支架"
    if "knob_0" in body or "knob_0" in geom:
        return "旋钮 0"
    if "knob_1" in body or "knob_1" in geom:
        return "旋钮 1"
    if "door" in body or "microwave_door" in geom:
        return "微波炉门"
    if "microwave" in body or "microwave" in geom:
        return "微波炉机身"
    if any(body == group or body.startswith(f"{group}_") for group in frozen_groups):
        return "其他机械臂"
    if body.startswith("robot_body") or geom.startswith("robot_body"):
        return "机器人本体"
    return "其他障碍"


class PickBowlScene:
    """持有 Viser 节点，使用当前 data 更新机器人、夹爪和物体的实际位姿。"""

    def __init__(self, server: Any, model: Any, data: Any, metadata: dict, snapshot: dict):
        self.server = server
        self.model = model
        self.metadata = metadata
        self.visual_handles: dict[int, Any] = {}
        self.collision_handles: dict[str, Any] = {}
        self.collision_groups: dict[str, tuple[str, ...]] = {}
        self._collision_visible = False
        self._selected_collision: str | None = None
        self._selected_collision_group: str | None = None
        self._isolate_collision = False
        self._robot_spheres_visible = True
        self._visual_visible = True
        self._collision_wireframe = False
        self._collision_snapshot: dict = {}
        self._build_visuals(data)
        self._build_collision(snapshot)

    def _build_visuals(self, data: Any) -> None:
        scene = self.server.scene
        for geom_id in range(self.model.ngeom):
            rgba = np.asarray(self.model.geom_rgba[geom_id], dtype=float)
            # 源资产把接触代理放在 group 3 且设为透明；视觉层只显示正常可见几何。
            if int(self.model.geom_group[geom_id]) == 3 or rgba[3] <= 0:
                continue
            kind = int(self.model.geom_type[geom_id])
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or f"geom_{geom_id}"
            node = f"/pick_bowl/visual/{geom_id}_{name}"
            size = np.asarray(self.model.geom_size[geom_id], dtype=float)
            body_name = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_BODY,
                int(self.model.geom_bodyid[geom_id]),
            ) or "world"
            color = _visual_color(name, body_name, rgba[:3])
            pose = _geom_pose(data, geom_id)
            wxyz, position = _pose(pose)
            common = dict(wxyz=wxyz, position=position, color=color, opacity=float(rgba[3]))
            if kind == int(mujoco.mjtGeom.mjGEOM_MESH):
                mesh = _compiled_mesh(self.model, geom_id)
                mesh.visual.face_colors = np.r_[np.asarray(color, dtype=np.uint8), np.uint8(round(rgba[3] * 255))]
                handle = scene.add_mesh_trimesh(node, mesh, wxyz=wxyz, position=position)
            elif kind == int(mujoco.mjtGeom.mjGEOM_BOX):
                handle = scene.add_box(node, dimensions=tuple(2 * size[:3]), **common)
            elif kind == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
                handle = scene.add_cylinder(node, radius=float(size[0]), height=float(2 * size[1]), **common)
            elif kind == int(mujoco.mjtGeom.mjGEOM_SPHERE):
                handle = scene.add_icosphere(node, radius=float(size[0]), **common)
            elif kind == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
                mesh = trimesh.creation.capsule(radius=float(size[0]), height=float(2 * size[1]))
                mesh.visual.face_colors = np.r_[np.asarray(color, dtype=np.uint8), np.uint8(round(rgba[3] * 255))]
                handle = scene.add_mesh_trimesh(node, mesh, wxyz=wxyz, position=position)
            elif kind == int(mujoco.mjtGeom.mjGEOM_ELLIPSOID):
                mesh = trimesh.creation.icosphere(subdivisions=3)
                mesh.apply_scale(size[:3])
                mesh.visual.face_colors = np.r_[np.asarray(color, dtype=np.uint8), np.uint8(round(rgba[3] * 255))]
                handle = scene.add_mesh_trimesh(node, mesh, wxyz=wxyz, position=position)
            else:
                # plane 等非实体几何不应被替换成臆造的障碍盒。
                continue
            self.visual_handles[geom_id] = handle

    def _build_collision(self, snapshot: dict) -> None:
        self._collision_snapshot = snapshot
        for handle in self.collision_handles.values():
            handle.remove()
        self.collision_handles.clear()
        grouped: dict[str, list[str]] = {}
        frozen_groups = set(snapshot.get("frozen_groups", []))
        if not frozen_groups and snapshot.get("active_group") in ("left", "right"):
            frozen_groups.add("left" if snapshot["active_group"] == "right" else "right")
        scene = self.server.scene
        scene_model = snapshot.get("scene_model", {})
        for record in snapshot.get("records", []):
            obstacle = record["obstacle_name"]
            node = f"/pick_bowl/collision/{obstacle}"
            pose = np.asarray(record["T_world_geom"], dtype=float)
            wxyz, position = _pose(pose)
            if record["export_kind"] == "cuboid":
                dimensions = scene_model.get("cuboid", {}).get(obstacle, {}).get("dims")
                if dimensions is None:
                    dimensions = (2 * np.asarray(record["geom_size"], dtype=float)).tolist()
                handle = scene.add_box(node, dimensions=dimensions, color=(235, 91, 63), opacity=.27,
                                       wireframe=self._collision_wireframe,
                                       wxyz=wxyz, position=position)
            elif record["export_kind"] == "mesh":
                mesh = trimesh.load_mesh(Path(record["mesh_path"]), process=False)
                if self._collision_wireframe:
                    handle = scene.add_mesh_simple(
                        node, np.asarray(mesh.vertices), np.asarray(mesh.faces),
                        color=(235, 91, 63), opacity=.35, wireframe=True,
                        wxyz=wxyz, position=position,
                    )
                else:
                    mesh.visual.face_colors = np.array([235, 91, 63, 70], dtype=np.uint8)
                    handle = scene.add_mesh_trimesh(node, mesh, wxyz=wxyz, position=position)
            else:
                raise ValueError(f"未知碰撞导出类型: {record['export_kind']}")
            self.collision_handles[obstacle] = handle
            grouped.setdefault(_collision_group(record, frozen_groups), []).append(obstacle)
        for index, sphere in enumerate(snapshot.get("active_robot_spheres", [])):
            if float(sphere["radius_m"]) <= 0:
                continue
            name = f"robot_sphere_{index}_{sphere['link']}"
            handle = scene.add_icosphere(f"/pick_bowl/collision/{name}", radius=float(sphere["radius_m"]),
                                         color=(54, 125, 230), opacity=.16,
                                         wireframe=self._collision_wireframe,
                                         position=tuple(sphere["center_world_m"]))
            self.collision_handles[name] = handle
            grouped.setdefault("活动机械臂碰撞球", []).append(name)
        self.collision_groups = {label: tuple(names) for label, names in grouped.items()}
        if self._selected_collision_group not in self.collision_groups:
            self._selected_collision_group = None
            self._isolate_collision = False
        self._update_collision_visibility()

    def _update_collision_visibility(self) -> None:
        """同时应用总开关、单项筛选和起始姿态碰撞球开关。"""
        selected_group = self.collision_groups.get(self._selected_collision_group, ())
        for name, handle in self.collision_handles.items():
            sphere = name.startswith("robot_sphere_")
            handle.visible = (self._collision_visible
                              and (not sphere or self._robot_spheres_visible)
                              and (not self._isolate_collision
                                   or name == self._selected_collision
                                   or name in selected_group))

    def select_collision(self, name: str | None, *, isolate: bool) -> None:
        """按快照中的资产名称筛选，避免下拉框仅改变文字。"""
        self._selected_collision = name
        self._selected_collision_group = None
        self._isolate_collision = bool(isolate)
        self._update_collision_visibility()

    def select_collision_group(self, group: str | None, *, isolate: bool) -> None:
        """按语义组查看碰撞代理；空组恢复全部，碰撞球总开关仍独立生效。"""
        self._selected_collision_group = group
        self._selected_collision = None
        self._isolate_collision = bool(isolate)
        self._update_collision_visibility()

    def update_visuals(self, data: Any) -> None:
        """只更新已有 visual 的位姿，供轨迹逐帧播放，避免反复加载碰撞 mesh。"""
        for geom_id, handle in self.visual_handles.items():
            handle.wxyz, handle.position = _pose(_geom_pose(data, geom_id))

    def set_visual_visible(self, visible: bool) -> None:
        """visual 与规划碰撞各自独立，方便比较资产外形和代理。"""
        self._visual_visible = bool(visible)
        for handle in self.visual_handles.values():
            handle.visible = self._visual_visible

    def refresh(self, data: Any, snapshot: dict) -> None:
        """刷新当前 MuJoCo 状态，并用本轮新快照重建规划碰撞层。"""
        self.update_visuals(data)
        self._build_collision(snapshot)

    def set_collision_visible(self, visible: bool) -> None:
        self._collision_visible = bool(visible)
        self._update_collision_visibility()

    def set_collision_wireframe(self, wireframe: bool) -> None:
        """整体切换碰撞代理外观；先删除旧节点，再仅操作新 handle。"""
        mode = bool(wireframe)
        if mode == self._collision_wireframe:
            return
        self._collision_wireframe = mode
        self._build_collision(self._collision_snapshot)

    def set_robot_spheres_visible(self, visible: bool) -> None:
        """碰撞球只对应起始关节状态，轨迹后续帧隐藏以免造成误判。"""
        self._robot_spheres_visible = bool(visible)
        self._update_collision_visibility()
