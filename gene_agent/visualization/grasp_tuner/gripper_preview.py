"""从当前 MuJoCo 场景复制夹爪 visual mesh，显示在 Knowledge 候选 TCP 上。"""
from __future__ import annotations

from typing import Any

import mujoco
import numpy as np

from gene_agent.visualization.pick_bowl.scene import _compiled_mesh, _geom_pose, _pose


class GripperMeshPreview:
    """只复制 visual 外观；初始开口取快照，选中候选后随 closing width 调整。"""

    def __init__(self, server: Any, case: Any, *, prefix: str = "/scene_knowledge/real_gripper",
                 color_rgba: tuple[int, int, int, int] = (245, 190, 55, 255)) -> None:
        self.server = server
        self.handles: list[Any] = []
        self.palm_handles: list[Any] = []
        self.palm_specs: list[tuple[str, Any, Any, Any]] = []
        self.finger_geoms: list[tuple[int, Any]] = []
        self.palm_geom_ids: list[int] = []
        self.root = None
        self.opening_width_m: float | None = None
        model = case.model
        self.model = model
        robot = case.metadata.get("robot", {})
        group = case.snapshot.get("active_group")
        site_name = robot.get("tcp_sites", {}).get(group)
        if not group or not site_name or mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name) < 0:
            return

        data = mujoco.MjData(model)
        mujoco.mj_copyData(data, model, case.data)
        self.data = data
        gripper = robot.get("grippers", {}).get(group, {})
        self.gripper = gripper
        self.site_name = site_name
        width = float((case.snapshot.get("gripper_snapshot") or {}).get("width_m", gripper.get("home_width_m", .08)))
        if gripper.get("mode") == "parallel_slide_proxy":
            low, high = gripper["width_range_m"]
            width = float(np.clip(width, low, high))
            for joint_name in gripper["joint_names"]:
                data.qpos[model.joint(joint_name).qposadr[0]] = width / 2
        mujoco.mj_forward(model, data)
        site = data.site(site_name)
        world_tcp = np.eye(4)
        world_tcp[:3, :3] = site.xmat.reshape(3, 3)
        world_tcp[:3, 3] = site.xpos
        tcp_world = np.linalg.inv(world_tcp)
        # Viser 的 frame 即使 axes_length=0 仍会画默认黄色原点球；
        # 它会遮挡整个碗。这里仅借用 frame 的父子变换，不绘制标记。
        self.root = server.scene.add_frame(prefix, show_axes=False, origin_radius=0.0)
        self.handles.append(self.root)
        for geom_id in range(model.ngeom):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
            if (not name.startswith(f"{group}_") or not any(word in name for word in ("gripper", "finger", "jaw"))
                    or int(model.geom_group[geom_id]) == 3 or model.geom_rgba[geom_id, 3] <= 0
                    or int(model.geom_type[geom_id]) != int(mujoco.mjtGeom.mjGEOM_MESH)):
                continue
            mesh = _compiled_mesh(model, geom_id)
            # 用户可单独隐藏掌部外壳以检查指尖与目标之间的距离。
            alpha = min(75, color_rgba[3]) if "jaw_base" in name else color_rgba[3]
            mesh.visual.face_colors = np.array([*color_rgba[:3], alpha], dtype=np.uint8)
            wxyz, position = _pose(tcp_world @ _geom_pose(data, geom_id))
            node = f"{prefix}/{geom_id}_{name}"
            if "jaw_base" in name:
                # 掌部由开关按需创建；关闭时删除节点，避免半透明状态残留。
                self.palm_specs.append((node, mesh, wxyz, position))
                self.palm_geom_ids.append(geom_id)
                continue
            handle = server.scene.add_mesh_trimesh(node, mesh, wxyz=wxyz, position=position)
            self.handles.append(handle)
            self.finger_geoms.append((geom_id, handle))
        if len(self.handles) == 1:
            self.root.remove()
            self.root = None
            self.handles.clear()
        self.opening_width_m = width

    @property
    def available(self) -> bool:
        return self.root is not None

    def select(self, pose_world: np.ndarray, *, opening_width_m: float | None = None) -> None:
        if opening_width_m is not None:
            self.set_opening_width(opening_width_m)
        if self.root is not None:
            self.root.wxyz, self.root.position = _pose(pose_world)

    def set_opening_width(self, requested_width_m: float) -> None:
        """按候选 closing width 更新真实夹爪指尖位姿，而不重建 mesh。"""
        if self.root is None or self.gripper.get("mode") != "parallel_slide_proxy":
            return
        low, high = self.gripper["width_range_m"]
        width = float(np.clip(requested_width_m, low, high))
        if self.opening_width_m is not None and abs(width - self.opening_width_m) < 1e-7:
            return
        for joint_name in self.gripper["joint_names"]:
            self.data.qpos[self.model.joint(joint_name).qposadr[0]] = width / 2
        mujoco.mj_forward(self.model, self.data)
        site = self.data.site(self.site_name)
        world_tcp = np.eye(4)
        world_tcp[:3, :3] = site.xmat.reshape(3, 3)
        world_tcp[:3, 3] = site.xpos
        tcp_world = np.linalg.inv(world_tcp)
        for geom_id, handle in self.finger_geoms:
            handle.wxyz, handle.position = _pose(tcp_world @ _geom_pose(self.data, geom_id))
        updated = []
        for index, (name, mesh, _, _) in enumerate(self.palm_specs):
            wxyz, position = _pose(tcp_world @ _geom_pose(self.data, self.palm_geom_ids[index]))
            updated.append((name, mesh, wxyz, position))
            if index < len(self.palm_handles):
                self.palm_handles[index].wxyz = wxyz
                self.palm_handles[index].position = position
        self.palm_specs = updated
        self.opening_width_m = width

    def set_visible(self, visible: bool) -> None:
        if self.root is not None:
            self.root.visible = visible

    def on_click(self, callback) -> None:
        """允许用户点击代表夹爪，直接切换当前候选。"""
        for _, handle in self.finger_geoms:
            handle.on_click(callback)

    def set_palm_visible(self, visible: bool) -> None:
        if visible and not self.palm_handles:
            for node, mesh, wxyz, position in self.palm_specs:
                self.palm_handles.append(self.server.scene.add_mesh_trimesh(node, mesh, wxyz=wxyz, position=position))
        elif not visible:
            for handle in self.palm_handles:
                handle.remove()
            self.palm_handles.clear()


__all__ = ["GripperMeshPreview"]
