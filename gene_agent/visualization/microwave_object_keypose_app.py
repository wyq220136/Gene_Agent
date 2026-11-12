"""在当前场景中叠加旧放入/取出 keypose，供资产 frame 迁移前目视审核。

该界面只展示几何目标，不调用 IK 或轨迹规划。颜色区分实际场景、旧资产
轮廓、炉内碗预览和目标夹爪，防止把尚未标定的叠图误认成成功 Skill。
"""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import trimesh

from gene_agent.scene.microwave_frame import LegacyFrameAlignment
from gene_agent.tasks.policies.microwave_task.operations.bowl import (
    fixed_extract_targets,
    fixed_place_targets,
)
from gene_agent.visualization.grasp_tuner.gripper_preview import GripperMeshPreview
from gene_agent.visualization.pick_bowl.scene import PickBowlScene, _pose


PLACE_LABELS = (
    "放入 / pre-place", "放入 / place", "放入 / post-place 1", "放入 / post-place 2",
)
EXTRACT_LABELS = (
    "取出 / pre-extract 1", "取出 / pre-extract 2", "取出 / grasp",
    "取出 / post-extract 1", "取出 / post-extract 2",
)


def projected_keyposes(
    T_world_calibrated: np.ndarray,
    alignment: LegacyFrameAlignment,
    profile: dict,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """把旧 profile 仅投影到当前 visual frame，保留旧 keypose 顺序。"""
    T_world_legacy = T_world_calibrated @ alignment.T_calibrated_legacy
    # 这里只展示 TCP 目标。传入单位 attachment 不影响四个 TCP 点；真正
    # 放入 Skill 必须用本次选中抓取的 T_tcp_object 计算物体终点。
    place = fixed_place_targets(T_world_legacy, profile, np.eye(4))
    extract = fixed_extract_targets(T_world_legacy, profile)
    targets = dict(zip(PLACE_LABELS, (
        place.pre_place_tcp, place.place_tcp,
        place.post_place_1_tcp, place.post_place_2_tcp,
    ), strict=True))
    targets.update(zip(EXTRACT_LABELS, extract, strict=True))
    placed_object = T_world_legacy @ np.asarray(profile["microwave_from_placed_object"], dtype=float)
    return targets, placed_object


class MicrowaveObjectKeyposeApp:
    """复用现有 MuJoCo GT visual 和真实夹爪 mesh 的 Viser 叠图。"""

    def __init__(self, server, case, *, task_dir: Path, profile: dict,
                 alignment: LegacyFrameAlignment,
                 initial_target: str = PLACE_LABELS[0],
                 initial_view: str = "炉门正面"):
        if initial_target not in PLACE_LABELS + EXTRACT_LABELS:
            raise ValueError(f"未知 keypose：{initial_target}")
        if initial_view not in ("全景", "炉门正面", "夹爪侧面"):
            raise ValueError(f"未知视角：{initial_view}")
        self.server = server
        self.case = case
        self.task_dir = task_dir
        self.profile = profile
        self.alignment = alignment
        self.data = mujoco.MjData(case.model)
        mujoco.mj_copyData(self.data, case.model, case.data)
        server.scene.set_up_direction("+z")
        server.scene.set_background_image(np.full((8, 8, 3), 238, dtype=np.uint8), format="png")
        server.gui.configure_theme(
            dark_mode=False, control_width="large", show_logo=False,
            show_share_button=False, brand_color=(53, 112, 169),
        )
        self.scene = PickBowlScene(server, case.model, self.data, case.metadata, {})
        self.preview_case = SimpleNamespace(
            model=case.model, data=self.data, metadata=case.metadata,
            snapshot={"active_group": "right"},
        )
        self.ghost_gripper = GripperMeshPreview(
            server, self.preview_case, prefix="/object_keypose/target_gripper",
            color_rgba=(245, 190, 55, 145),
        )
        if not self.ghost_gripper.available:
            raise ValueError("当前场景没有可展示的右侧 gripper visual")
        self.ghost_gripper.set_palm_visible(True)
        self.ghost_gripper.set_opening_width(float(profile["grasp_pregrasp_opening_width_m"]))
        self.microwave_pose = self._body_pose("microwave")
        self.targets, self.placed_object_pose = projected_keyposes(
            self.microwave_pose, alignment, profile,
        )
        self.keypose_frames = {}
        self.keypose_labels = {}
        for index, (label, pose) in enumerate(self.targets.items()):
            wxyz, position = _pose(pose)
            node = f"/object_keypose/target_{index:02d}"
            self.keypose_frames[label] = server.scene.add_frame(
                node, axes_length=.038, axes_radius=.0012,
                wxyz=wxyz, position=position,
            )
            self.keypose_labels[label] = server.scene.add_label(
                node + "_label", text=label, position=position,
                anchor="bottom-center",
            )
        self.placed_bowl = self._add_placed_bowl_preview()
        bowl_label_position = self.placed_object_pose[:3, 3] + np.array([0., 0., .12])
        self.placed_bowl_label = server.scene.add_label(
            "/object_keypose/placed_bowl_label", text="旧记录的炉内碗目标",
            position=tuple(bowl_label_position), anchor="bottom-center",
        )
        self.old_asset = self._add_old_asset_preview()
        self.old_asset.visible = False
        gui = server.gui
        gui.add_markdown(
            "### 微波炉 Bowl 旧 keypose ↔ 当前 URDF 叠图\n"
            "这是**几何诊断**：黄色夹爪是旧目标的投影，不表示 IK、路径或接触成功。"
            "青色半透明碗是旧记录的炉内目标；当前桌上的碗是场景初态。"
        )
        self.report = gui.add_markdown(
            f"**门/把手叠图残差：** 把手中心 {alignment.handle_center_residual_m*1000:.1f} mm；"
            f"门尺寸最大差 {alignment.door_size_max_residual_m*1000:.1f} mm。"
            "该变换尚未批准用于规划。"
        )
        self.selection = gui.add_dropdown(
            "关键 TCP", options=tuple(self.targets), initial_value=initial_target,
        )
        self.mode = gui.add_dropdown(
            "显示范围", options=("当前目标", "全部目标"), initial_value="当前目标",
        )
        self.door_angle = gui.add_slider(
            "门角（仅视觉检查，°）", min=0, max=100, step=1, initial_value=85,
        )
        self.camera = gui.add_dropdown(
            "视角", options=("全景", "炉门正面", "夹爪侧面"), initial_value=initial_view,
        )
        self.show_old_asset = gui.add_checkbox("叠加旧资产（门关闭时有意义）", initial_value=False)
        self.show_bowl = gui.add_checkbox("显示炉内碗预览", initial_value=True)
        self.target_info = gui.add_markdown("")
        self.selection.on_update(lambda _: self._refresh_target())
        self.mode.on_update(lambda _: self._refresh_target())
        self.door_angle.on_update(lambda _: self._refresh_door())
        self.camera.on_update(lambda _: self._set_camera())
        self.show_old_asset.on_update(lambda _: self._toggle_old_asset())
        self.show_bowl.on_update(lambda _: self._toggle_bowl())
        self._refresh_door()
        self._refresh_target()

    def _body_pose(self, name: str) -> np.ndarray:
        body = self.data.body(name)
        pose = np.eye(4)
        pose[:3, :3] = body.xmat.reshape(3, 3)
        pose[:3, 3] = body.xpos
        return pose

    def _add_placed_bowl_preview(self):
        mesh = trimesh.load_mesh(
            self.task_dir / "objs/Bowl/Bowl_0/meshes/obj_mesh.obj", process=False,
        )
        mesh.visual.face_colors = np.array([156, 78, 203, 125], dtype=np.uint8)
        wxyz, position = _pose(self.placed_object_pose)
        return self.server.scene.add_mesh_trimesh(
            "/object_keypose/placed_bowl_preview", mesh, wxyz=wxyz, position=position,
        )

    def _add_old_asset_preview(self):
        mesh = trimesh.load_mesh(
            self.task_dir / "objs/Microwave/Microwave_0/meshes/obj_mesh.obj",
            process=False,
        )
        mesh.visual.face_colors = np.array([98, 204, 212, 45], dtype=np.uint8)
        T_world_old = self.microwave_pose @ self.alignment.T_calibrated_legacy
        wxyz, position = _pose(T_world_old)
        return self.server.scene.add_mesh_trimesh(
            "/object_keypose/old_microwave_preview", mesh,
            wxyz=wxyz, position=position,
        )

    def _refresh_door(self):
        joint = self.case.model.joint("microwave_door_joint")
        self.data.qpos[joint.qposadr[0]] = math.radians(float(self.door_angle.value))
        mujoco.mj_forward(self.case.model, self.data)
        self.scene.update_visuals(self.data)

    def _toggle_old_asset(self):
        self.old_asset.visible = bool(self.show_old_asset.value)

    def _toggle_bowl(self):
        self.placed_bowl.visible = bool(self.show_bowl.value)
        self.placed_bowl_label.visible = bool(self.show_bowl.value)

    def _refresh_target(self):
        selected = self.selection.value
        target = self.targets[selected]
        self.ghost_gripper.select(target)
        all_visible = self.mode.value == "全部目标"
        for label, handle in self.keypose_frames.items():
            handle.visible = all_visible or label == selected
            self.keypose_labels[label].visible = all_visible or label == selected
        position = target[:3, 3]
        self.target_info.content = (
            f"**{selected}** · TCP 世界位置 "
            f"`({position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}) m`。"
            "路径类型、碰撞白名单与真实轨迹须在后续 Skill 规划中检查。"
        )
        self._set_camera()

    def _set_camera(self):
        target = self.targets[self.selection.value]
        look = target[:3, 3]
        if self.camera.value == "全景":
            robot_base = self.data.body("right_base_link").xpos
            look = .55 * look + .45 * robot_base
            position = look + np.array([-.82, .66, .60])
        elif self.camera.value == "炉门正面":
            position = look + np.array([-.48, .18, .20])
        else:
            # 侧视仍必须站在炉门开口外；若只沿夹爪局部 Y 取机位，取出
            # 目标会把镜头放进柜体。门朝铰链侧打开，因此选夹爪两侧中
            # 更靠近把手/旋钮的一侧，避免敞开的门板遮住指尖。
            front_world = -self.microwave_pose[:3, 1]
            gripper_side = target[:3, 1]
            if np.dot(gripper_side, self.microwave_pose[:3, 0]) < 0:
                gripper_side = -gripper_side
            position = (look + .33 * front_world + .24 * gripper_side
                        + np.array([0., 0., .13]))
        self.server.initial_camera.position = position.tolist()
        self.server.initial_camera.look_at = look.tolist()
        self.server.initial_camera.up = [0., 0., 1.]
        for client in self.server.get_clients().values():
            client.camera.position = position.tolist()
            client.camera.look_at = look.tolist()
            client.camera.up_direction = [0., 0., 1.]


__all__ = ["MicrowaveObjectKeyposeApp", "projected_keyposes"]
