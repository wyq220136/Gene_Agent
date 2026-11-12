"""在线 Skill/Task 共用的运动学场景与 Viser 显示层。

这里只把真实规划轨迹和状态事件投影到 MuJoCo 模型做正运动学，
不执行物理仿真，也不生成新的动作。碰撞层读取每段规划实际导出的快照。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from gene_agent.visualization.pick_bowl.scene import PickBowlScene, _pose

from .model import VisualStage
from .playback import VisualSample


class KinematicSceneState:
    """把任意时间的完整状态写回 MuJoCo qpos，并执行一次正运动学。"""

    def __init__(self, model: Any, data: Any, *,
                 gripper: Mapping[str, Any] | None,
                 door_joint_name: str | None = None):
        self.model = model
        self.data = data
        self.gripper = gripper
        self._base_qpos = data.qpos.copy()
        self._free_addresses: dict[str, int] = {}
        self._door_address = (None if door_joint_name is None
                              else self._joint_address(door_joint_name))
        self._gripper_addresses: tuple[int, ...] = ()
        if gripper is not None:
            if gripper.get("mode") != "parallel_slide_proxy":
                raise ValueError(f"尚无夹爪运动学显示驱动：{gripper.get('mode')}")
            self._gripper_addresses = tuple(
                self._joint_address(name) for name in gripper["joint_names"]
            )

    def _joint_address(self, name: str) -> int:
        if mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name) < 0:
            raise ValueError(f"场景中不存在关节：{name}")
        return int(self.model.joint(name).qposadr[0])

    def _free_address(self, name: str) -> int:
        if name in self._free_addresses:
            return self._free_addresses[name]
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id < 0:
            raise ValueError(f"场景中不存在物体：{name}")
        body = self.model.body(body_id)
        if int(body.jntnum[0]) != 1:
            raise ValueError(f"物体 {name} 的视觉位姿需要 free joint")
        joint_id = int(body.jntadr[0])
        if int(self.model.jnt_type[joint_id]) != int(mujoco.mjtJoint.mjJNT_FREE):
            raise ValueError(f"物体 {name} 的视觉位姿需要 free joint")
        address = int(self.model.jnt_qposadr[joint_id])
        self._free_addresses[name] = address
        return address

    def capture_base(self) -> None:
        """场景编辑成功并重新规划后，保存新 revision 的 qpos 基线。"""
        self._base_qpos = self.data.qpos.copy()

    def apply(self, sample: VisualSample) -> None:
        """从初态重建每一帧，拖动时间轴时不会残留后续阶段的门角或物体状态。"""
        state = sample.state
        qpos = self._base_qpos.copy()
        for name, value in zip(state.joint_names, state.joint_positions_rad, strict=True):
            qpos[self._joint_address(name)] = float(value)
        if self._door_address is not None and state.door_angle_rad is not None:
            qpos[self._door_address] = state.door_angle_rad
        for name, value in state.object_joint_positions_rad.items():
            if name in state.joint_names:
                raise ValueError(f"场景关节不能覆盖机器人关节：{name}")
            address = self._joint_address(name)
            joint = self.model.joint(name)
            if int(joint.type[0]) != int(mujoco.mjtJoint.mjJNT_HINGE):
                raise ValueError(f"弧度状态只支持标量 hinge 关节：{name}")
            if address in self._gripper_addresses:
                raise ValueError(f"场景关节不能覆盖夹爪驱动：{name}")
            if address == self._door_address and state.door_angle_rad is not None:
                if not np.isclose(value, state.door_angle_rad, atol=1e-9):
                    raise ValueError("通用关节与门角状态互相冲突")
            qpos[address] = value
        if self._gripper_addresses:
            lower, upper = self.gripper["width_range_m"]
            opening = float(np.clip(state.gripper_width_m, lower, upper))
            for address in self._gripper_addresses:
                qpos[address] = opening / 2.
        for name, pose in state.object_poses_world.items():
            address = self._free_address(name)
            qpos[address:address + 3] = pose[:3, 3]
            qpos[address + 3:address + 7] = Rotation.from_matrix(
                pose[:3, :3]
            ).as_quat()[[3, 0, 1, 2]]
        self.data.qpos[:] = qpos
        mujoco.mj_forward(self.model, self.data)


class ViserVisualScene:
    """Skill 和 Task 共用同一个 visual、碰撞快照及目标显示实现。"""

    def __init__(self, server: Any, model: Any, data: Any, metadata: dict, *,
                 gripper_group: str, tcp_site_name: str,
                 door_joint_name: str | None = None):
        self.server = server
        self.model = model
        self.data = data
        self.tcp_site_name = tcp_site_name
        gripper = metadata["robot"]["grippers"].get(gripper_group)
        self.state = KinematicSceneState(model, data, gripper=gripper,
                                         door_joint_name=door_joint_name)
        self.renderer = PickBowlScene(server, model, data, metadata, {})
        self.renderer.set_collision_visible(False)
        self._shown_snapshot_ref: str | None = None
        self._path_handle = None
        self._target_handle = server.scene.add_frame(
            "/viser_sim/target_tcp", axes_length=.06, axes_radius=.002)
        self._target_handle.visible = False
        self.attached_object_names: tuple[str, ...] = ()

    def set_stage(self, stage: VisualStage) -> None:
        """按该阶段真实的 collision snapshot 显示，而不是用上一阶段的代理。"""
        if stage.collision_snapshot_ref != self._shown_snapshot_ref:
            snapshot = ({} if stage.collision_snapshot_ref is None else json.loads(
                Path(stage.collision_snapshot_ref).read_text(encoding="utf-8")
            ))
            self.renderer._build_collision(snapshot)
            self._shown_snapshot_ref = stage.collision_snapshot_ref
        self._target_handle.visible = stage.target_tcp_world is not None
        if stage.target_tcp_world is not None:
            self._target_handle.wxyz, self._target_handle.position = _pose(
                stage.target_tcp_world
            )
        if self._path_handle is not None:
            self._path_handle.remove()
            self._path_handle = None
        if stage.trajectory is not None:
            points = self._trajectory_tcp_points(stage)
            if len(points) >= 2:
                self._path_handle = self.server.scene.add_line_segments(
                    "/viser_sim/tcp_path",
                    points=np.stack((points[:-1], points[1:]), axis=1),
                    colors=(26, 131, 196), thickness=.003,
                    thickness_units="world",
                )

    def _trajectory_tcp_points(self, stage: VisualStage) -> np.ndarray:
        scratch = mujoco.MjData(self.model)
        scratch.qpos[:] = self.data.qpos
        points = []
        for row in stage.trajectory.positions_rad:
            for name, value in zip(stage.trajectory.joint_names, row, strict=True):
                scratch.qpos[self.model.joint(name).qposadr[0]] = float(value)
            mujoco.mj_forward(self.model, scratch)
            points.append(np.asarray(scratch.site(self.tcp_site_name).xpos).copy())
        return np.asarray(points)

    def show_sample(self, sample: VisualSample) -> None:
        self.state.apply(sample)
        self.renderer.update_visuals(self.data)
        # 静态起点碰撞球不代表运动中的机器人位置。
        self.renderer.set_robot_spheres_visible(sample.time_s <= 0.)
        self.attached_object_names = sample.state.attachments

    def set_visual_visible(self, visible: bool) -> None:
        self.renderer.set_visual_visible(visible)

    def set_collision_visible(self, visible: bool) -> None:
        self.renderer.set_collision_visible(visible)


__all__ = ["KinematicSceneState", "ViserVisualScene"]
