"""Skill 检查界面的只读场景层。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from gene_agent.feedback.skill_run import StageView
from gene_agent.visualization.pick_bowl.scene import PickBowlScene, _geom_pose, _pose
from gene_agent.visualization.replay import ReplayFrame


def read_json(root: Path, ref: str | None) -> dict[str, Any]:
    if ref is None:
        return {}
    path = (root / ref).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"产物引用越过运行目录：{ref}")
    return json.loads(path.read_text(encoding="utf-8"))


def stage_facts(root: Path, stage: StageView) -> dict[str, Any]:
    """从原始记录提取阶段事实，缺失值保持为空，供界面和自动检查共用。"""
    plan = read_json(root, stage.result_ref)
    if stage.result_ref in {"skill_pick_bowl_result.json", "skill_insert_bowl_result.json", "skill_rotate_knob_0_result.json",
                            "skill_open_door_result.json", "skill_close_door_result.json"}:
        plan = plan["stages"][int(stage.stage_id.split("_", 1)[0])]
    execution = read_json(root, stage.execution_ref)
    if stage.execution_ref == "result.json":
        execution = execution.get("stages", {}).get(stage.name, {})
    snapshot = read_json(root, stage.collision_snapshot_ref)
    result = plan.get("result") or plan
    request = plan.get("request") or {}
    excluded = snapshot.get("excluded", [])
    facts = {
        "stage_id": stage.stage_id,
        "path_kind": stage.path_kind or stage.operation,
        "planning_status": stage.planning_status,
        "execution_status": stage.execution_status,
        "target_pose": (plan.get("target") or request.get("target") or
                        request.get("goal") or {}).get("matrix"),
        "planning_duration_s": result.get("duration_s"),
        "execution_duration_s": execution.get("sim_duration_s"),
        "failure": (result.get("failure_category") or result.get("raw_error") or
                    result.get("status") or stage.planning_status
                    if stage.planning_status in {"failed", "unsupported", "not_executed"} else
                    execution.get("failure_type") or execution.get("status")
                    if stage.execution_status == "failed" else None),
        "collision_sha256": snapshot.get("snapshot_sha256"),
        "collision_objects": [item.get("obstacle_name") for item in snapshot.get("records", [])],
        "excluded": [{"geom": item.get("geom_name"), "reason": item.get("reason")}
                     for item in excluded],
        "limitations": snapshot.get("limitations", []),
        "mounting_contact_policy": snapshot.get("mounting_contact_policy"),
        "attachment": read_json(root, stage.attachment_ref),
        "sample_format": stage.actual_sample_format,
        "articulation_start_angle_rad": plan.get("start_angle_rad"),
        "articulation_end_angle_rad": plan.get("end_angle_rad"),
        "axis_path_validation": plan.get("axis_path_validation"),
        "moving_body_names": plan.get("moving_body_names") or snapshot.get("moving_body_names") or [],
        "dynamic_door_contact_check": plan.get("dynamic_door_contact_check"),
        "kinematic_preview": stage.result_ref in {"skill_rotate_knob_0_result.json",
                                                  "skill_open_door_result.json",
                                                  "skill_close_door_result.json"},
    }
    if plan.get("dynamic_door_contact_failure") or plan.get("axis_path_failure"):
        facts["failure"] = plan.get("dynamic_door_contact_failure") or plan.get("axis_path_failure")
    return facts


class SkillInspectorScene:
    """复用现有 MuJoCo visual 和碰撞代理渲染，按记录中的状态更新位姿。"""

    def __init__(self, server: Any, root: Path, model: Any, data: Any, *,
                 planned_base_qpos: np.ndarray | None = None):
        self.server, self.root, self.model, self.data = server, root, model, data
        self._planned_base_qpos = (data.qpos.copy() if planned_base_qpos is None
                                   else np.asarray(planned_base_qpos, dtype=float).copy())
        bowl_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "bowl")
        self._bowl_free_qpos_adr = None
        if bowl_id >= 0:
            body = model.body(bowl_id)
            if int(body.jntnum[0]) == 1:
                joint_id = int(body.jntadr[0])
                if int(model.jnt_type[joint_id]) == int(mujoco.mjtJoint.mjJNT_FREE):
                    self._bowl_free_qpos_adr = int(model.jnt_qposadr[joint_id])
        self.renderer = PickBowlScene(server, model, data, {}, {})
        self.renderer.set_collision_visible(False)
        self.target_handle = server.scene.add_frame(
            "/skill_inspector/target", axes_length=.07, axes_radius=.002)
        self.target_handle.visible = False
        self.path_handle = None
        self._snapshot_ref = None
        self._collision_snapshot: dict[str, Any] = {}
        self._collision_records: dict[str, dict[str, Any]] = {}
        self._collision_base_poses: dict[str, np.ndarray] = {}
        self.current_target_pose = None
        self._preview_poses: dict[str, np.ndarray] = {}

    def object_pose(self, name: str) -> np.ndarray:
        """读取当前回放帧中的 body 世界位姿，不读取或修改记录文件。"""
        if name == "table":
            geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "scene_table")
            if geom_id >= 0:
                return _geom_pose(self.data, geom_id)
        if mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name) < 0:
            raise KeyError(f"场景中没有物体 body：{name}")
        body = self.data.body(name)
        pose = np.eye(4)
        pose[:3, :3] = np.asarray(body.xmat).reshape(3, 3)
        pose[:3, 3] = body.xpos
        return pose

    def _visual_belongs_to(self, geom_id: int, name: str) -> bool:
        """按 MuJoCo body 树归属 visual；微波炉支架是本资产的独立同组 body。"""
        if name == "table":
            return mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) == "scene_table"
        body_id = int(self.model.geom_bodyid[geom_id])
        while body_id > 0:
            body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            if body_name == name or (name == "microwave" and body_name == "microwave_support"):
                return True
            body_id = int(self.model.body_parentid[body_id])
        return False

    def _apply_preview_poses(self, *, restore_collision: bool = False) -> None:
        for name, target in self._preview_poses.items():
            delta = target @ np.linalg.inv(self.object_pose(name))
            for geom_id, handle in self.renderer.visual_handles.items():
                if self._visual_belongs_to(geom_id, name):
                    handle.wxyz, handle.position = _pose(delta @ _geom_pose(self.data, geom_id))
        # 碰撞快照本身保持冻结；这里只改变与 visual 同屏显示的代理节点。
        # 用户提交编辑后会重新导出真正供 cuRobo 使用的新快照。
        if not self._preview_poses and not restore_collision:
            return
        for obstacle, record in self._collision_records.items():
            handle = self.renderer.collision_handles.get(obstacle)
            if handle is None:
                continue
            owner = self._collision_owner(record)
            if owner not in self._preview_poses and not restore_collision:
                continue
            original = np.asarray(record["T_world_geom"], dtype=float)
            if owner in self._preview_poses and owner in self._collision_base_poses:
                if (not restore_collision and np.allclose(
                    self._preview_poses[owner], self._collision_base_poses[owner], atol=1e-8
                )):
                    continue
                delta = self._preview_poses[owner] @ np.linalg.inv(self._collision_base_poses[owner])
                original = delta @ original
            handle.wxyz, handle.position = _pose(original)

    @staticmethod
    def _collision_owner(record: dict[str, Any]) -> str | None:
        body = str(record.get("body_name") or "").lower()
        geom = str(record.get("geom_name") or "").lower()
        if "bowl" in body or "bowl" in geom:
            return "bowl"
        if "table" in body or "table" in geom:
            return "table"
        if "microwave" in body or "microwave" in geom or body.startswith(("door", "knob")):
            return "microwave"
        return None

    def set_collision_snapshot(self, snapshot: dict[str, Any], *,
                               base_poses: dict[str, np.ndarray] | None = None) -> None:
        """切换可见碰撞代理及其位姿基线；规划快照数据永不原地修改。"""
        self._collision_snapshot = snapshot
        self.renderer._build_collision(snapshot)
        self._collision_records = {row["obstacle_name"]: row for row in snapshot.get("records", [])}
        self._collision_base_poses = (
            {name: np.asarray(pose, dtype=float).copy() for name, pose in base_poses.items()}
            if base_poses is not None else
            {name: self.object_pose(name) for name in ("bowl", "microwave", "table")}
        )
        self._apply_preview_poses()

    def preview_object_pose(self, name: str, pose_world: np.ndarray) -> None:
        """只移动 visual 预览，不伪称更新了实际 qpos、规划轨迹或碰撞快照。"""
        pose = np.asarray(pose_world, dtype=float)
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            raise ValueError("预览物体 pose 必须是有限的 4×4 矩阵")
        self.object_pose(name)
        self._preview_poses[name] = pose.copy()
        self.renderer.update_visuals(self.data)
        self._apply_preview_poses()

    def clear_object_preview(self, name: str | None = None) -> None:
        if name is None:
            self._preview_poses.clear()
        else:
            self._preview_poses.pop(name, None)
        self.renderer.update_visuals(self.data)
        self._apply_preview_poses(restore_collision=True)

    def set_stage(self, stage: StageView, *, show_collision: bool, tcp_site: str) -> None:
        if stage.collision_snapshot_ref != self._snapshot_ref:
            snapshot = read_json(self.root, stage.collision_snapshot_ref)
            self.set_collision_snapshot(snapshot)
            self._snapshot_ref = stage.collision_snapshot_ref
        self.renderer.set_collision_visible(show_collision)
        self.renderer.set_robot_spheres_visible(True)
        facts = stage_facts(self.root, stage)
        pose = facts["target_pose"]
        # 旧执行报告只保存关节轨迹；其终点 TCP 由同一 MuJoCo 模型正运动学求出。
        if pose is None and stage.result_ref and stage.result_ref not in {
                "skill_pick_bowl_result.json", "skill_insert_bowl_result.json", "skill_rotate_knob_0_result.json",
                "skill_open_door_result.json", "skill_close_door_result.json"}:
            plan = read_json(self.root, stage.result_ref)
            rows = plan.get("positions_rad") or []
            names = plan.get("joint_names") or []
            if rows and names:
                saved = self.data.qpos.copy()
                for name, value in zip(names, rows[-1]):
                    self.data.qpos[self.model.joint(name).qposadr[0]] = value
                mujoco.mj_forward(self.model, self.data)
                site = self.data.site(tcp_site)
                pose = np.eye(4)
                pose[:3, :3] = site.xmat.reshape(3, 3)
                pose[:3, 3] = site.xpos
                self.data.qpos[:] = saved
                mujoco.mj_forward(self.model, self.data)
        self.current_target_pose = np.asarray(pose, dtype=float) if pose is not None else None
        self.target_handle.visible = pose is not None
        if pose is not None:
            self.target_handle.wxyz, self.target_handle.position = _pose(np.asarray(pose, dtype=float))

    def set_path(self, frames: list[ReplayFrame], tcp_site: str) -> None:
        """利用保存的关节样本计算 TCP 轨迹；线表示实际或规划样本而非插值。"""
        if not frames:
            self.set_path_points([])
            return
        points = []
        saved = self.data.qpos.copy()
        for frame in frames:
            if frame.qpos is not None:
                self.data.qpos[:] = frame.qpos
            elif frame.joint_positions_rad is not None:
                for name, value in zip(frame.joint_names, frame.joint_positions_rad):
                    self.data.qpos[self.model.joint(name).qposadr[0]] = value
            else:
                continue
            mujoco.mj_forward(self.model, self.data)
            points.append(np.asarray(self.data.site(tcp_site).xpos).copy())
        self.data.qpos[:] = saved
        mujoco.mj_forward(self.model, self.data)
        self.set_path_points(points)

    def set_path_points(self, points: Any) -> None:
        """显示明确来源的 TCP 路径；旧回放与新规划均使用同一条线。"""
        if self.path_handle is not None:
            self.path_handle.remove()
            self.path_handle = None
        if len(points) >= 2:
            arr = np.asarray(points, dtype=float)
            self.path_handle = self.server.scene.add_line_segments(
                "/skill_inspector/tcp_path", points=np.stack((arr[:-1], arr[1:]), axis=1),
                colors=(26, 131, 196), thickness=.003, thickness_units="world")

    def show_frame(self, frame: ReplayFrame, *, base_state_ref: str | None = None,
                   gripper_group: str = "right", tcp_site: str | None = None) -> None:
        if frame.qpos is not None:
            if len(frame.qpos) != self.model.nq:
                raise ValueError("回放 qpos 长度与 MuJoCo 模型不一致")
            self.data.qpos[:] = frame.qpos
            if frame.sim_time_s is not None:
                self.data.time = float(frame.sim_time_s)
        elif frame.mode == "planned":
            # 规划帧只有活动臂轨迹，每次 seek 先恢复同一场景的门角和机器人 home。
            self.data.qpos[:] = self._planned_base_qpos
        elif frame.state_ref or base_state_ref:
            # 部分实测帧没有物体 qpos；先恢复本阶段真实起点，随后只叠加已采到的关节。
            state_path = self.root / Path(frame.state_ref or base_state_ref).parent / "state.npz"
            if state_path.is_file():
                with np.load(state_path) as state:
                    mujoco.mj_setState(self.model, self.data, state["state"], int(state["spec"]))
        if frame.qpos is None and frame.joint_positions_rad is not None:
            for name, value in zip(frame.joint_names, frame.joint_positions_rad):
                self.data.qpos[self.model.joint(name).qposadr[0]] = value
        if frame.qpos is None and frame.gripper_width_m is not None:
            for side in ("positive", "negative"):
                name = f"{gripper_group}_gripper_{side}_slide"
                if mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name) >= 0:
                    self.data.qpos[self.model.joint(name).qposadr[0]] = frame.gripper_width_m / 2
        mujoco.mj_forward(self.model, self.data)
        attachment = frame.attachment or {}
        if (frame.mode == "planned" and attachment.get("object_instance_id") == "bowl"
                and attachment.get("T_tcp_object") is not None):
            if tcp_site is None or self._bowl_free_qpos_adr is None:
                raise ValueError("附着碗的视觉帧需要 TCP site 和 Bowl free joint")
            site = self.data.site(tcp_site)
            world_tcp = np.eye(4)
            world_tcp[:3, :3] = site.xmat.reshape(3, 3)
            world_tcp[:3, 3] = site.xpos
            world_bowl = world_tcp @ np.asarray(attachment["T_tcp_object"], dtype=float)
            adr = self._bowl_free_qpos_adr
            self.data.qpos[adr:adr + 3] = world_bowl[:3, 3]
            self.data.qpos[adr + 3:adr + 7] = Rotation.from_matrix(
                world_bowl[:3, :3],
            ).as_quat()[[3, 0, 1, 2]]
            mujoco.mj_forward(self.model, self.data)
        elif frame.mode == "planned" and frame.qpos is None and frame.object_poses_world:
            for instance, raw_pose in frame.object_poses_world.items():
                pose = np.asarray(raw_pose, dtype=float)
                if pose.shape != (4, 4) or not np.isfinite(pose).all():
                    raise ValueError(f"{instance} 的规划物体位姿无效")
                joint_id = int(self.model.body(instance).jntadr[0])
                if int(self.model.jnt_type[joint_id]) != int(mujoco.mjtJoint.mjJNT_FREE):
                    raise ValueError(f"{instance} 的视觉位姿要求 free joint")
                adr = int(self.model.jnt_qposadr[joint_id])
                self.data.qpos[adr:adr + 3] = pose[:3, 3]
                self.data.qpos[adr + 3:adr + 7] = Rotation.from_matrix(
                    pose[:3, :3]).as_quat()[[3, 0, 1, 2]]
            mujoco.mj_forward(self.model, self.data)
        self.renderer.update_visuals(self.data)
        if self._collision_records:
            self._collision_base_poses = {
                name: self.object_pose(name) for name in ("bowl", "microwave", "table")
            }
        self._apply_preview_poses()
        self.renderer.set_robot_spheres_visible(frame.local_time_s <= 0)
