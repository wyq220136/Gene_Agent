"""只读现场初态上的运动学规划影子；本模块没有设备或执行器接口。"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from gene_agent.motion.curobo.scene import export_collision_scene
from .prediction import SOURCE, report_basis, assumption_check


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")


class PlanningFailed(RuntimeError):
    def __init__(self, code, message, details=None):
        super().__init__(message)
        self.failure = {"code": code, "message": message, "details": details}


class PlanOnlyWorld:
    def __init__(self, initial_scene, output_dir, *, max_start_error_rad):
        self.output = Path(output_dir).resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.initial = deepcopy(initial_scene)
        source = initial_scene["provenance"]
        if (source.get("physics_stepped") is not False
                or source.get("pose_frame_verified") is not True
                or source.get("collision_snapshot_sha256") != initial_scene["snapshot"].get("snapshot_sha256")
                or source.get("state_source") == SOURCE):
            raise ValueError("完整预览必须从已核对的现场初态开始，不能把预测场景当新实测")
        self.model = mujoco.MjModel.from_xml_path(str(initial_scene["scene_xml_path"]))
        self.data = mujoco.MjData(self.model)
        self.data.qpos[:] = np.asarray(initial_scene["snapshot"]["qpos_snapshot"])
        mujoco.mj_forward(self.model, self.data)
        self.robot = initial_scene["robot_metadata"]
        self.group = initial_scene["snapshot"]["active_group"]
        if self.group != "right":
            raise ValueError("当前完整微波炉预览与真机任务均只支持右臂")
        self.names = tuple(self.robot["joint_groups"][self.group])
        self.max_start_error_rad = float(max_start_error_rad)
        self.revision = 0
        self.attachment = None
        self.stages = []
        self.assumptions = []
        self._scene = None
        self._collision_refs = {}
        self._indexed_snapshots = set()
        write_json(self.output / "initial_scene.json", initial_scene)

    def pose(self, name, *, site=False):
        item = self.data.site(name) if site else self.data.body(name)
        pose = np.eye(4)
        pose[:3, :3] = item.xmat.reshape(3, 3)
        pose[:3, 3] = item.xpos
        return pose

    def tcp(self):
        return self.pose(self.robot["tcp_sites"][self.group], site=True)

    def set_object(self, name, pose):
        body = self.model.body(name)
        jid = int(body.jntadr[0])
        if int(self.model.jnt_type[jid]) != int(mujoco.mjtJoint.mjJNT_FREE):
            raise ValueError(f"{name} 必须由 free joint 驱动")
        adr = int(self.model.jnt_qposadr[jid])
        self.data.qpos[adr:adr+3] = pose[:3, 3]
        self.data.qpos[adr+3:adr+7] = Rotation.from_matrix(pose[:3, :3]).as_quat()[[3, 0, 1, 2]]

    def forward(self):
        mujoco.mj_forward(self.model, self.data)
        if self.attachment is not None:
            self.set_object("bowl", self.tcp() @ self.attachment)
            mujoco.mj_forward(self.model, self.data)

    def joint(self, name):
        return float(self.data.qpos[self.model.joint(name).qposadr[0]])

    def set_joint(self, name, value):
        j = self.model.joint(name)
        value = float(value)
        if not np.isfinite(value):
            raise ValueError("预测关节非有限")
        if self.model.jnt_limited[j.id]:
            low, high = self.model.jnt_range[j.id]
            if not low-1e-6 <= value <= high+1e-6:
                raise ValueError(f"预测关节 {name}={value} 越限 [{low}, {high}]")
        self.data.qpos[j.qposadr[0]] = value

    def width(self):
        return sum(self.joint(n) for n in self.robot["grippers"][self.group]["joint_names"])

    def state(self):
        return {"qpos": self.data.qpos.tolist(), "joint_names": list(self.names),
                "joint_positions_rad": [self.joint(n) for n in self.names],
                "gripper_width_m": self.width(),
                "door_angle_rad": self.joint("microwave_door_joint"),
                "knob_angle_rad": self.joint("knob_0_joint"),
                "object_poses_world": {n: self.pose(n).tolist() for n in ("bowl", "microwave")},
                "attachment": None if self.attachment is None else {
                    "object_instance_id": "bowl", "T_tcp_object": self.attachment.tolist()},
                "predicted": True, "prediction_step": self.revision}

    def scene(self):
        if self._scene is not None:
            return self._scene
        self.forward()
        snapshot = export_collision_scene(
            self.model, self.data, self.robot, self.group,
            self.output / "prediction_scenes" / f"{self.revision:04d}")
        source = deepcopy(self.initial["provenance"])
        source.update(state_source=SOURCE, prediction_step=self.revision,
            all_objects_same_frame=False,
            pose_source_modes={key:"plan_only_prediction" for key in source["aligned_object_poses_world"]},
            prediction_origin_sha256=self.initial["snapshot"]["snapshot_sha256"],
            collision_snapshot_sha256=snapshot["snapshot_sha256"],
            aligned_object_poses_world={
                "Bowl/Bowl_0": self.pose("bowl").tolist(),
                "Microwave/Microwave_0": self.pose("microwave").tolist()},
            door_angle_rad=self.joint("microwave_door_joint"),
            knob_angle_rad=self.joint("knob_0_joint"),
            joint_positions_by_group_rad={
                g: [self.joint(n) for n in self.robot["joint_groups"][g]]
                for g in self.robot["joint_groups"]},
            gripper_widths_m={
                g: sum(self.joint(n) for n in self.robot["grippers"][g]["joint_names"])
                for g in self.robot["grippers"]},
            claim="初始现场帧派生的预测状态；没有新相机帧、SDK实测或物理执行")
        self._scene = {**self.initial, "snapshot": snapshot, "provenance": source}
        return self._scene

    def sample(self, time_s):
        return {**self.state(), "time_s": float(time_s), "robot_group": self.group,
                "tcp_pose_world": self.tcp().tolist()}

    def _commit(self, row):
        self.stages.append(row)
        self.revision += 1
        self._scene = None

    def event(self, name, *, width=None, attach=None, duration_s=.2, assumption=None):
        scene = self.scene()
        before = self.sample(0.)
        if attach is False:
            self.attachment = None
        if width is not None:
            lo, hi = self.robot["grippers"][self.group]["width_range_m"]
            if not lo <= float(width) <= hi:
                raise ValueError("预览夹爪开口越界")
            joints = self.robot["grippers"][self.group]["joint_names"]
            for n in joints:
                self.set_joint(n, float(width)/len(joints))
        self.forward()
        if attach is True:
            self.attachment = np.linalg.inv(self.tcp()) @ self.pose("bowl")
        if assumption:
            self.assumptions.append({"stage": name, "message": assumption, "verified": False})
        self._commit({"name": name, "operation": "state_event", "status": "planned",
            "predicted": True, "assumption": assumption,
            "collision_snapshot_ref": scene["snapshot"]["snapshot_path"] if
                "snapshot_path" in scene["snapshot"] else
                str(self.output / "prediction_scenes" / f"{self.revision:04d}" / "collision_snapshot.json"),
            "planned_replay_samples": [before, self.sample(max(.001, float(duration_s)))]})

    def collision_ref(self, snapshot_sha256):
        if snapshot_sha256 in self._collision_refs:
            return self._collision_refs[snapshot_sha256]
        for path in self.output.rglob("collision_snapshot.json"):
            if path not in self._indexed_snapshots:
                record = json.loads(path.read_text(encoding="utf-8"))
                self._collision_refs[record["snapshot_sha256"]] = str(path)
                self._indexed_snapshots.add(path)
        if snapshot_sha256 not in self._collision_refs:
            raise ValueError("预览缺少本段实际使用的碰撞快照，不能用完整场景冒充")
        return self._collision_refs[snapshot_sha256]

    def motion(self, name, stage, *, collision_ref=None, articulation=None):
        scene = self.scene()
        # 保存原 planner 失败，失败阶段不播放不存在的轨迹。
        if stage.status != "planned" or stage.result is None or not stage.result.success:
            self.stages.append({**stage.to_dict(), "name": name, "status": "failed",
                "operation": None if stage.request is None else stage.request.operation.value,
                "planned_replay_samples": [], "collision_snapshot_ref": collision_ref})
            failure = stage.to_dict()
            raise PlanningFailed("motion_planning_failed", name, failure)
        result, request = stage.result, stage.request
        trajectory = result.trajectory
        if trajectory is None or request is None or result.backend != "curobo":
            raise ValueError("正式预览运动必须包含真实 cuRobo 轨迹与请求")
        if result.request_id != request.request_id or result.operation != request.operation:
            raise ValueError("规划请求与结果身份不同")
        if tuple(trajectory.joint_names) != self.names:
            raise ValueError("预览轨迹关节顺序不同")
        positions = np.asarray(trajectory.positions_rad)
        times = np.asarray(trajectory.times_s)
        start = np.array([self.joint(n) for n in self.names])
        if (positions.ndim != 2 or positions.shape[1] != len(self.names)
                or len(positions) < 2 or len(times) != len(positions)
                or not np.isfinite(positions).all() or not np.isfinite(times).all()
                or np.any(np.diff(times) <= 0)
                or np.max(np.abs(positions[0]-start)) > self.max_start_error_rad
                or np.max(np.abs(np.asarray(request.start.positions_rad)-start)) > 1e-6):
            raise ValueError("预览轨迹起点不连续或轨迹数据无效")
        if result.goal_state is None or not np.allclose(
                result.goal_state.positions_rad, positions[-1], atol=1e-6, rtol=0):
            raise ValueError("规划 goal_state 与实际轨迹末帧不同")
        # 先验证整条轨迹，避免在后续越限时留下未记录的半段预测终态。
        for index, joint_name in enumerate(self.names):
            joint = self.model.joint(joint_name)
            if self.model.jnt_limited[joint.id]:
                lo, hi = self.model.jnt_range[joint.id]
                if np.any(positions[:, index] < lo-1e-6) or np.any(positions[:, index] > hi+1e-6):
                    raise ValueError(f"轨迹关节 {joint_name} 越限")
        collision_ref = collision_ref or self.collision_ref(request.scene.snapshot_sha256)
        if json.loads(Path(collision_ref).read_text())["snapshot_sha256"] != request.scene.snapshot_sha256:
            raise ValueError("预览碰撞显示层与本阶段实际规划请求不一致")
        samples = []
        initial_angle = self.joint(articulation[0]) if articulation else None
        duration = float(times[-1]-times[0])
        for t, q in zip(times, positions):
            for n, v in zip(self.names, q):
                self.set_joint(n, v)
            if articulation:
                alpha = float((t-times[0])/duration)
                self.set_joint(articulation[0], initial_angle +
                               alpha*(articulation[1]-initial_angle))
            self.forward()
            samples.append(self.sample(t-times[0]))
        if articulation:
            self.assumptions.append({"stage": name, "message": "物体关节按阶段门角/旋钮角目标随动；未验证物理接触", "verified": False})
        self._commit({**stage.to_dict(), "name": name,
            "operation": request.operation.value, "execution_forbidden": True,
            "details": {**stage.details, "execution_forbidden": True, "state_source": SOURCE},
            "predicted": True, "planned_replay_samples": samples,
            "collision_snapshot_ref": collision_ref or
                str(self.output / "prediction_scenes" / f"{self.revision:04d}" / "collision_snapshot.json")})

    def hold_report(self, candidate):
        scene = self.scene()
        return {**report_basis(scene, "hold"), "selected_candidate_id": candidate,
                "physical_grasp_check": assumption_check(scene).to_dict(),
                "grasp_success": False, "pick_success": False,
                "postcondition": assumption_check(scene).to_dict()}

    def release_report(self):
        scene = self.scene()
        return {**report_basis(scene, "release"), "release_success": False,
                "source_frame_number": scene["provenance"]["source_frame_number"],
                "source_timestamp_ms": scene["provenance"]["source_timestamp_ms"],
                "bowl_pose_world_after_release": self.pose("bowl").tolist(),
                "postcondition": assumption_check(scene).to_dict()}

    def save_replay(self):
        write_json(self.output / "plan_preview.json", {
            "schema_version": 1, "source_format": SOURCE,
            "skill": "microwave_task1", "instance_id": "microwave",
            "robot_group": self.group, "model_ref": self.initial["scene_xml_path"],
            "initial_scene_snapshot_sha256": self.initial["snapshot"]["snapshot_sha256"],
            "stages": self.stages, "assumptions": self.assumptions,
            "claim": "cuRobo规划轨迹及显式运动学假设；无真机执行与物理成功证据"})
