"""MuJoCo 的同步运行边界：Pipeline 只读取可迁移的具名状态。"""
from __future__ import annotations

from collections.abc import Iterable, Mapping

import mujoco
import numpy as np

from gene_agent.feedback.state import state_id
from gene_agent.robots.gripper import measure_gripper_width


def _transform(position: np.ndarray, rotation: np.ndarray) -> list[list[float]]:
    transform = np.eye(4)
    transform[:3, :3] = np.asarray(rotation).reshape(3, 3)
    transform[:3, 3] = position
    return transform.tolist()


class MujocoRuntime:
    """持有一个同步 MuJoCo 实例，并区分 operational 与诊断信息。"""

    def __init__(self, model, data, metadata: Mapping, *, scene_revision: int = 0):
        self.model = model
        self.data = data
        self.metadata = metadata
        self.robot_metadata = metadata.get("robot", metadata)
        self.scene_revision = int(scene_revision)
        self._simulation_attachment_evidence = {}

    def simulation_attachments(self) -> list[dict]:
        """只报告当前已启用的仿真 weld；规划碰撞代理与此状态无关。"""
        active = []
        for group, object_name in self.metadata.get("simulation_attachment_pairs", {}).items():
            name = f"sim_attach_{group}_{object_name}"
            eq_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_EQUALITY, name)
            if eq_id >= 0 and self.data.eq_active[eq_id]:
                active.append({"mode": "simulation_privileged_weld", "robot_group": group,
                               "object_instance_id": object_name,
                               "activation_contact": self._simulation_attachment_evidence.get(name)})
        return active

    def attach_simulation_object(self, group: str, object_name: str,
                                 *, contact_evidence: Mapping) -> dict:
        """仅在双侧指垫实际接触后，按当前相对位姿启用仿真 weld，不瞬移物体。"""
        if (not contact_evidence.get("success")
                or not contact_evidence.get("bilateral_pad_contact")
                or contact_evidence.get("undesired_gripper_contacts")):
            raise ValueError("启用仿真固定需要有效双侧指垫接触证据")
        if self.metadata.get("simulation_attachment_pairs", {}).get(group) != object_name:
            raise ValueError(f"未配置仿真固定：{group} → {object_name}")
        name = f"sim_attach_{group}_{object_name}"
        eq_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_EQUALITY, name)
        if eq_id < 0 or int(self.data.eq_active[eq_id]):
            raise ValueError(f"仿真固定缺失或已启用：{name}")
        mujoco.mj_forward(self.model, self.data)
        body1 = int(self.model.eq_obj1id[eq_id])
        body2 = int(self.model.eq_obj2id[eq_id])
        rotation1 = self.data.xmat[body1].reshape(3, 3)
        rotation2 = self.data.xmat[body2].reshape(3, 3)
        relative_rotation = rotation1.T @ rotation2
        relative_quat = np.empty(4)
        mujoco.mju_mat2Quat(relative_quat, relative_rotation.reshape(9))
        relative_pos = rotation1.T @ (self.data.xpos[body2] - self.data.xpos[body1])
        self.model.eq_data[eq_id, 3:6] = relative_pos
        self.model.eq_data[eq_id, 6:10] = relative_quat
        self.data.eq_active[eq_id] = 1
        self._simulation_attachment_evidence[name] = {
            "bilateral_pad_contact": True,
            "contact_count": int(contact_evidence.get("contact_count", 0)),
            "sim_time_s": float(self.data.time),
        }
        mujoco.mj_forward(self.model, self.data)
        return self.simulation_attachments()[0]

    def release_simulation_object(self, group: str, object_name: str) -> None:
        """取消仿真固定；保留碗当前位姿与速度，下一步由接触物理决定运动。"""
        name = f"sim_attach_{group}_{object_name}"
        eq_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_EQUALITY, name)
        if eq_id < 0 or not int(self.data.eq_active[eq_id]):
            raise ValueError(f"仿真固定未启用：{name}")
        self.data.eq_active[eq_id] = 0
        self._simulation_attachment_evidence.pop(name, None)
        mujoco.mj_forward(self.model, self.data)

    def copy_data(self):
        """冻结当前状态并刷新副本的派生量，不改变在线 data。"""
        copied = mujoco.MjData(self.model)
        mujoco.mj_copyData(copied, self.model, self.data)
        mujoco.mj_forward(self.model, copied)
        return copied

    def query(
        self,
        *,
        object_names: Iterable[str] = (),
        robot_groups: Iterable[str] = (),
    ) -> dict:
        """按名称返回 Pipeline 可使用的 GT；不包含接触、力或内部数字 ID。"""
        object_names = tuple(object_names)
        robot_groups = tuple(robot_groups)
        self._validate_query_names(object_names, robot_groups)
        copied = self.copy_data()

        objects = {}
        for name in object_names:
            body = copied.body(name)
            objects[name] = {
                "T_world_object": _transform(body.xpos, body.xmat),
            }

        robots = {}
        for group in robot_groups:
            joint_names = list(self.robot_metadata["joint_groups"][group])
            positions = []
            velocities = []
            for name in joint_names:
                joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                positions.append(float(copied.qpos[self.model.jnt_qposadr[joint_id]]))
                velocities.append(float(copied.qvel[self.model.jnt_dofadr[joint_id]]))
            tcp = copied.site(self.robot_metadata["tcp_sites"][group])
            robots[group] = {
                "joint_names": joint_names,
                "joint_positions_rad": positions,
                "joint_velocities_rad_s": velocities,
                "T_world_tcp": _transform(tcp.xpos, tcp.xmat),
                "gripper": self._query_gripper(copied, group),
            }

        return {
            "schema_version": 1,
            "state_id": state_id(self.model, self.data),
            "scene_revision": self.scene_revision,
            "sim_time_s": float(self.data.time),
            "world_frame": self.metadata.get(
                "world_frame", self.robot_metadata.get("world_frame", "world")
            ),
            "objects": objects,
            "robots": robots,
            "simulation_attachments": self.simulation_attachments(),
        }

    def replay_sample(self, *, object_names: Iterable[str], robot_groups: Iterable[str],
                      attachment: dict | None = None) -> dict:
        """取得同一仿真时刻的完整可视状态，供实际执行的中间帧使用。

        state_id 使用当前积分状态计算；qpos 与 query 返回的物体/TCP 均从
        同一个未被外部线程写入的同步执行步读取。调用者不能在回放时插值补造物体 pose。
        """
        observation = self.query(object_names=object_names, robot_groups=robot_groups)
        row = {
            **observation,
            "qpos": self.data.qpos.tolist(),
            "attachment": attachment,
        }
        from gene_agent.feedback.live_record import record_actual_sample
        record_actual_sample(self,row)
        return row

    def execute_joint_trajectory(self, **kwargs) -> dict:
        """执行短轨迹；具体输入验证与 actuator 映射集中在 execution 模块。"""
        from .execution import (
            execute_joint_trajectory, temporary_position_actuator_kp_scale,
        )

        gain_scale = kwargs.pop("position_kp_scale", None)
        if gain_scale is None:
            return execute_joint_trajectory(self, **kwargs)
        group = kwargs.get("group")
        joint_names = self.robot_metadata.get("joint_groups", {}).get(group)
        if joint_names is None:
            raise ValueError(f"未知机器人组: {group}")
        with temporary_position_actuator_kp_scale(
                self.model, joint_names, gain_scale):
            return execute_joint_trajectory(self, **kwargs)

    def execute_gripper_command(self, **kwargs) -> dict:
        """执行米制夹爪开口命令；输入验证集中在 gripper_execution。"""
        from .gripper_execution import execute_gripper_command
        from .execution import temporary_position_actuator_kp_scale

        gain_scale = kwargs.pop("position_kp_scale", None)
        if gain_scale is None:
            return execute_gripper_command(self, **kwargs)
        group = kwargs.get("group")
        joint_names = self.robot_metadata.get("joint_groups", {}).get(group)
        if joint_names is None:
            raise ValueError(f"未知机器人组: {group}")
        with temporary_position_actuator_kp_scale(
                self.model, joint_names, gain_scale):
            result = execute_gripper_command(self, **kwargs)
        return {**result, "position_kp_scale": float(gain_scale)}

    def execute_hold(self, **kwargs) -> dict:
        """保持当前执行器目标并真实推进仿真，用于接触后的保持判定。"""
        from .execution import execute_hold, temporary_position_actuator_kp_scale

        gain_scale = kwargs.pop("position_kp_scale", None)
        if gain_scale is None:
            return execute_hold(self, **kwargs)
        group = kwargs.get("replay_group", "right")
        joint_names = self.robot_metadata.get("joint_groups", {}).get(group)
        if joint_names is None:
            raise ValueError(f"未知机器人组: {group}")
        with temporary_position_actuator_kp_scale(
                self.model, joint_names, gain_scale):
            return execute_hold(self, **kwargs)

    def _validate_query_names(self, object_names: tuple[str, ...], robot_groups: tuple[str, ...]):
        if len(set(object_names)) != len(object_names):
            raise ValueError("object_names 不能重复")
        if len(set(robot_groups)) != len(robot_groups):
            raise ValueError("robot_groups 不能重复")
        for name in object_names:
            if mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name) < 0:
                raise ValueError(f"未知物体 body: {name}")
        groups = self.robot_metadata.get("joint_groups", {})
        tcp_sites = self.robot_metadata.get("tcp_sites", {})
        for group in robot_groups:
            if group not in groups:
                raise ValueError(f"未知机器人组: {group}")
            if group not in tcp_sites or mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_SITE, tcp_sites[group]
            ) < 0:
                raise ValueError(f"机器人组 {group} 缺少有效 TCP site")
            for name in groups[group]:
                joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                if joint_id < 0 or int(self.model.jnt_type[joint_id]) != int(
                    mujoco.mjtJoint.mjJNT_HINGE
                ):
                    raise ValueError(f"机器人组 {group} 仅支持具名 hinge 关节: {name}")

    def _query_gripper(self, copied, group: str) -> dict:
        descriptor = self.robot_metadata.get("grippers", {}).get(group)
        if not descriptor or descriptor.get("mode") not in {"parallel_slide_proxy", "pika_urdf"}:
            return {"available": False}
        measured = measure_gripper_width(
            self.model, copied, self.robot_metadata, group
        )
        return {
            "available": True,
            "width_m": measured["width_m"],
            "symmetry_error_m": measured["symmetry_error_m"],
            "joint_positions_m": measured["joint_positions_m"],
        }
