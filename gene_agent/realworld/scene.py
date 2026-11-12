"""真机感知 pose 到已校验场景资产 frame 的纯几何边界。

这里只计算 ``T_world_scene_asset``，不生成 cuRobo 障碍物。要把结果
用于规划，还必须单独核对桌面/支撑的真实几何及阶段接触规则。
"""

from __future__ import annotations

from typing import Mapping

import mujoco
import numpy as np

from gene_agent.knowledge.geometry import validate_pose
from gene_agent.motion.model import JointState

from .perception import ObservedScene


def align_scene_asset_poses(
    observed: ObservedScene, *, T_world_sdk: np.ndarray,
    T_perception_object_scene_asset: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """逐实例应用已标定的物体 frame 对齐变换。

    FoundationPose 返回的是 ``T_sdk_perception_object``；资产碰撞代理
    可能采用另一局部原点。两者的关系必须逐实例显式给出，不能默认为
    单位矩阵，也不能把左 SDK base 误认为右臂/仿真 world。
    """
    if not observed.pose_frame_verified:
        raise ValueError("缺少与相机同帧的 pose 来源，不能构建现场规划场景")
    world_from_sdk = validate_pose(T_world_sdk, field_name="T_world_sdk")
    observed_ids = set(observed.object_poses_sdk)
    alignment_ids = set(T_perception_object_scene_asset)
    if observed_ids != alignment_ids:
        raise ValueError(
            f"每个物体都必须有显式资产对齐变换；缺少 "
            f"{sorted(observed_ids - alignment_ids)}，多余 {sorted(alignment_ids - observed_ids)}"
        )
    aligned = {}
    for object_id, T_sdk_object in observed.object_poses_sdk.items():
        local = validate_pose(
            T_perception_object_scene_asset[object_id],
            field_name=f"{object_id} 资产对齐",
        )
        aligned[object_id] = validate_pose(
            world_from_sdk @ T_sdk_object @ local,
            field_name=f"T_world_{object_id}_scene_asset",
        )
    return aligned


def observed_joint_state(scene: dict) -> JointState:
    group = scene["snapshot"]["active_group"]
    names = tuple(scene["robot_metadata"]["joint_groups"][group])
    values = scene["provenance"]["joint_positions_by_group_rad"][group]
    return JointState(names, tuple(values))


def observed_tcp_pose(scene: Mapping) -> np.ndarray:
    """用现场 qpos 在私有 MuJoCo 数据中求 FK；不推进物理。"""
    snapshot = scene["snapshot"]
    group = snapshot["active_group"]
    model = mujoco.MjModel.from_xml_path(str(scene["scene_xml_path"]))
    qpos = np.asarray(snapshot["qpos_snapshot"], dtype=float)
    if qpos.shape != (model.nq,) or not np.isfinite(qpos).all():
        raise ValueError("现场 qpos 与 FK 资产不一致")
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    site = data.site(scene["robot_metadata"]["tcp_sites"][group])
    pose = np.eye(4)
    pose[:3, :3] = site.xmat.reshape(3, 3)
    pose[:3, 3] = site.xpos
    return validate_pose(pose, field_name="T_world_tcp")
