"""把现场碰撞快照接到已有 MotionPlanner；不下发真机命令。"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from gene_agent.motion.curobo.core import CuroboPlannerAdapter, load_realman_motion_config
from gene_agent.motion.curobo.session import CuroboPlannerPool
from gene_agent.motion.model import JointState, PlannerMode, PlanningSceneRef
from gene_agent.motion.configuration import CollisionMode, resolve_collision_mode, collision_mode_metadata


def prepare_plan_only_context(
    collision_scene: Mapping, *, group: str,
    planner_pool: CuroboPlannerPool, scene_revision: int = 0,
    seed: int = 123, planner_mode: PlannerMode | str = PlannerMode.CUDA_GRAPH_BENCHMARK,
    attachment_spheres_tool: Sequence[Sequence[float]] | None = None,
) -> dict:
    """复用当前 cuRobo 配置与 pool，返回 Skill 所需的 planner/起点/场景。

    调用方保有 ``planner_pool``，可在同一场景的多个阶段复用 solver。
    本函数不生成目标、不改变阶段碰撞策略，也不调用机器人 SDK。
    现场发生任何位姿、关节、夹爪或门角变化后，须重新导出碰撞快照。
    """
    snapshot = collision_scene["snapshot"]
    provenance = collision_scene["provenance"]
    robot = collision_scene["robot_metadata"]
    if provenance.get("physics_stepped") is not False:
        raise ValueError("现场规划不能采用 MuJoCo 动力学更新后的状态")
    if provenance.get("pose_frame_verified") is not True:
        raise ValueError("现场规划缺少同帧验证的感知 pose")
    if provenance.get("collision_snapshot_sha256") != snapshot.get("snapshot_sha256"):
        raise ValueError("现场来源与规划碰撞快照 hash 不一致，必须重新建场景")
    if group != snapshot.get("active_group") or group not in robot.get("joint_groups", {}):
        raise ValueError("活动机器人组与现场碰撞快照不一致")
    if snapshot.get("world_frame") != robot.get("world_frame"):
        raise ValueError("机器人和现场碰撞快照的 world frame 不一致")
    groups = set(robot["joint_groups"])
    measured = provenance.get("joint_positions_by_group_rad")
    if not isinstance(measured, dict) or set(measured) != groups:
        raise ValueError("所有机器人组都须提供实测关节，另一臂不能沿用 home")
    if not snapshot.get("other_arm_frozen") or set(snapshot.get("frozen_groups", ())) != groups - {group}:
        raise ValueError("另一臂必须按当前实测状态冻结在规划障碍中")
    gripper = snapshot.get("gripper_snapshot")
    if not isinstance(gripper, dict) or not gripper.get("snapshot_id"):
        raise ValueError("现场规划缺少当前活动夹爪的碰撞球快照")
    if gripper.get("group", group) != group:
        raise ValueError("夹爪碰撞快照与活动机器人组不一致")
    names = tuple(robot["joint_groups"][group])
    start = JointState(names, tuple(measured[group]))
    # 其他臂不作为当前规划关节，但必须存在于快照。JointState 在这里
    # 一并检验它们的顺序、数量和有限值，避免错误输入进入 planner。
    for other in groups - {group}:
        JointState(tuple(robot["joint_groups"][other]), tuple(measured[other]))
    scene = PlanningSceneRef(
        scene_revision, snapshot["snapshot_sha256"], snapshot["scope"]
    )
    resolved_mode = PlannerMode(planner_mode)
    settings = load_realman_motion_config(
        group, robot, gripper_snapshot=gripper,
    )
    collision_mode = resolve_collision_mode(snapshot.get("collision_mode"))
    if attachment_spheres_tool is not None and collision_mode is CollisionMode.FULL:
        spheres = np.asarray(attachment_spheres_tool, dtype=float)
        if (spheres.ndim != 2 or spheres.shape[1] != 4 or len(spheres) == 0
                or not np.isfinite(spheres).all() or np.any(spheres[:, 3] <= 0)):
            raise ValueError("现场附着物必须提供 [N,4] 有限正半径工具坐标球")
        # 原 cuRobo 构造函数已有 attachment_spheres_tool；只在物体已
        # 通过现场夹持检查时显式传入，普通规划仍保持原 settings 和缓存键。
        settings = {**settings, "attachment_spheres_tool": spheres.tolist()}
    lease = planner_pool.acquire(
        settings, snapshot, planner_mode=resolved_mode,
        random_seed=int(seed),
    )
    planner = CuroboPlannerAdapter(
        lease.planner, group=group, scene_revision=scene.revision,
    )
    return {
        "planner": planner,
        **collision_mode_metadata(collision_mode),
        "scene": scene,
        "start": start,
        "planner_cache_hit": lease.cache_hit,
        "planner_setup_duration_s": lease.setup_duration_s,
        "planner_rebuild_reason": lease.rebuild_reason,
        "collision_snapshot_sha256": snapshot["snapshot_sha256"],
    }
