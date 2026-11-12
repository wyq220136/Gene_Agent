"""闭爪后将 Bowl 从静态 world 转成末端附着物的纯视觉规划快照。

这里复用旧版 32 球表面采样和 ``T_link_object`` 坐标链；MuJoCo 仅提供
当前 GT 几何和机器人 FK，不做接触或物理成功判断。
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import trimesh

from gene_agent.knowledge.geometry import validate_pose
from gene_agent.motion.curobo.attachment import (
    build_attachment_surface_spheres_from_mesh,
    transform_object_centers_to_link,
)
from gene_agent.motion.curobo.scene import export_collision_scene
from gene_agent.motion.model import JointState
from gene_agent.motion.configuration import CollisionMode, resolve_collision_mode

from gene_agent.tasks.policies.microwave_task.planning.bowl import PickBowlCase


ROOT = Path(__file__).resolve().parents[5]
BOWL_MESH = ROOT / "assets/objects/bowl_visual.obj"


def build_bowl_attachment_snapshot(
    case: PickBowlCase,
    end_state: JointState,
    T_world_tcp: np.ndarray,
    *,
    close_width_m: float,
    output_dir: Path,
) -> dict:
    """按旧 attach 时机冻结闭爪状态，并显式记录物体的碰撞转移。

    ``end_state`` 是 approach 的实际关节终点；``T_world_tcp`` 应来自
    同一终点的规划器 FK。抓取物体跟随的是实际末端，而非 Knowledge 的
    标称 grasp pose。该函数不推断接触，也不自行修改原始 ``case.data``。
    """
    group = "right"
    robot = case.metadata["robot"]
    names = tuple(robot["joint_groups"][group])
    if end_state.joint_names != names:
        raise ValueError("附着时的实际关节组与场景机器人不一致")
    width = float(close_width_m)
    gripper = robot["grippers"][group]
    lower, upper = map(float, gripper["width_range_m"])
    if not np.isfinite(width) or not lower <= width <= upper:
        raise ValueError("附着时夹爪开口超出当前机器人范围")
    world_tcp = validate_pose(T_world_tcp, field_name="T_world_tcp")
    link_from_tcp = validate_pose(
        robot["transforms"][group]["link_from_tcp"], field_name="T_link_tcp",
    )

    frozen = mujoco.MjData(case.model)
    mujoco.mj_copyData(frozen, case.model, case.data)
    for name, position in zip(names, end_state.positions_rad):
        frozen.qpos[case.model.joint(name).qposadr[0]] = float(position)
    for name in gripper["joint_names"]:
        frozen.qpos[case.model.joint(name).qposadr[0]] = width / 2.0
    mujoco.mj_forward(case.model, frozen)

    body = frozen.body("bowl")
    world_object = np.eye(4)
    world_object[:3, :3] = body.xmat.reshape(3, 3)
    world_object[:3, 3] = body.xpos
    world_link = world_tcp @ np.linalg.inv(link_from_tcp)
    mesh = (trimesh.load_mesh(BOWL_MESH, process=False)
            if resolve_collision_mode() is CollisionMode.FULL else None)
    centers_object, radii, component_removed = build_attachment_surface_spheres_from_mesh(
        mesh, num_spheres=32, radius=.002,
    )
    centers_world = (world_object[:3, :3] @ centers_object.T).T + world_object[:3, 3]
    centers_link = transform_object_centers_to_link(
        centers_object, link_pose_planning=world_link,
        object_pose_planning=world_object,
    )
    spheres_world = np.column_stack((centers_world, radii))
    spheres_link = np.column_stack((centers_link, radii))
    output = Path(output_dir).resolve()
    snapshot = export_collision_scene(
        case.model, frozen, robot, group, output,
        transferred_objects={"bowl": spheres_world.tolist()},
    )
    return {
        "snapshot": snapshot,
        "planning_attachment_enabled": resolve_collision_mode() is CollisionMode.FULL,
        "collision_snapshot_ref": str(output / "collision_snapshot.json"),
        "spheres_world_m": spheres_world.tolist(),
        "spheres_link_m": spheres_link.tolist(),
        "T_world_tcp": world_tcp.tolist(),
        "T_tcp_bowl": (np.linalg.inv(world_tcp) @ world_object).tolist(),
        "component_removed": component_removed,
        "gripper_width_m": width,
    }


__all__ = ["build_bowl_attachment_snapshot"]
