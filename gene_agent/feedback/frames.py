"""Codex 专用坐标关系；不进入 Pipeline operational observation。"""
from __future__ import annotations

import mujoco
import numpy as np

from gene_agent.feedback.state import state_id


def _pose(position, rotation) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, :3] = np.asarray(rotation, dtype=float).reshape(3, 3)
    transform[:3, 3] = np.asarray(position, dtype=float)
    return transform


def query_frame_diagnostics(
    runtime, *, object_name: str, group: str, target_pose=None,
) -> dict:
    """在同一冻结状态中比较物体与 TCP，显式给出 world/TCP 两种分量。"""
    model = runtime.model
    if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, object_name) < 0:
        raise ValueError(f"未知物体 body: {object_name}")
    sites = runtime.robot_metadata.get("tcp_sites", {})
    if group not in sites or mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SITE, sites[group]
    ) < 0:
        raise ValueError(f"机器人组 {group} 缺少有效 TCP site")
    frozen = runtime.copy_data()
    body = frozen.body(object_name)
    tcp = frozen.site(sites[group])
    world_from_object = _pose(body.xpos, body.xmat)
    world_from_tcp = _pose(tcp.xpos, tcp.xmat)
    delta_world = world_from_object[:3, 3] - world_from_tcp[:3, 3]
    delta_tcp = world_from_tcp[:3, :3].T @ delta_world
    world_frame = runtime.metadata.get(
        "world_frame", runtime.robot_metadata.get("world_frame", "world")
    )
    result = {
        "schema_version": 1,
        "state_id": state_id(model, frozen),
        "sim_time_s": float(frozen.time),
        "world_frame": world_frame,
        "object_name": object_name,
        "robot_group": group,
        "T_world_object": world_from_object.tolist(),
        "T_world_tcp": world_from_tcp.tolist(),
        "object_minus_tcp_world_m": delta_world.tolist(),
        "object_minus_tcp_tcp_m": delta_tcp.tolist(),
        "frames": {
            "WORLD": {
                "description": f"world / {world_frame}",
                "T_world_frame": np.eye(4).tolist(),
            },
            "TCP": {
                "description": f"{group} TCP",
                "T_world_frame": world_from_tcp.tolist(),
            },
        },
    }
    if target_pose is not None:
        world_from_target = np.asarray(target_pose, dtype=float)
        if world_from_target.shape != (4, 4) or not np.all(np.isfinite(world_from_target)):
            raise ValueError("target_pose 必须是有限 4x4 矩阵")
        if not np.allclose(world_from_target[3], [0, 0, 0, 1]) or not np.allclose(
            world_from_target[:3, :3].T @ world_from_target[:3, :3],
            np.eye(3), atol=1e-6,
        ) or np.linalg.det(world_from_target[:3, :3]) < 0:
            raise ValueError("target_pose 必须是合法刚体变换")
        target_delta_world = world_from_target[:3, 3] - world_from_tcp[:3, 3]
        result.update(
            T_world_target=world_from_target.tolist(),
            target_minus_tcp_world_m=target_delta_world.tolist(),
            target_minus_tcp_tcp_m=(
                world_from_tcp[:3, :3].T @ target_delta_world
            ).tolist(),
        )
        result["frames"]["TARGET"] = {
            "description": "requested TCP target",
            "T_world_frame": world_from_target.tolist(),
        }
    return result


__all__ = ["query_frame_diagnostics"]
