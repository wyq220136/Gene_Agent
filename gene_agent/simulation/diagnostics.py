"""Codex 专用 MuJoCo 诊断；这些字段不得作为正式 Pipeline 输入。"""
from __future__ import annotations

import mujoco

from gene_agent.feedback.state import snapshot


def query_diagnostics(runtime, *, body_names=()) -> dict:
    """从冻结状态返回精确接触、力和积分状态摘要。"""
    requested = tuple(body_names)
    if len(set(requested)) != len(requested):
        raise ValueError("body_names 不能重复")
    for name in requested:
        if mujoco.mj_name2id(runtime.model, mujoco.mjtObj.mjOBJ_BODY, name) < 0:
            raise ValueError(f"未知诊断 body: {name}")

    frozen = runtime.copy_data()
    full = snapshot(runtime.model, frozen)
    selected = set(requested)
    contacts = [
        {**contact, "penetration_depth_m": max(0.0, -float(contact["dist_m"]))}
        for contact in full["contacts"]
        if not selected or selected.intersection(contact["body_names"])
    ]
    bodies = (
        {name: full["bodies"][name] for name in requested}
        if requested
        else full["bodies"]
    )
    return {
        "schema_version": 1,
        "state_id": full["state_id"],
        "sim_time_s": full["sim_time"],
        "qpos": full["qpos"],
        "qvel": full["qvel"],
        "ctrl": full["ctrl"],
        "joint_names": full["joint_names"],
        "bodies": bodies,
        "contacts": contacts,
    }
