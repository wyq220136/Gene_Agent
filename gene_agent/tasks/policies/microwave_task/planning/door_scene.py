"""门 Skill 阶段的规划碰撞范围；Viewer 与现场接同一条规则。"""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

from gene_agent.motion.curobo.scene import export_collision_scene
from gene_agent.skills.library.door.sequence import DoorPhase


def door_stage_collision_options(phase: DoorPhase) -> dict:
    """保持已验证的门阶段 world 切换，不改变自碰撞设置。

    把手阶段沿用旧 Task1 的整机微波炉 world 关闭规则；推门时只将
    运动门板移出静态障碍，仍保留炉体、桌面、另一臂等障碍及排除记录。
    """
    return {
        "disabled_body_names": (
            ("microwave", "microwave_support")
            if phase.skill_id == "handle_pull" else ()
        ),
        "moving_body_names": (
            ("microwave_door",)
            if phase.contact_policy == "door_contact" else ()
        ),
    }


def export_observed_door_stage_scene(
    collision_scene: dict, phase: DoorPhase, *, output_dir: str | Path,
) -> dict:
    """从同一现场状态派生一个阶段的碰撞快照，不执行 MuJoCo 步进。"""
    if phase.motion_kind == "gripper":
        raise ValueError("夹爪命令阶段没有独立轨迹规划碰撞快照")
    snapshot = collision_scene["snapshot"]
    provenance = collision_scene["provenance"]
    if provenance.get("physics_stepped") is not False or provenance.get("pose_frame_verified") is not True:
        raise ValueError("门阶段碰撞需要同帧现场观测，不能使用 MuJoCo 动力学结果")
    if provenance.get("collision_snapshot_sha256") != snapshot.get("snapshot_sha256"):
        raise ValueError("门阶段来源碰撞快照 hash 不一致")
    model = mujoco.MjModel.from_xml_path(str(collision_scene["scene_xml_path"]))
    data = mujoco.MjData(model)
    recorded = np.asarray(snapshot["qpos_snapshot"], dtype=float)
    if recorded.shape != (model.nq,) or not np.isfinite(recorded).all():
        raise ValueError("现场 qpos 与场景资产不一致")
    data.qpos[:] = recorded
    mujoco.mj_forward(model, data)
    options = door_stage_collision_options(phase)
    output = Path(output_dir).resolve()
    derived = export_collision_scene(
        model, data, collision_scene["robot_metadata"],
        snapshot["active_group"], output / "collision", **options,
    )
    stage_provenance = {
        **provenance,
        "parent_collision_snapshot_sha256": snapshot["snapshot_sha256"],
        "collision_snapshot_sha256": derived["snapshot_sha256"],
        "stage_id": phase.phase_id,
        "stage_collision_options": {name: list(value)
                                    for name, value in options.items()},
    }
    (output / "observed_stage_provenance.json").write_text(
        json.dumps(stage_provenance, ensure_ascii=False, indent=2,
                   allow_nan=False) + "\n", encoding="utf-8",
    )
    return {**collision_scene, "snapshot": derived,
            "provenance": stage_provenance}
