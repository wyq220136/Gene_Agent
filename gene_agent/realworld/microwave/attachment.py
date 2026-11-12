"""真实持物证据与规划附着；不以仿真辅助代替真实夹持。"""

from __future__ import annotations
from gene_agent.motion.configuration import CollisionMode, resolve_collision_mode

import hashlib
import json
from pathlib import Path
from typing import Mapping

import mujoco
import numpy as np
import trimesh

from gene_agent.tasks.policies.microwave_task.planning.attachment import BOWL_MESH
from gene_agent.knowledge.geometry import validate_pose
from gene_agent.motion.curobo.attachment import (
    build_attachment_surface_spheres_from_mesh,
)
from gene_agent.motion.curobo.scene import export_collision_scene
from gene_agent.tasks.checks import CheckResult

_BOWL = "Bowl/Bowl_0"
_SIMULATION_SOURCES = {
    "simulation_privileged_weld", "mujoco_contact_geometry_at_planned_waypoints",
}


def same_observation_as_hold_report(
    scene: Mapping, *, snapshot_sha256: str,
    frame_number: int, timestamp_ms: float,
) -> bool:
    """区分碰撞状态相同与真正的同一观测帧。

    哈希只表示规划几何；即使后来读到完全相同的几何，也不能证明
    Bowl 仍被夹住。新帧必须同时具有递增的帧号和时间，并重新检查。
    """
    snapshot = scene["snapshot"]
    source = scene["provenance"]
    frame = int(source["source_frame_number"])
    timestamp = float(source["source_timestamp_ms"])
    old_frame = int(frame_number)
    old_timestamp = float(timestamp_ms)
    if frame == old_frame and timestamp == old_timestamp:
        if snapshot.get("snapshot_sha256") != snapshot_sha256:
            raise ValueError("同一观测帧的碰撞快照与原持碗记录不一致")
        return True
    if frame <= old_frame or timestamp <= old_timestamp:
        raise ValueError("当前观测必须新于原持碗确认帧")
    return False


def bind_current_hold_check(scene: Mapping, check: CheckResult) -> CheckResult:
    """把现场检查明确绑定到传给检查器的碰撞快照和观测帧。"""
    if not isinstance(check, CheckResult):
        raise TypeError("现场持碗检查器必须返回 CheckResult")
    expected = {
        "collision_snapshot_sha256": scene["snapshot"]["snapshot_sha256"],
        "source_frame_number": scene["provenance"]["source_frame_number"],
        "source_timestamp_ms": scene["provenance"]["source_timestamp_ms"],
    }
    evidence = dict(check.evidence)
    if any(key in evidence and evidence[key] != value
           for key, value in expected.items()):
        raise ValueError("持碗检查证据与当前现场快照不一致")
    evidence.update(expected)
    return CheckResult(check.status, check.source, evidence, check.reason)


def require_current_hold_check(scene: Mapping, check: Mapping) -> None:
    """直接调用目标生成器时也拒绝未绑定或过期的持碗证据。"""
    evidence = check.get("evidence") or {}
    expected = {
        "collision_snapshot_sha256": scene["snapshot"]["snapshot_sha256"],
        "source_frame_number": scene["provenance"]["source_frame_number"],
        "source_timestamp_ms": scene["provenance"]["source_timestamp_ms"],
    }
    if not isinstance(evidence, Mapping) or any(
        evidence.get(key) != value for key, value in expected.items()
    ):
        raise ValueError("持碗检查不是当前现场快照，必须重新确认")


def make_manual_held_bowl_report(scene: dict, check: CheckResult) -> dict:
    """记录人准备场景后同帧确认持碗的事实，不声明抓取动作已执行。"""
    if not isinstance(check, CheckResult) or not check.passed:
        raise ValueError("人工准备的持碗前置条件必须有通过的现场检查")
    if check.source in _SIMULATION_SOURCES:
        raise ValueError("仿真证据不能作为真机持碗前置条件")
    snapshot = scene["snapshot"]
    source = scene["provenance"]
    if (source.get("physics_stepped") is not False
            or source.get("pose_frame_verified") is not True
            or source.get("collision_snapshot_sha256") != snapshot.get("snapshot_sha256")):
        raise ValueError("人工持碗确认需要同帧现场碰撞快照")
    if snapshot.get("active_group") != "right":
        raise NotImplementedError("原 task_1 Bowl 放入当前只验收右臂")
    aligned = source.get("aligned_object_poses_world") or {}
    if _BOWL not in aligned:
        raise ValueError("人工持碗确认缺少 Bowl pose")
    validate_pose(aligned[_BOWL], field_name="T_world_bowl")
    # 顶层记录和人工/视觉检查必须来自同一观测；不能把旧回答换上新帧号。
    check = bind_current_hold_check(scene, check)
    return {
        "schema_version": 1,
        "source_format": "manual_prepared_held_bowl",
        "object_id": _BOWL,
        "source_collision_snapshot_sha256": snapshot["snapshot_sha256"],
        "source_frame_number": source["source_frame_number"],
        "source_timestamp_ms": source["source_timestamp_ms"],
        "postcondition": check.to_dict(),
        "claim": "人工准备并确认当前持碗；未执行本项目 Pick 或抓取 Skill",
    }


def validate_manual_held_bowl_report(
    scene: dict, report: dict, *, current_hold_check: dict | None = None,
) -> dict:
    """核对人工持碗来源、观测新旧关系，并返回本帧持碗检查。"""
    if report.get("source_format") != "manual_prepared_held_bowl":
        raise ValueError("持碗来源不是人工准备记录")
    if report.get("object_id") != _BOWL or report.get("schema_version") != 1:
        raise ValueError("人工持碗记录的物体或版本不匹配")
    snapshot = scene["snapshot"]
    source = scene["provenance"]
    if (source.get("physics_stepped") is not False
            or source.get("pose_frame_verified") is not True
            or source.get("collision_snapshot_sha256") != snapshot.get("snapshot_sha256")):
        raise ValueError("人工持碗复核需要同帧现场碰撞快照")
    if snapshot.get("active_group") != "right":
        raise NotImplementedError("原 task_1 Bowl 放入当前只验收右臂")
    aligned = source.get("aligned_object_poses_world") or {}
    if _BOWL not in aligned:
        raise ValueError("人工持碗复核缺少 Bowl pose")
    validate_pose(aligned[_BOWL], field_name="T_world_bowl")
    same = same_observation_as_hold_report(
        scene, snapshot_sha256=report["source_collision_snapshot_sha256"],
        frame_number=report["source_frame_number"],
        timestamp_ms=report["source_timestamp_ms"],
    )
    if not same and current_hold_check is None:
        raise ValueError("更新后的放碗观测必须重新确认当前仍持碗")
    check = current_hold_check if current_hold_check is not None else report.get("postcondition") or {}
    if current_hold_check is not None or same:
        require_current_hold_check(scene, check)
    if check.get("status") != "pass" or not check.get("source"):
        raise ValueError("人工持碗记录缺少通过且有来源的现场检查")
    if check["source"] in _SIMULATION_SOURCES:
        raise ValueError("仿真证据不能作为真机持碗前置条件")
    return check


_ROOT = Path(__file__).resolve().parents[3]
_PROFILE_PATH = _ROOT / "data/task_1/skills/skill_profiles.json"
_BOWL_INSTANCE = "Bowl/Bowl_0"


def prepare_observed_bowl_attachment(
    collision_scene: dict, grasp_report: dict, *, output_dir: str | Path,
    hold_evidence: dict | None = None,
    sphere_source: str = "pick_profile_5",
) -> dict:
    """按显式来源构造 Bowl 的规划碰撞球，不模拟物理粘连。

    抓取检查须通过且含可审计的真机或人工证据。Bowl 从规划 world
    移除，再按当前实测 Bowl pose 和 TCP FK 转到末端工具坐标；机械臂
    执行 lift 后仍需通过新的实测/人工后置条件，不可把附着代理当成功。
    默认保留原抓碗 5 球；取碗撤离显式选择旧 mesh 表面 32 球。
    """
    manual = grasp_report.get("source_format") == "manual_prepared_held_bowl"
    if manual:
        evidence = validate_manual_held_bowl_report(
            collision_scene, grasp_report, current_hold_check=hold_evidence,
        )
        grasp_evidence = grasp_report["postcondition"]
    else:
        grasp_evidence = grasp_report.get("physical_grasp_check") or {}
        evidence = grasp_evidence if hold_evidence is None else hold_evidence
    if not manual and (grasp_report.get("execute") is not True
            or grasp_report.get("status") != "executed"
            or grasp_report.get("grasp_success") is not True
            or grasp_evidence.get("status") != "pass"
            or not grasp_evidence.get("source")
            or evidence.get("status") != "pass"
            or not evidence.get("source")):
        raise ValueError("Bowl 附着代理需要已执行且有来源的现场夹持通过证据")
    if grasp_evidence["source"] in {
        "simulation_privileged_weld", "mujoco_contact_geometry_at_planned_waypoints",
    } or evidence["source"] in {
        "simulation_privileged_weld", "mujoco_contact_geometry_at_planned_waypoints",
    }:
        raise ValueError("仿真夹持证据不能用于现场 Bowl 附着规划")
    snapshot = collision_scene["snapshot"]
    provenance = collision_scene["provenance"]
    if (provenance.get("physics_stepped") is not False
            or provenance.get("pose_frame_verified") is not True
            or provenance.get("collision_snapshot_sha256") != snapshot.get("snapshot_sha256")):
        raise ValueError("Bowl 附着需要同帧现场观测及一致的碰撞快照")
    aligned = provenance.get("aligned_object_poses_world")
    if not isinstance(aligned, dict) or _BOWL_INSTANCE not in aligned:
        raise ValueError("Bowl 附着缺少当前资产对齐的物体 pose")
    T_world_bowl = validate_pose(aligned[_BOWL_INSTANCE], field_name="T_world_bowl")
    robot = collision_scene["robot_metadata"]
    group = snapshot["active_group"]
    if group not in robot["joint_groups"] or group not in robot["tcp_sites"]:
        raise ValueError("活动臂缺少已验证的 TCP site 或关节配置")
    if not manual and (grasp_report.get("selected_candidate_id") is None):
        raise ValueError("现场抓取未记录所选 Bowl 候选")
    model = mujoco.MjModel.from_xml_path(str(collision_scene["scene_xml_path"]))
    qpos = np.asarray(snapshot["qpos_snapshot"], dtype=float)
    if qpos.shape != (model.nq,) or not np.isfinite(qpos).all():
        raise ValueError("Bowl 附着的现场 qpos 与资产模型不一致")
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    tcp = data.site(robot["tcp_sites"][group])
    T_world_tcp = np.eye(4)
    T_world_tcp[:3, :3] = tcp.xmat.reshape(3, 3)
    T_world_tcp[:3, 3] = tcp.xpos
    T_link_tcp = validate_pose(
        robot["transforms"][group]["link_from_tcp"],
        field_name=f"{group} link_from_tcp",
    )
    T_world_tool = T_world_tcp @ np.linalg.inv(T_link_tcp)
    robot_only = resolve_collision_mode(snapshot.get("collision_mode")) is CollisionMode.ROBOT_ONLY
    if robot_only:
        local, radii = np.empty((0,3)), np.empty(0)
        coverage, profile_sha256, removed_mesh_component = "disabled_by_robot_only", None, None
    elif sphere_source == "pick_profile_5":
        profile_bytes = _PROFILE_PATH.read_bytes()
        profile = json.loads(profile_bytes)["pick_bowl_attachment_proxy"]
        local = np.asarray(profile["sphere_centers_object_m"], dtype=float)
        radius = float(profile["sphere_radius_m"])
        radii = np.full(len(local), radius)
        coverage = profile["coverage"]
        profile_sha256 = hashlib.sha256(profile_bytes).hexdigest()
        removed_mesh_component = None
    elif sphere_source == "legacy_extract_surface_32":
        mesh = trimesh.load_mesh(BOWL_MESH, process=False)
        local, radii, removed_mesh_component = build_attachment_surface_spheres_from_mesh(
            mesh, num_spheres=32, radius=.002,
        )
        coverage = sphere_source
        profile_sha256 = hashlib.sha256(
            BOWL_MESH.read_bytes() + b"legacy_extract_surface_32:.002"
        ).hexdigest()
    else:
        raise ValueError(f"未知 Bowl 附着碰撞球来源：{sphere_source}")
    if not robot_only and (local.ndim != 2 or local.shape[1] != 3 or len(local) == 0
            or radii.shape != (len(local),) or not np.isfinite(local).all()
            or not np.isfinite(radii).all() or np.any(radii <= 0)):
        raise ValueError("Bowl 附着碰撞球必须是有限中心和正半径")
    centers_world = (T_world_bowl[:3, :3] @ local.T).T + T_world_bowl[:3, 3]
    spheres_world = np.column_stack((centers_world, radii))
    T_tool_world = np.linalg.inv(T_world_tool)
    centers_tool = (T_tool_world[:3, :3] @ centers_world.T).T + T_tool_world[:3, 3]
    spheres_tool = np.column_stack((centers_tool, spheres_world[:, 3]))
    output = Path(output_dir).resolve()
    transferred = export_collision_scene(
        model, data, robot, group, output / "collision",
        transferred_objects={"bowl": spheres_world.tolist()},
    )
    attached_provenance = {
        **provenance,
        "collision_snapshot_sha256": transferred["snapshot_sha256"],
        "derived_from_collision_snapshot_sha256": snapshot["snapshot_sha256"],
        "attachment_object_id": "bowl",
        "attachment_evidence": evidence,
        "attachment_profile_sha256": profile_sha256,
    }
    record = {
        "schema_version": 1,
        "object_id": _BOWL_INSTANCE,
        "selected_candidate_id": grasp_report.get("selected_candidate_id"),
        "hold_origin": ("manual_prepared_held_bowl" if manual else "executed_grasp"),
        "source_frame_number": provenance["source_frame_number"],
        "base_collision_snapshot_sha256": snapshot["snapshot_sha256"],
        "attached_collision_snapshot_sha256": transferred["snapshot_sha256"],
        "physical_hold_evidence": evidence,
        "profile_sha256": profile_sha256,
        "coverage": coverage,
        "sphere_source": sphere_source,
        "planning_attachment_enabled": not robot_only,
        "removed_mesh_component": removed_mesh_component,
        "spheres_world_m": spheres_world.tolist(),
        "spheres_tool_m": spheres_tool.tolist(),
        "T_world_tool": T_world_tool.tolist(),
        "claim": "只有规划碰撞代理转移；尚未执行 lift 或验证 Bowl 随机械臂运动",
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "attachment.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return {
        **record,
        "snapshot": transferred,
        "provenance": attached_provenance,
        "scene_xml_path": collision_scene["scene_xml_path"],
        "scene_metadata": collision_scene["scene_metadata"],
        "robot_metadata": robot,
    }


__all__ = ['make_manual_held_bowl_report', 'validate_manual_held_bowl_report', 'same_observation_as_hold_report', 'bind_current_hold_check', 'require_current_hold_check', 'prepare_observed_bowl_attachment']
