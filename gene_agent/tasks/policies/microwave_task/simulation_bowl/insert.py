"""由实际夹持关系生成带稳定性限幅的入炉目标并执行。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from gene_agent.tasks.policies.microwave_task.operations.bowl import (
    bounded_insert_tilt_for_tcp_clearance,
    object_relative_insert_targets,
    rotate_retreat_keyposes,
)

from gene_agent.feedback.grasp_contact import classify_grasp_contacts
from gene_agent.feedback.state import save_state
from gene_agent.motion.curobo.runtime_bridge import CuroboRuntimeBridge
from gene_agent.simulation.diagnostics import query_diagnostics
from gene_agent.skills.runtime.operation import CallbackOperationBackend
from gene_agent.skills.contracts import OperationState
from gene_agent.skills.library.transfer.skill import run_transfer_flow
from gene_agent.tasks.checks import CheckResult

from gene_agent.tasks.policies.microwave_task.planning.insert import _legacy_place_targets
from gene_agent.tasks.policies.microwave_task.planning.bowl import PickBowlCase


ROOT = Path(__file__).resolve().parents[5]
INSERT_STAGE_ORDER = ("preplace", "insert", "release", "detach", "retreat_1", "retreat_2")


def insertion_contact_check(model, data, robot_metadata, *,
                            allow_pad_bowl_contact: bool = True) -> dict:
    """保持炉体/门严格避碰；按阶段决定是否允许指垫继续接触碗。"""
    import mujoco
    roots = {name: model.body(name).id for name in ('bowl', 'microwave')}
    roots['robot'] = model.body(robot_metadata['group_root_bodies']['right']).id
    pads = set(robot_metadata['grippers']['right'].get('pad_geom_names', ()))
    forbidden = []
    def role(body):
        while body:
            for name, root in roots.items():
                if body == root:
                    return name
            body = int(model.body_parentid[body])
        return None
    for contact in data.contact:
        if contact.dist > 0:
            continue
        ids = [int(contact.geom1), int(contact.geom2)]
        roles = [role(int(model.geom_bodyid[g])) for g in ids]
        names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) for g in ids]
        pair = set(roles)
        reject = pair == {'robot', 'microwave'}
        if pair == {'bowl', 'microwave'}:
            reject = names[roles.index('microwave')] != 'microwave_turntable_collision'
        if pair == {'bowl', 'robot'}:
            reject = (not allow_pad_bowl_contact
                      or names[roles.index('robot')] not in pads)
        if reject:
            forbidden.append({'geom_names': names,
                              'penetration_m': max(0., -float(contact.dist))})
    return {'success': not forbidden, 'forbidden_contacts': forbidden,
            'allow_pad_bowl_contact': bool(allow_pad_bowl_contact),
            'source': 'mujoco_contact_geometry'}


def validate_insertion_plan(runtime, stage: str, plan: dict, tcp_bowl: np.ndarray,
                            *, max_tilt_deg: float | None = 15.) -> dict:
    """逐点检查实际几何和持碗倾角；不改在线状态或正式规划的碰撞世界。"""
    import mujoco
    frozen = runtime.copy_data()
    model = runtime.model
    held = stage in {'preplace', 'insert'}
    adr = model.joint('bowl_free').qposadr[0]
    maximum_tilt = 0.
    for index, positions in enumerate(plan['positions_rad']):
        for name, value in zip(plan['joint_names'], positions, strict=True):
            frozen.qpos[model.joint(name).qposadr[0]] = value
        mujoco.mj_forward(model, frozen)
        if held:
            site = frozen.site(runtime.robot_metadata['tcp_sites']['right'])
            tcp = np.eye(4)
            tcp[:3, :3], tcp[:3, 3] = site.xmat.reshape(3, 3), site.xpos
            bowl = tcp @ tcp_bowl
            frozen.qpos[adr:adr+3] = bowl[:3, 3]
            mujoco.mju_mat2Quat(frozen.qpos[adr+3:adr+7], bowl[:3, :3].ravel())
            mujoco.mj_forward(model, frozen)
            maximum_tilt = max(maximum_tilt, float(np.rad2deg(np.arccos(
                np.clip(bowl[2, 1], -1., 1.)))))
        contact = insertion_contact_check(
            model, frozen, runtime.robot_metadata,
            allow_pad_bowl_contact=held,
        )
        if not contact['success']:
            return {**contact, 'success': False, 'waypoint_index': index,
                    'status': 'pre_execution_contact_rejected',
                    'failure_category': 'collision_postcheck_rejected',
                    'reason_code': 'forbidden_insert_contact',
                    'maximum_bowl_tilt_deg': maximum_tilt,
                    'maximum_allowed_tilt_deg': max_tilt_deg}
        if max_tilt_deg is not None and maximum_tilt > max_tilt_deg:
            return {**contact, 'success': False, 'waypoint_index': index,
                    'status': 'pre_execution_task_constraint_rejected',
                    'failure_category': 'task_constraint_postcheck_rejected',
                    'reason_code': 'bowl_tilt_limit_exceeded',
                    'maximum_bowl_tilt_deg': maximum_tilt,
                    'maximum_allowed_tilt_deg': max_tilt_deg}
    return {'success': True, 'checked_waypoints': len(plan['positions_rad']),
            'maximum_bowl_tilt_deg': maximum_tilt,
            'source': 'mujoco_geometry_and_actual_hold_transform'}


def raise_retreat_for_released_pad_clearance(
    model, data, robot_metadata, targets: dict, inward_world: np.ndarray, *,
    clearance_margin_m: float = .005,
    minimum_lift_m: float = 0.,
) -> tuple[dict, dict]:
    """按松爪后的实际碰撞几何抬高第一段撤离，避免指垫扫过碗沿。"""
    import mujoco

    inward = np.asarray(inward_world, dtype=float)
    if (inward.shape != (3,) or not np.isfinite(inward).all()
            or not np.isclose(np.linalg.norm(inward), 1.)
            or not np.isfinite(clearance_margin_m) or clearance_margin_m < 0
            or not np.isfinite(minimum_lift_m) or minimum_lift_m < 0):
        raise ValueError('撤离净空需要单位进炉方向、非负有限余量和最小抬升')
    poses = {name: np.asarray(pose, dtype=float).copy()
             for name, pose in targets.items()}
    if set(poses) != {'preplace', 'insert', 'retreat_1', 'retreat_2'}:
        raise ValueError('撤离净空需要完整的四个插入目标')

    bowl_root = model.body('bowl').id
    pads = tuple(robot_metadata['grippers']['right'].get('pad_geom_names', ()))
    if not pads:
        raise ValueError('撤离净空缺少右夹爪指垫碰撞几何')

    def belongs_to_bowl(body_id: int) -> bool:
        while body_id:
            if body_id == bowl_root:
                return True
            body_id = int(model.body_parentid[body_id])
        return False

    def vertical_extent(geom_id: int) -> tuple[float, float]:
        rotation = np.asarray(data.geom_xmat[geom_id], dtype=float).reshape(3, 3)
        size = np.asarray(model.geom_size[geom_id], dtype=float)
        geom_type = int(model.geom_type[geom_id])
        if geom_type == int(mujoco.mjtGeom.mjGEOM_BOX):
            half = float(np.sum(np.abs(rotation[2]) * size))
        elif geom_type == int(mujoco.mjtGeom.mjGEOM_SPHERE):
            half = float(size[0])
        elif geom_type in {
            int(mujoco.mjtGeom.mjGEOM_CAPSULE),
            int(mujoco.mjtGeom.mjGEOM_CYLINDER),
        }:
            half = float(
                size[0] * np.sqrt(max(0., 1. - rotation[2, 2] ** 2))
                + size[1] * abs(rotation[2, 2])
            )
        else:
            half = float(model.geom_rbound[geom_id])
        center = float(data.geom_xpos[geom_id, 2])
        return center - half, center + half

    bowl_geoms = [
        geom_id for geom_id in range(model.ngeom)
        if belongs_to_bowl(int(model.geom_bodyid[geom_id]))
        and (int(model.geom_contype[geom_id]) != 0
             or int(model.geom_conaffinity[geom_id]) != 0)
    ]
    if not bowl_geoms:
        raise ValueError('撤离净空没有找到 bowl 碰撞几何')
    bowl_top = max(vertical_extent(geom_id)[1] for geom_id in bowl_geoms)
    bowl_center = np.asarray(data.body('bowl').xpos, dtype=float)
    pad_rows = []
    for name in pads:
        geom_id = model.geom(name).id
        lower, upper = vertical_extent(geom_id)
        projection = float(
            inward @ (np.asarray(data.geom_xpos[geom_id]) - bowl_center))
        pad_rows.append({
            'geom_name': name,
            'inward_projection_from_bowl_center_m': projection,
            'lower_world_z_m': lower,
            'upper_world_z_m': upper,
        })
    # 沿 -inward 撤离时，最靠炉内/靠碗的指垫会扫过近侧碗沿；另一指垫
    # 已位于出口方向，继续撤离只会远离碗，不能用其较低高度抬高整条路径。
    critical = max(pad_rows, key=lambda row: row[
        'inward_projection_from_bowl_center_m'])
    geometric_required_lift = max(
        0., bowl_top + float(clearance_margin_m) - critical['lower_world_z_m'])
    required_lift = max(geometric_required_lift, float(minimum_lift_m))
    available_lift = float(poses['preplace'][2, 3] - poses['insert'][2, 3])
    if available_lift < -1e-9:
        raise ValueError('preplace 高度低于 insert，无法构造向上撤离通道')
    if required_lift > available_lift + 1e-9:
        raise ValueError(
            f'实际指垫清碗沿需要抬升 {required_lift:.6f}m，超过已验证通道 '
            f'{available_lift:.6f}m')
    retreat_lift = poses['insert'].copy()
    retreat_lift[2, 3] = poses['insert'][2, 3] + required_lift
    poses['retreat_lift'] = retreat_lift
    poses['retreat_1'][2, 3] = retreat_lift[2, 3]
    evidence = {
        'policy': 'released_open_pad_clearance_from_actual_mujoco_geometry',
        'critical_pad': critical,
        'all_pad_extents': pad_rows,
        'bowl_collision_geom_count': len(bowl_geoms),
        'bowl_collision_top_world_z_m': bowl_top,
        'clearance_margin_m': float(clearance_margin_m),
        'geometric_required_lift_m': geometric_required_lift,
        'minimum_lift_m': float(minimum_lift_m),
        'required_lift_m': required_lift,
        'available_preplace_lift_m': available_lift,
        'retreat_lift_world_z_m': float(retreat_lift[2, 3]),
        'retreat_1_world_z_m': float(poses['retreat_1'][2, 3]),
        'retreat_strategy': 'linear_vertical_clearance_then_collision_free_pose_exit',
        'post_clearance_motion_policy': 'full_scene_collision_checked_pose_motion',
        'retreat_2_unchanged_for_downstream_compatibility': True,
    }
    return poses, evidence


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def box_footprint_margins(T_world_region: np.ndarray, half_size_xy_m,
                          centers_world_m, radii_m) -> dict:
    """以每个碰撞体的保守外接球检查区域占用，不能只检查物体中心。"""
    transform = np.asarray(T_world_region, dtype=float)
    centers = np.asarray(centers_world_m, dtype=float)
    radii = np.asarray(radii_m, dtype=float)
    half = np.asarray(half_size_xy_m, dtype=float)
    if transform.shape != (4, 4) or centers.ndim != 2 or centers.shape[1] != 3 \
            or len(centers) == 0 or radii.shape != (len(centers),) \
            or half.shape != (2,) or not np.isfinite(transform).all() \
            or not np.isfinite(centers).all() or not np.isfinite(radii).all() \
            or np.any(radii <= 0) or np.any(half <= 0):
        raise ValueError("区域占用检查需要有限的变换、碰撞中心、半径和长宽")
    local = (transform[:3, :3].T @ (centers - transform[:3, 3]).T).T
    margins = half[None, :] - np.abs(local[:, :2]) - radii[:, None]
    return {"success": bool(np.all(margins >= 0)),
            "minimum_margin_m": float(np.min(margins)),
            "axis_minimum_margin_m": np.min(margins, axis=0).tolist(),
            "geom_count": len(centers)}


def cavity_footprint_margins(model, data, bowl_geom_ids,
                             *, door_closed_angle_rad: float = 0.) -> dict:
    """用实际 Bowl 碰撞形状及关门后的门板内侧，检查炉腔水平占用。

    地板代理可能比门板内侧稍短；用地板边界或每个小碰撞盒的外接球
    会把能关门的碗误判为伸出。这里只在 data 副本中闭门求边界，
    绝不改在线物理状态，也不把关门当成实际执行成功。
    """
    import mujoco

    floor_id = model.geom("microwave_floor_collision").id
    door_id = model.geom("microwave_door_panel_collision").id
    floor = np.asarray(data.geom_xmat[floor_id], dtype=float).reshape(3, 3)
    floor_pos = np.asarray(data.geom_xpos[floor_id], dtype=float)
    lower = -np.asarray(model.geom_size[floor_id, :2], dtype=float).copy()
    upper = -lower.copy()
    closed = mujoco.MjData(model)
    mujoco.mj_copyData(closed, model, data)
    closed.qpos[model.joint("microwave_door_joint").qposadr[0]] = float(
        door_closed_angle_rad)
    mujoco.mj_forward(model, closed)
    door_pos = np.asarray(closed.geom_xpos[door_id], dtype=float)
    door_rot = np.asarray(closed.geom_xmat[door_id], dtype=float).reshape(3, 3)
    door_local = floor.T @ (door_pos - floor_pos)
    front_axis = int(np.argmax(np.abs(door_local[:2]) / upper))
    sign = float(np.sign(door_local[front_axis]))
    if sign == 0:
        raise ValueError("关门后门板无法确定炉腔前沿")
    door_half = float(np.sum(np.abs(floor[:, front_axis] @ door_rot)
                             * model.geom_size[door_id]))
    boundary = float(door_local[front_axis] - sign * door_half)
    if sign > 0:
        upper[front_axis] = boundary
    else:
        lower[front_axis] = boundary
    if np.any(upper <= lower):
        raise ValueError("门板与炉腔地板的边界不构成有效区域")

    margins = []
    for geom_id in bowl_geom_ids:
        index = int(geom_id)
        center = floor.T @ (data.geom_xpos[index] - floor_pos)
        rotation = np.asarray(data.geom_xmat[index], dtype=float).reshape(3, 3)
        extents = []
        for axis in range(2):
            projections = floor[:, axis] @ rotation
            geom_type = int(model.geom_type[index])
            if geom_type == int(mujoco.mjtGeom.mjGEOM_BOX):
                half_extent = np.sum(np.abs(projections) * model.geom_size[index])
            elif geom_type == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
                half_extent = (model.geom_size[index, 0]
                               * np.linalg.norm(projections[:2])
                               + model.geom_size[index, 1] * abs(projections[2]))
            else:
                # 其他碰撞形状暂用保守外接球，不假装已支持精确投影。
                half_extent = model.geom_rbound[index]
            extents.append(float(half_extent))
        margins.append(np.minimum(center[:2] - extents - lower,
                                  upper - center[:2] - extents))
    if not margins:
        raise ValueError("炉腔检查没有 Bowl 碰撞体")
    margins = np.asarray(margins)
    return {"success": bool(np.all(margins >= 0)),
            "minimum_margin_m": float(np.min(margins)),
            "axis_minimum_margin_m": np.min(margins, axis=0).tolist(),
            "geom_count": len(margins),
            "front_boundary_source": "closed_door_panel",
            "front_axis": front_axis,
            "front_sign": sign,
            "front_boundary_local_m": boundary,
            "door_closed_angle_rad": float(door_closed_angle_rad)}


def evaluate_bowl_upright_path(observations: dict, stages: dict, *,
                               max_tilt_deg: float = 15.) -> dict:
    """检查搬运期间的实际逐帧姿态，不能用落稳后的直立掩盖中途翻碗。"""
    if not np.isfinite(max_tilt_deg) or not 0 < max_tilt_deg < 90:
        raise ValueError("碗口倾角阈值必须在 (0, 90) 度内")
    minimum_cos = 1.0
    worst_stage = None
    checked = 0
    missing = []
    groups = [("held_start", [observations.get("held_start")])]
    for name in ("preplace", "insert", "release"):
        row = stages.get(name) or {}
        execution = row.get("execution") if name != "release" else row
        groups.append((name, (execution or {}).get("replay_samples") or []))
    for stage, samples in groups:
        if not samples or any(sample is None or "bowl" not in sample.get("objects", {})
                              for sample in samples):
            missing.append(stage)
            continue
        for sample in samples:
            pose = np.asarray(sample["objects"]["bowl"]["T_world_object"], dtype=float)
            if pose.shape != (4, 4) or not np.isfinite(pose).all():
                missing.append(stage)
                continue
            cosine = float(np.clip(pose[2, 1], -1., 1.))
            checked += 1
            if cosine < minimum_cos:
                minimum_cos = cosine
                worst_stage = stage
    return {
        "success": bool(not missing and checked > 0
                        and minimum_cos >= np.cos(np.deg2rad(max_tilt_deg))),
        "source": "actual_mujoco_replay_samples",
        "maximum_allowed_tilt_deg": max_tilt_deg,
        "maximum_observed_tilt_deg": (float(np.rad2deg(np.arccos(minimum_cos)))
                                      if checked else None),
        "worst_stage": worst_stage,
        "checked_sample_count": checked,
        "missing_stages": sorted(set(missing)),
    }


def evaluate_bowl_retreat_motion(observations: dict, stages: dict, *,
                                 max_translation_m: float = .005) -> dict:
    """逐帧检查松爪后的碗，避免被碰歪后又回到终点附近而漏报。"""
    if not np.isfinite(max_translation_m) or max_translation_m <= 0:
        raise ValueError("撤离期间容许的 Bowl 位移必须为正有限值")
    origin = observations.get("settled", {}).get("objects", {}).get("bowl")
    final = observations.get("retreat_2", {}).get("objects", {}).get("bowl")
    missing = []
    if not origin or not final:
        missing.append("settled_or_final")
        return {"success": False, "source": "actual_mujoco_replay_samples",
                "maximum_allowed_translation_m": max_translation_m,
                "maximum_observed_translation_m": None,
                "final_translation_m": None, "checked_sample_count": 0,
                "missing_stages": missing}
    origin_pose = np.asarray(origin["T_world_object"], dtype=float)
    final_pose = np.asarray(final["T_world_object"], dtype=float)
    if origin_pose.shape != (4, 4) or final_pose.shape != (4, 4) \
            or not np.isfinite(origin_pose).all() or not np.isfinite(final_pose).all():
        raise ValueError("撤离位移检查需要有限的 Bowl 4x4 pose")
    distances = []
    for name in ("retreat_1", "retreat_2"):
        samples = ((stages.get(name) or {}).get("execution") or {}).get(
            "replay_samples") or []
        if not samples:
            missing.append(name)
        for sample in samples:
            bowl = (sample.get("objects") or {}).get("bowl")
            if not bowl:
                missing.append(name)
                continue
            pose = np.asarray(bowl["T_world_object"], dtype=float)
            if pose.shape != (4, 4) or not np.isfinite(pose).all():
                missing.append(name)
                continue
            distances.append(float(np.linalg.norm(
                pose[:3, 3] - origin_pose[:3, 3])))
    maximum = max(distances) if distances else None
    final_distance = float(np.linalg.norm(
        final_pose[:3, 3] - origin_pose[:3, 3]))
    return {
        "success": bool(not missing and maximum is not None
                        and maximum <= max_translation_m
                        and final_distance <= max_translation_m),
        "source": "actual_mujoco_replay_samples",
        "maximum_allowed_translation_m": max_translation_m,
        "maximum_observed_translation_m": maximum,
        "final_translation_m": final_distance,
        "checked_sample_count": len(distances),
        "missing_stages": sorted(set(missing)),
    }


def evaluate_bowl_inside_microwave(runtime, observations: dict,
                                   stages: dict | None = None) -> dict:
    """在实际 MuJoCo 终态检查占用、朝向、支撑、稳定及解除附着。"""
    import mujoco

    model = runtime.model
    data = runtime.copy_data()
    floor_id = model.geom("microwave_floor_collision").id
    turntable_id = model.geom("microwave_turntable_collision").id
    bowl_body_id = model.body("bowl").id
    if int(model.geom_type[floor_id]) != int(mujoco.mjtGeom.mjGEOM_BOX):
        raise ValueError("微波炉地板占用检查需要 box 碰撞体")
    floor = np.eye(4)
    floor[:3, :3] = data.geom_xmat[floor_id].reshape(3, 3)
    floor[:3, 3] = data.geom_xpos[floor_id]
    bowl_geoms = np.flatnonzero(
        (model.geom_bodyid == bowl_body_id)
        & ((model.geom_contype != 0) | (model.geom_conaffinity != 0))
    )
    footprint = box_footprint_margins(
        floor, model.geom_size[floor_id, :2], data.geom_xpos[bowl_geoms],
        model.geom_rbound[bowl_geoms],
    )
    cavity_footprint = cavity_footprint_margins(model, data, bowl_geoms)
    # Bowl_0 的局部 +Y 是碗口朝向；以 world +Z 判断倾斜。该语义应在
    # 更换物体实例时从任务配置取得，不能推广为所有容器的默认轴。
    upright_cos = float(data.body("bowl").xmat.reshape(3, 3)[2, 1])
    max_tilt_deg = 15.0
    upright = upright_cos >= float(np.cos(np.deg2rad(max_tilt_deg)))
    support_contacts = []
    other_contacts = []
    bowl_geom_set = set(int(index) for index in bowl_geoms)
    for index in range(data.ncon):
        contact = data.contact[index]
        first, second = int(contact.geom1), int(contact.geom2)
        if first not in bowl_geom_set and second not in bowl_geom_set:
            continue
        other = second if first in bowl_geom_set else first
        if float(contact.dist) > .002:
            continue
        item = {"geom_name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, other),
                "distance_m": float(contact.dist)}
        (support_contacts if other == turntable_id else other_contacts).append(item)
    early = observations.get("retreat_1")
    late = observations.get("retreat_2")
    if early is None or late is None:
        stable = False
        displacement = None
        duration = None
    else:
        early_pose = np.asarray(early["objects"]["bowl"]["T_world_object"])
        late_pose = np.asarray(late["objects"]["bowl"]["T_world_object"])
        displacement = float(np.linalg.norm(early_pose[:3, 3] - late_pose[:3, 3]))
        duration = float(late["sim_time_s"] - early["sim_time_s"])
        stable = duration >= .3 and displacement <= .005
    detached = not runtime.simulation_attachments()
    transfer_upright = evaluate_bowl_upright_path(observations, stages or {})
    retreat_motion = evaluate_bowl_retreat_motion(observations, stages or {})
    success = bool(cavity_footprint["success"] and upright
                   and transfer_upright["success"]
                   and support_contacts
                   and not other_contacts and stable
                   and retreat_motion["success"] and detached)
    return {
        "success": success, "source": "actual_mujoco_final_state",
        "cavity_footprint": cavity_footprint,
        "footprint": footprint, "bowl_upright_cos": upright_cos,
        "maximum_tilt_deg": max_tilt_deg, "upright": bool(upright),
        "transfer_upright": transfer_upright,
        "retreat_motion": retreat_motion,
        "support_geom_name": "microwave_turntable_collision",
        "support_contacts": support_contacts,
        "other_bowl_contacts": other_contacts,
        "stable_duration_s": duration, "stable_translation_m": displacement,
        "stable_translation_limit_m": .005, "stable": bool(stable),
        "simulation_attachment_released": detached,
    }


def execute_insertion_stages(runtime, bridge, targets: dict, output_dir: Path, *,
                             release_width_m: float, visual_recorder=None,
                             check_hold=None, check_final=None, stop_after=None,
                             prepare_released_retreat=None) -> dict:
    """共享插入编排；低层仅报告动作，完整 Skill 注入实际任务后置检查。"""
    if set(targets) != {"preplace", "insert", "retreat_1", "retreat_2"}:
        raise ValueError("放入动作需要 preplace、insert 和两段撤离 TCP keypose")
    if not np.isfinite(release_width_m) or release_width_m <= 0:
        raise ValueError("松爪开口必须为正有限值")
    output = Path(output_dir)
    report = {"stage_order": list(INSERT_STAGE_ORDER), "stages": {},
              "stage_state_ids": {}, "observations": {},
              "success": False, "failure_stage": None,
              "goal_check_enabled": check_final is not None}

    def observe(stage):
        observation = runtime.query(robot_groups=("right",), object_names=("bowl",))
        report["stage_state_ids"][stage] = observation["state_id"]
        report["observations"][stage] = observation
        if hasattr(runtime, "model") and hasattr(runtime, "copy_data"):
            save_state(runtime.model, runtime.copy_data(), output / "states" / f"{stage}.npz")
        return observation

    state = observe("held_start")
    observed = OperationState(state['state_id'], 'mujoco', state)
    retreat_prepared = False
    def perform(name):
        nonlocal state, observed, retreat_prepared
        if name == 'hold_check':
            return {'success': True, 'check': check_hold()}
        if name == 'release_check':
            detached = not runtime.simulation_attachments()
            return {'success': True, 'check': CheckResult(
                'pass' if detached else 'fail', 'mujoco_attachment_state',
                {'simulation_attachment_released': detached},
                '松爪后仿真固定应已解除；最终支撑由完整 Skill 检查')}
        if name == 'final_check':
            return {'success': True, 'check': check_final(report)}
        if name in targets:
            if name == 'retreat_1' and prepare_released_retreat is not None \
                    and not retreat_prepared:
                adjusted, evidence = prepare_released_retreat(targets, state)
                substages = ('retreat_lift',)
                if set(adjusted) != set(targets) | set(substages):
                    raise ValueError('松爪后撤离调整必须返回完整目标和 retreat_lift')
                targets.update(adjusted)
                report['retreat_target_adjustment'] = evidence
                insertion_index = report['stage_order'].index('retreat_1')
                report['stage_order'][insertion_index:insertion_index] = list(substages)
                retreat_prepared = True
            if name == 'retreat_1' and 'retreat_lift' in targets:
                for substage in ('retreat_lift',):
                    substage_outcome = bridge(
                        substage, targets[substage], state['state_id'])
                    report['stages'][substage] = substage_outcome
                    state = observe(substage)
                    observed = OperationState(state['state_id'], 'mujoco', state)
                    if not substage_outcome.get('success'):
                        outcome = {
                            'success': False,
                            'status': f'blocked_by_{substage}',
                            substage: substage_outcome,
                        }
                        report['stages'][name] = outcome
                        return outcome
            outcome = bridge(name, targets[name], state['state_id'])
            report['stages'][name] = outcome
        elif name == 'release':
            command = dict(group='right', width_m=release_width_m, duration_s=.6,
                           planned_state_id=state['state_id'], timeout_s=1.0,
                           replay_object_names=('bowl',))
            if visual_recorder is not None:
                visual_recorder.begin_stage(name, runtime.model, runtime.data)
                command['step_observer'] = visual_recorder.observer(name)
            try:
                outcome = runtime.execute_gripper_command(**command)
            finally:
                if visual_recorder is not None:
                    visual_recorder.end_stage(name, runtime.model, runtime.data)
            report['stages'][name] = outcome
        elif name == 'detach':
            active = [row for row in runtime.simulation_attachments()
                      if row.get('object_instance_id') == 'bowl']
            if active:
                runtime.release_simulation_object('right', 'bowl')
            outcome = {'success': True, 'mode':
                       'simulation_weld_released' if active else 'no_simulation_weld'}
            report['stages'][name] = outcome
        elif name == 'settle':
            command = dict(duration_s=.5, planned_state_id=state['state_id'],
                           replay_object_names=('bowl',), replay_group='right')
            if visual_recorder is not None:
                visual_recorder.begin_stage(name, runtime.model, runtime.data)
                command['step_observer'] = visual_recorder.observer(name)
            try:
                outcome = runtime.execute_hold(**command)
            finally:
                if visual_recorder is not None:
                    visual_recorder.end_stage(name, runtime.model, runtime.data)
            report['settle'] = outcome
        else:
            raise ValueError(f'未知插入阶段：{name}')
        state = observe('settled' if name == 'settle' else name)
        observed = OperationState(state['state_id'], 'mujoco', state)
        return {**outcome, 'success': bool(outcome.get('success'))}

    def guarded_perform(name):
        nonlocal observed
        try:
            return perform(name)
        except Exception as error:
            row = {'success': False, 'exception_type': type(error).__name__,
                   'message': str(error)}
            if name == 'settle':
                report['settle'] = row
            else:
                report['stages'][name] = {**report['stages'].get(name, {}), **row}
            # 动作可能已部分执行；异常之后不能把发令前状态当成当前事实。
            try:
                current = observe(f'{name}_failure')
                observed = OperationState(current['state_id'], 'mujoco', current)
            except Exception as observation_error:
                observed = OperationState(f'unavailable:{name}', 'unavailable',
                    {'state_unavailable': True, 'reason': str(observation_error)})
            return {'success': False}
    flow = run_transfer_flow(CallbackOperationBackend(
        observe=lambda: observed, perform=guarded_perform, evidence_mode='mujoco', failure_details=lambda stage: report.get('first_failure')),
        verify_hold=check_hold is not None, approve_path=False,
        verify_final=check_final is not None, stop_after=stop_after)
    report['shared_flow'] = flow.to_dict()
    report['requested_scope'] = flow.requested_scope
    report['operation_success'] = flow.success
    report['actual_final'] = flow.to_dict()['end_state']
    report['success'] = flow.success
    report['failure_stage'] = flow.failure_stage
    if flow.success:
        report['final_state_id'] = state['state_id']
    _write_json(output / 'insert_result.json', report)
    return report


def run_physical_insert_bowl(runtime, output_dir: Path, visual_recorder,
                             pick_result: dict, *,
                             insert_depth_offset_m: float = 0.,
                             insert_tcp_table_angle_max_deg: float = 70.,
                             insert_max_bowl_tilt_deg: float = 14.,
                             retreat_clearance_margin_m: float = .005,
                             retreat_minimum_lift_m: float = 0.,
                             retreat_rotation_local_deg=(0., 0., 0.), stop_after=None) -> dict:
    """消费实际 held 状态，以受限预倾斜目标和完整场景避碰执行入炉。"""
    if not pick_result.get("success") or pick_result.get("instance_id") != "bowl":
        raise ValueError("只有实际抓碗阶段成功后才能继续放入")
    output = Path(output_dir) / "physical_insert"
    output.mkdir(parents=True, exist_ok=False)
    observation = runtime.query(robot_groups=("right",), object_names=("bowl",))
    world_tcp = np.asarray(observation["robots"]["right"]["T_world_tcp"], dtype=float)
    world_bowl = np.asarray(observation["objects"]["bowl"]["T_world_object"], dtype=float)
    tcp_bowl = np.linalg.inv(world_tcp) @ world_bowl
    case = PickBowlCase(output, runtime.scene_revision, runtime.model, runtime.data,
                        runtime.metadata, {}, world_bowl, [])
    geometry, alignment = _legacy_place_targets(case, tcp_bowl)
    old_targets = {
        "preplace": geometry.pre_place_tcp,
        "insert": geometry.place_tcp,
        "retreat_1": geometry.post_place_1_tcp,
        "retreat_2": geometry.post_place_2_tcp,
    }
    microwave_body = runtime.data.body("microwave")
    world_microwave = np.eye(4)
    world_microwave[:3, :3] = microwave_body.xmat.reshape(3, 3)
    world_microwave[:3, 3] = microwave_body.xpos
    retreat_rotation = np.asarray(retreat_rotation_local_deg, dtype=float)
    # 旧 profile 只作参考，不能把为某一次抓取标定的 TCP 用于任意抓法。
    # 碗是旋转对称容器；令 TCP→碗心的水平向量指向炉内，让腕部留在炉前。
    inward = world_microwave[:3, 1]
    heading = world_bowl[:3, :3].copy()
    up = np.array([0., 0., 1.])
    x_axis = heading[:, 0] - up * heading[2, 0]
    x_axis /= np.linalg.norm(x_axis)
    heading = np.column_stack((x_axis, up, np.cross(x_axis, up)))
    horizontal_hold = (heading @ tcp_bowl[:3, :3].T @ tcp_bowl[:3, 3])[:2]
    yaw = np.arctan2(inward[1], inward[0]) - np.arctan2(horizontal_hold[1], horizontal_hold[0])
    heading = Rotation.from_euler('z', yaw).as_matrix() @ heading
    nominal_object_goal = np.eye(4)
    nominal_object_goal[:3, :3] = heading
    turntable = runtime.model.geom('microwave_turntable_collision').id
    turn_rotation = runtime.data.geom_xmat[turntable].reshape(3, 3)
    turn_half_height = np.sum(np.abs(turn_rotation[2]) * runtime.model.geom_size[turntable])
    body = runtime.model.body('bowl').id
    geoms = np.flatnonzero((runtime.model.geom_bodyid == body) &
        ((runtime.model.geom_contype != 0) | (runtime.model.geom_conaffinity != 0)))
    # 实际碰撞几何在目标朝向下的最低点；不依赖每次抓取带来的旧 TCP 偏差。
    import mujoco
    def lowest_collision_offset(rotation_world_object: np.ndarray) -> float:
        lower_points = []
        for g in geoms:
            local_rotation = np.empty(9)
            mujoco.mju_quat2Mat(local_rotation, runtime.model.geom_quat[g])
            rotation = rotation_world_object @ local_rotation.reshape(3, 3)
            if int(runtime.model.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
                extent = (runtime.model.geom_size[g, 0] * np.linalg.norm(rotation[2, :2])
                          + runtime.model.geom_size[g, 1] * abs(rotation[2, 2]))
            else:
                extent = np.sum(np.abs(rotation[2]) * runtime.model.geom_size[g])
            lower_points.append(
                (rotation_world_object @ runtime.model.geom_pos[g])[2] - extent
            )
        return float(min(lower_points))
    nominal_object_goal[:3, 3] = runtime.data.geom_xpos[turntable]
    # 碗完整落在转盘前半部，让掌部靠炉口一侧；保留显式深度偏移。
    # 默认 +20 mm 时碗心位于转盘中心前方 25 mm。
    nominal_object_goal[:3, 3] += inward * (float(insert_depth_offset_m) - .045)
    # 目标不是单一固定倾角。按从小到大的必要预倾斜生成候选，稍平的
    # 候选只在更直立的目标与当前炉腔碰撞时才会被采用。
    nominal_tcp = nominal_object_goal @ np.linalg.inv(tcp_bowl)
    nominal_tcp_table_angle = float(np.rad2deg(np.arcsin(
        np.clip(abs(nominal_tcp[2, 2]), 0., 1.))))
    minimum_reachable_angle = max(
        0., nominal_tcp_table_angle - float(insert_max_bowl_tilt_deg))
    angle_limits = np.arange(
        float(insert_tcp_table_angle_max_deg), minimum_reachable_angle, -2.,
    ).tolist()
    angle_limits.append(minimum_reachable_angle)
    candidate_families = []
    seen_rotations = []
    for angle_limit in angle_limits:
        object_goal, tilt_evidence = bounded_insert_tilt_for_tcp_clearance(
            nominal_object_goal, tcp_bowl, inward,
            max_tcp_table_angle_deg=float(angle_limit),
            max_object_tilt_deg=float(insert_max_bowl_tilt_deg),
        )
        if any(np.allclose(object_goal[:3, :3], rotation, atol=1e-9)
               for rotation in seen_rotations):
            continue
        seen_rotations.append(object_goal[:3, :3].copy())
        # 预倾斜会改变碗沿最低点；每个候选按自身姿态重新抬高，保留2mm净空。
        object_goal[2, 3] = (
            runtime.data.geom_xpos[turntable][2] + turn_half_height
            - lowest_collision_offset(object_goal[:3, :3]) + .002
        )
        family_targets = object_relative_insert_targets(object_goal, tcp_bowl, inward)
        family_targets = rotate_retreat_keyposes(family_targets, {
            'retreat_1': retreat_rotation, 'retreat_2': retreat_rotation})
        candidate_families.append({
            'object_goal': object_goal,
            'targets': family_targets,
            'orientation_policy': tilt_evidence,
        })
    if not candidate_families:
        raise RuntimeError('插入姿态候选生成结果为空')
    place_delta = world_microwave[:3, 1] * float(insert_depth_offset_m)
    profile = json.loads((ROOT / "data/task_1/skills/skill_profiles.json").read_text(
        encoding="utf-8"))["pick_bowl_attachment_proxy"]

    def attachment_for_stage(stage: str, _observation: dict):
        if stage not in {"preplace", "insert", "insert_sequence_screen"}:
            return None
        active = [row for row in runtime.simulation_attachments()
                  if row.get('object_instance_id') == 'bowl']
        if active and not (len(active) == 1 and
                (active[0].get('activation_contact') or {}).get('bilateral_pad_contact')):
            raise ValueError(f'{stage} 缺少仿真辅助固定的历史双侧指垫接触证据')
        contact = classify_grasp_contacts(
            query_diagnostics(runtime, body_names=("bowl",))["contacts"],
            runtime.robot_metadata["grippers"]["right"], object_body="bowl",
        )
        if not active and not contact["success"]:
            raise ValueError(f"{stage} 之前没有有效物理夹持或显式仿真固定")
        pose = np.asarray(runtime.query(object_names=("bowl",))[
            "objects"]["bowl"]["T_world_object"], dtype=float)
        local = np.asarray(profile["sphere_centers_object_m"], dtype=float)
        centers = (pose[:3, :3] @ local.T).T + pose[:3, 3]
        spheres = np.column_stack((centers, np.full(len(centers),
                                                   float(profile["sphere_radius_m"]))))
        return {
            "object_instance_id": "bowl", "spheres_world_m": spheres.tolist(),
            "coverage": profile["coverage"],
            "physical_hold_evidence": (
                {"mode": "simulation_privileged_weld",
                 "activation_contact": active[0].get("activation_contact")}
                if active else {"bilateral_pad_contact": True,
                                "contact_count": contact["contact_count"]}
            ),
        }

    bridge = CuroboRuntimeBridge(
        runtime, output, group="right", visual_recorder=visual_recorder,
        attachment_for_stage=attachment_for_stage,
        # 接触任务中的 insert 与脱离碗沿的竖直净空必须保持直线；净空之后
        # 由完整场景碰撞约束的 pose motion 退出，避免强迫局部不可达直线。
        linear_tcp_stages=("insert", "retreat_lift"),
        # 松爪后碗重新成为完整规划障碍物；撤离不再保留指垫接触例外。
        plan_postcheck_for_stage=lambda stage, plan, _: (
            validate_insertion_plan(runtime, stage, plan, tcp_bowl)),
        execution_stop_condition_for_stage=lambda stage, model, data, _: (
            None if (check := insertion_contact_check(
                model, data, runtime.robot_metadata,
                allow_pad_bowl_contact=stage in {'preplace', 'insert'},
            ))['success']
            else {'failure_type': 'forbidden_insert_contact', **check}),
        endpoint_tolerance_rad=.03, settle_timeout_s=1.5,
        replay_object_names=("bowl",),
        # 降低持碗阶段的动态跟踪误差，避免碗底未抬过地板前沿便进入。
        execution_time_scale_by_stage={'preplace': 2.5, 'insert': 3.,
                                       'retreat_lift': 2.,
                                       'retreat_1': 2.,
                                       'retreat_2': 2.},
        execute_rejected_plan_prefixes=True,
    )
    screen = bridge.plan_only_candidate_sequences(
        'insert_sequence_screen',
        tuple((('preplace', family['targets']['preplace']),
               ('insert', family['targets']['insert']))
              for family in candidate_families),
        observation['state_id'],
    )
    _write_json(output / 'insert_sequence_screen.json', screen)
    selected_index = (screen.get('plan') or {}).get('selected_candidate_index')
    if not screen.get('success') or not isinstance(selected_index, int):
        bridge.close()
        report = {
            'success': False,
            'failure_stage': 'insert_sequence_screen',
            'stages': {'insert_sequence_screen': screen},
            'physical_place_success': False,
            'source_pick_state_id': observation['state_id'],
        }
        _write_json(output / 'insert_result.json', report)
        return report
    selected_family = candidate_families[selected_index]
    bridge.preferred_trajectory_by_stage.update(
        (screen.get('plan') or {}).get('selected_stage_trajectories') or {})
    effective_object_goal = selected_family['object_goal']
    targets = selected_family['targets']
    insert_tilt_evidence = selected_family['orientation_policy']
    _write_json(output / "targets.json", {
        "target_policy": "staged_collision_free_bounded_tilt_with_released_vertical_lift",
        "goal_position_source": "turntable_collision_center_and_bowl_collision_bottom",
        "grasp_wrist_policy": "palm_outside_bowl_toward_microwave_front",
        "insert_clearance_orientation_policy": insert_tilt_evidence,
        "selected_candidate_index": selected_index,
        "candidate_orientation_policies": [
            family['orientation_policy'] for family in candidate_families],
        "insert_sequence_screen_ref": "insert_sequence_screen.json",
        "grasp_approach_to_bowl_axis_deg": float(np.rad2deg(np.arccos(
            np.clip(tcp_bowl[2, 1], -1., 1.)))),
        "grasp_axis_unsigned_deviation_deg": float(np.rad2deg(np.arccos(
            np.clip(abs(tcp_bowl[2, 1]), 0., 1.)))),
        "source_state_id": observation["state_id"],
        "T_tcp_bowl_from_actual_hold": tcp_bowl.tolist(),
        "frame_alignment": alignment,
        "keyposes_world_tcp": {name: pose.tolist() for name, pose in targets.items()},
        "original_keyposes_world_tcp": {name: pose.tolist()
                                         for name, pose in old_targets.items()},
        "insert_depth_offset_microwave_y_m": float(insert_depth_offset_m),
        "retreat_rotation_tcp_local_xyz_deg": retreat_rotation.tolist(),
        "insert_depth_offset_world_m": place_delta.tolist(),
        "original_placed_object_world": geometry.placed_object.tolist(),
        "desired_placed_object_world": effective_object_goal.tolist(),
    })
    release_width = float(runtime.robot_metadata["grippers"]["right"][
        "width_range_m"][1])
    def hold_check():
        evidence = attachment_for_stage('insert', {})['physical_hold_evidence']
        return CheckResult('pass', 'mujoco_contact_or_recorded_assistance', evidence)

    def final_check(report):
        check = evaluate_bowl_inside_microwave(
            runtime, report['observations'], report['stages'])
        report['physical_goal_check'] = check
        return CheckResult('pass' if check['success'] else 'fail',
                           check['source'], check)

    try:
        def prepare_released_retreat(current_targets, _state):
            adjusted, evidence = raise_retreat_for_released_pad_clearance(
                runtime.model, runtime.data, runtime.robot_metadata,
                current_targets, inward,
                clearance_margin_m=retreat_clearance_margin_m,
                minimum_lift_m=retreat_minimum_lift_m,
            )
            _write_json(output / 'released_retreat_targets.json', {
                'evidence': evidence,
                'keyposes_world_tcp': {
                    name: pose.tolist() for name, pose in adjusted.items()},
            })
            return adjusted, evidence

        report = execute_insertion_stages(
            runtime, bridge, targets, output, release_width_m=release_width,
            visual_recorder=visual_recorder, check_hold=hold_check, check_final=final_check,
            stop_after=stop_after,
            prepare_released_retreat=prepare_released_retreat,
        )
    finally:
        bridge.close()
    after = runtime.query(robot_groups=("right",), object_names=("bowl",))
    report["actual_final_object_world"] = after["objects"]["bowl"]["T_world_object"]
    if 'physical_goal_check' not in report:
        report['physical_goal_check'] = {'success': None if stop_after and report['success'] else False,
            'code': 'not_checked_debug_scope' if stop_after and report['success'] else 'stage_execution_incomplete'}
    report['physical_place_success'] = bool(
        report['success'] and report['physical_goal_check']['success'])
    report["fully_physical_pick_insert_success"] = bool(
        report["physical_place_success"] and pick_result.get("physical_pick_success")
    )
    report["source_pick_state_id"] = observation["state_id"]
    _write_json(output / "insert_result.json", report)
    return report


__all__ = ["execute_insertion_stages", "run_physical_insert_bowl"]
