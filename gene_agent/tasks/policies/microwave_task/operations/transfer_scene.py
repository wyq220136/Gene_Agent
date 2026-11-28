"""与仿真共用的目标及冻结场景几何；无物理 step，也不提供实测接触事实。"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from gene_agent._compat import strict_zip
from gene_agent.motion.model import PoseTarget
from .bowl import (bounded_insert_tilt_for_tcp_clearance, clearance_adjusted_insert_goal,
                   object_relative_insert_targets, rotate_retreat_keyposes)
from gene_agent.knowledge import bowl_azimuth_for_world_approach, bowl_grasp_candidates

def _write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n")

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
            'source': 'reconstructed_geometry_not_measured_contact'}

def held_turn_tcp(world_bowl, tcp_bowl, yaw_deg):
    """绕实际碗心和世界竖直轴转向，保留当前夹持刚体关系与倾角。"""
    from scipy.spatial.transform import Rotation
    turn_bowl = world_bowl.copy()
    turn_bowl[:3, :3] = Rotation.from_euler('z', yaw_deg, degrees=True).as_matrix() @ world_bowl[:3, :3]
    return turn_bowl @ np.linalg.inv(tcp_bowl)

def validate_insertion_plan(runtime, stage: str, plan: dict, tcp_bowl: np.ndarray,
                            *, max_tilt_deg: float | None = 15.) -> dict:
    """逐点检查实际几何和持碗倾角；不改在线状态或正式规划的碰撞世界。"""
    import mujoco
    frozen = runtime.copy_data()
    model = runtime.model
    held = stage in {'held_turn', 'preplace', 'insert'}
    adr = model.joint('bowl_free').qposadr[0]
    maximum_tilt = 0.
    for index, positions in enumerate(plan['positions_rad']):
        for name, value in strict_zip(plan['joint_names'], positions):
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
            'source': 'reconstructed_geometry_and_observed_hold'}

def _set_frozen_gripper_width(model, data, robot_metadata, group: str,
                              width_m: float) -> None:
    """仅修改诊断副本的夹爪 qpos，用于 release/retreat 前视。"""
    gripper = robot_metadata['grippers'][group]
    low, high = map(float, gripper['width_range_m'])
    width = float(width_m)
    if (gripper.get('mode') != 'parallel_slide_proxy'
            or not np.isfinite(width) or not low <= width <= high):
        raise ValueError('release 前视只支持范围内的 parallel_slide_proxy 开口')
    for name in gripper['joint_names']:
        data.qpos[model.joint(name).qposadr[0]] = width / 2.
    for name in gripper.get('actuator_names', ()):
        data.ctrl[model.actuator(name).id] = width / 2.

def predict_released_insert_state(
    runtime, insert_trajectory: dict, T_world_bowl: np.ndarray,
    release_width_m: float, *, release_samples: int = 9,
):
    """预测 insert 末端的完整松爪包络，并构造开爪、脱离后的规划状态。

    该函数只操作 MuJoCo 数据副本；在线 runtime、历史 run 和正式成功条件
    均不改变。release 期间保留预期的指垫-碗接触，但任何机器人-微波炉或
    碗-非转盘接触仍会淘汰当前插入候选。
    """
    import mujoco

    names = tuple(insert_trajectory.get('joint_names') or ())
    positions = np.asarray(insert_trajectory.get('positions_rad') or (), dtype=float)
    bowl = np.asarray(T_world_bowl, dtype=float)
    if (not names or positions.ndim != 2 or positions.shape[1] != len(names)
            or len(positions) < 2 or not np.isfinite(positions).all()
            or bowl.shape != (4, 4) or not np.isfinite(bowl).all()
            or release_samples < 2):
        raise ValueError('release 前视需要有效 insert 轨迹、碗目标和至少两个采样')

    predicted = runtime.copy_data()
    model = runtime.model
    for name, value in strict_zip(names, positions[-1]):
        predicted.qpos[model.joint(name).qposadr[0]] = value
    bowl_adr = model.joint('bowl_free').qposadr[0]
    predicted.qpos[bowl_adr:bowl_adr + 3] = bowl[:3, 3]
    mujoco.mju_mat2Quat(
        predicted.qpos[bowl_adr + 3:bowl_adr + 7], bowl[:3, :3].ravel())
    attach_name = 'sim_attach_right_bowl'
    attach_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_EQUALITY, attach_name)
    if attach_id >= 0:
        predicted.eq_active[attach_id] = 0
    predicted.qvel[:] = 0.

    gripper = runtime.robot_metadata['grippers']['right']
    start_width = float(sum(
        predicted.qpos[model.joint(name).qposadr[0]]
        for name in gripper['joint_names']))
    widths = np.linspace(start_width, float(release_width_m), release_samples)
    samples = []
    final_data = None
    for index, width in enumerate(widths):
        sample = mujoco.MjData(model)
        mujoco.mj_copyData(sample, model, predicted)
        _set_frozen_gripper_width(
            model, sample, runtime.robot_metadata, 'right', float(width))
        mujoco.mj_forward(model, sample)
        contact = insertion_contact_check(
            model, sample, runtime.robot_metadata,
            allow_pad_bowl_contact=True,
        )
        samples.append({
            'index': index,
            'width_m': float(width),
            'contact_check': contact,
        })
        if not contact['success']:
            return None, {
                'success': False,
                'reason_code': 'release_envelope_forbidden_contact',
                'start_width_m': start_width,
                'release_width_m': float(release_width_m),
                'sample_count': release_samples,
                'first_invalid_release_sample': index,
                'samples': samples,
                'source': 'predicted_insert_endpoint_mujoco_geometry',
            }
        final_data = sample
    assert final_data is not None
    released_runtime = FrozenGeometry(model, final_data, runtime.robot_metadata)
    return released_runtime, {
        'success': True,
        'start_width_m': start_width,
        'release_width_m': float(release_width_m),
        'sample_count': release_samples,
        'samples': samples,
        'attachment_active_after_prediction': False,
        'source': 'predicted_insert_endpoint_mujoco_geometry',
    }

def raise_retreat_for_released_pad_clearance(
    model, data, robot_metadata, targets: dict, inward_world: np.ndarray, *,
    clearance_margin_m: float = .005,
    minimum_lift_m: float = 0.,
    outward_fraction: float = 0.,
) -> tuple[dict, dict]:
    """按松爪后的实际几何抬高并可同步外撤，避免指垫扫过碗沿。"""
    import mujoco

    inward = np.asarray(inward_world, dtype=float)
    if (inward.shape != (3,) or not np.isfinite(inward).all()
            or not np.isclose(np.linalg.norm(inward), 1.)
            or not np.isfinite(clearance_margin_m) or clearance_margin_m < 0
            or not np.isfinite(minimum_lift_m) or minimum_lift_m < 0
            or not np.isfinite(outward_fraction)
            or not 0. <= outward_fraction <= 1.):
        raise ValueError(
            '撤离净空需要单位进炉方向、非负有限余量和最小抬升，'
            '外撤比例必须在 [0, 1]')
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
    outward_delta = poses['retreat_1'][:3, 3] - poses['insert'][:3, 3]
    outward_delta[2] = 0.
    retreat_lift[:3, 3] += float(outward_fraction) * outward_delta
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
        'outward_fraction': float(outward_fraction),
        'outward_translation_m': float(
            np.linalg.norm(float(outward_fraction) * outward_delta)),
        'retreat_lift_world_z_m': float(retreat_lift[2, 3]),
        'retreat_1_world_z_m': float(poses['retreat_1'][2, 3]),
        'retreat_strategy': (
            'linear_vertical_clearance_then_collision_free_pose_exit'
            if outward_fraction == 0.
            else 'linear_diagonal_outward_clearance_then_collision_free_pose_exit'
        ),
        'post_clearance_motion_policy': 'full_scene_collision_checked_pose_motion',
        'retreat_2_unchanged_for_downstream_compatibility': True,
    }
    return poses, evidence

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

def released_ceiling_clearance_snapshot(snapshot: dict, margin_m: float, output_path=None) -> dict:
    """释放前瞻保留炉顶净空，避免名义末端刚好可行而实测落稳态起点碰撞。"""
    from copy import deepcopy
    import hashlib
    if not np.isfinite(margin_m) or margin_m < 0:
        raise ValueError("释放前瞻净空必须非负有限")
    result = deepcopy(snapshot)
    for row in result["records"]:
        if row["geom_name"] != "microwave_ceiling_collision":
            continue
        box = result["scene_model"]["cuboid"][row["obstacle_name"]]
        # 对炉顶所有边界保留空间余量，包括炉口前缘；只收紧预测世界。
        box["dims"] = [float(d) + 2. * margin_m for d in box["dims"]]
        row["prediction_clearance_margin_m"] = float(margin_m)
    result["source_snapshot_sha256"] = snapshot["snapshot_sha256"]
    result["prediction_ceiling_clearance_margin_m"] = float(margin_m)
    result["scope"] += "; release prediction ceiling clearance margin"
    result["snapshot_sha256"] = hashlib.sha256(
        json.dumps(result["scene_model"], sort_keys=True).encode()).hexdigest()
    if output_path is not None:
        _write_json(Path(output_path), result)
    return result
class FrozenGeometry:
    """仅做 FK/碰撞几何检查的数据副本，不提供执行及物理 step 接口。"""
    def __init__(self, model, data, robot_metadata):
        self.model, self.data, self.robot_metadata = model, data, robot_metadata
        self.metadata = {"robot": robot_metadata}
        self.scene_revision = 0
    def copy_data(self):
        import mujoco
        result = mujoco.MjData(self.model)
        mujoco.mj_copyData(result, self.model, self.data)
        return result

def insert_candidate_families(runtime, world_bowl, tcp_bowl, *,
        insert_depth_offset_m=.02, insert_tcp_table_angle_max_deg=70.,
        insert_max_bowl_tilt_deg=14., retreat_clearance_margin_m=.005,
        retreat_rotation_local_deg=(0., 0., 0.)):
    """同仿真候选顺序：必要预倾斜、支撑内净空修正、两侧45度持碗转向。"""
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
    # 原候选优先；为顶板前沿卡腕的持物姿态补充几何推导的支撑内目标。
    # 不改抓取、不降低转盘净空，也不把碰撞检查失败直接当成可执行目标。
    from gene_agent.motion.curobo.core import load_realman_motion_config
    motion_config = load_realman_motion_config('right', runtime.robot_metadata)
    tcp_from_link = np.linalg.inv(np.asarray(motion_config['link_from_tcp']))
    wrist_spheres = motion_config['robot_config']['kinematics']['collision_spheres']['link_7']
    spheres_tcp = [
        [*(tcp_from_link[:3, :3] @ sphere['center'] + tcp_from_link[:3, 3]), sphere['radius']]
        for sphere in wrist_spheres if sphere['radius'] > 0
    ]
    ceiling_id = runtime.model.geom('microwave_ceiling_collision').id
    world_ceiling = np.eye(4)
    world_ceiling[:3, :3] = runtime.data.geom_xmat[ceiling_id].reshape(3, 3)
    world_ceiling[:3, 3] = runtime.data.geom_xpos[ceiling_id]
    adapted_families = []
    for family in candidate_families:
        probe = runtime.copy_data()
        adr = runtime.model.joint('bowl_free').qposadr[0]
        goal = family['object_goal']
        probe.qpos[adr:adr+3] = goal[:3, 3]
        mujoco.mju_mat2Quat(probe.qpos[adr+3:adr+7], goal[:3, :3].ravel())
        mujoco.mj_forward(runtime.model, probe)
        cavity = cavity_footprint_margins(runtime.model, probe, geoms)
        # 整碗仍可关门；底面中心及完整底半径仍在转盘投影内，预留原5mm余量。
        front_axis = cavity['front_axis']
        cavity_room = cavity['axis_minimum_margin_m'][front_axis]
        turn_radius = float(np.sum(np.abs(inward @ turn_rotation) * runtime.model.geom_size[turntable]))
        bowl_bottom = runtime.model.geom('bowl_bottom').id
        bottom_radius = float(runtime.model.geom_size[bowl_bottom, 0])
        support_room = (float(np.dot(goal[:3, 3] - runtime.data.geom_xpos[turntable], inward))
                        + turn_radius - bottom_radius)
        maximum_outward = max(0., min(cavity_room, support_room) - retreat_clearance_margin_m)
        adjusted_goal, clearance_policy = clearance_adjusted_insert_goal(
            goal, tcp_bowl, inward, spheres_tcp, world_ceiling,
            runtime.model.geom_size[ceiling_id] + runtime.model.geom_margin[ceiling_id],
            margin_m=retreat_clearance_margin_m, max_outward_m=maximum_outward,
        )
        family['clearance_policy'] = {**clearance_policy, 'adjusted': False,
                                      'outward_distance_m': 0., 'candidate_variant': 'nominal'}
        if clearance_policy['adjusted']:
            adjusted_targets = object_relative_insert_targets(adjusted_goal, tcp_bowl, inward)
            adjusted_targets = rotate_retreat_keyposes(adjusted_targets, {
                'retreat_1': retreat_rotation, 'retreat_2': retreat_rotation})
            adapted_families.append({
                'object_goal': adjusted_goal, 'targets': adjusted_targets,
                'orientation_policy': family['orientation_policy'],
                'clearance_policy': {**clearance_policy, 'support_check': cavity},
            })
    candidate_families.extend(adapted_families)
    # 炉外先在实际碗心绕世界竖直轴转向，保持碗口倾角和相对夹持位姿。
    # 末轴只用于优先排序；由全臂求解整段运动，禁止直接修改关节或夹持状态。
    wrist_name = runtime.robot_metadata['joint_groups']['right'][-1]
    wrist = runtime.model.joint(wrist_name)
    midpoint = float(np.mean(wrist.range))
    wrist_position = float(runtime.data.qpos[wrist.qposadr[0]])
    vertical_projection = float(runtime.data.xaxis[wrist.id, 2])
    preferred_sign = np.sign((midpoint - wrist_position) * vertical_projection) or 1.
    original_families = tuple(candidate_families)
    for direction in (preferred_sign, -preferred_sign):
        turn_tcp = held_turn_tcp(world_bowl, tcp_bowl, direction * 45.)
        candidate_families.extend({**family, 'held_turn': turn_tcp} for family in original_families)
    if not candidate_families:
        raise RuntimeError('插入姿态候选生成结果为空')
    return candidate_families

def bowl_exterior_clearance(runtime) -> dict:
    """用 Bowl 全部碰撞体的保守外接球检查它是否完全越过炉门前沿。

    前沿由关门姿态的门板与炉腔地板确定，只在 MuJoCo 副本中计算；
    不改变实际门角。正余量表示整个碰撞代理已退出，而非仅碗中心退出。
    """
    model, data = runtime.model, runtime.data
    bowl_id = model.body("bowl").id
    geom_ids = [index for index in range(model.ngeom)
                if int(model.geom_bodyid[index]) == bowl_id
                and int(model.geom_contype[index]) != 0]
    if not geom_ids:
        raise ValueError("取碗终点缺少 Bowl 碰撞体")
    cavity = cavity_footprint_margins(model, data, geom_ids)
    floor_id = model.geom("microwave_floor_collision").id
    axis = int(cavity["front_axis"])
    sign = float(cavity["front_sign"])
    boundary = float(cavity["front_boundary_local_m"])
    floor_rotation = np.asarray(data.geom_xmat[floor_id]).reshape(3, 3)
    floor_origin = np.asarray(data.geom_xpos[floor_id])
    projections = [float(floor_rotation[:, axis] @
                         (data.geom_xpos[index] - floor_origin))
                   for index in geom_ids]
    if sign < 0:
        nearest_to_cavity = max(
            value + float(model.geom_rbound[index])
            for index, value in zip(geom_ids, projections))
        clearance = boundary - nearest_to_cavity
    else:
        nearest_to_cavity = min(
            value - float(model.geom_rbound[index])
            for index, value in zip(geom_ids, projections))
        clearance = nearest_to_cavity - boundary
    return {
        "clearance_m": float(clearance),
        "front_axis": axis,
        "front_sign": sign,
        "front_boundary_local_m": boundary,
        "nearest_bowl_proxy_local_m": nearest_to_cavity,
        "bowl_collision_geom_count": len(geom_ids),
        "method": "all_bowl_collision_geom_bounding_spheres_vs_closed_door_front",
    }

def actual_exterior_retreat_target(runtime, current, nominal_target, previous_target,
                                  *, group, required_clearance_m):
    """与实际验收使用相同炉口前沿及全部碗碰撞geom，消费抬升后的实测持物状态。"""
    from types import SimpleNamespace
    from gene_agent.motion.curobo.runtime_bridge import _pose_error
    data = runtime.copy_data()
    boundary = bowl_exterior_clearance(SimpleNamespace(model=runtime.model, data=data))
    floor_id = runtime.model.geom("microwave_floor_collision").id
    floor_rotation = np.asarray(data.geom_xmat[floor_id]).reshape(3, 3)
    outward = float(boundary["front_sign"]) * floor_rotation[:, int(boundary["front_axis"])]
    plane_offset = (float(outward @ data.geom_xpos[floor_id])
                    + float(boundary["front_sign"]) * boundary["front_boundary_local_m"])
    bowl_id = runtime.model.body("bowl").id
    ids = [i for i in range(runtime.model.ngeom)
           if int(runtime.model.geom_bodyid[i]) == bowl_id and int(runtime.model.geom_contype[i]) != 0]
    actual = np.asarray(current["robots"][group]["T_world_tcp"], dtype=float)
    observed_position, observed_rotation = _pose_error(np.asarray(previous_target), actual)
    # LINEAR_TCP既有5mm端点/.05rad朝向规划预算，外加上一段实测终态偏差。
    # 后者是观测，不声称后续控制误差必被它界定；实际5mm净空仍为最终门控。
    target, evidence = exterior_envelope_retreat_target(
        actual, nominal_target, data.geom_xpos[ids], runtime.model.geom_rbound[ids],
        outward, plane_offset, required_clearance_m=required_clearance_m,
        position_budget_m=.005 + observed_position,
        rotation_budget_rad=.05 + observed_rotation,
    )
    return target, {**evidence, "source_state_id": current["state_id"],
                    "clearance_geometry": boundary,
                    "planning_endpoint_position_budget_m": .005,
                    "planning_endpoint_rotation_budget_rad": .05,
                    "preceding_lift_observed_position_error_m": observed_position,
                    "preceding_lift_observed_rotation_error_rad": observed_rotation}

def object_relative_table_target(runtime, options, bowl, tcp):
    """在炉前桌面区域放下直立的碗；由真实夹持关系反算腕部姿态。"""
    up = np.array([0., 0., 1.])
    x = bowl[:3, 0] - up * bowl[2, 0]
    x /= np.linalg.norm(x)
    rotation = np.column_stack((x, up, np.cross(x, up)))
    yaw = np.deg2rad(float(options['yaw_offset_deg']))
    rotation = np.array([[np.cos(yaw), -np.sin(yaw), 0.],
                         [np.sin(yaw), np.cos(yaw), 0.], [0., 0., 1.]]) @ rotation
    bottom = []
    body_id = runtime.model.body('bowl').id
    for g in range(runtime.model.ngeom):
        if (runtime.model.geom_bodyid[g] != body_id or
                not (runtime.model.geom_contype[g] or runtime.model.geom_conaffinity[g])):
            continue
        local = np.empty(9)
        mujoco.mju_quat2Mat(local, runtime.model.geom_quat[g])
        r = rotation @ local.reshape(3, 3)
        size = runtime.model.geom_size[g]
        if runtime.model.geom_type[g] == mujoco.mjtGeom.mjGEOM_CYLINDER:
            extent = size[0] * np.linalg.norm(r[2, :2]) + size[1] * abs(r[2, 2])
        else:
            extent = np.sum(np.abs(r[2]) * size)
        bottom.append(float((rotation @ runtime.model.geom_pos[g])[2] - extent))
    if not bottom:
        raise ValueError('桌面放置需要实际碗碰撞几何')
    table = runtime.data.geom('scene_table')
    normal = table.xmat.reshape(3, 3)[:, 2]
    top = float((table.xpos + normal * runtime.model.geom('scene_table').size[2])[2])
    goal = np.eye(4)
    goal[:3, :3] = rotation
    # 碗绕直立轴转动不会改变任务语义；调整heading避开腕部折回基座。
    # 这是显式任务工作区参数，不从历史成功轨迹/seed读取关节或目标。
    goal[:2, 3] = np.asarray(options['table_xy_world_m'], dtype=float)
    clearance = float(options['release_clearance_m'])
    if not np.isfinite(goal[:2, 3]).all() or not 0 < clearance <= .005:
        raise ValueError('桌面放置区域及松爪高度配置无效')
    goal[2, 3] = top - min(bottom) + clearance
    attachment = np.linalg.inv(tcp) @ bowl
    target = PoseTarget('world_tcp', goal @ np.linalg.inv(attachment))
    return target.matrix, attachment, {
        'policy': 'upright_object_goal_from_actual_grasp_in_front_table_region',
        'xy_source': 'physical_table_place.object_goal.table_xy_world_m',
        'yaw_offset_deg': float(options['yaw_offset_deg']),
        'desired_bowl_goal': goal.tolist(),
        'T_tcp_bowl_from_actual_hold': attachment.tolist(),
        'collision_bottom_local_z_m': min(bottom),
        'table_top_world_z_m': top,
        'release_clearance_m': clearance,
    }


def actual_bowl_extract_targets(
    bowl_world, tcp_world, pad_center_world, inward_world, *, tilt_deg=-45.,
) -> tuple[dict[str, np.ndarray], dict]:
    """从炉口侧斜抓碗缘；X开口、Z接近，倾斜绕指垫中心进行。

    bowl pose来自在线感知/GT；pad支点来自机器人几何，不使用成功轨迹。
    方位由进炉方向在实际圆环平面内连续求解，避免碗的任意绕轴姿态
    改变抓取方向；仍保留实际碗倾角和原接近/接触参数。
    """
    bowl = np.asarray(bowl_world, dtype=float)
    tcp = np.asarray(tcp_world, dtype=float)
    inward = np.array(inward_world, dtype=float, copy=True)
    azimuth, azimuth_alignment = bowl_azimuth_for_world_approach(
        bowl, inward, tip_y_rad=float(np.deg2rad(tilt_deg)))
    inward = np.asarray(azimuth_alignment["world_approach_direction"])
    pivot = tcp[:3, :3].T @ (np.asarray(pad_center_world) - tcp[:3, 3])
    candidates = bowl_grasp_candidates(bowl, parameters={
        "azimuths_rad": [azimuth],
        "tip_y_rad": [float(np.deg2rad(tilt_deg))],
        "radial_offset_m": .006,
        "tcp_world_offset_m": [0., 0., -.012],
        "tilt_pivot_tcp_m": pivot.tolist(),
    })
    chosen = candidates[0]
    grasp = np.asarray(chosen["T_world_tcp"])
    pre2 = grasp.copy()
    pre2[:3, 3] -= grasp[:3, 2] * .045
    pre1 = pre2.copy()
    pre1[:3, 3] -= inward * .15
    return {"pre_extract_1": pre1, "pre_extract_2": pre2, "grasp": grasp}, {
        "target_policy": "actual_bowl_rim_contact_pivot",
        "selected_candidate": chosen,
        "azimuth_alignment": azimuth_alignment,
        "inward_world": inward.tolist(),
        "pad_pivot_tcp_m": pivot.tolist(),
        "pose_source": "online_actual_bowl_pose",
    }

def actual_hold_retreat_targets(tcp_world, inward_world) -> dict[str, np.ndarray]:
    """先小幅抬离转盘，再保持实际夹持姿态完全退到炉口外。"""
    start = np.asarray(tcp_world, dtype=float)
    inward = np.array(inward_world, dtype=float, copy=True)
    inward /= np.linalg.norm(inward)
    lift = start.copy()
    lift[2, 3] += .018
    outside = lift.copy()
    outside[:3, 3] -= inward * .19
    return {"post_extract_1": lift, "post_extract_2": outside}

def exterior_envelope_retreat_target(
    actual_tcp_world, nominal_target_world, centers_world, radii_m,
    outward_world, front_plane_offset_m, *, required_clearance_m,
    position_budget_m, rotation_budget_rad,
) -> tuple[np.ndarray, dict]:
    """把名义目标沿炉口外法向延长，使全部持物外接球留出声明余量。

    球心通过当前TCP持物变换预测到名义终点；球半径旋转不变。
    位姿误差余量只用于目标构造，不能替代实际执行后的净空验收。
    """
    actual = np.asarray(actual_tcp_world, dtype=float)
    nominal = np.asarray(nominal_target_world, dtype=float)
    centers = np.asarray(centers_world, dtype=float)
    radii = np.asarray(radii_m, dtype=float)
    outward = np.asarray(outward_world, dtype=float)
    scalars = [front_plane_offset_m, required_clearance_m,
               position_budget_m, rotation_budget_rad]
    if (actual.shape != (4, 4) or nominal.shape != (4, 4)
            or centers.ndim != 2 or centers.shape[1] != 3 or len(centers) == 0
            or radii.shape != (len(centers),) or outward.shape != (3,)
            or not all(np.isfinite(x).all() for x in [actual, nominal, centers, radii, outward])
            or not np.isfinite(scalars).all()
            or not np.isclose(np.linalg.norm(outward), 1.)
            or np.any(radii < 0) or required_clearance_m < 0
            or position_budget_m < 0 or not 0 <= rotation_budget_rad < np.pi):
        raise ValueError("撤出包络、炉口单位法向与误差预算必须有效")
    local = (actual[:3, :3].T @ (centers - actual[:3, 3]).T).T
    predicted = (nominal[:3, :3] @ local.T).T + nominal[:3, 3]
    clearance = predicted @ outward - float(front_plane_offset_m) - radii
    rotation_displacement = 2. * np.linalg.norm(local, axis=1) * np.sin(rotation_budget_rad / 2.)
    robust = clearance - float(position_budget_m) - rotation_displacement
    extension = max(0., float(required_clearance_m) - float(np.min(robust)))
    target = nominal.copy()
    target[:3, 3] += outward * extension
    return target, {
        "policy": "extend_nominal_retreat_from_actual_held_full_collision_envelope",
        "outward_world": outward.tolist(), "front_plane_offset_world_m": float(front_plane_offset_m),
        "collision_geom_count": len(centers), "required_actual_clearance_m": float(required_clearance_m),
        "position_budget_m": float(position_budget_m), "rotation_budget_rad": float(rotation_budget_rad),
        "max_rotation_envelope_displacement_m": float(np.max(rotation_displacement)),
        "nominal_predicted_clearance_m": float(np.min(clearance)),
        "nominal_budgeted_clearance_m": float(np.min(robust)),
        "extension_m": extension, "target_budgeted_clearance_m": float(np.min(robust) + extension),
        "nominal_target_world_tcp": nominal.tolist(), "actual_start_world_tcp": actual.tolist(),
        "claim": "目标余量不是执行误差硬保证；仍须全路径避碰和实际物体包络净空验收",
    }
