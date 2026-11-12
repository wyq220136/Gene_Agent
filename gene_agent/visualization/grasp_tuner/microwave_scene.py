"""微波炉 Knowledge 的场景校准与纯位姿 IK 结果，不依赖 Viser。"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import mujoco
from scipy.spatial.transform import Rotation

from gene_agent.knowledge.geometry import CylinderConcept, load_concept
from gene_agent.knowledge.grasp import generate_grasps, parse_grasp_profiles
from gene_agent.knowledge.operation import (
    AxisSpec,
    OperationPath,
    OperationWaypoint,
    generate_handle_pull_candidates,
    generate_handle_pull_path,
    generate_rotation_path,
    knob_axis_from_concept,
)
from gene_agent.knowledge.operation.door_push import (
    generate_mode_door_push_candidate_sets,
    generate_profile_door_push_candidate_sets,
    make_door_push_precontact,
    select_door_push_path,
)
from .pose_ik import MuJoCoPoseIK, PoseIKResult


def _make_curobo_pose_ik(case):
    """显式选择 cuRobo 时才加载具体适配，便于纯几何独立导入。"""
    from .curobo_pose_ik import CuroboPoseIK
    return CuroboPoseIK(case)


@dataclass(frozen=True)
class MicrowaveKnowledge:
    """一个冻结场景中的 grasp 和 operation 检查结果。"""

    microwave_pose_world: np.ndarray
    handle_pose_world: np.ndarray
    knob_pose_world: np.ndarray
    handle_grasp: object
    knob_grasps: tuple
    knob_grasp: object
    handle_grasp_ik: PoseIKResult
    handle_pull_candidates: tuple
    handle_pull_selected_candidate_id: str
    handle_pull_path: object
    handle_pull_ik: tuple[PoseIKResult, ...]
    handle_pull_labels: tuple[str, ...]
    handle_pull_widths_m: tuple[float, ...]
    knob_grasp_ik: tuple[PoseIKResult, ...]
    door_candidate_sets: tuple
    door_selected_candidate_ids: tuple[str, ...]
    door_path: object
    knob_path: object
    door_ik: tuple[PoseIKResult, ...]
    knob_ik: tuple[PoseIKResult, ...]
    door_waypoint_parameters: tuple[dict, ...]
    door_push_labels: tuple[str, ...]
    settings: dict
    door_push_parameters: dict


def _body_pose(data, name: str) -> np.ndarray:
    body = data.body(name)
    pose = np.eye(4)
    pose[:3, :3] = body.xmat.reshape(3, 3)
    pose[:3, 3] = body.xpos
    return pose


def set_preview_microwave_pose(case, pose_world: np.ndarray) -> None:
    """只在已加载的预览模型中移动微波炉和垫台，然后重算所有子部件 FK。

    微波炉与垫台均是 world 下的固定 body；门和旋钮是微波炉子 body。
    因此必须更新 MuJoCo 编译模型的根 body 位姿并执行 mj_forward，
    不能仅平移 Viser mesh，否则 Knowledge、IK 与画面会使用不同场景。
    每次以当前姿态计算垫台的相对位姿，连续拖拽也不会累积错位。
    """
    pose = np.asarray(pose_world, dtype=float)
    if pose.shape != (4, 4) or not np.isfinite(pose).all() or not np.allclose(pose[3], [0, 0, 0, 1]):
        raise ValueError("微波炉预览位姿必须是有限的 4×4 齐次矩阵")
    rotation = pose[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(rotation), 1, atol=1e-5):
        raise ValueError("微波炉预览位姿的旋转矩阵无效")
    model, data = case.model, case.data
    microwave = model.body("microwave")
    support = model.body("microwave_support")
    if int(model.body_parentid[microwave.id]) != 0 or int(model.body_parentid[support.id]) != 0:
        raise ValueError("预览摆位要求微波炉与垫台都是 world 下的固定 body")
    old_microwave = _body_pose(data, "microwave")
    old_support = _body_pose(data, "microwave_support")
    support_pose = pose @ np.linalg.inv(old_microwave) @ old_support
    microwave.pos[:] = pose[:3, 3]
    microwave.quat[:] = Rotation.from_matrix(rotation).as_quat()[[3, 0, 1, 2]]
    support.pos[:] = support_pose[:3, 3]
    support.quat[:] = Rotation.from_matrix(support_pose[:3, :3]).as_quat()[[3, 0, 1, 2]]
    mujoco.mj_forward(model, data)


def build_microwave_knowledge(
    root: str | Path, case, *, handle_parameters: dict | None = None,
    knob_parameters: dict | None = None, door_push_parameters: dict | None = None,
    door_push_waypoint_profile: dict | None = None,
    door_target_angle_rad: float | None = None, knob_target_angle_rad: float | None = None,
    door_start_angle_rad: float | None = None,
    door_push_mode: str | None = None,
    handle_pull_target_angle_rad: float | None = None,
    handle_pull_open_width_m: float = .08,
    handle_pull_close_width_m: float = .015,
    handle_pull_pregrasp_distance_m: float = .03,
    handle_pull_retreat_distance_m: float = .015,
    handle_pull_retreat_up_m: float = 0.,
    handle_pull_selected_candidate_id: str | None = None,
    handle_pull_candidate_rank: int = 0,
    door_push_pre_contact_distance_m: float = .05,
    door_push_retract_up_m: float | None = None,
    door_push_retract_local_z_m: float | None = None,
    door_push_selected_candidate_ids: tuple[str, ...] | None = None,
    door_push_start_candidate_rank: int = 0,
    selected_knob_index: int = 0,
    ik_scope: str = "all", selected_mode: str | None = None,
    selected_waypoint: int = 0,
    ik_backend: str = "mujoco",
) -> MicrowaveKnowledge:
    """从校准 URDF 与 task 配置生成可检查的 TCP 姿态和绕轴路径。

    旧采集 Concept 的 part pose 属于旧重建资产坐标，不能直接用于当前校准
    URDF。这里仅在可视化装配层把把手对齐当前 grip visual、把旋钮对齐
    场景元数据；通用 Knowledge 生成器及原始 grasp profile 都不被改写。
    """
    root = Path(root).resolve()
    settings = json.loads((root / "data/task_1/knowledge/microwave_knowledge_view.json").read_text(encoding="utf-8"))
    if settings.get("schema_version") != 1:
        raise ValueError("microwave_knowledge_view.json 的 schema_version 必须为 1")
    if ik_scope not in {"all", "selected", "none"}:
        raise ValueError("ik_scope 只能为 all、selected 或 none")
    if ik_backend not in {"mujoco", "curobo"}:
        raise ValueError("ik_backend 只能为 mujoco 或 curobo")
    if ik_scope == "selected" and selected_mode not in {"handle_grasp", "handle_pull", "door_push", "knob_grasp", "knob_turn"}:
        raise ValueError("selected 模式必须指定当前检查内容")
    microwave_pose = _body_pose(case.data, "microwave")
    raw_profiles = json.loads((root / "data/task_1/knowledge/grasp_profiles.json").read_text(encoding="utf-8"))
    profiles_json = copy.deepcopy(raw_profiles)
    for name, parameters in settings["profile_overrides"].items():
        profiles_json["profiles"][name]["parameters"].update(parameters)
    if handle_parameters:
        target = profiles_json["profiles"][settings["handle_profile"]]["parameters"]
        target.update({name: {"values": [float(value)]} for name, value in handle_parameters.items()})
    if knob_parameters:
        target = profiles_json["profiles"][settings["knob_profile"]]["parameters"]
        for name, value in knob_parameters.items():
            if name == "azimuth_shift_deg":
                values = target["azimuth_deg"]["values"]
                target["azimuth_deg"] = {"values": [(float(item) + float(value)) % 360 for item in values]}
            else:
                target[name] = {"values": [float(value)]}
    profiles = parse_grasp_profiles(profiles_json)

    # 把手主抓持段的中心由 URDF visual geom 给出；局部 +Y 沿竖直握持段，
    # 局部 +Z 从门板指向用户。这个映射只描述当前资产与 Concept 的对应关系。
    handle_geom = case.data.geom(settings["handle_visual_geom"])
    handle_original = load_concept(root / "assets/objects/microwave_concept.json", part_name="handle").concept
    handle_rotation = np.column_stack((microwave_pose[:3, 0], microwave_pose[:3, 2], -microwave_pose[:3, 1]))
    grip_offset = float(handle_original.grip_segment.pose_in_trifold[2, 3])
    handle_pose = np.eye(4)
    handle_pose[:3, :3] = handle_rotation
    handle_pose[:3, 3] = handle_geom.xpos - handle_rotation[:, 2] * grip_offset
    from dataclasses import replace
    handle = replace(handle_original, pose_in_object=np.linalg.inv(microwave_pose) @ handle_pose)

    # 旋钮 part pose 与半径直接取校准 URDF 的场景快照；采集 Concept 的旧
    # pose 会把夹爪画到机身内部，不能作为这个 run 的几何真值。
    knob_data = case.metadata["objects"]["knob_concept"]
    knob = CylinderConcept(knob_data["size"][:2], knob_data["size"][2], knob_data["pose_matrix"])
    knob_pose = microwave_pose @ knob.pose_in_object
    knob_grasps = generate_grasps(knob, microwave_pose, profiles[settings["knob_profile"]], part_name="knob_0").proposals
    handle_grasps = generate_grasps(handle, microwave_pose, profiles[settings["handle_profile"]], part_name="handle").proposals
    if len(handle_grasps) != 1 or not knob_grasps:
        raise ValueError("展示配置必须生成一个把手候选和至少一个旋钮候选")
    if not 0 <= selected_knob_index < len(knob_grasps):
        raise ValueError("旋钮候选序号超出当前参数生成范围")
    handle_grasp, knob_grasp = handle_grasps[0], knob_grasps[selected_knob_index]

    robot = case.metadata["robot"]
    group = case.snapshot["active_group"]
    if ik_scope == "none":
        solver = None
    elif ik_backend == "curobo":
        # 纯几何和 MuJoCo IK 不应因模块导入而依赖 cuRobo。
        # 只在用户显式选择 cuRobo 诊断时加载该适配。
        solver = _make_curobo_pose_ik(case)
    else:
        solver = MuJoCoPoseIK(case.model, case.data, tuple(robot["joint_groups"][group]), robot["tcp_sites"][group])

    def inspect(pose, *, joint_name=None, angle_rad=None, opening_width_m=None):
        if solver is None:
            return None
        if ik_backend == "curobo":
            return solver.solve(pose, joint_name=joint_name, angle_rad=angle_rad,
                                opening_width_m=opening_width_m)
        return solver.solve(pose)

    door_joint = case.model.joint(settings["door_joint"])
    door_body = case.data.body("microwave_door")
    door_axis = AxisSpec(door_body.xpos, door_body.xmat.reshape(3, 3) @ door_joint.axis,
                         "world", f"urdf:{settings['door_joint']}")
    knob_axis = knob_axis_from_concept(knob, source="calibrated_urdf:knob_0")
    step = float(settings["waypoint_step_rad"])
    door_step = float(settings.get("door_step_rad", step))
    door_start = float(settings.get("door_start_angle_rad", 0.) if door_start_angle_rad is None else door_start_angle_rad)
    door_target = float(settings["door_target_angle_rad"] if door_target_angle_rad is None else door_target_angle_rad)
    knob_target = float(settings["knob_target_angle_rad"] if knob_target_angle_rad is None else knob_target_angle_rad)
    def within_urdf_limit(angle: float, joint_name: str) -> float:
        """允许十进制 URDF 与整度滑块之间的微小舍入差，然后取精确限位。"""
        upper = float(case.model.joint(joint_name).range[1])
        if not np.isfinite(angle) or angle < -1e-5 or angle > upper + 1e-5:
            raise ValueError(f"展示路径目标角超出当前 URDF 关节限位：{joint_name}，有效范围 0–{np.degrees(upper):.3f}°")
        return float(np.clip(angle, 0., upper))
    door_start = within_urdf_limit(door_start, settings["door_joint"])
    door_target = within_urdf_limit(door_target, settings["door_joint"])
    handle_pull_start = float(case.data.qpos[door_joint.qposadr[0]])
    handle_pull_target = within_urdf_limit(
        np.deg2rad(25) if handle_pull_target_angle_rad is None else handle_pull_target_angle_rad,
        settings["door_joint"],
    )
    handle_pull_candidates = generate_handle_pull_candidates(
        handle_grasp.tcp_pose_world,
        # 当前校准中 handle 局部 +Z 从门板指向用户。
        toward_panel_direction_world=-handle_pose[:3, 2],
    )
    handle_candidate_rank = int(handle_pull_candidate_rank)
    if handle_candidate_rank < 0:
        raise ValueError("handle_pull_candidate_rank 不能为负数")
    if handle_pull_selected_candidate_id is not None and handle_candidate_rank != 0:
        raise ValueError("不能同时传入 handle candidate ID 与非零顺序")
    if handle_pull_selected_candidate_id is None:
        if handle_candidate_rank >= len(handle_pull_candidates):
            raise ValueError("把手路线候选序号超出范围")
        selected_handle_candidate = handle_pull_candidates[handle_candidate_rank]
    else:
        selected_handle_candidate = next(
            (
                item for item in handle_pull_candidates
                if item.candidate_id == handle_pull_selected_candidate_id
            ),
            None,
        )
        if selected_handle_candidate is None:
            raise ValueError(
                f"把手路线候选 {handle_pull_selected_candidate_id!r} 不属于当前集合"
            )
    handle_pull_path, handle_pull_labels, handle_pull_widths_m = generate_handle_pull_path(
        selected_handle_candidate.grasp_tcp_pose_world, door_axis,
        start_angle_rad=handle_pull_start, target_angle_rad=handle_pull_target,
        rotation_steps=3, tip_rotation_x_scale=1.,
        pregrasp_distance_m=handle_pull_pregrasp_distance_m,
        retreat_distance_m=handle_pull_retreat_distance_m,
        retreat_up_m=handle_pull_retreat_up_m,
        open_width_m=handle_pull_open_width_m, close_width_m=handle_pull_close_width_m,
    )
    knob_target = within_urdf_limit(knob_target, settings["knob_joint"])
    push_mode = str(settings.get("door_push_mode", "right_open") if door_push_mode is None else door_push_mode)
    settings["door_push_mode"] = push_mode
    if push_mode.endswith("_open") and not door_start < door_target:
        raise ValueError("开门模式要求起始门角小于目标门角")
    if push_mode.endswith("_close") and not door_start > door_target:
        raise ValueError("关门模式要求起始门角大于目标门角")
    push = dict(settings["door_push"])
    if door_push_parameters:
        push.update(door_push_parameters)
    panel_name = settings["door_panel_geom"]
    panel_geom = case.data.geom(panel_name)
    panel_pose = np.eye(4)
    panel_pose[:3, :3] = panel_geom.xmat.reshape(3, 3)
    panel_pose[:3, 3] = panel_geom.xpos
    panel_size = case.model.geom(panel_name).size
    if door_push_waypoint_profile is None:
        door_candidate_sets = generate_mode_door_push_candidate_sets(
            panel_pose, panel_width_m=float(2 * panel_size[0]),
            panel_height_m=float(2 * panel_size[2]),
            panel_thickness_m=float(2 * panel_size[1]), axis=door_axis,
            mode=push_mode, start_angle_rad=door_start,
            target_angle_rad=door_target, step_rad=door_step, overrides=push,
            reference_origin_world=case.data.body(f"{group}_base_link").xpos.copy(),
        )
    else:
        if door_push_parameters:
            raise ValueError("逐角推门 profile 与自动候选偏移不能同时指定")
        profile = dict(door_push_waypoint_profile)
        door_candidate_sets = generate_profile_door_push_candidate_sets(
            panel_pose, panel_width_m=float(2 * panel_size[0]),
            panel_height_m=float(2 * panel_size[2]),
            panel_thickness_m=float(2 * panel_size[1]), axis=door_axis,
            mode=push_mode, start_angle_rad=door_start,
            target_angle_rad=door_target,
            surface=str(profile["surface"]),
            waypoints=tuple(profile["push_waypoints"]),
            source=str(profile["source"]),
        )
    selected_candidate_ids = door_push_selected_candidate_ids
    start_candidate_rank = int(door_push_start_candidate_rank)
    if start_candidate_rank < 0:
        raise ValueError("door_push_start_candidate_rank 不能为负数")
    if selected_candidate_ids is not None and start_candidate_rank != 0:
        raise ValueError("不能同时传入完整 candidate ID 与非零起始候选序号")
    if selected_candidate_ids is None and start_candidate_rank != 0:
        if start_candidate_rank >= len(door_candidate_sets[0].candidates):
            raise ValueError("起始门角候选序号超出当前候选集合")
        selected_candidate_ids = tuple(
            (
                candidate_set.candidates[start_candidate_rank].candidate_id
                if index == 0
                else candidate_set.candidates[0].candidate_id
            )
            for index, candidate_set in enumerate(door_candidate_sets)
        )
    contact_path, contact_parameters = select_door_push_path(
        door_candidate_sets,
        selected_candidate_ids=selected_candidate_ids,
    )
    door_selected_candidate_ids = tuple(
        value["candidate_id"] for value in contact_parameters
    )
    first_selected = next(
        item for item in door_candidate_sets[0].candidates
        if item.candidate_id == door_selected_candidate_ids[0]
    )
    precontact_pose = make_door_push_precontact(
        first_selected,
        distance_m=door_push_pre_contact_distance_m,
    )
    contact_labels = (
        "contact",
        *(f"push_{index:02d}" for index in range(1, len(contact_path.waypoints))),
    )
    door_push_labels = ("pre_contact", *contact_labels)
    door_waypoint_parameters = (
        {**contact_parameters[0], "phase": "pre_contact"},
        *(
            {**parameters, "phase": label}
            for parameters, label in zip(contact_parameters, contact_labels, strict=True)
        ),
    )
    door_waypoints = (
        OperationWaypoint(contact_path.start_angle_rad, precontact_pose),
        *contact_path.waypoints,
    )

    # 旧的 live door 调试装配会显式传入这两个距离。保留该调用的短撤离
    # 兼容行为，但 Knowledge viewer 默认不传入；Task1 的真实撤离由 Task
    # 返回默认关节 FK 得到的 TCP，不属于本模块。
    legacy_retract_values = (
        door_push_retract_up_m,
        door_push_retract_local_z_m,
    )
    if any(value is not None for value in legacy_retract_values):
        if any(value is None for value in legacy_retract_values):
            raise ValueError("兼容短撤离必须同时给出 retract_up_m 和 retract_local_z_m")
        retract_up = float(door_push_retract_up_m)
        retract_local_z = float(door_push_retract_local_z_m)
        if not np.isfinite([retract_up, retract_local_z]).all() or min(retract_up, retract_local_z) < 0.:
            raise ValueError("兼容短撤离距离必须为有限非负数")
        final = contact_path.waypoints[-1]
        up = final.tcp_pose_world.copy()
        up[2, 3] += retract_up
        out = up.copy()
        out[:3, 3] += retract_local_z * final.tcp_pose_world[:3, 2]
        door_waypoints = (
            *door_waypoints,
            OperationWaypoint(final.angle_rad, up),
            OperationWaypoint(final.angle_rad, out),
        )
        door_push_labels = (*door_push_labels, "retract_up", "retract_local_z")
        door_waypoint_parameters = (
            *door_waypoint_parameters,
            {**contact_parameters[-1], "phase": "retract_up"},
            {**contact_parameters[-1], "phase": "retract_local_z"},
        )
    door_path = OperationPath(
        contact_path.kind,
        contact_path.axis_point_world,
        contact_path.axis_direction_world,
        contact_path.axis_source,
        contact_path.start_angle_rad,
        contact_path.target_angle_rad,
        tuple(door_waypoints),
    )
    knob_path = generate_rotation_path("knob", knob_grasp.tcp_pose_world, knob_axis,
                                       target_angle_rad=knob_target, step_rad=step,
                                       object_pose_world=microwave_pose)
    # 拖动 Viser 滑块时只求当前可见姿态，避免 20 多次 IK 阻塞每一帧
    # 交互；用户点击“检查全部”时再同步计算所有候选和 waypoint。
    handle_ik = inspect(handle_grasp.tcp_pose_world) if ik_scope == "all" or (ik_scope == "selected" and selected_mode == "handle_grasp") else None
    knob_grasp_ik = tuple(
        inspect(item.tcp_pose_world, opening_width_m=item.closing_width_m) if ik_scope == "all" or
        (ik_scope == "selected" and selected_mode == "knob_grasp" and index == selected_knob_index) else None
        for index, item in enumerate(knob_grasps)
    )
    def path_ik(path, mode):
        chosen = min(max(int(selected_waypoint), 0), len(path.waypoints) - 1)
        joint_name = settings["knob_joint"] if mode == "knob_turn" else settings["door_joint"]
        return tuple(
            inspect(item.tcp_pose_world, joint_name=joint_name, angle_rad=item.angle_rad,
                    opening_width_m=(handle_pull_widths_m[index] if mode == "handle_pull" else
                                     .06 if mode == "door_push" else knob_grasp.closing_width_m)) if ik_scope == "all" or
            (ik_scope == "selected" and selected_mode == mode and index == chosen) else None
            for index, item in enumerate(path.waypoints)
        )
    door_ik = path_ik(door_path, "door_push")
    handle_pull_ik = path_ik(handle_pull_path, "handle_pull")
    knob_ik = path_ik(knob_path, "knob_turn")
    return MicrowaveKnowledge(
        microwave_pose_world=microwave_pose,
        handle_pose_world=handle_pose,
        knob_pose_world=knob_pose,
        handle_grasp=handle_grasp,
        knob_grasps=tuple(knob_grasps),
        knob_grasp=knob_grasp,
        handle_grasp_ik=handle_ik,
        handle_pull_candidates=handle_pull_candidates,
        handle_pull_selected_candidate_id=selected_handle_candidate.candidate_id,
        handle_pull_path=handle_pull_path,
        handle_pull_ik=handle_pull_ik,
        handle_pull_labels=handle_pull_labels,
        handle_pull_widths_m=handle_pull_widths_m,
        knob_grasp_ik=knob_grasp_ik,
        door_candidate_sets=door_candidate_sets,
        door_selected_candidate_ids=door_selected_candidate_ids,
        door_path=door_path,
        knob_path=knob_path,
        door_ik=door_ik,
        knob_ik=knob_ik,
        door_waypoint_parameters=door_waypoint_parameters,
        door_push_labels=door_push_labels,
        settings=settings,
        door_push_parameters=push,
    )


__all__ = ["MicrowaveKnowledge", "build_microwave_knowledge", "set_preview_microwave_pose"]
