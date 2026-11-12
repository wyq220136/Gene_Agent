"""微波炉现场观测与碰撞场景适配。"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import replace
from itertools import count
from pathlib import Path
from typing import Callable, Mapping, Sequence
from xml.etree import ElementTree

import numpy as np

from gene_agent.knowledge.geometry import validate_pose
from gene_agent.motion.curobo.scene import export_collision_scene
from gene_agent.realworld.perception import ObservedScene
from gene_agent.realworld.pose_tracking import (
    capture_frozen_scene_frame,
    track_observed_scene_from_capture,
)
from gene_agent.realworld.scene import align_scene_asset_poses
from gene_agent.realworld.scene import observed_joint_state as _observed_start
from gene_agent.simulation import build_scene, set_joint_positions

_ROOT = Path(__file__).resolve().parents[3]
_SCENE_CONFIG = _ROOT / "config/scenes/task_1.json"
_INSTANCE_TO_SCENE = {
    "Bowl/Bowl_0": "bowl",
    "Microwave/Microwave_0": "microwave",
}


def microwave_articulated_joint_limits() -> dict[str, tuple[float, float]]:
    """从当前场景所用 URDF 读取门和 knob_0 的物理关节限位。

    现场状态与规划碰撞必须使用同一资产范围；特别是 knob_0 的上限
    大于半圈，不能把门角或通用 180° 规则套到旋钮上。每次读取资产，
    避免进程中更换 URDF 后沿用旧限位。
    """
    scene = json.loads(_SCENE_CONFIG.read_text(encoding="utf-8"))
    urdf = _ROOT / scene["microwave_asset"]["urdf"]
    root = ElementTree.parse(urdf).getroot()
    limits = {}
    for part, joint_name in (("door", "door_hinge"),
                             ("knob", "power_knob_rotation")):
        limit = root.find(f".//joint[@name='{joint_name}']/limit")
        if limit is None:
            raise ValueError(f"微波炉 URDF 缺少 {joint_name} 关节限位")
        lower, upper = float(limit.attrib["lower"]), float(limit.attrib["upper"])
        if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
            raise ValueError(f"微波炉 URDF 的 {joint_name} 关节限位无效")
        limits[part] = (lower, upper)
    return limits


def _positive_vector(values: Sequence[float], *, count: int, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.shape != (count,) or not np.isfinite(array).all() or np.any(array <= 0):
        raise ValueError(f"{name} 必须是 {count} 个正的有限米数值")
    return array


def _finite_scalar(value: float, *, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} 必须是有限值")
    return number


def build_observed_collision_scene(
    observed: ObservedScene, *, T_world_sdk: np.ndarray,
    T_perception_object_scene_asset: Mapping[str, np.ndarray],
    table_pose_world: np.ndarray, table_size_xy_m: Sequence[float],
    accepted_support_size_xyz_m: Sequence[float],
    accepted_support_center_xy_local_m: Sequence[float],
    accepted_table_thickness_m: float,
    joint_positions_by_group: Mapping[str, Sequence[float]],
    gripper_widths_m: Mapping[str, float], door_angle_rad: float,
    knob_angle_rad: float, turntable_height_offset_m: float,
    microwave_floor_height_offset_m: float, active_group: str,
    output_dir: str | Path, microwave_root_anchor: Mapping | None = None,
) -> dict:
    """将现场状态投到现有资产上，导出当前时刻的 cuRobo 静态场景。

    仅支持本任务当前的双臂 Realman 与可动标准夹爪。桌面、支撑尺寸
    是调用方已核对的规划代理；本函数要求它们与当前场景资产一致，
    避免界面显示一套几何、规划导出另一套几何。微波炉/转盘的仿真
    专用高度补偿不能进入真机规划。返回的快照需要在场景变化后重建。
    """
    anchor_record = None
    source_paths = dict(observed.sources)
    if microwave_root_anchor is None:
        aligned = align_scene_asset_poses(
            observed, T_world_sdk=T_world_sdk,
            T_perception_object_scene_asset=T_perception_object_scene_asset,
        )
    else:
        # 旧版开门后跟踪的 Microwave pose 实际来自把手 mesh。它可用于
        # 单独测门角，却不能覆盖仍静止的机身根 pose。这里只从当前帧
        # 取碗等动态物体；机身使用另行确认未移动的 world 锚点。
        microwave_id = "Microwave/Microwave_0"
        current_poses = {name: pose for name, pose in observed.object_poses_sdk.items()
                         if name != microwave_id}
        source_paths.pop(microwave_id, None)
        current = replace(observed, object_poses_sdk=current_poses,
                          sources=source_paths)
        current_alignment = {
            name: transform for name, transform in T_perception_object_scene_asset.items()
            if name != microwave_id
        }
        aligned = align_scene_asset_poses(
            current, T_world_sdk=T_world_sdk,
            T_perception_object_scene_asset=current_alignment,
        )
        source = microwave_root_anchor.get("source")
        verified_at = float(microwave_root_anchor["verified_at_unix_ms"])
        if not isinstance(source, str) or not source.strip() or not math.isfinite(verified_at):
            raise ValueError("微波炉本体锚点须有有效现场核对时间和来源")
        anchor_pose = validate_pose(
            microwave_root_anchor["T_world_scene_asset"],
            field_name="T_world_microwave_stationary_anchor",
        )
        aligned[microwave_id] = anchor_pose
        anchor_record = {"T_world_scene_asset": anchor_pose.tolist(),
                         "verified_at_unix_ms": verified_at, "source": source}
    if set(aligned) != set(_INSTANCE_TO_SCENE):
        raise ValueError(
            f"task_1 现场碰撞需要且只需要 {sorted(_INSTANCE_TO_SCENE)}；"
            f"收到 {sorted(aligned)}"
        )
    if active_group not in {"left", "right"}:
        raise ValueError("当前 Realman 规划组必须为 left 或 right")
    if set(joint_positions_by_group) != {"left", "right"}:
        raise ValueError("双臂关节必须全部实测，另一臂不能沿用仿真 home")
    if set(gripper_widths_m) != {"left", "right"}:
        raise ValueError("双臂夹爪开口必须全部实测")
    groups = {}
    for group, raw in joint_positions_by_group.items():
        values = np.asarray(raw, dtype=float)
        if values.shape != (7,) or not np.isfinite(values).all():
            raise ValueError(f"{group} 必须提供 SDK 顺序的七轴有限弧度值")
        groups[group] = values
    widths = {group: _finite_scalar(value, name=f"{group} 夹爪开口")
              for group, value in gripper_widths_m.items()}
    table_pose = validate_pose(table_pose_world, field_name="T_world_table")
    table_size = _positive_vector(table_size_xy_m, count=2, name="桌面长宽")
    support_size = _positive_vector(
        accepted_support_size_xyz_m, count=3, name="微波炉支撑代理尺寸"
    )
    support_center = np.asarray(accepted_support_center_xy_local_m, dtype=float)
    if support_center.shape != (2,) or not np.isfinite(support_center).all():
        raise ValueError("微波炉支撑中心必须是两个有限局部坐标")
    thickness = _finite_scalar(accepted_table_thickness_m, name="桌面厚度")
    if thickness <= 0:
        raise ValueError("桌面厚度必须为正")
    door = _finite_scalar(door_angle_rad, name="门角")
    knob = _finite_scalar(knob_angle_rad, name="旋钮角")
    turntable_offset = _finite_scalar(turntable_height_offset_m, name="转盘高度补偿")
    floor_offset = _finite_scalar(microwave_floor_height_offset_m, name="炉腔地板高度补偿")
    if turntable_offset != 0 or floor_offset != 0:
        raise ValueError("仿真专用转盘/炉底高度补偿不能用于真机规划影子")

    raw_config = _SCENE_CONFIG.read_bytes()
    config = json.loads(raw_config)
    # 场景配置文件的 hash 不包含现场标定。即使重新计算后的物体 world
    # pose 恰好相同，外参或 mesh frame 对齐值被更换也必须留下新版本，
    # 供执行前复核拒绝跨标定版本使用旧轨迹。
    alignment_inputs = {
        "T_world_sdk": validate_pose(T_world_sdk, field_name="T_world_sdk").tolist(),
        "T_sdk_camera": validate_pose(
            observed.T_sdk_camera, field_name="T_sdk_camera",
        ).tolist(),
        "T_perception_object_scene_asset": {
            name: validate_pose(transform, field_name=f"{name} mesh 对齐").tolist()
            for name, transform in sorted(T_perception_object_scene_asset.items())
        },
    }
    alignment_sha256 = hashlib.sha256(json.dumps(
        alignment_inputs, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")).hexdigest()
    expected_support = np.asarray([
        *config["microwave_support"]["size_xy_m"],
        config["microwave_support"]["height_m"],
    ], dtype=float)
    if not np.allclose(support_size, expected_support, rtol=0, atol=1e-9):
        raise ValueError("已核对支撑尺寸与场景资产不一致；请先校验代理资产")
    if not np.allclose(support_center, config["microwave_support"]["center_xy_local_m"],
                       rtol=0, atol=1e-9):
        raise ValueError("已核对支撑中心与场景资产不一致；请先校验代理资产")
    if not math.isclose(thickness, config["table"]["thickness_m"], rel_tol=0, abs_tol=1e-9):
        raise ValueError("已核对桌面厚度与场景资产不一致；请先校验代理资产")

    # 场景只用来计算已有资产的 FK 和碰撞几何；不进行 mj_step、落稳或
    # 接触模拟。物体 pose 来自同一观测帧，桌面由现场单独标定。
    poses = {_INSTANCE_TO_SCENE[object_id]: transform
             for object_id, transform in aligned.items()}
    poses["table"] = table_pose
    destination = Path(output_dir).resolve()
    model, data, metadata = build_scene(
        destination / "scene", seed=0, perturb=False,
        pose_overrides=poses, table_size_xy_m=tuple(table_size),
        turntable_height_offset_m=0.0, microwave_floor_height_offset_m=0.0,
        robot_profile="realman_dual_standard",
    )
    robot = metadata["robot"]
    positions = {}
    for group, values in groups.items():
        names = robot["joint_groups"][group]
        if len(names) != len(values):
            raise ValueError(f"{group} 实测关节数与模型不符")
        positions.update(zip(names, values))
        descriptor = robot["grippers"][group]
        if descriptor["mode"] != "parallel_slide_proxy":
            raise ValueError(f"{group} 夹爪模式不支持现场开口影子：{descriptor['mode']}")
        low, high = descriptor["width_range_m"]
        if not low <= widths[group] <= high:
            raise ValueError(f"{group} 夹爪开口超出模型范围 [{low}, {high}] m")
        positions.update({name: widths[group] / 2
                          for name in descriptor["joint_names"]})
    positions[metadata["objects"]["door_joint"]] = door
    positions[metadata["objects"]["knob_joint"]] = knob
    # MuJoCo 的 reset helper 只保证数值有限，不检查关节限位；现场快照
    # 不能把越界读数静默塞进 FK，否则导出的碰撞位置会看似合理但不可信。
    for name, value in positions.items():
        joint = model.joint(name)
        if model.jnt_limited[joint.id]:
            low, high = model.jnt_range[joint.id]
            if not low - 1e-6 <= value <= high + 1e-6:
                raise ValueError(
                    f"现场关节 {name}={value:.6f} 超出模型限位 "
                    f"[{low:.6f}, {high:.6f}]"
                )
    set_joint_positions(model, data, positions)
    snapshot = export_collision_scene(
        model, data, robot, active_group, destination / "collision"
    )
    provenance = {
        "claim": ("同帧物体和实测状态投影到现有 MuJoCo 资产的规划碰撞代理；非物理执行"
                  if anchor_record is None else
                  "当前帧动态物体加现场确认静止的微波炉本体锚点；非同时曝光、非物理执行"),
        "physics_stepped": False,
        "source_frame_number": observed.frame_number,
        "source_timestamp_ms": observed.timestamp_ms,
        "pose_frame_verified": observed.pose_frame_verified,
        "source_camera_id": observed.camera_id,
        "source_side": observed.side,
        "source_paths": source_paths,
        "pose_source_modes": {
            name: ("verified_stationary_anchor" if name == "Microwave/Microwave_0"
                   and anchor_record is not None else "same_frame")
            for name in aligned
        },
        "microwave_root_anchor": anchor_record,
        "aligned_object_poses_world": {name: pose.tolist() for name, pose in aligned.items()},
        "table_pose_world": table_pose.tolist(),
        "table_size_xy_m": table_size.tolist(),
        "accepted_table_thickness_m": thickness,
        "accepted_support_size_xyz_m": support_size.tolist(),
        "accepted_support_center_xy_local_m": support_center.tolist(),
        "joint_positions_by_group_rad": {name: values.tolist() for name, values in groups.items()},
        "gripper_widths_m": widths,
        "door_angle_rad": door,
        "knob_angle_rad": knob,
        "active_group": active_group,
        "scene_config_sha256": hashlib.sha256(raw_config).hexdigest(),
        "alignment_input_sha256": alignment_sha256,
        "collision_snapshot_sha256": snapshot["snapshot_sha256"],
        "limitations": [
            "桌面与微波炉支撑是调用方核对的规划代理；支撑按微波炉局部中心放置，未证明是真实接触几何。",
            "碰撞快照只表示当前静态状态；门、旋钮、另一臂或物体移动后必须重建。",
            "未验证 FoundationPose 精度、资产 frame 标定或真实机器人 FK/TCP。",
        ],
    }
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "observed_scene_provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return {
        "snapshot": snapshot,
        "scene_xml_path": metadata["xml_path"],
        "scene_metadata": metadata,
        "robot_metadata": robot,
        "provenance": provenance,
    }


def build_live_microwave_collision_scene(
    *, observed: ObservedScene, robots: Mapping[str, object],
    grippers: Mapping[str, object], angles: Mapping[str, Mapping],
    scene_parameters: Mapping, max_age_s: float,
    microwave_pose_mesh_path: str | Path,
    microwave_root_anchor: Mapping | None = None,
) -> dict:
    """读两臂与夹爪后调用现有碰撞导出，不运行 MuJoCo 动力学。

    ``angles`` 的 door/knob 各含 ``angle_rad``、
    ``observed_at_unix_ms`` 和 ``source``。可以由传感器或现场人工
    检查提供，但不能使用上轮缓存。``scene_parameters`` 仅包含
    标定、桌面/支撑代理、活动臂和输出目录等静态参数。
    ``microwave_pose_mesh_path`` 应与本次 FoundationPose session 的注册
    mesh 一致。旧项目的把手 mesh 在门打开后只表示门的运动，不能继续
    作为整个微波炉本体的 pose；注册来源真实性还需现场核对。
    若另给本体锚点，须在本次现场确认机身仍静止并提供确认时间；当前
    帧的把手 pose 仍不得写进机身根 pose。
    """
    age_limit = float(max_age_s)
    if not math.isfinite(age_limit) or age_limit <= 0:
        raise ValueError("现场状态有效期必须是正有限秒数")
    if not observed.pose_frame_verified:
        raise ValueError("物体 pose 未验证同帧来源")
    if set(robots) != {"left", "right"} or set(grippers) != {"left", "right"}:
        raise ValueError("必须读取两臂关节和两臂夹爪，不能沿用仿真默认状态")
    if set(angles) != {"door", "knob"}:
        raise ValueError("必须提供门角与旋钮角的现场读数及来源")
    reserved = {
        "observed", "joint_positions_by_group", "gripper_widths_m",
        "door_angle_rad", "knob_angle_rad", "turntable_height_offset_m",
        "microwave_floor_height_offset_m", "microwave_root_anchor",
    }
    if reserved.intersection(scene_parameters):
        raise ValueError("静态场景参数不能覆盖现场动态状态或仿真高度补偿")

    now_ms = time.time() * 1000.

    def check_age(timestamp_ms: float, name: str) -> None:
        value = float(timestamp_ms)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} 时间无效")
        age_ms = now_ms - value
        if age_ms < -1000. or age_ms > age_limit * 1000.:
            raise ValueError(f"{name}过期或时间异常：{age_ms / 1000.:.3f} s")

    check_age(observed.timestamp_ms, "物体同帧 pose ")
    joint_limits = microwave_articulated_joint_limits()
    checked_angles = {}
    for key, title in (("door", "门角"), ("knob", "旋钮角")):
        reading = angles[key]
        angle = float(reading["angle_rad"])
        source = reading["source"]
        if not math.isfinite(angle) or not isinstance(source, str) or not source.strip():
            raise ValueError(f"{title}必须是有限弧度并有非空来源")
        lower, upper = joint_limits[key]
        if not lower - 1e-6 <= angle <= upper + 1e-6:
            raise ValueError(f"{title}规划角超出当前微波炉资产关节限位 [{lower}, {upper}] rad")
        check_age(reading["observed_at_unix_ms"], title)
        checked_angles[key] = {
            "angle_rad": angle,
            "observed_at_unix_ms": float(reading["observed_at_unix_ms"]),
            "source": source,
        }
        if "measured_angle_rad" in reading:
            measured = float(reading["measured_angle_rad"])
            if (not math.isfinite(measured)
                    or not lower - 1e-6 <= measured <= upper + 1e-6
                    or abs(measured - angle) > math.radians(5.) + 1e-9):
                raise ValueError(f"{title}实测值与规划角度不一致")
            checked_angles[key]["measured_angle_rad"] = measured
    mesh_name = Path(microwave_pose_mesh_path).name.lower()
    if not mesh_name:
        raise ValueError("必须记录微波炉 FoundationPose 注册 mesh")
    anchor = None
    if microwave_root_anchor is not None:
        # 几何和来源字段由下层碰撞场景再次校验。这里先检查本次人工或
        # 传感器对机身静止的核对时间，拒绝把旧确认无限期沿用。
        check_age(microwave_root_anchor["verified_at_unix_ms"], "微波炉本体锚点")
        anchor = dict(microwave_root_anchor)
    if abs(checked_angles["door"]["angle_rad"]) > math.radians(1.) and (
        "handle" in mesh_name or "door" in mesh_name
    ) and anchor is None:
        raise ValueError(
            "门已打开：把手 mesh 的 pose 不能作为微波炉本体 pose；"
            "请沿用经检查的固定本体锚点并单独测门角，或跟踪静止柜体"
        )

    joints = {}
    widths = {}
    for group in ("left", "right"):
        state = robots[group].read_joint_state()
        expected = tuple(f"{group}_joint_{index}" for index in range(1, 8))
        if tuple(state.joint_names) != expected:
            raise ValueError(f"{group} 实测关节顺序与 RealMan 模型不一致")
        values = np.asarray(state.positions_rad, dtype=float)
        if values.shape != (7,) or not np.isfinite(values).all():
            raise ValueError(f"{group} 必须有七个有效实测关节角")
        joints[group] = values.tolist()
        widths[group] = float(grippers[group].read_width_m())
    sampled_at_ms = time.time() * 1000.
    if sampled_at_ms - now_ms > age_limit * 1000.:
        raise ValueError("读取两臂状态耗时超过现场快照有效期")

    result = build_observed_collision_scene(
        observed, **dict(scene_parameters),
        joint_positions_by_group=joints, gripper_widths_m=widths,
        door_angle_rad=checked_angles["door"]["angle_rad"],
        knob_angle_rad=checked_angles["knob"]["angle_rad"],
        turntable_height_offset_m=0., microwave_floor_height_offset_m=0.,
        microwave_root_anchor=anchor,
    )
    completed_at_ms = time.time() * 1000.
    if completed_at_ms - observed.timestamp_ms > age_limit * 1000.:
        raise ValueError("碰撞场景导出完成时物体观测已过期，必须重新采集")
    for key, title in (("door", "门角"), ("knob", "旋钮角")):
        if completed_at_ms - checked_angles[key]["observed_at_unix_ms"] > age_limit * 1000.:
            raise ValueError(f"碰撞场景导出完成时{title}已过期，必须重新测量")
    if anchor is not None and completed_at_ms - float(anchor["verified_at_unix_ms"]) > age_limit * 1000.:
        raise ValueError("碰撞场景导出完成时微波炉本体锚点已过期")
    result["provenance"]["live_state"] = {
        "robot_sampled_at_unix_ms": sampled_at_ms,
        "declared_microwave_pose_mesh_path": str(microwave_pose_mesh_path),
        "door": checked_angles["door"], "knob": checked_angles["knob"],
        "claim": "物体同帧；机器人与关节物体分别在有效期内读取，非同时曝光",
    }
    output = Path(scene_parameters["output_dir"]).resolve()
    (output / "observed_scene_provenance.json").write_text(
        json.dumps(result["provenance"], ensure_ascii=False, indent=2,
                   allow_nan=False) + "\n", encoding="utf-8",
    )
    return result


def make_live_microwave_observer(
    *, output_dir: str | Path, camera_endpoint: str, expected_serial: str,
    pose_client, sessions: Mapping[str, str], side: str, camera_id: str,
    T_sdk_camera: np.ndarray, robots: Mapping[str, object],
    grippers: Mapping[str, object],
    angle_reader: Callable[[ObservedScene], Mapping[str, Mapping]],
    scene_parameters: Mapping, max_age_s: float,
    microwave_pose_mesh_path: str | Path,
    microwave_root_anchor_reader: Callable[[ObservedScene], Mapping | None] | None = None,
) -> Callable[[], dict]:
    """生成子任务与完整 Task 共用的同步现场观测回调。

    构造回调时不拍照、不连接机器人、不规划。每次调用都另建目录，
    冻结新 RGB-D，再跟踪、读取关节物体和两臂状态、导出碰撞；上段
    结束后重新调用即可消费实际终态，而不是预测的终点。
    """
    static = dict(scene_parameters)
    if "output_dir" in static:
        raise ValueError("静态场景参数不能预设 output_dir；每次观测需要独立目录")
    base = Path(output_dir).expanduser().resolve()
    sequence = count()

    def observe() -> dict:
        folder = base / f"{next(sequence):04d}"
        capture = capture_frozen_scene_frame(
            output_dir=folder / "capture", endpoint=camera_endpoint,
            expected_serial=expected_serial,
        )
        observed = track_observed_scene_from_capture(
            capture_dir=capture, pose_client=pose_client, sessions=sessions,
            side=side, camera_id=camera_id, T_sdk_camera=T_sdk_camera,
            expected_serial=expected_serial, max_age_s=max_age_s,
            output_dir=folder / "tracked",
        )
        angles = angle_reader(observed)
        anchor = (None if microwave_root_anchor_reader is None else
                  microwave_root_anchor_reader(observed))
        return build_live_microwave_collision_scene(
            observed=observed, robots=robots, grippers=grippers,
            angles=angles, scene_parameters={**static, "output_dir": folder / "scene"},
            max_age_s=max_age_s, microwave_pose_mesh_path=microwave_pose_mesh_path,
            microwave_root_anchor=anchor,
        )

    return observe


__all__ = ['build_live_microwave_collision_scene', 'make_live_microwave_observer']


def bowl_state_summary(scene: dict) -> dict:
    group = scene["snapshot"]["active_group"]
    source = scene["provenance"]
    return {
        "collision_snapshot_sha256": scene["snapshot"]["snapshot_sha256"],
        "source_frame_number": source["source_frame_number"],
        "source_timestamp_ms": source["source_timestamp_ms"],
        "joint_state": _observed_start(scene).to_dict(),
        "gripper_width_m": float(source["gripper_widths_m"][group]),
        "bowl_pose_world": source["aligned_object_poses_world"]["Bowl/Bowl_0"],
    }

