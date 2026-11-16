"""由感知/标定产物与设备命令行参数装配 Task1；不要求操作者维护站点文件。"""

from __future__ import annotations

import ipaddress
import json
import math
from pathlib import Path

import numpy as np

from gene_agent.knowledge.geometry import validate_pose
from gene_agent.realworld.scene_drift import SceneDriftLimits
from .perception_setup import ROOT, BOWL, MICROWAVE, KNOB, load_perception_setup, sha256


def _positive(value, name):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} 必须是正有限数")
    return value


def runtime_configuration(*, perception: str | Path, left_ip: str, right_ip: str,
                          left_port=8080, right_port=8080, max_age_s=5., speed_scale=.1,
                          gripper_config=None, no_knob=True) -> dict:
    """默认值属于设备适配层；操作 profile 和阶段算法保持原样。"""
    source, setup = load_perception_setup(perception)
    objects = (BOWL, MICROWAVE) if no_knob else (BOWL, MICROWAVE, KNOB)
    if any(name not in setup["sessions"] for name in objects):
        raise ValueError("启用旋钮需要先用 prepare_microwave_realworld.py --with-knob 重新准备感知")
    ips = {}
    ports = {}
    for side, ip, port in (("left", left_ip, left_port), ("right", right_ip, right_port)):
        try:
            ips[side] = str(ipaddress.IPv4Address(ip))
        except (ValueError, TypeError) as error:
            raise ValueError(f"{side} 机械臂需要有效 IPv4 地址") from error
        if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
            raise ValueError(f"{side} 端口必须在 1~65535 范围")
        ports[side] = port
    if ips["left"] == ips["right"]:
        raise ValueError("两臂 IP 不能相同")
    max_age_s = _positive(max_age_s, "max_age_s")
    speed_scale = _positive(speed_scale, "speed_scale")
    if speed_scale > 1:
        raise ValueError("speed_scale 不得大于 1")
    standard = ROOT / "config/robots/standard_gripper.json"
    gripper = Path(gripper_config).expanduser() if gripper_config else standard
    if not gripper.is_absolute():
        gripper = ROOT / gripper
    gripper = gripper.resolve()
    profile = json.loads(gripper.read_text(encoding="utf-8"))
    default_profile = json.loads(standard.read_text(encoding="utf-8"))
    # 当前规划影子固定使用标准夹爪，拒绝设备侧悄悄换成另一套 TCP/开口。
    for key in ("max_opening_width_m", "flange_to_tcp_z_m", "flange_to_tcp_rotation_z_deg"):
        if profile.get(key) != default_profile.get(key):
            raise ValueError(f"gripper_config.{key} 与当前标准夹爪规划模型不一致")
    assets, table = setup["assets"], setup["table"]
    table_pose = validate_pose(table["table_pose_world"])
    size = np.asarray(table["table_size_xy_m"], float)
    if size.shape != (2,) or not np.isfinite(size).all() or np.any(size <= 0):
        raise ValueError("感知记录的桌面尺寸无效")
    if not math.isclose(float(table["accepted_table_thickness_m"]), assets["table_thickness_m"], abs_tol=1e-9):
        raise ValueError("桌面碰撞厚度与当前代理资产不一致")
    alignment = validate_pose(assets["alignment"])
    frame_alignment = {"accepted_for_planning": True, "source": f"{source}:paired_asset_frames",
                       "T_scene_asset_profile": np.linalg.inv(alignment).tolist()}
    table_alignment = {"accepted_for_planning": True, "source": f"{source}:confirmed_rgbd_table",
                       "height_bias_m": 0.}
    common = {"max_start_error_rad": .02, "max_gripper_width_error_m": .005}
    microwave = {"max_microwave_translation_m": .005, "max_microwave_rotation_rad": .15}
    bowl = {"max_bowl_translation_m": .005, "max_bowl_rotation_rad": .15}
    hold = {"max_hold_translation_m": .005, "max_hold_rotation_rad": .15}
    options = {
        "door": {**common, **microwave, "max_door_angle_error_rad": .05,
                 "config_path": str(ROOT / "data/task_1/skills/door_skill_debug.json")},
        "pick": {**common, **bowl},
        "insert": {**common, **microwave, **hold, "microwave_frame_alignment": frame_alignment},
        "knob": {**common, **microwave, "max_knob_angle_error_rad": .05,
                 "task_path": str(ROOT / "data/task_1/realworld/knob_0_task_realworld_tuned.json")},
        "extract": {**common, **microwave, **hold, **bowl, "microwave_frame_alignment": frame_alignment},
        "table": {**common, **hold, **bowl, "max_table_height_delta_m": .005,
                  "max_table_translation_m": .005, "max_table_rotation_rad": .15,
                  "table_height_alignment": table_alignment},
    }
    if no_knob:
        options.pop("knob")
    camera = setup["camera"]
    return {
        "source_path": str(source), "source_sha256": sha256(source), "perception_setup": setup,
        "active_group": "right", "robot_ips": ips, "robot_ports": ports,
        "gripper_config": gripper, "gripper_config_sha256": sha256(gripper),
        "camera": {"endpoint": setup["camera_endpoint"], "serial": camera["serial"],
                   "camera_id": "0", "side": "right", "T_sdk_camera": np.array(camera["T_sdk_camera"]),
                   "max_age_s": max_age_s},
        "pose_server": setup["pose_server"], "pose_sessions": {name: setup["sessions"][name] for name in objects},
        "microwave_pose_mesh_path": Path(setup["meshes"][MICROWAVE]),
        "scene_parameters": {"T_world_sdk": np.eye(4),
            "T_perception_object_scene_asset": {BOWL: np.eye(4), MICROWAVE: alignment},
            "table_pose_world": table_pose, "table_size_xy_m": size.tolist(),
            "accepted_table_thickness_m": assets["table_thickness_m"],
            "accepted_support_size_xyz_m": [*assets["support"]["size_xy_m"], assets["support"]["height_m"]],
            "accepted_support_center_xy_local_m": assets["support"]["center_xy_local_m"],
            "active_group": "right"},
        "runtime": {"max_start_error_rad": .02,
                    "min_tcp_z_m": float(table_pose[2, 3] + assets["table_thickness_m"] / 2 + .02),
                    "motion_speed_scale": speed_scale, "max_tcp_speed_mps": .1,
                    "canfd_frequency_hz": 100.},
        "subtask_options": options, "include_knob": not no_knob,
        "scene_drift_limits": SceneDriftLimits(.005, .15, .02, .005, .05),
        "prepared_hold_report": None,
    }
