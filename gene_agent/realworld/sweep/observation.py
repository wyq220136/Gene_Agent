"""清扫现场快照；相机/SDK 只读采集与离线输入共用验证。"""
from __future__ import annotations

import json
import os
from pathlib import Path
import time
import subprocess
import sys

import numpy as np

from gene_agent.knowledge.geometry import validate_pose

ROLES = ("brush", "cube", "dustpan")
JOINT_NAMES = tuple(f"joint_{i}" for i in range(1,8))


def resolve_preview_robot_ip(site, robot_ip, *, calibration=None, assume_standby=False, saved_capture=False):
    """新拍照默认只读实机；离线图不可悄悄混入当前关节，也不可静默假定待机。"""
    if assume_standby:
        if robot_ip:
            raise ValueError("假定待机与实际机器人地址不能同时指定")
        return None
    if saved_capture:
        raise ValueError("离线图不能混用当前关节；请显式 --assume-standby 查看假定初态")
    configured = str(site.get("devices", {}).get("robot_ip", "")).strip()
    calibrated = str((calibration or {}).get("communication", {}).get("robot_ip", "")).strip()
    value = robot_ip or calibrated or (configured if configured and not configured.startswith("FILL") else "169.254.128.19")
    from gene_agent.realworld.devices.arms.realman.connection import normalize_realman_ip
    return normalize_realman_ip(value)


def _read_flange(device):
    sdk_python=device.get("sdk_python")
    if sdk_python:
        # cuRobo 与厂家 SDK 可分处两个环境。子进程只走已有的只读连接。
        code='''import json,sys
from gene_agent.realworld.devices.arms.realman.connection import connect_realman_read_only
with connect_realman_read_only(sys.argv[1],group="left",connect_retries=3) as c:
    pose,q=c.read_flange_pose()
print("SWEEP_READ_ONLY="+json.dumps({"pose":pose.tolist(),"joints":q.tolist()}))
'''
        root=Path(__file__).resolve().parents[3]
        try:
            result=subprocess.run([str(sdk_python),"-c",code,device["robot_ip"]],
                capture_output=True,text=True,encoding="utf-8",timeout=30,check=True,
                env={**os.environ,"PYTHONPATH":str(root),"PYTHONNOUSERSITE":"1","PYTHONUTF8":"1"})
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"Realman 只读状态失败 ({device['robot_ip']})；不采用假定姿态。\n"
                               f"{exc.stdout or ''}\n{exc.stderr or ''}") from exc
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"Realman 只读状态超时 ({device['robot_ip']})；不采用假定姿态") from exc
        rows=[line.split("=",1)[1] for line in result.stdout.splitlines() if line.startswith("SWEEP_READ_ONLY=")]
        if len(rows)!=1:raise ValueError("SDK 只读进程没有返回唯一状态")
        row=json.loads(rows[0])
        return validate_pose(row["pose"]),np.asarray(row["joints"],float)
    from gene_agent.realworld.devices.arms.realman.connection import connect_realman_read_only
    with connect_realman_read_only(device["robot_ip"],group="left",connect_retries=3) as connection:
        return connection.read_flange_pose()


def validate_snapshot(snapshot: dict, *, max_age_s: float, allow_synthetic: bool = False,
                      allow_first_frame_preview: bool = False) -> None:
    synthetic = snapshot.get("source") == "synthetic_fixture"
    preview = snapshot.get("source") == "first_frame_preview"
    if preview and (not allow_first_frame_preview or snapshot.get("execution_enabled") is not False
                    or snapshot.get("initial_brush_state") != "on_table"):
        raise ValueError("首帧桌面预览必须显式启用，且禁止执行")
    if synthetic and not allow_synthetic:
        raise ValueError("synthetic 输入需要显式 --demo/--allow-synthetic")
    if snapshot.get("source") not in {"synthetic_fixture", "realworld_observation", "first_frame_preview"}:
        raise ValueError("未知的快照来源")
    if snapshot.get("schema_version") != 1 or tuple(snapshot.get("joint_names", ())) != JOINT_NAMES:
        raise ValueError("快照版本或 SDK 关节 1–7 顺序不符")
    joints = np.asarray(snapshot["joint_positions_rad"], dtype=float)
    if joints.shape != (7,) or not np.isfinite(joints).all():
        raise ValueError("需要七个有限关节角（rad）")
    width = float(snapshot["gripper_width_m"])
    if not np.isfinite(width) or not 0 <= width <= .0941:
        raise ValueError("Pika 开口超出当前模型范围")
    if snapshot.get("pose_frame_verified") is not True:
        raise ValueError("缺少同帧 pose 验证")
    validate_pose(snapshot["T_sdk_flange"], field_name="T_sdk_flange")
    if set(snapshot["T_sdk_object"]) != set(ROLES):
        raise ValueError("单方块快照必须且只能包含 brush/cube/dustpan 三个角色")
    for role, pose in snapshot["T_sdk_object"].items():
        validate_pose(pose, field_name=f"T_sdk_{role}")
    now = time.time()*1000
    observed = float(snapshot["observed_at_unix_ms"])
    if not np.isfinite(observed) or not np.isfinite(max_age_s) or max_age_s <= 0:
        raise ValueError("采集时间或有效期无效")
    if not synthetic and not preview and not -1000 <= now-observed <= max_age_s*1000:
        raise ValueError("现场快照已过期或时间异常")
    hold = snapshot.get("hold", {})
    if preview:
        if hold.get("confirmed") is not False:
            raise ValueError("桌面首帧不能声称已夹持")
        if snapshot.get("robot_state_source") not in {"assumed_template_standby", "sdk_read_only"}:
            raise ValueError("首帧预览缺少机器人初态来源")
        if not str(snapshot.get("camera_serial", "")).strip() or int(snapshot["camera_frame_number"]) < 0:
            raise ValueError("缺少相机来源")
        return
    if hold.get("confirmed") is not True or not str(hold.get("source", "")).strip():
        raise ValueError("需要本次人工夹持确认，不能凭规划 attachment 判断已夹稳")
    confirmed = float(hold["confirmed_at_unix_ms"])
    if not np.isfinite(confirmed) or (not synthetic and not -1000 <= now-confirmed <= max_age_s*1000):
        raise ValueError("夹持确认已过期")
    if not str(snapshot.get("camera_serial", "")).strip() or int(snapshot["camera_frame_number"]) < 0:
        raise ValueError("缺少相机来源")


def capture_snapshot(config: dict, output: Path, *, held_brush_confirmed: bool) -> dict:
    """显式只读采集；不导入 Pika motor driver，不使能或配置夹爪。

开口由用户现场测量填写；连续读取两次关节检查采集期间是否移动。
"""
    from gene_agent.realworld.pose_tracking import capture_frozen_scene_frame, track_observed_scene_from_capture
    from gene_agent.perception_modules.pose_est.pose_client import PoseClient
    if not held_brush_confirmed:
        raise ValueError("采集持刷场景需要 --held-brush-confirmed")
    device = config["devices"]
    camera = device["camera"]
    from gene_agent.realworld.devices.arms.realman.connection import normalize_realman_ip
    from gene_agent.robots.pika_frames import load_frame_profile
    normalize_realman_ip(device["robot_ip"])
    if camera.get("mount","fixed")!="fixed":
        raise ValueError("当前清扫采集只支持已标定的固定相机；手腕相机不能填一个固定外参")
    if str(camera["serial"]).startswith("FILL") or any(
        not row["session"] or row["session"].startswith("FILL") for row in config["objects"].values()):
        raise ValueError("请先填写实际相机序列号和三个已注册的 Pose session")
    before, joints_before = _read_flange(device)
    capture = capture_frozen_scene_frame(output_dir=output / "capture",
                    endpoint=camera["endpoint"], expected_serial=camera["serial"])
    observed = track_observed_scene_from_capture(capture_dir=capture,
            pose_client=PoseClient(device["pose_server"], timeout=120),
            sessions={row["object_id"]: row["session"] for row in config["objects"].values()},
            side="left", camera_id="0", T_sdk_camera=config["T_sdk_camera"],
            expected_serial=camera["serial"], max_age_s=config["max_snapshot_age_s"],
            output_dir=output / "pose_observation")
    after, joints_after = _read_flange(device)
    if np.max(np.abs(joints_after-joints_before)) > .005:
        raise ValueError("采集期间机械臂移动，关节与图像不能作为同一静止场景")
    now = time.time()*1000
    snapshot = {"schema_version":1, "source":"realworld_observation",
        "joint_names":list(JOINT_NAMES), "joint_positions_rad":joints_after.tolist(),
        "T_sdk_flange":after.tolist(), "T_sdk_flange_before":before.tolist(),
        "gripper_width_m":device["measured_gripper_width_m"],
        "observed_at_unix_ms":observed.timestamp_ms,
        "camera_frame_number":observed.frame_number, "camera_serial":camera["serial"],
        "site_sha256":config["site_sha256"],
        "frame_profile_sha256":load_frame_profile(config["frame_profile"])["sha256"],
        "pose_frame_verified":observed.pose_frame_verified,
        "T_sdk_object":{role: observed.object_poses_sdk[row["object_id"]].tolist()
                        for role,row in config["objects"].items()},
        "hold":{"confirmed":True,"source":"manual_cli","confirmed_at_unix_ms":now},
        "pose_sources":dict(observed.sources)}
    validate_snapshot(snapshot, max_age_s=config["max_snapshot_age_s"])
    (output / "observation.json").write_text(json.dumps(snapshot,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    return snapshot
