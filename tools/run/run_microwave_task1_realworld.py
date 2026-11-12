"""微波炉现场子任务/完整 Task 的薄入口；默认仅规划首段。

此脚本只装配现场观测、现有 cuRobo pool、真机 Runtime 和已经存在的
子任务函数，不定义第二套操作顺序。现场标定和角度来源必须显式给出。
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
from dataclasses import asdict
from datetime import datetime
import hashlib
import inspect
import json
import math
from pathlib import Path
import sys
import traceback
from typing import Mapping

import numpy as np

# 与现有 Viewer/MuJoCo 命令保持一致：从项目根目录直接运行脚本时，
# Python 默认只把 tools/run 放进搜索路径，需要显式加入项目根目录。
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from gene_agent.motion.configuration import configured_motion, current_motion_config

from gene_agent.realworld.microwave.task import SUBTASK_NAMES
from gene_agent.realworld.microwave.task import run_observed_microwave_subtask
from gene_agent.realworld.microwave.task import run_observed_microwave_task1
from gene_agent.knowledge.geometry import validate_pose
from gene_agent.realworld.scene_drift import SceneDriftLimits


_OBJECT_IDS = ("Bowl/Bowl_0", "Microwave/Microwave_0")
_OPTION_KEY = {
    "open_door": "door", "close_door": "door", "pick_bowl": "pick",
    "insert_bowl": "insert", "rotate_knob": "knob",
    "extract_bowl": "extract", "place_bowl_on_table": "table",
}
_COMBO_KINDS = {'load_bowl': ('pick', 'insert'), 'unload_bowl': ('extract', 'table')}
_RIGHT_ONLY_TASK1_SUBTASKS = frozenset({
    "pick_bowl", "insert_bowl", "rotate_knob", "extract_bowl",
    "place_bowl_on_table", "all", "load_bowl", "unload_bowl",
})


def _nonempty(value, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} 必须是非空字符串")
    return value.strip()


def _require(mapping, fields: tuple[str, ...], name: str) -> None:
    if not isinstance(mapping, Mapping):
        raise ValueError(f"{name} 必须是 JSON 对象")
    missing = [field for field in fields if field not in mapping]
    if missing:
        raise ValueError(f"{name} 缺少字段：{', '.join(missing)}")


def _finite(value, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} 必须是有限数，不能留空") from error
    if not math.isfinite(number):
        raise ValueError(f"{name} 必须是有限数")
    return number


def _positive(value, name: str) -> float:
    number = _finite(value, name)
    if number <= 0:
        raise ValueError(f"{name} 必须是正有限数")
    return number


def _path(value, *, base: Path, name: str) -> Path:
    raw = Path(_nonempty(value, name)).expanduser()
    path = raw if raw.is_absolute() else base / raw
    path = path.resolve()
    if not path.is_file():
        raise ValueError(f"{name} 文件不存在：{path}")
    return path


def _required_options(kind: str, options: Mapping) -> None:
    """只核对现有子任务函数实际要求的参数，不复制它们的策略。"""
    from gene_agent.realworld.microwave.pick import run_observed_bowl_pick
    from gene_agent.realworld.microwave.door import run_observed_door_subtask
    from gene_agent.realworld.microwave.extract import run_observed_bowl_extract
    from gene_agent.realworld.microwave.insert import run_observed_bowl_insert
    from gene_agent.realworld.microwave.knob import run_observed_knob_skill
    from gene_agent.realworld.microwave.place import run_observed_table_place
    from gene_agent.realworld.microwave.task import run_observed_door_return_default_tcp

    functions = {
        "door": (run_observed_door_subtask, run_observed_door_return_default_tcp),
        "pick": (run_observed_bowl_pick,), "insert": (run_observed_bowl_insert,),
        "knob": (run_observed_knob_skill,), "extract": (run_observed_bowl_extract,),
        "table": (run_observed_table_place,),
    }
    injected = {
        "observe_collision_scene", "make_runtime", "planner_pool", "output_dir",
        "execute", "part", "pick_report", "grasp_report", "ask_manual",
    }
    allowed = set().union(*(set(inspect.signature(function).parameters)
                            for function in functions[kind])) - injected
    # 现场检查器是运行时函数，JSON 只能保存数值与路径。允许它穿过
    # 静态检查会在已执行前段动作后才抛出“bool 不可调用”等错误。
    callbacks = sorted(name for name in options
                       if name.startswith("check_") or name == "grasp_check")
    if callbacks:
        raise ValueError(f"subtask_options.{kind}.{callbacks[0]} 不能写入 JSON；现场检查由入口注入")
    unexpected = sorted(set(options) - allowed)
    if unexpected:
        raise ValueError(f"subtask_options.{kind} 含未知参数：{unexpected}")
    for function in functions[kind]:
        missing = [
            name for name, parameter in inspect.signature(function).parameters.items()
            if parameter.default is inspect.Parameter.empty
            and name not in injected and name not in options
        ]
        if missing:
            raise ValueError(f"subtask_options.{kind} 缺少 {function.__name__} 参数：{missing}")
        for name, parameter in inspect.signature(function).parameters.items():
            if (parameter.default is inspect.Parameter.empty
                    and name not in injected and options[name] is None):
                raise ValueError(f"subtask_options.{kind}.{name} 尚未标定")
            if name in options and name.startswith("max_"):
                _positive(options[name], f"subtask_options.{kind}.{name}")
    if kind in {"insert", "extract"}:
        name = f"subtask_options.{kind}.microwave_frame_alignment"
        alignment = options["microwave_frame_alignment"]
        if not isinstance(alignment, Mapping) or alignment.get("accepted_for_planning") is not True:
            raise ValueError(f"{name} 未验收，不能用于真机规划")
        _nonempty(alignment.get("source"), f"{name}.source")
        validate_pose(alignment.get("T_scene_asset_profile"),
                      field_name=f"{name}.T_scene_asset_profile")
    if kind == "table":
        name = "subtask_options.table.table_height_alignment"
        alignment = options["table_height_alignment"]
        if not isinstance(alignment, Mapping) or alignment.get("accepted_for_planning") is not True:
            raise ValueError(f"{name} 未验收，不能用于真机规划")
        _nonempty(alignment.get("source"), f"{name}.source")
        try:
            bias = float(alignment.get("height_bias_m"))
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name}.height_bias_m 必须是有限米数值") from error
        if not math.isfinite(bias):
            raise ValueError(f"{name}.height_bias_m 必须是有限米数值")


def _check_prepared_hold_file(path: Path) -> None:
    """连接设备前检查人工持碗文件结构；当前夹持状态仍由新观测判断。"""
    record = json.loads(path.read_text(encoding="utf-8"))
    name = "prepared_hold_report"
    _require(record, ("source_format", "schema_version", "object_id",
                      "source_collision_snapshot_sha256", "source_frame_number",
                      "source_timestamp_ms", "postcondition"), name)
    if (record["source_format"] != "manual_prepared_held_bowl"
            or record["schema_version"] != 1
            or record["object_id"] != "Bowl/Bowl_0"):
        raise ValueError(f"{name} 必须是当前任务 Bowl 的人工准备记录")
    _nonempty(record["source_collision_snapshot_sha256"],
              f"{name}.source_collision_snapshot_sha256")
    frame = record["source_frame_number"]
    if isinstance(frame, bool) or not isinstance(frame, int) or frame < 0:
        raise ValueError(f"{name}.source_frame_number 必须是非负整数")
    stamp = _finite(record["source_timestamp_ms"], f"{name}.source_timestamp_ms")
    if stamp < 0:
        raise ValueError(f"{name}.source_timestamp_ms 不能为负")
    check = record["postcondition"]
    _require(check, ("status", "source"), f"{name}.postcondition")
    if check["status"] != "pass":
        raise ValueError(f"{name}.postcondition 尚未确认持碗")
    source = _nonempty(check["source"], f"{name}.postcondition.source")
    if source in {"simulation_privileged_weld",
                  "mujoco_contact_geometry_at_planned_waypoints"}:
        raise ValueError(f"{name}.postcondition 不能使用仿真证据")
    evidence = check.get("evidence")
    if evidence is not None:
        if not isinstance(evidence, Mapping):
            raise ValueError(f"{name}.postcondition.evidence 必须是 JSON 对象")
        expected = {
            "collision_snapshot_sha256": record["source_collision_snapshot_sha256"],
            "source_frame_number": frame,
            "source_timestamp_ms": stamp,
        }
        # 老文件可能没有内部来源字段，此处只拒绝明确矛盾的证据；
        # 是否仍为当前夹持状态，运行时必须用新观测重新确认。
        if any(key in evidence and evidence[key] != value
               for key, value in expected.items()):
            raise ValueError(f"{name}.postcondition.evidence 与顶层来源不一致")


def load_site_config(path: str | Path, *, subtask: str, execute: bool,
                     prepare_held_bowl: bool = False,
                     no_knob: bool = False) -> dict:
    """静态检查站点输入；不连接相机、模型服务或机械臂。"""
    source = Path(path).expanduser().resolve()
    source_bytes = source.read_bytes()
    payload = json.loads(source_bytes.decode("utf-8"))
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("site config 必须是 schema_version=1 的 JSON 对象")
    if subtask != "all" and subtask not in SUBTASK_NAMES:
        raise ValueError(f"subtask 必须是 all 或 {SUBTASK_NAMES}")
    if no_knob and subtask != "all":
        raise ValueError("--no-knob 只适用于 --subtask all")
    if prepare_held_bowl and subtask not in {"insert_bowl", "place_bowl_on_table"}:
        raise ValueError("--prepare-held-bowl 只适用于独立放碗或桌面 Place")
    _require(payload, (
        "active_group", "robot_ips", "camera", "pose_server", "pose_sessions",
        "microwave_pose_mesh_path", "state_file", "scene_parameters",
        "runtime", "subtask_options",
    ), "site config")
    group = payload.get("active_group")
    if group not in {"left", "right"}:
        raise ValueError("active_group 必须是 left 或 right")
    if group == "left" and subtask in _RIGHT_ONLY_TASK1_SUBTASKS:
        # 旧 task_1 的 Bowl 与 knob_0 在下层仍使用右臂 profile；
        # 不能让左臂配置通过静态检查后才在现场规划中途报错。
        raise ValueError(f"旧 task_1 的 {subtask} 当前只验收右臂；左臂需单独适配与验证")
    ips = payload["robot_ips"]
    if not isinstance(ips, dict) or set(ips) != {"left", "right"}:
        raise ValueError("robot_ips 必须显式包含 left 和 right")
    for side in ("left", "right"):
        _nonempty(ips[side], f"robot_ips.{side}")
    _require(payload["camera"], (
        "endpoint", "serial", "camera_id", "side", "T_sdk_camera", "max_age_s",
    ), "camera")
    camera = dict(payload["camera"])
    for name in ("endpoint", "serial", "camera_id"):
        camera[name] = _nonempty(camera[name], f"camera.{name}")
    if not camera["camera_id"].isdigit():
        raise ValueError("camera.camera_id 必须是数字编号")
    if camera.get("side") not in {"left", "right"}:
        raise ValueError("camera.side 必须是 left 或 right")
    camera["T_sdk_camera"] = validate_pose(
        camera["T_sdk_camera"], field_name="T_sdk_camera",
    )
    camera["max_age_s"] = _positive(camera["max_age_s"], "camera.max_age_s")
    sessions = payload["pose_sessions"]
    if not isinstance(sessions, dict) or set(sessions) != set(_OBJECT_IDS):
        raise ValueError(f"pose_sessions 必须恰好包含 {_OBJECT_IDS}")
    for object_id in _OBJECT_IDS:
        _nonempty(sessions[object_id], f"pose_sessions.{object_id}")
    if len(set(sessions.values())) != len(sessions):
        raise ValueError("两个物体不能共用 FoundationPose session")
    _nonempty(payload["pose_server"], "pose_server")
    mesh_path = _path(payload["microwave_pose_mesh_path"], base=source.parent,
                      name="microwave_pose_mesh_path")
    state_file = _path(payload["state_file"], base=source.parent, name="state_file")
    state = json.loads(state_file.read_text(encoding="utf-8"))
    _require(state, ("door", "knob"), "state_file")
    from gene_agent.realworld.microwave.scene import microwave_articulated_joint_limits
    joint_limits = microwave_articulated_joint_limits()
    for name in ("door", "knob"):
        reading = state[name]
        _require(reading, ("angle_rad", "observed_at_unix_ms", "source"),
                 f"state_file.{name}")
        angle = _finite(reading["angle_rad"], f"state_file.{name}.angle_rad")
        lower, upper = joint_limits[name]
        if not lower - 1e-6 <= angle <= upper + 1e-6:
            raise ValueError(f"state_file.{name}.angle_rad 超出资产关节限位 [{lower}, {upper}] rad")
        timestamp = _finite(reading["observed_at_unix_ms"],
                            f"state_file.{name}.observed_at_unix_ms")
        if timestamp < 0:
            raise ValueError(f"state_file.{name}.observed_at_unix_ms 不能为负")
        _nonempty(reading["source"], f"state_file.{name}.source")
        if "measured_angle_rad" in reading:
            measured = _finite(reading["measured_angle_rad"],
                               f"state_file.{name}.measured_angle_rad")
            if (not lower - 1e-6 <= measured <= upper + 1e-6
                    or abs(measured - angle) > math.radians(5.) + 1e-9):
                raise ValueError(
                    f"state_file.{name}.measured_angle_rad 与规划角相差超过 5°"
                    "或超出资产关节限位")
    if "microwave_root_anchor" in state:
        anchor = state["microwave_root_anchor"]
        _require(anchor, ("T_world_scene_asset", "verified_at_unix_ms", "source"),
                 "state_file.microwave_root_anchor")
        validate_pose(anchor["T_world_scene_asset"],
                      field_name="state_file.microwave_root_anchor.T_world_scene_asset")
        verified_at = _finite(anchor["verified_at_unix_ms"],
                              "state_file.microwave_root_anchor.verified_at_unix_ms")
        if verified_at < 0:
            raise ValueError("state_file.microwave_root_anchor.verified_at_unix_ms 不能为负")
        _nonempty(anchor["source"], "state_file.microwave_root_anchor.source")
    # 这里只核对字段与单位；每次真正观测时仍由现场观察器核对帧龄。

    _require(payload["scene_parameters"], (
        "active_group", "T_world_sdk", "T_perception_object_scene_asset",
        "table_pose_world", "table_size_xy_m", "accepted_support_size_xyz_m",
        "accepted_support_center_xy_local_m", "accepted_table_thickness_m",
    ), "scene_parameters")
    scene = dict(payload["scene_parameters"])
    if scene.get("active_group") != group:
        raise ValueError("scene_parameters.active_group 与活动臂不一致")
    for key in ("T_world_sdk", "table_pose_world"):
        scene[key] = validate_pose(scene[key], field_name=key)
    alignment = scene["T_perception_object_scene_asset"]
    if not isinstance(alignment, dict) or set(alignment) != set(_OBJECT_IDS):
        raise ValueError("T_perception_object_scene_asset 必须覆盖 Bowl 与 Microwave")
    scene["T_perception_object_scene_asset"] = {
        key: validate_pose(alignment[key], field_name=f"mesh frame {key}")
        for key in _OBJECT_IDS
    }
    for key, length in (("table_size_xy_m", 2), ("accepted_support_size_xyz_m", 3)):
        values = np.asarray(scene[key], dtype=float)
        if values.shape != (length,) or not np.isfinite(values).all() or np.any(values <= 0):
            raise ValueError(f"scene_parameters.{key} 必须是 {length} 个正有限数")
    center = np.asarray(scene["accepted_support_center_xy_local_m"], dtype=float)
    if center.shape != (2,) or not np.isfinite(center).all():
        raise ValueError("accepted_support_center_xy_local_m 必须是两个有限数")
    _positive(scene["accepted_table_thickness_m"], "accepted_table_thickness_m")

    _require(payload["runtime"], (
        "max_start_error_rad", "min_tcp_z_m", "motion_speed_scale",
        "max_tcp_speed_mps", "canfd_frequency_hz",
    ), "runtime")
    runtime = dict(payload["runtime"])
    for key in ("max_start_error_rad", "min_tcp_z_m", "motion_speed_scale",
                "max_tcp_speed_mps", "canfd_frequency_hz"):
        runtime[key] = _positive(runtime[key], f"runtime.{key}")
    if runtime["motion_speed_scale"] > 1.:
        raise ValueError("runtime.motion_speed_scale 不得大于 1")
    if "tcp_speed_mps" in runtime and runtime["tcp_speed_mps"] is not None:
        runtime["tcp_speed_mps"] = _positive(runtime["tcp_speed_mps"], "tcp_speed_mps")
        if runtime["tcp_speed_mps"] > runtime["max_tcp_speed_mps"]:
            raise ValueError("tcp_speed_mps 不得超过 max_tcp_speed_mps")
    limits = payload.get("scene_drift_limits")
    if execute and not isinstance(limits, dict):
        raise ValueError("执行模式必须提供 scene_drift_limits")
    if limits is not None:
        names = tuple(SceneDriftLimits.__dataclass_fields__)
        _require(limits, names, "scene_drift_limits")
        extra = sorted(set(limits) - set(names))
        if extra:
            raise ValueError(f"scene_drift_limits 含未知字段：{extra}")
        for name in names:
            value = _finite(limits[name], f"scene_drift_limits.{name}")
            if value < 0:
                raise ValueError(f"scene_drift_limits.{name} 不能为负")
    drift = None if limits is None else SceneDriftLimits(**limits)

    options = payload["subtask_options"]
    if not isinstance(options, dict):
        raise ValueError("subtask_options 必须是子任务参数映射")
    required_kinds = (set(_OPTION_KEY.values()) if subtask == "all" else
                      set(_COMBO_KINDS[subtask]) if subtask in _COMBO_KINDS else {_OPTION_KEY[subtask]})
    if no_knob:
        required_kinds.discard("knob")
    if any(not isinstance(values, dict) for values in options.values()):
        raise ValueError("每个 subtask_options 子项都必须是 JSON 对象")
    options = {kind: dict(values) for kind, values in options.items()}
    for kind in required_kinds:
        if kind not in options or not isinstance(options[kind], dict):
            raise ValueError(f"缺少 subtask_options.{kind}")
        _required_options(kind, options[kind])
        for key in ("config_path", "task_path"):
            if key in options[kind] and options[kind][key] is not None:
                options[kind][key] = str(_path(
                    options[kind][key], base=source.parent,
                    name=f"subtask_options.{kind}.{key}",
                ))
    prepared_report = payload.get("prepared_hold_report")
    if subtask in _COMBO_KINDS and prepared_report is not None:
        raise ValueError('组合操作不能注入旧持碗报告；请使用本轮实际 Pick/Extract 结果')
    if prepare_held_bowl and prepared_report is not None:
        raise ValueError("现场准备持碗与 prepared_hold_report 文件只能选择一种")
    if subtask in {"insert_bowl", "place_bowl_on_table"}:
        if prepared_report is None and not prepare_held_bowl:
            raise ValueError("独立放碗/桌面 Place 需要 prepared_hold_report 或 --prepare-held-bowl")
        if prepared_report is not None:
            prepared_report = _path(prepared_report, base=source.parent,
                                    name="prepared_hold_report")
            _check_prepared_hold_file(prepared_report)
    return {**payload, "source_path": str(source),
            "source_sha256": source_sha256,
            "camera": camera, "scene_parameters": scene,
            "runtime": runtime, "subtask_options": options,
            "include_knob": not no_knob,
            "scene_drift_limits": drift,
            "state_file": state_file, "microwave_pose_mesh_path": mesh_path,
            "prepared_hold_report": prepared_report}


def _prepare_current_held_bowl_report(scene: dict, *, output: Path,
                                      ask) -> dict:
    """把操作者对本次现场帧的持碗确认保存成可追溯的前置证据。

    回答只确认当前看到的物体状态，不代表本程序执行了抓取。后续
    Skill 首次观测必须复用同一帧；发令前仍须由漂移保护采集新帧。
    """
    from gene_agent.realworld.microwave.attachment import make_manual_held_bowl_report
    from gene_agent.realworld.checks import resolve_realworld_check
    from gene_agent.tasks.checks import CheckResult

    check = resolve_realworld_check(
        CheckResult("unknown", "not_configured", {}, "等待操作者确认当前持碗状态"),
        ask=ask, question="当前 Bowl 是否已由夹爪稳定夹持？[y/n/?] ",
    )
    if not check.passed:
        raise ValueError(f"未确认当前 Bowl 稳定夹持：{check.reason}")
    report = make_manual_held_bowl_report(scene, check)
    (output / "prepared_hold_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report



@contextmanager
def _record_entry_errors(output: Path, site_config_sha256: str):
    """现场入口异常也写入可追溯记录；已有更具体的连接报告优先。"""
    try:
        yield
    except Exception as error:
        result_path = output / "entry_result.json"
        if not result_path.exists():
            result_path.write_text(json.dumps({
                "schema_version": 1, "status": "failed", "success": False,
                "site_config_sha256": site_config_sha256,
                "failure": {
                    "code": "entry_failed",
                    "message": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc(),
                },
            }, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8")
        raise


@configured_motion("config")
def run_site(config: dict, *, subtask: str, output_dir: str | Path,
             execute: bool, prepare_held_bowl: bool = False) -> dict:
    """接通设备并调用现有子任务/Task；只在 execute=True 时允许 SDK 运动。"""
    source = Path(config["source_path"])
    if hashlib.sha256(source.read_bytes()).hexdigest() != config["source_sha256"]:
        raise ValueError("现场配置文件在检查后已改变，请重新检查再运行")
    output = Path(output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("本次输出目录已有内容，不能覆盖已有现场运行记录")
    output.mkdir(parents=True, exist_ok=True)
    drift = config["scene_drift_limits"]
    (output / "site_input.json").write_text(json.dumps({
        "schema_version": 1,
        "site_config_path": str(source),
        "site_config_sha256": config["source_sha256"],
        "subtask": subtask, "execute": bool(execute),
        "include_knob": config.get("include_knob", True),
        "prepare_held_bowl": bool(prepare_held_bowl),
        "active_group": config["active_group"],
        "motion": current_motion_config().to_dict(),
        "camera_serial": config["camera"]["serial"],
        "pose_sessions": config["pose_sessions"],
        "runtime": config["runtime"],
        "scene_drift_limits": None if drift is None else asdict(drift),
        "claim": "仅记录输入来源；设备连接、规划和实际执行需另看阶段报告",
    }, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    from gene_agent.realworld.microwave.scene import make_live_microwave_observer
    from gene_agent.motion.curobo.core import SynchronousPlanner
    from gene_agent.motion.curobo.session import CuroboPlannerPool
    from gene_agent.perception_modules.pose_est.pose_client import PoseClient
    from gene_agent.realworld.devices.arms.realman.connection import connect_realman
    from gene_agent.realworld.runtime import RealWorldRuntime

    state_cache: dict = {}

    def read_angles(_observed):
        state = json.loads(config["state_file"].read_text(encoding="utf-8"))
        if not isinstance(state, dict) or not all(key in state for key in ("door", "knob")):
            raise ValueError("state_file 必须含有带来源和 Unix 时间的 door/knob")
        state_cache["state"] = state
        return {"door": state["door"], "knob": state["knob"]}

    def read_anchor(_observed):
        return state_cache["state"].get("microwave_root_anchor")

    with _record_entry_errors(output, config["source_sha256"]), ExitStack() as stack:
        try:
            connections = {
                side: stack.enter_context(connect_realman(config["robot_ips"][side], group=side))
                for side in ("left", "right")
            }
        except Exception as error:
            (output / "entry_result.json").write_text(json.dumps({
                "schema_version": 1, "status": "failed", "success": False,
                "site_config_sha256": config["source_sha256"],
                "failure": {
                    "code": "connection_failed",
                    "message": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc(),
                },
            }, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            raise
        robots = {side: connection.robot for side, connection in connections.items()}
        grippers = {side: connection.gripper for side, connection in connections.items()}
        camera = config["camera"]
        observer = make_live_microwave_observer(
            output_dir=output / "observations",
            camera_endpoint=camera["endpoint"], expected_serial=camera["serial"],
            pose_client=PoseClient(config["pose_server"]),
            sessions=config["pose_sessions"], side=camera["side"],
            camera_id=camera["camera_id"], T_sdk_camera=camera["T_sdk_camera"],
            robots=robots, grippers=grippers, angle_reader=read_angles,
            scene_parameters=config["scene_parameters"],
            max_age_s=camera["max_age_s"],
            microwave_pose_mesh_path=config["microwave_pose_mesh_path"],
            microwave_root_anchor_reader=read_anchor,
        )
        group = config["active_group"]
        speed = config["runtime"]

        def make_runtime(snapshot_sha256: str) -> RealWorldRuntime:
            return RealWorldRuntime(
                robots[group], gripper=grippers[group],
                scene_snapshot_sha256=snapshot_sha256,
                max_start_error_rad=speed["max_start_error_rad"],
                tcp_height_fn=robots[group].tcp_height_from_sdk_fk,
                min_tcp_z_m=speed["min_tcp_z_m"],
                motion_speed_scale=speed["motion_speed_scale"],
                max_tcp_speed_mps=speed["max_tcp_speed_mps"],
                canfd_frequency_hz=speed["canfd_frequency_hz"],
                tcp_speed_mps=speed.get("tcp_speed_mps"),
            )

        pool = CuroboPlannerPool(
            planner_factory=lambda **kwargs: SynchronousPlanner(**kwargs),
            max_entries=6,
        )
        try:
            options = {kind: dict(values) for kind, values in config["subtask_options"].items()}
            if execute or (subtask in {"insert_bowl", "place_bowl_on_table"}
                           and config.get("prepared_hold_report") is not None):
                # 从旧持碗报告开始的独立预检会采集新帧，仍需人工重新
                # 确认当前夹持，不能把历史回答当成此帧的事实。
                for kind in ("pick", "insert", "knob", "extract", "table"):
                    if kind in options:
                        options[kind]["ask_manual"] = input
            if subtask == "all":
                return run_observed_microwave_task1(
                    observe_collision_scene=observer, make_runtime=make_runtime,
                    planner_pool=pool, output_dir=output, subtask_options=options,
                    execute=execute, scene_drift_limits=config["scene_drift_limits"],
                    include_knob=config.get("include_knob", True),
                )
            hold = config.get("prepared_hold_report")
            prepared = None if hold is None else json.loads(hold.read_text(encoding="utf-8"))
            if prepare_held_bowl:
                if subtask not in {"insert_bowl", "place_bowl_on_table"} or hold is not None:
                    raise ValueError("现场持碗准备只用于独立放碗/桌面 Place，且不能混用证据文件")
                first_scene = observer()
                prepared = _prepare_current_held_bowl_report(
                    first_scene, output=output, ask=input,
                )
                live_observer = observer

                def observe_first_then_live():
                    nonlocal first_scene
                    if first_scene is not None:
                        scene, first_scene = first_scene, None
                        return scene
                    return live_observer()

                observer = observe_first_then_live
            return run_observed_microwave_subtask(
                name=subtask, observe_collision_scene=observer,
                make_runtime=make_runtime, planner_pool=pool,
                output_dir=output, options=options if subtask in _COMBO_KINDS else options[_OPTION_KEY[subtask]],
                execute=execute, prepared_hold_report=prepared,
                scene_drift_limits=config["scene_drift_limits"],
            )
        finally:
            pool.clear()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="现场微波炉子任务/Task；默认只预检首段")
    parser.add_argument("--config", required=True, help="现场标定、服务和参数 JSON")
    parser.add_argument("--collision-mode", choices=("full", "robot_only"))
    parser.add_argument("--subtask", choices=("all", *SUBTASK_NAMES), required=True)
    parser.add_argument("--output", help="本次记录目录；省略时自动创建带时间戳的目录")
    parser.add_argument("--check-config", action="store_true", help="只检查静态配置，不连接设备")
    parser.add_argument("--no-knob", action="store_true",
                        help="仅完整 Task：跳过旋钮，保留第二次开门前过渡")
    parser.add_argument("--execute", action="store_true", help="逐段向机器人发令")
    parser.add_argument("--yes", action="store_true", help="确认执行模式；仍需现场后置检查")
    parser.add_argument("--prepare-held-bowl", action="store_true",
                        help="独立放碗/桌面 Place：观察当前持碗场景并由操作者确认")
    args = parser.parse_args(argv)
    if args.execute and not args.yes:
        raise ValueError("执行真机运动必须显式加 --yes")
    if args.no_knob and args.subtask != "all":
        raise ValueError("--no-knob 只适用于 --subtask all")
    config = load_site_config(args.config, subtask=args.subtask, execute=args.execute,
                              prepare_held_bowl=args.prepare_held_bowl,
                              no_knob=args.no_knob)
    if args.collision_mode is not None:
        config['motion'] = {**config.get('motion', {}), 'collision_mode': args.collision_mode}
    if args.check_config:
        print("配置检查通过：静态字段齐全；尚未验证相机、Pose session、标定精度或真机")
        return 0
    saved_hold_recheck = (args.subtask in {"insert_bowl", "place_bowl_on_table"}
                          and config.get("prepared_hold_report") is not None)
    if (args.execute or args.prepare_held_bowl or saved_hold_recheck) and not sys.stdin.isatty():
        raise ValueError("现场执行或持碗人工复核需要交互终端")
    output = (Path(args.output) if args.output else
              Path("runs/realworld") / (
                  f"{args.subtask}-{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}"
              ))
    report = run_site(config, subtask=args.subtask,
                      output_dir=output, execute=args.execute,
                      prepare_held_bowl=args.prepare_held_bowl)
    print(json.dumps({"status": report.get("status"), "success": report.get("success"),
                      "requested_scope": report.get("requested_scope", args.subtask),
                      "task_success": report.get("task_success"),
                      "output_dir": str(output.resolve())},
                     ensure_ascii=False))
    return 0 if report.get("success") is True or report.get("status") == "validated_prefix" else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError) as error:
        # 现场配置错误在终端只打印关键字段；阶段运行的详细 traceback
        # 仍由各子任务报告保存，避免操作者从长栈里找真正原因。
        print(f"配置或前置条件错误：{error}", file=sys.stderr)
        raise SystemExit(2) from None
