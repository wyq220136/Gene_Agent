"""微波炉现场子任务/完整 Task 的薄入口；默认仅规划首段。

此脚本只装配现场观测、现有 cuRobo pool、真机 Runtime 和已经存在的
子任务函数，不定义第二套操作顺序。正式入口从自动感知准备记录读取标定，
每次观测实时计算门角和旋钮角，不要求手填站点或角度 JSON。
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
from gene_agent.feedback.runtime_timing import runtime_timing_scope, timed_runtime

from gene_agent.realworld.microwave.task import SUBTASK_NAMES
from gene_agent.realworld.microwave.task import run_observed_microwave_subtask
from gene_agent.realworld.microwave.task import run_observed_microwave_task1
from gene_agent.knowledge.geometry import validate_pose
from gene_agent.realworld.scene_drift import SceneDriftLimits
from gene_agent.realworld.microwave.configuration import runtime_configuration


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
_DEFAULT_GRIPPER_CONFIG = PROJECT_ROOT / "config/robots/standard_gripper.json"


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


def _project_path(value, *, name: str, default: Path) -> Path:
    """Resolve a hardware profile like the AffordToolAct CLI does.

    Site files live below ``data/task_1`` while robot/gripper profiles live at
    the repository root, so relative profile paths must not depend on cwd or
    the site file's directory.
    """
    if value is None:
        path = default
    else:
        raw = Path(_nonempty(value, name)).expanduser()
        path = raw if raw.is_absolute() else PROJECT_ROOT / raw
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
    """仅供旧记录的离线兼容检查；正式 CLI 不再接受手填站点文件。"""
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
    ports = payload.get("robot_ports", {"left": 8080, "right": 8080})
    if not isinstance(ports, dict) or set(ports) != {"left", "right"}:
        raise ValueError("robot_ports 必须显式包含 left 和 right")
    normalized_ports = {}
    for side in ("left", "right"):
        value = ports[side]
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 65535:
            raise ValueError(f"robot_ports.{side} 必须是 1~65535 的整数")
        normalized_ports[side] = value
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
    gripper_config = _project_path(
        payload.get("gripper_config"), name="gripper_config",
        default=_DEFAULT_GRIPPER_CONFIG,
    )
    gripper_config_bytes = gripper_config.read_bytes()
    try:
        gripper_payload = json.loads(gripper_config_bytes.decode("utf-8"))
        max_width = float(gripper_payload["max_opening_width_m"])
        tcp_z = float(gripper_payload["flange_to_tcp_z_m"])
        tcp_rotation_z = float(gripper_payload["flange_to_tcp_rotation_z_deg"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(
            "gripper_config 必须包含有效的 max_opening_width_m、"
            "flange_to_tcp_z_m 和 flange_to_tcp_rotation_z_deg"
        ) from error
    if not math.isfinite(max_width) or max_width <= 0:
        raise ValueError("gripper_config.max_opening_width_m 必须为正有限数")
    if not math.isfinite(tcp_z) or not math.isfinite(tcp_rotation_z):
        raise ValueError("gripper_config 的 TCP 位姿参数必须为有限数")
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
            "robot_ports": normalized_ports,
            "gripper_config": gripper_config,
            "gripper_config_sha256": hashlib.sha256(gripper_config_bytes).hexdigest(),
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


@contextmanager
def _planning_acceleration(config, output):
    """只管理规划资源；不参与观测、阶段选择或机器人执行。"""
    from gene_agent.motion.curobo.optimization import optimization_scope
    compatible = (config["active_group"] == "right"
                  and current_motion_config().collision_mode.value == "full")
    mode = config.get("curobo_optimization") or ("single_reuse" if compatible else "baseline")
    if mode == "single_reuse" and not compatible:
        raise ValueError("single_reuse仅支持右臂full碰撞配置")
    owner = None
    try:
        with optimization_scope(mode) as owner:
            yield
    finally:
        payload = owner.to_dict() if owner is not None else {"mode": mode}
        (output / "planning_optimization.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8")


@timed_runtime("output_dir")
@configured_motion("config")
def run_site(config: dict, *, subtask: str, output_dir: str | Path,
             execute: bool, prepare_held_bowl: bool = False, plan_only: bool = False) -> dict:
    """接通设备并调用现有子任务/Task；只在 execute=True 时允许 SDK 运动。"""
    if plan_only and (execute or subtask != "all" or prepare_held_bowl):
        raise ValueError("--plan-only 只用于 --subtask all，不能与 --execute/--prepare-held-bowl 混用")
    source = Path(config["source_path"])
    if hashlib.sha256(source.read_bytes()).hexdigest() != config["source_sha256"]:
        raise ValueError("现场配置文件或感知记录在检查后已改变，请重新检查再运行")
    setup = config.get("perception_setup")
    if subtask == "rotate_knob" and not config.get("include_knob", False):
        raise ValueError("当前为 no_knob 流程，不允许 rotate_knob")
    if setup is not None:
        from gene_agent.realworld.microwave.perception_setup import load_perception_setup
        load_perception_setup(source)
    gripper_config = Path(config["gripper_config"])
    if hashlib.sha256(gripper_config.read_bytes()).hexdigest() != config["gripper_config_sha256"]:
        raise ValueError("gripper_config 在检查后已改变，请重新检查再运行")
    output = Path(output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("本次输出目录已有内容，不能覆盖已有现场运行记录")
    output.mkdir(parents=True, exist_ok=True)
    drift = config["scene_drift_limits"]
    (output / ("runtime_input.json" if setup is not None else "site_input.json")).write_text(json.dumps({
        "schema_version": 1,
        "site_config_path": str(source),
        "site_config_sha256": config["source_sha256"],
        "subtask": subtask, "execute": bool(execute), "plan_only": bool(plan_only),
        "include_knob": config.get("include_knob", False),
        "prepare_held_bowl": bool(prepare_held_bowl),
        "active_group": config["active_group"],
        "robot_ports": config["robot_ports"],
        "gripper_config": str(config["gripper_config"]),
        "gripper_config_sha256": config["gripper_config_sha256"],
        "motion": current_motion_config().to_dict(),
        "camera_serial": config["camera"]["serial"],
        "pose_sessions": config["pose_sessions"],
        "runtime": config["runtime"],
        "scene_drift_limits": None if drift is None else asdict(drift),
        "input_mode": "automatic_perception" if setup is not None else "legacy_offline_compatibility",
        "angle_source": "same_frame_foundationpose" if setup is not None else "legacy_state_file",
        "subtask_options": config["subtask_options"],
        "claim": "仅记录输入来源；设备连接、规划和实际执行需另看阶段报告",
    }, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    from gene_agent.realworld.microwave.scene import make_live_microwave_observer
    from gene_agent.motion.curobo.optimization import create_planner
    from gene_agent.motion.curobo.session import CuroboPlannerPool
    from gene_agent.perception_modules.pose_est.pose_client import PoseClient
    from gene_agent.realworld.devices.arms.realman.connection import connect_realman
    from gene_agent.realworld.devices.arms.realman.adapter import (
        load_standard_gripper_flange_tcp,
    )
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

    observer_options = {}
    if setup is not None:
        from gene_agent.realworld.microwave.live_perception import collision_objects
        from gene_agent.realworld.microwave.operation_perception import OperationPerception
        scoped_perception = OperationPerception(setup, include_knob=config.get("include_knob", False))
        reader = scoped_perception.reader
        read_angles = reader
        read_anchor = reader.anchor
        from gene_agent.realworld.microwave.session_recovery import make_session_recovery
        observer_options = {"recover_session": make_session_recovery(setup),
                            "operation_perception": scoped_perception,
                            "observation_filter": collision_objects, "camera_calibration": setup["camera"],
                            "include_knob": config.get("include_knob", False)}

    with _record_entry_errors(output, config["source_sha256"]), ExitStack() as stack:
        from gene_agent.feedback.realworld_diagnostics import RealworldRunDiagnostics
        diagnostics = stack.enter_context(RealworldRunDiagnostics(
            output, perception_paths=(source.parent if setup is not None else source,
                                      output / "observations" / "0000"),
            options=config.get("diagnostics"), failure_only=True,
            metadata={"subtask": subtask, "execute": bool(execute),
                      "plan_only": bool(plan_only), "active_group": config["active_group"],
                      "motion": current_motion_config().to_dict()}))
        stack.enter_context(_planning_acceleration(config, output))
        try:
            gripper_payload = json.loads(
                Path(config["gripper_config"]).read_text(encoding="utf-8")
            )
            T_flange_tcp = load_standard_gripper_flange_tcp(config["gripper_config"])
            max_width_m = float(gripper_payload["max_opening_width_m"])

            def connect_side(side):
                kwargs = {
                    "group": side,
                    "port": config["robot_ports"][side],
                    "T_flange_tcp": T_flange_tcp,
                    "max_width_m": max_width_m,
                    **({"configure_gripper": False} if plan_only else {}),
                }
                # This keeps lightweight injected connectors usable while the
                # production connector receives the complete hardware profile.
                accepted = inspect.signature(connect_realman).parameters
                kwargs = {key: value for key, value in kwargs.items()
                          if key in accepted or any(
                              parameter.kind is inspect.Parameter.VAR_KEYWORD
                              for parameter in accepted.values()
                          )}
                return connect_realman(config["robot_ips"][side], **kwargs)

            connections = {
                side: stack.enter_context(connect_side(side))
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
            **observer_options,
        )
        observer = diagnostics.wrap_observer(observer)
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
            planner_factory=create_planner,
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
            if plan_only:
                from gene_agent.realworld.microwave.operation_perception import begin_observation_operation
                from gene_agent.realworld.microwave.plan_only import plan_microwave_task1
                begin_observation_operation(observer, "plan_only")
                initial_scene = observer()
                if setup is not None:
                    from gene_agent.realworld.microwave.observed_geometry import reconstruct_plan_scene
                    initial_scene = reconstruct_plan_scene(
                        initial_scene, perception_dir=source.parent, setup=setup,
                        output_dir=output / "observed_geometry",
                        T_world_sdk=config["scene_parameters"]["T_world_sdk"])
                return diagnostics.finish(plan_microwave_task1(
                    initial_scene, planner_pool=pool, output_dir=output,
                    subtask_options=options, include_knob=config.get("include_knob", False)))
            if subtask == "all":
                return diagnostics.finish(run_observed_microwave_task1(
                    observe_collision_scene=observer, make_runtime=make_runtime,
                    planner_pool=pool, output_dir=output, subtask_options=options,
                    execute=execute, scene_drift_limits=config["scene_drift_limits"],
                    include_knob=config.get("include_knob", False),
                ))
            hold = config.get("prepared_hold_report")
            prepared = None if hold is None else json.loads(hold.read_text(encoding="utf-8"))
            if prepare_held_bowl:
                if subtask not in {"insert_bowl", "place_bowl_on_table"} or hold is not None:
                    raise ValueError("现场持碗准备只用于独立放碗/桌面 Place，且不能混用证据文件")
                from gene_agent.realworld.microwave.operation_perception import begin_observation_operation
                begin_observation_operation(observer, subtask)
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

                from gene_agent.realworld.microwave.operation_perception import forward_observation_hooks
                forward_observation_hooks(live_observer, observe_first_then_live)
                observer = observe_first_then_live
            return diagnostics.finish(run_observed_microwave_subtask(
                name=subtask, observe_collision_scene=observer,
                make_runtime=make_runtime, planner_pool=pool,
                output_dir=output, options=options if subtask in _COMBO_KINDS else options[_OPTION_KEY[subtask]],
                execute=execute, prepared_hold_report=prepared,
                scene_drift_limits=config["scene_drift_limits"],
            ))
        finally:
            pool.clear()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="现场微波炉子任务/Task；默认只预检首段")
    parser.add_argument("--perception", required=True, help="prepare_microwave_realworld.py 自动生成的感知记录目录")
    parser.add_argument("--left-ip", required=True, help="左臂 RealMan IPv4 地址")
    parser.add_argument("--right-ip", required=True, help="右臂 RealMan IPv4 地址")
    parser.add_argument("--left-port", type=int, default=8080)
    parser.add_argument("--right-port", type=int, default=8080)
    parser.add_argument("--gripper-config", default=str(_DEFAULT_GRIPPER_CONFIG))
    parser.add_argument("--max-age-s", type=float, default=5., help="真实采集帧有效期；不刷新旧时间戳")
    parser.add_argument("--speed-scale", type=float, default=.1)
    parser.add_argument("--collision-mode", choices=("full", "robot_only"))
    parser.add_argument("--curobo-optimization", choices=("baseline", "single_reuse"),
                        help="默认右臂full碰撞复用任务内引擎；baseline用于性能对照")
    parser.add_argument("--no-freeze-import-gc", action="store_true",
                        help="关闭独立命令启动时的导入对象GC冻结")
    parser.add_argument("--diagnostic-mode", choices=("off", "on_failure"), default="on_failure",
                        help="默认补充失败 FK/约束证据；off 仍保留原始请求与候选清单")
    parser.add_argument("--diagnostic-budget-s", type=float, default=5., help="单次失败的诊断软预算，秒")
    parser.add_argument("--subtask", choices=("all", *SUBTASK_NAMES), required=True)
    parser.add_argument("--output", help="本次记录目录；省略时自动创建带时间戳的目录")
    parser.add_argument("--check-config", action="store_true", help="只检查静态配置，不连接设备")
    knobs = parser.add_mutually_exclusive_group()
    knobs.add_argument("--no-knob", dest="no_knob", action="store_true",
                       help="默认：不感知、不操作旋钮，保留第二次开门前过渡")
    knobs.add_argument("--with-knob", dest="no_knob", action="store_false",
                       help="显式启用旋钮；感知记录也必须包含旋钮 session")
    parser.set_defaults(no_knob=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true", help="逐段向机器人发令")
    mode.add_argument("--plan-only", "--plan_only", dest="plan_only", action="store_true",
                      help="完整 Task1 预测规划与回放；只读取初始现场，不发送运动或夹爪命令")
    parser.add_argument("--video", action="store_true", help="plan_only 后导出 planned 预览 MP4")
    parser.add_argument("--yes", action="store_true", help="确认执行模式；仍需现场后置检查")
    parser.add_argument("--prepare-held-bowl", action="store_true",
                        help="独立放碗/桌面 Place：观察当前持碗场景并由操作者确认")
    args = parser.parse_args(argv)
    if args.plan_only and (args.subtask != "all" or args.prepare_held_bowl):
        raise ValueError("--plan-only 需要 --subtask all，且不能准备人工持碗")
    if args.video and not args.plan_only:
        raise ValueError("--video 只用于 --plan-only 的规划预览")
    if args.execute and not args.yes:
        raise ValueError("执行真机运动必须显式加 --yes")
    if args.no_knob and args.subtask == "rotate_knob":
        raise ValueError("默认 no_knob，不执行 rotate_knob；启用旋钮需显式 --with-knob")
    if args.prepare_held_bowl and args.subtask not in {"insert_bowl", "place_bowl_on_table"}:
        raise ValueError("--prepare-held-bowl 只适用于独立放碗或桌面 Place")
    if args.subtask in {"insert_bowl", "place_bowl_on_table"} and not args.prepare_held_bowl:
        raise ValueError("独立放碗/桌面 Place 需要 --prepare-held-bowl 现场确认")
    config = runtime_configuration(perception=args.perception,
        left_ip=args.left_ip, right_ip=args.right_ip, left_port=args.left_port, right_port=args.right_port,
        max_age_s=args.max_age_s, speed_scale=args.speed_scale, gripper_config=args.gripper_config,
        no_knob=args.no_knob)
    kinds = (set(_OPTION_KEY.values()) if args.subtask == "all" else
             set(_COMBO_KINDS[args.subtask]) if args.subtask in _COMBO_KINDS else {_OPTION_KEY[args.subtask]})
    if args.no_knob:
        kinds.discard("knob")
    for kind in kinds:
        _required_options(kind, config["subtask_options"][kind])
    if args.collision_mode is not None:
        config['motion'] = {**config.get('motion', {}), 'collision_mode': args.collision_mode}
    from gene_agent.motion.curobo.diagnostics import DiagnosticOptions
    diagnostic_options = DiagnosticOptions(mode=args.diagnostic_mode, budget_s=args.diagnostic_budget_s)
    config["diagnostics"] = asdict(diagnostic_options)
    config["curobo_optimization"] = args.curobo_optimization
    if args.check_config:
        print("配置检查通过：已从感知/标定产物自动装配；尚未验证在线相机、Pose session 或真机")
        return 0
    saved_hold_recheck = (args.subtask in {"insert_bowl", "place_bowl_on_table"}
                          and config.get("prepared_hold_report") is not None)
    if (args.execute or args.prepare_held_bowl or saved_hold_recheck) and not sys.stdin.isatty():
        raise ValueError("现场执行或持碗人工复核需要交互终端")
    output = (Path(args.output) if args.output else
              Path("runs/realworld") / (
                  f"{args.subtask}-{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}"
              ))
    print("=" * 78, flush=True)
    print(f"[task] 开始：subtask={args.subtask} "
          f"mode={'execute' if args.execute else 'plan_only' if args.plan_only else 'preflight'}",
          flush=True)
    print(f"[task] collision={args.collision_mode or config.get('motion', {}).get('collision_mode', 'configured')} "
          f"diagnostics={args.diagnostic_mode} output={output.resolve()}", flush=True)
    print(f"[perception] source={Path(args.perception).expanduser().resolve()}", flush=True)
    print("=" * 78, flush=True)
    from gene_agent.motion.curobo.bootstrap import curobo_import_gc_scope
    bootstrap = None
    bootstrap_output_available = not output.exists() or not any(output.iterdir())
    with runtime_timing_scope(output):
        try:
            with curobo_import_gc_scope(enabled=not args.no_freeze_import_gc) as bootstrap:
                report = run_site(config, subtask=args.subtask,
                                  output_dir=output, execute=args.execute,
                                  prepare_held_bowl=args.prepare_held_bowl,
                                  **({"plan_only": True} if args.plan_only else {}))
        finally:
            if bootstrap is not None and bootstrap_output_available and output.is_dir():
                (output / "planning_bootstrap.json").write_text(
                    json.dumps(bootstrap, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")
        video_failed = False
        if args.video:
            try:
                from gene_agent.visualization.video_export import export_video
                export_video(output, output / "plan_preview.mp4", mode="planned",
                             cameras=(("observed", "gripper_side") if report.get("observed_geometry")
                                      else ("task", "gripper_side")), profile="microwave_task1")
            except Exception as error:
                video_failed = True
                write = {"status": "failed", "message": f"{type(error).__name__}: {error}"}
                (output / "video_result.json").write_text(json.dumps(write, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
                print(f"规划报告已保存，视频导出失败：{error}", file=sys.stderr)
    timing_path = output.resolve() / "runtime_timing.json"
    timing = json.loads(timing_path.read_text(encoding="utf-8")) if timing_path.is_file() else {}
    from gene_agent.feedback.realworld_diagnostics import print_realworld_console_summary
    print_realworld_console_summary(
        output, report, subtask=args.subtask, execute=args.execute,
        collision_mode=args.collision_mode or config.get("motion", {}).get("collision_mode"),
        plan_only=args.plan_only)
    print(json.dumps({"status": report.get("status"), "success": report.get("success"),
                      "requested_scope": report.get("requested_scope", args.subtask),
                      "task_success": report.get("task_success"),
                      "output_dir": str(output.resolve()),
                      "failure_analysis_ref": str(output.resolve() / "failure_analysis.json"),
                      "initial_scene_ref": str(output.resolve() / "initial_scene/manifest.json"),
                      "runtime_timing_ref": str(timing_path),
                      "started_at": timing.get("started_at"),
                      "finished_at": timing.get("finished_at"),
                      "total_duration_s": timing.get("total_duration_s"),
                      "timing_bottleneck": timing.get("bottleneck"),
                      "timing_comparison": timing.get("comparison")},
                     ensure_ascii=False))
    return 0 if not video_failed and (report.get("success") is True or report.get("status") == "validated_prefix") else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError) as error:
        # 现场配置错误在终端只打印关键字段；阶段运行的详细 traceback
        # 仍由各子任务报告保存，避免操作者从长栈里找真正原因。
        print(f"配置或前置条件错误：{error}", file=sys.stderr)
        raise SystemExit(2) from None
