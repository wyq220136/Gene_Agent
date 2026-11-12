"""固定板完整外参向导：MoveL 到旧记录位姿、当前双相机观测和 SDK base 点云预览。"""

from __future__ import annotations

from .storage import standard_camera_parameters, publish_result, default_run_dir, resolve_cli_paths, same_recorded_config

import argparse
import json
import os
import sys
import unicodedata
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from gene_agent.knowledge.geometry.transforms import validate_pose
from gene_agent.perception_modules.camera import CameraClient
from gene_agent.realworld.devices.arms.realman.connection import (
    connect_realman_pose_replay, connect_realman_read_only, normalize_realman_ip,
)
from gene_agent.realworld.devices.arms.realman.calibration_replay import ControllerReadError
from .guided_third_view_crosscheck import (
    _WINDOW, _check_camera_health, _jsonable, _load_pair, _prompt_pair, _record_pair, _save_json,
)
from .third_view_crosscheck import _rotation_deg, solve_fixed_camera_from_pair
from .wrist_handeye import board_object_points


def normalize_command(raw: str) -> str:
    """兼容全角键入和粘贴边界格式字符；仍要求完整、精确的命令词。"""
    return unicodedata.normalize("NFKC", raw).strip().strip("\ufeff\u200b\u200c\u200d").strip().upper()


def load_replay_targets(path: Path, indices: list[int], *, ip: str, group: str) -> list[dict]:
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    if record.get("robot_group") != group:
        raise ValueError("旧姿态的机械臂 group 与当前不一致")
    if normalize_realman_ip(record.get("robot_sdk_ip", "")) != normalize_realman_ip(ip):
        raise ValueError("旧姿态记录的机器人 IP 与当前不一致")
    if not indices or len(set(indices)) != len(indices):
        raise ValueError("请选择至少一个且不重复的旧样本编号")
    samples = record.get("samples", [])
    selected = []
    for index in indices:
        matches = [s for s in samples if s.get("index") == index]
        if len(matches) != 1:
            raise ValueError(f"旧姿态中没有唯一的样本 {index}")
        joints = np.asarray(matches[0]["joints_rad"], dtype=float)
        if joints.shape != (7,) or not np.isfinite(joints).all():
            raise ValueError(f"样本 {index} 缺少七个有限关节角，单位 rad")
        if "T_sdk_flange" not in matches[0]:
            raise ValueError(f"样本 {index} 缺少 T_sdk_flange，不能用于 MoveL 姿态回放")
        flange = validate_pose(np.asarray(matches[0]["T_sdk_flange"], dtype=float), field_name="recorded_T_sdk_flange")
        selected.append({"source_index": index, "joints_rad": joints.tolist(), "T_sdk_flange": flange.tolist()})
    return selected


def evaluate_samples(samples: list[dict], T_flange_wrist, *, max_mm=10.0, max_deg=3.0,
                     max_board_mm=5.0, max_board_deg=2.0) -> dict:
    """首个当前板位求解，其余样本独立检查；不以重复拟合消除误差。"""
    if not samples:
        raise ValueError("至少需要一组新拍摄的配对样本")
    limits = np.asarray([max_mm, max_deg, max_board_mm, max_board_deg], dtype=float)
    if not np.isfinite(limits).all() or np.any(limits <= 0):
        raise ValueError("外参与板位一致性阈值必须是有限正数")
    wrist = validate_pose(np.asarray(T_flange_wrist), field_name="T_flange_wrist")
    first = samples[0]
    solved = solve_fixed_camera_from_pair(
        np.asarray(first["T_sdk_flange"]), wrist,
        np.asarray(first["wrist"]["T_camera_board"]), np.asarray(first["third"]["T_camera_board"]),
    )
    third0 = validate_pose(np.asarray(first["third"]["T_camera_board"]))
    checks = []
    for sample in samples:
        board_wrist = validate_pose(np.asarray(sample["T_sdk_flange"]) @ wrist
                                    @ np.asarray(sample["wrist"]["T_camera_board"]))
        observed = validate_pose(np.asarray(sample["third"]["T_camera_board"]))
        board_third = validate_pose(solved["T_sdk_third"] @ observed)
        checks.append({
            "chain_translation_mm": float(np.linalg.norm(board_wrist[:3, 3] - board_third[:3, 3]) * 1000),
            "chain_rotation_deg": _rotation_deg(board_wrist, board_third),
            "third_board_shift_mm": float(np.linalg.norm(observed[:3, 3] - third0[:3, 3]) * 1000),
            "third_board_rotation_deg": _rotation_deg(third0, observed),
            "sdk_board_shift_mm": float(np.linalg.norm(board_wrist[:3, 3] - solved["T_sdk_board"][:3, 3]) * 1000),
            "sdk_board_rotation_deg": _rotation_deg(solved["T_sdk_board"], board_wrist),
        })
    mm = max(c["chain_translation_mm"] for c in checks)
    deg = max(c["chain_rotation_deg"] for c in checks)
    board_mm = max(c["third_board_shift_mm"] for c in checks)
    board_deg = max(c["third_board_rotation_deg"] for c in checks)
    sdk_mm = max(c["sdk_board_shift_mm"] for c in checks)
    sdk_deg = max(c["sdk_board_rotation_deg"] for c in checks)
    rotations = [_rotation_deg(np.asarray(first["T_sdk_flange"]), np.asarray(s["T_sdk_flange"]))
                 for s in samples]
    validation_count = sum(angle >= 10.0 for angle in rotations[1:])
    validation_error = (None if validation_count else
                        "缺少相对首帧手腕实际旋转至少10°的新样本；重复同一姿态不能独立验证手眼误差")
    return {
        **solved, "sample_count": len(samples), "validation_count": validation_count,
        "fresh_pair_count_after_anchor": len(samples) - 1, "validation_error": validation_error,
        "chain_translation_max_mm": mm, "chain_rotation_max_deg": deg,
        "third_board_spread_mm": board_mm, "third_board_spread_deg": board_deg,
        "sdk_board_spread_mm": sdk_mm, "sdk_board_spread_deg": sdk_deg,
        "flange_rotation_span_deg": max(rotations),
        "thresholds": {"chain_translation_mm": max_mm, "chain_rotation_deg": max_deg,
                       "third_board_translation_mm": max_board_mm, "third_board_rotation_deg": max_board_deg,
                       "validation_rotation_min_deg": 10.0},
        "quality_passed": (mm <= max_mm and deg <= max_deg and sdk_mm <= max_mm
                           and sdk_deg <= max_deg and board_mm <= max_board_mm
                           and board_deg <= max_board_deg and validation_count >= 1),
        "checks": checks,
        "note": "首样本求解；其余样本验证。数值一致性不是外部真值精度。",
    }


def preview_session(data_dir: Path, session: dict, report: dict, *, max_depth: float = 2.5) -> None:
    import open3d as o3d
    from .compute_extrinsic import _depth_to_points

    geometries = [o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15)]
    def cloud(item, transform):
        points, colors = _depth_to_points(item["depth_m"], item["rgb"], item["K"],
                                          max_depth=max_depth, max_points=50000)
        result = o3d.geometry.PointCloud()
        result.points = o3d.utility.Vector3dVector(points)
        result.colors = o3d.utility.Vector3dVector(colors.astype(float) / 255)
        result.transform(transform)
        return result

    samples = session["samples"]
    selected = sorted({0, len(samples) // 2, len(samples) - 1})
    for index in selected:
        pair = _load_pair(data_dir, samples[index])
        transform = pair["T_sdk_flange"] @ np.asarray(session["T_flange_wrist"])
        geometries.append(cloud(pair["wrist"], transform))
        if index == 0:
            geometries.append(cloud(pair["third"], np.asarray(report["T_sdk_third"])))
    board_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.12)
    board_axis.transform(np.asarray(report["T_sdk_board"]))
    geometries.append(board_axis)
    print("Open3D：所有点云位于 SDK base；手腕和第三视角均显示原始 RGB 颜色。")
    print("只显示基座与当前板两组坐标轴。检查棋盘格和静态桌面有无双层；运动中的机械臂不要求重合。")
    if not report["quality_passed"]:
        print("当前显示的是未通过质量检查的候选，用于定位误差，不能作为正式外参。")
    o3d.visualization.draw_geometries(geometries, window_name="Wrist + third-view RGB in SDK base")


def _parser():
    parser = argparse.ArgumentParser(description="固定板完整外参：可选回放旧关节姿态，重新拍双相机，组合并预览")
    parser.add_argument("--handeye", type=Path)
    parser.add_argument("--replay-samples", type=Path, help="旧 samples.json，默认取 handeye 同目录")
    parser.add_argument("--pose-indices", nargs="+", type=int, default=[1, 11, 20])
    parser.add_argument("--ip", default="192.168.1.18")
    parser.add_argument("--group", choices=("left", "right"), default="right")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--wrist-endpoint", default="tcp://127.0.0.1:18082")
    parser.add_argument("--third-endpoint", default="tcp://127.0.0.1:18083")
    parser.add_argument("--third-serial", required=False)
    parser.add_argument("--execute", action="store_true", help="启用 SDK MoveL 直线到旧法兰姿态；逐点回车确认")
    parser.add_argument("--speed-percent", type=int, default=5, help="SDK MoveL 速度百分比，整数1–10，默认5")
    parser.add_argument("--move-timeout", type=float, default=60.0)
    parser.add_argument("--state-source", choices=("udp", "tcp"), default="udp",
                        help="默认使用与 UMI 一致的 UDP 状态；tcp 仅作为显式诊断选择")
    parser.add_argument("--sdk-timeout-ms", type=int, default=2000, help="SDK 接收超时，毫秒")
    parser.add_argument("--udp-target-ip", help="机器人回传到本机的 IP，默认根据路由自动选择")
    parser.add_argument("--udp-port", type=int, default=8888)
    parser.add_argument("--udp-timeout", type=float, default=0.5, help="等待新鲜 UDP 状态的最大秒数")
    parser.add_argument("--max-joint-change-deg", type=float, default=0.2)
    parser.add_argument("--max-reprojection-px", type=float, default=2.0)
    parser.add_argument("--max-pair-gap-ms", type=float, default=250.0)
    parser.add_argument("--max-validation-mm", type=float, default=10.0)
    parser.add_argument("--max-validation-deg", type=float, default=3.0)
    parser.add_argument("--connect-retries", type=int, default=30)
    parser.add_argument("--connect-retry-interval", type=float, default=0.5)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--list-poses", action="store_true", help="仅列出所选旧目标，不连接设备")
    parser.add_argument("--no-preview", action="store_true", help="终端回车/y采样，代替实时OpenCV")
    parser.add_argument("--visualize", action="store_true", help="采完在Open3D叠加点云，包括失败候选")
    parser.add_argument("--preview-only", action="store_true", help="只读取--data-dir预览，不连接任何设备")
    return parser


def _report(session, args):
    return evaluate_samples(session["samples"], session["T_flange_wrist"],
                            max_mm=args.max_validation_mm, max_deg=args.max_validation_deg)


@contextmanager
def _capture_window(enabled: bool):
    """只在取帧时拥有窗口；终端 input/SDK 到位等待期间没有 GUI 事件循环。"""
    if not enabled:
        yield
        return
    import cv2
    os.environ.setdefault("QT_QPA_FONTS_DIR", "/usr/share/fonts")
    cv2.namedWindow(_WINDOW, cv2.WINDOW_NORMAL)
    try:
        cv2.resizeWindow(_WINDOW, 1400, 700)
        waiting = np.full((160, 900, 3), 35, dtype=np.uint8)
        cv2.putText(waiting, "Loading wrist + third-view frames... Q to quit",
                    (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow(_WINDOW, waiting)
        cv2.waitKey(1)
        yield
    finally:
        cv2.destroyWindow(_WINDOW)
        cv2.waitKey(1)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    resolve_cli_paths(args, "handeye", "data_dir", "replay_samples")
    if not args.preview_only and args.data_dir and (args.data_dir / "RUN_FAILED.json").exists():
        raise ValueError("该目录已被判定为失败，不能续采；请省略 --data-dir 和 --resume，创建新一轮")
    if args.preview_only:
        if args.data_dir is None:
            raise ValueError("--preview-only 需要 --data-dir")
        session = json.loads((args.data_dir / "session.json").read_text())
        report = json.loads((args.data_dir / "quality_report.json").read_text())
        preview_session(args.data_dir, session, report)
        return 0
    if args.handeye is None:
        raise ValueError("需要 --handeye 指定已有手眼标定")
    handeye = json.loads(args.handeye.read_text(encoding="utf-8"))
    if handeye.get("calibration_type") != "eye_in_hand_fixed_board" or handeye.get("robot_group") != args.group:
        raise ValueError("手眼标定类型或机械臂 group 不一致")
    wrist = validate_pose(np.asarray(handeye["T_flange_camera"]))
    args.ip = normalize_realman_ip(args.ip)
    source = args.replay_samples or args.handeye.parent / handeye.get("source_samples", "samples.json")
    targets = load_replay_targets(source, args.pose_indices, ip=args.ip, group=args.group)
    for target in targets:
        print(f"旧样本 {target['source_index']}，MoveL 法兰目标位置（米）："
              f"{np.round(np.asarray(target['T_sdk_flange'])[:3, 3], 4).tolist()}；"
              f"原关节记录（度）：{np.round(np.rad2deg(target['joints_rad']), 2).tolist()}")
    if args.list_poses:
        return 0
    values = [args.move_timeout, args.max_joint_change_deg, args.max_reprojection_px,
              args.max_pair_gap_ms, args.max_validation_mm, args.max_validation_deg, args.udp_timeout]
    if not np.isfinite(values).all() or min(values) <= 0 or not 1 <= args.speed_percent <= 10:
        raise ValueError("运动/采集阈值须为有限正数，回放速度限1–10%")
    if args.sdk_timeout_ms < 100 or not 1 <= args.udp_port <= 65535:
        raise ValueError("SDK 接收超时至少100ms，UDP端口须为1–65535")
    if args.wrist_endpoint == args.third_endpoint:
        raise ValueError("两台相机必须使用不同 endpoint")
    if args.third_serial == handeye["camera_serial"]:
        raise ValueError("第三相机不能与手腕相机为同一设备")
    if (not args.no_preview or args.visualize) and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        raise RuntimeError("需要图形桌面；无桌面时加 --no-preview 并去掉 --visualize")
    pattern = tuple(handeye["pattern_size_inner_corners"])
    square = float(handeye["square_size_m"])
    board_object_points(pattern, square)
    args.data_dir = args.data_dir or default_run_dir("fixed_board")
    path = args.data_dir / "session.json"
    config = {
        "schema_version": 1, "calibration_type": "fixed_board_from_handeye_and_joint_targets",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "handeye_file": str(args.handeye.resolve()), "T_flange_wrist": wrist.tolist(),
        "robot_sdk_ip": args.ip, "robot_group": args.group, "replay_samples": str(source.resolve()),
        "targets": targets, "wrist_endpoint": args.wrist_endpoint, "third_endpoint": args.third_endpoint,
        "motion_backend": "realman_movel",
        "communication_requested": {"state_source": args.state_source, "sdk_timeout_ms": args.sdk_timeout_ms,
                                    "udp_target_ip": args.udp_target_ip, "udp_port": args.udp_port,
                                    "udp_timeout_s": args.udp_timeout},
        "wrist_serial": handeye["camera_serial"], "third_serial": args.third_serial,
        "pattern_size_inner_corners": list(pattern), "square_size_m": square,
        "samples": [], "motion_events": [], "source_handeye": handeye,
    }
    if args.resume:
        session = json.loads(path.read_text(encoding="utf-8"))
        for key in ("calibration_type", "handeye_file", "T_flange_wrist", "robot_sdk_ip", "robot_group",
                    "targets", "motion_backend", "communication_requested", "wrist_endpoint", "third_endpoint",
                    "wrist_serial", "pattern_size_inner_corners", "square_size_m"):
            if not same_recorded_config({key: session.get(key)}, {key: config[key]}, ("handeye_file", "replay_samples", "board_pose_source")):
                raise ValueError(f"续采参数 {key} 与已有记录不一致")
        if args.third_serial and session["third_serial"] not in (None, args.third_serial):
            raise ValueError("续采第三相机 serial 不一致")
    else:
        if args.data_dir.exists() and any(args.data_dir.iterdir()):
            raise FileExistsError(f"目录非空：{args.data_dir}；请使用新目录或 --resume")
        session = config
    if (args.data_dir / "third_view_extrinsic.json").exists():
        raise FileExistsError("已有正式外参；预览请加 --preview-only，新采集请用新目录")
    print("\n棋盘格已经移动也可标定：现在将板和第三相机固定，此后全程不要移动。")
    print("每组都重新读取机器人实际法兰，并拍摄两相机；不复用旧照片或旧板基座位姿。")
    print("本次：SDK MoveL 直线回放，每点按回车确认。" if args.execute else "本次只读：请人工摆姿态，回车开始采样。")
    if args.execute:
        print("以控制器当前工具 TCP 走直线，到达记录的法兰姿态；未做现场障碍碰撞规划，失败不降级。")
    print(f"控制连接使用同一个 SDK；状态源 {args.state_source}，SDK 接收超时 {args.sdk_timeout_ms}ms。")
    with ExitStack() as stack:
        cameras = {}
        for key, endpoint in (("wrist", args.wrist_endpoint), ("third", args.third_endpoint)):
            camera = CameraClient(endpoint)
            stack.callback(camera.close)
            _check_camera_health(camera, name="手腕" if key == "wrist" else "第三视角", endpoint=endpoint)
            cameras[key] = camera
        connector = connect_realman_pose_replay if args.execute else connect_realman_read_only
        robot = stack.enter_context(connector(args.ip, group=args.group, port=args.port,
                                              connect_retries=args.connect_retries,
                                              connect_retry_interval_s=args.connect_retry_interval,
                                              state_source=args.state_source, sdk_timeout_ms=args.sdk_timeout_ms,
                                              udp_target_ip=args.udp_target_ip, udp_port=args.udp_port,
                                              udp_timeout_s=args.udp_timeout))
        session["communication_actual"] = robot.communication_info
        args.data_dir.mkdir(parents=True, exist_ok=True)
        _save_json(path, session)
        print(f"输出目录：{args.data_dir}；中断后同一命令指定此目录并加 --resume。")
        print(f"本次选取 {len(targets)} 个记录位姿；每个单独确认。"
              "双相机窗口在回放到位或选择当前姿态后打开，采完该组后关闭。")
        while len(session["samples"]) < len(targets):
            index = len(session["samples"])
            target = targets[index]
            _, before = robot.read_flange_pose()
            difference = np.rad2deg(np.asarray(target["joints_rad"]) - before)
            print(f"\n[{index + 1}/{len(targets)}] 旧样本 {target['source_index']}；原关节记录与当前差（度，仅参考）："
                  f"{np.round(difference, 2).tolist()}")
            if args.execute:
                raw_answer = input("检查通路后按回车：MoveL 回放；c+回车：当前姿态采样；q+回车：退出：")
            else:
                raw_answer = input("人工调整手腕并停稳。按回车：开始采样；q+回车：退出：")
            answer = normalize_command(raw_answer)
            if not raw_answer.strip():
                answer = "MOVE" if args.execute else "CURRENT"
            elif answer == "C":
                answer = "CURRENT"
            if answer == "Q":
                return 1
            if answer == "MOVE" and args.execute:
                event = {"source_index": target["source_index"], "target_T_sdk_flange": target["T_sdk_flange"],
                         "backend": "realman_movel",
                         "status": "command_requested", "requested_at_utc": datetime.now(timezone.utc).isoformat()}
                session["motion_events"].append(event)
                _save_json(path, session)
                try:
                    event["result"] = robot.move_to_flange(target["T_sdk_flange"], speed_percent=args.speed_percent,
                                                           timeout_s=args.move_timeout, expected_start_joints_rad=before)
                    event["status"] = "measured_reached"
                    event["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
                except ControllerReadError as exc:
                    event["status"] = "failed_or_interrupted" if exc.motion_sent else "not_sent"
                    event["error"] = repr(exc)
                    _save_json(path, session)
                    if not exc.motion_sent:
                        print(f"{exc}。本目标未运动；检查连接后回车重试，或 c+回车使用当前姿态。", flush=True)
                        continue
                    raise
                except BaseException as exc:
                    event["status"] = "failed_or_interrupted"
                    event["error"] = repr(exc)
                    _save_json(path, session)
                    raise
                _save_json(path, session)
            elif answer != "CURRENT":
                print(f"未接受输入 {raw_answer!r}；直接回车继续，c+回车使用当前姿态，q+回车退出。")
                continue
            while True:
                with _capture_window(not args.no_preview):
                    pair = _prompt_pair(cameras, robot, args, stage=f"固定板 {index + 1}/{len(targets)}",
                                        pattern_size=pattern, square_size=square,
                                        wrist_serial=session["wrist_serial"], third_serial=session["third_serial"],
                                        data_dir=args.data_dir)
                if pair is None:
                    return 1
                for key in ("wrist", "third"):
                    reference = (session["samples"][0][key]["K"] if session["samples"] else
                                 handeye.get("intrinsic_matrix") if key == "wrist" else None)
                    if reference is not None and not np.allclose(reference, pair[key]["K"], atol=1e-6, rtol=0):
                        raise ValueError(f"{key} 相机内参与标定/此前采样不一致")
                candidate = {"T_sdk_flange": pair["T_sdk_flange"].tolist(),
                             "wrist": {"T_camera_board": pair["wrist"]["T_camera_board"].tolist()},
                             "third": {"T_camera_board": pair["third"]["T_camera_board"].tolist()}}
                check = evaluate_samples(session["samples"] + [candidate], wrist)
                if check["third_board_spread_mm"] > 5 or check["third_board_spread_deg"] > 2:
                    print("此帧未收录：固定相机看到的板位发生变化。请检查板/固定相机是否移动或角点原点是否翻转。")
                    continue
                break
            metadata = _record_pair(args.data_dir, f"pose_{index + 1:03d}", pair)
            metadata["source_target"] = target
            metadata["recorded_vs_capture_max_joint_deg"] = float(np.rad2deg(np.max(
                np.abs(pair["joints_rad"] - target["joints_rad"]))))
            session["samples"].append(metadata)
            session["third_serial"] = pair["third"]["serial"]
            _save_json(path, session)
            report = _report(session, args)
            _save_json(args.data_dir / "quality_report.json", _jsonable(report))
            print(f"已收录；两条链最大偏差 {report['chain_translation_max_mm']:.2f} mm / "
                  f"{report['chain_rotation_max_deg']:.2f}°。板保持固定，继续改变手腕。")
        report = _report(session, args)
        _save_json(args.data_dir / "quality_report.json", _jsonable(report))
        print(f"\n独立验证 {report['validation_count']} 组；双链 {report['chain_translation_max_mm']:.2f} mm / "
              f"{report['chain_rotation_max_deg']:.2f}°；SDK板位变化 {report['sdk_board_spread_mm']:.2f} mm / "
              f"{report['sdk_board_spread_deg']:.2f}°。")
        if args.visualize:
            preview_session(args.data_dir, session, report)
        if not report["quality_passed"]:
            if report["validation_error"]:
                print(report["validation_error"])
            print("数值未通过；已保存候选和原始样本，不生成正式外参。可用 --preview-only 再检查。")
            return 2
        if input("检查点云与角点后，输入 ACCEPT 保存完整外参；其他输入取消：").strip() != "ACCEPT":
            return 1
        save_results(args.data_dir, session, report, visual_confirmed=args.visualize)
        print(f"已保存当前板位和完整第三视角外参：{args.data_dir / 'third_view_extrinsic.json'}")
        return 0


def save_results(data_dir: Path, session: dict, report: dict, *, visual_confirmed: bool) -> None:
    """只由最终 ACCEPT 路径调用，不把旧板位写回原手眼目录。"""
    if not report["quality_passed"]:
        raise ValueError("未通过质量检查，不能发布正式外参")
    sdk_board = np.asarray(report["T_sdk_board"])
    third_board = np.asarray(report["T_third_board"])
    sdk_third = np.asarray(report["T_sdk_third"])
    common = {"pose_convention": "T_A_B maps B coordinates into A; translation in meter",
              "source_session": "session.json", "quality_report": "quality_report.json"}
    _save_json(data_dir / "board_to_robot.json", {
        **common, "T_sdk_board": sdk_board.tolist(), "board_to_robot": sdk_board.tolist(),
        "robot_to_board": np.linalg.inv(sdk_board).tolist(), "robot_group": session["robot_group"],
    })
    _save_json(data_dir / "camera_to_board.json", {
        **common, "T_camera_board": third_board.tolist(), "T_board_camera": np.linalg.inv(third_board).tolist(),
        "board_to_camera": third_board.tolist(), "camera_to_board": np.linalg.inv(third_board).tolist(),
        "camera_serial": session["third_serial"], "origin_confirmed": True,
    })
    _save_json(data_dir / "third_view_extrinsic.json", standard_camera_parameters(data_dir, session["samples"][0]["third"], {
        **common, "calibration_type": "fixed_third_view_from_existing_handeye",
        "T_sdk_third": sdk_third.tolist(), "T_third_sdk": np.linalg.inv(sdk_third).tolist(),
        "T_sdk_board": sdk_board.tolist(), "T_third_board": third_board.tolist(),
        "camera_serial": session["third_serial"], "robot_group": session["robot_group"],
        "handeye_file": session["handeye_file"], "open3d_visual_confirmed": bool(visual_confirmed),
        "validation_count": report["validation_count"], "quality_passed": True,
        "communication": session.get("communication_actual", {}),
    }))
    publish_result(data_dir, "third_camera")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("已中断；若中断发生在回放中，已尝试停止并清除运动队列。", file=sys.stderr)
        raise SystemExit(130)
