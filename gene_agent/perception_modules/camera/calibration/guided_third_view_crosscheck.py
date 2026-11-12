"""默认固定板直接标定；--crosscheck 显式启用旧双相机移板检查。"""

from __future__ import annotations

from .storage import standard_camera_parameters, publish_result, default_run_dir, resolve_cli_paths, same_recorded_config

import argparse
import json
import os
import sys
import time
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from gene_agent.knowledge.geometry.transforms import validate_pose
from gene_agent.perception_modules.camera import CameraClient
from gene_agent.perception_modules.camera.calibration.guided_wrist_handeye import _draw_corners
from gene_agent.perception_modules.camera.calibration.third_view_crosscheck import (
    _rotation_deg, check_flange_pose_diversity, compare_board_poses,
    solve_fixed_camera_from_pair,
)
from gene_agent.perception_modules.camera.calibration.wrist_handeye import (
    check_stationary, detect_board_pose,
)
from gene_agent.realworld.devices.arms.realman.connection import connect_realman_read_only, normalize_realman_ip

_WINDOW = "Wrist + third-view chessboard cross-check"


def _save_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _check_camera_health(camera, *, name: str, endpoint: str,
                         retries: int = 2, retry_interval_s: float = 0.5) -> dict:
    """相机服务可能在处理上一条取帧请求；短暂超时后用新 REQ socket 重试。"""
    if retries < 1 or retry_interval_s < 0:
        raise ValueError("相机健康检查重试次数至少为 1，间隔不能为负")
    for attempt in range(1, retries + 1):
        try:
            health = camera.health()
        except TimeoutError as exc:
            if attempt == retries:
                raise RuntimeError(
                    f"{name}相机服务 {endpoint} 连续 {retries} 次 health 未响应；"
                    "请确认对应 camera_server 已启动、endpoint 正确，并查看服务终端有无采帧错误"
                ) from exc
            print(f"{name}相机服务 {endpoint} 暂未响应 health，"
                  f"{attempt}/{retries}，{retry_interval_s:.1f} 秒后重试。", flush=True)
            time.sleep(retry_interval_s)
            continue
        except RuntimeError as exc:
            raise RuntimeError(f"{name}相机服务 {endpoint} 报错：{exc}") from exc
        if not health.get("ok"):
            raise RuntimeError(f"{name}相机服务 {endpoint} 未就绪：{health}")
        print(f"{name}相机服务已连接：{endpoint}", flush=True)
        return health
    raise AssertionError("相机健康检查循环异常结束")


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value


def _detect(packet, pattern_size, square_size_m, max_reprojection_px, serial):
    if serial and packet.serial != serial:
        raise ValueError(f"相机 serial 不符：期望 {serial}，实际 {packet.serial}")
    found = detect_board_pose(packet.rgb, packet.intrinsic_matrix,
                              pattern_size, square_size_m, detector="sb")
    if found is None:
        raise ValueError("未检测到完整棋盘格")
    if found["reprojection_rmse_px"] > max_reprojection_px:
        raise ValueError(f"重投影误差 {found['reprojection_rmse_px']:.2f} px 超出阈值")
    return found


def capture_pair(
    wrist_camera, third_camera, robot, *, pattern_size: tuple[int, int],
    square_size_m: float, wrist_serial: str | None, third_serial: str | None,
    max_joint_change_deg: float, max_reprojection_px: float,
    max_pair_gap_ms: float = 250.0,
) -> dict:
    """用机器人前后状态包住两次新帧；不发送运动命令。"""
    wrist_camera.get_frame()
    third_camera.get_frame()
    _before_pose, before_joints = robot.read_flange_pose()
    start_ns = time.time_ns()
    wrist_packet = wrist_camera.get_frame()
    wrist_received_ns = time.time_ns()
    third_packet = third_camera.get_frame()
    third_received_ns = time.time_ns()
    end_ns = time.time_ns()
    flange, after_joints = robot.read_flange_pose()
    moved_deg = check_stationary(before_joints, after_joints,
                                 max_change_deg=max_joint_change_deg)
    gap_ms = (third_received_ns - wrist_received_ns) / 1e6
    if max_pair_gap_ms <= 0 or gap_ms < 0:
        raise ValueError("配对帧最大时间间隔必须为正，主机时间不可倒退")
    if gap_ms > max_pair_gap_ms:
        raise ValueError(f"两相机采帧时间间隔 {gap_ms:.1f} ms 超过 {max_pair_gap_ms:.1f} ms")
    if wrist_packet.serial == third_packet.serial:
        raise ValueError("手腕相机与第三视角相机 serial 相同，请检查两个 endpoint")
    wrist_found = _detect(wrist_packet, pattern_size, square_size_m,
                          max_reprojection_px, wrist_serial)
    third_found = _detect(third_packet, pattern_size, square_size_m,
                          max_reprojection_px, third_serial)
    def item(packet, found):
        return {
            "T_camera_board": found["T_camera_board"],
            "corners": found["corners"],
            "reprojection_rmse_px": found["reprojection_rmse_px"],
            "frame_number": packet.frame_number,
            "timestamp_ms": packet.timestamp_ms,
            "serial": packet.serial,
            "K": np.asarray(packet.intrinsic_matrix),
            "rgb": packet.rgb,
            "depth_m": packet.depth,
        }
    return {
        "T_sdk_flange": flange, "joints_rad": after_joints,
        "joint_change_deg": moved_deg,
        "host_capture_start_ns": start_ns, "host_capture_end_ns": end_ns,
        "host_frame_gap_ms": gap_ms,
        "wrist": item(wrist_packet, wrist_found),
        "third": item(third_packet, third_found),
    }


def _panel(packet, found, label: str, status: str):
    import cv2

    picture = (_draw_corners(packet.rgb, found["corners"])
               if found is not None else cv2.cvtColor(packet.rgb, cv2.COLOR_RGB2BGR))
    width = 650
    height = max(1, round(picture.shape[0] * width / picture.shape[1]))
    picture = cv2.resize(picture, (width, height), interpolation=cv2.INTER_AREA)
    picture = cv2.copyMakeBorder(picture, 66, 0, 0, 0, cv2.BORDER_CONSTANT)
    cv2.putText(picture, f"{label} | serial {packet.serial} | frame {packet.frame_number}",
                (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cv2.putText(picture, status[:65], (8, 51), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (50, 220, 50) if found is not None else (0, 170, 255), 2)
    return picture


def _montage(pair: dict, stage: str) -> np.ndarray:
    import cv2

    panels = []
    for key, title in (("wrist", "WRIST"), ("third", "THIRD VIEW")):
        item = pair[key]
        packet = type("Packet", (), {"rgb": item["rgb"], "serial": item["serial"],
                                     "frame_number": item["frame_number"]})()
        panels.append(_panel(packet, {"corners": item["corners"]}, title,
                             f"PnP {item['reprojection_rmse_px']:.2f} px | corner 0 in red"))
    height = max(panel.shape[0] for panel in panels)
    panels = [cv2.copyMakeBorder(panel, 0, height - panel.shape[0], 0, 0,
                                 cv2.BORDER_CONSTANT) for panel in panels]
    image = np.hstack(panels)
    image = cv2.copyMakeBorder(image, 42, 0, 0, 0, cv2.BORDER_CONSTANT)
    cv2.putText(image, f"{stage} | release pedal, SPACE confirm | N retry | Q quit",
                (12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    return image


def _confirm_candidate_gui(path: Path, *, window_name: str = _WINDOW) -> str:
    """冻结实际配对帧；连续空格需间隔安静期，过滤多数长按自动重复。"""
    import cv2

    display = cv2.imread(str(path))
    if display is None:
        raise RuntimeError(f"无法打开候选画面：{path}")
    cv2.imshow(window_name, display)
    last_space_at = time.monotonic()
    while True:
        raw = cv2.waitKey(20)
        now = time.monotonic()
        if raw < 0:
            continue
        key = raw & 0xFF
        if key == 32:
            if now - last_space_at >= 0.8:
                return "y"
            last_space_at = now
        if key in (ord("y"), ord("Y")):
            return "y"
        if key in (ord("n"), ord("N"), 27):
            return "n"
        if key in (ord("q"), ord("Q")):
            return "q"


def _wait_for_capture(cameras: dict, *, stage: str, pattern_size: tuple[int, int],
                      square_size_m: float, wrist_serial: str | None,
                      third_serial: str | None, max_reprojection_px: float) -> bool:
    import cv2

    last_error = None
    last_readiness = None
    while True:
        packets = {}
        found = {}
        statuses = {}
        try:
            for key in ("wrist", "third"):
                packets[key] = cameras[key].get_frame()
                expected = wrist_serial if key == "wrist" else third_serial
                try:
                    found[key] = _detect(packets[key], pattern_size, square_size_m,
                                         max_reprojection_px, expected)
                    statuses[key] = f"BOARD FOUND | {found[key]['reprojection_rmse_px']:.2f} px"
                except ValueError as exc:
                    found[key] = None
                    statuses[key] = str(exc)
        except (TimeoutError, RuntimeError) as exc:
            if str(exc) != last_error:
                print(f"实时取帧失败，继续等待：{exc}", flush=True)
                last_error = str(exc)
            if cv2.waitKey(50) & 0xFF in (ord("q"), ord("Q"), 27):
                return False
            continue
        last_error = None
        panels = [_panel(packets[key], found[key], key.upper(), statuses[key])
                  for key in ("wrist", "third")]
        height = max(panel.shape[0] for panel in panels)
        panels = [cv2.copyMakeBorder(panel, 0, height - panel.shape[0], 0, 0,
                                     cv2.BORDER_CONSTANT) for panel in panels]
        picture = np.hstack(panels)
        picture = cv2.copyMakeBorder(picture, 43, 0, 0, 0, cv2.BORDER_CONSTANT)
        same_camera = packets["wrist"].serial == packets["third"].serial
        ready = all(found.values()) and not same_camera
        # OpenCV 默认字体不能显示中文错误；终端明确解释为何采样键未生效。
        reasons = [f"{'手腕' if name == 'wrist' else '第三视角'}相机：{statuses[name]}"
                   for name in ("wrist", "third") if found[name] is None]
        if same_camera:
            reasons.append("两相机 serial 相同，请检查两个 endpoint 是否连接到同一台相机")
        readiness = (found["wrist"] is not None, found["third"] is not None, same_camera)
        if readiness != last_readiness:
            if ready:
                print("双相机棋盘格检测合格：点击图像窗口使其获得焦点，停稳后按空格或回车采样。", flush=True)
            else:
                print(f"等待采样条件：{'；'.join(reasons)}。要求内角点 {pattern_size[0]}×{pattern_size[1]}。", flush=True)
            last_readiness = readiness
        text = "READY | SPACE/ENTER capture | Q quit" if ready else "BLOCKED | see terminal for reason | Q quit"
        cv2.putText(picture, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (255, 255, 255), 2)
        cv2.imshow(_WINDOW, picture)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            return False
        if key in (32, 13, 10):
            if ready:
                return True
            print(f"采样未触发：{'；'.join(reasons)}。请调整视角，使两台相机都看到完整棋盘格后再按键。", flush=True)


def _record_pair(data_dir: Path, name: str, pair: dict) -> dict:
    import cv2

    preview = data_dir / f"{name}_corners.png"
    if not cv2.imwrite(str(preview), _montage(pair, name)):
        raise RuntimeError(f"无法保存预览：{preview}")
    metadata = {
        "name": name, "corner_preview_file": preview.name,
        "T_sdk_flange": pair["T_sdk_flange"].tolist(),
        "joints_rad": pair["joints_rad"].tolist(),
        "joint_change_deg": pair["joint_change_deg"],
        "host_capture_start_ns": pair["host_capture_start_ns"],
        "host_capture_end_ns": pair["host_capture_end_ns"],
        "host_frame_gap_ms": pair.get("host_frame_gap_ms"),
    }
    for key in ("wrist", "third"):
        item = pair[key]
        file = f"{name}_{key}.npz"
        np.savez_compressed(data_dir / file, rgb=item["rgb"], depth_m=item["depth_m"],
                            K=item["K"])
        metadata[key] = {
            "rgbd_file": file, "T_camera_board": item["T_camera_board"].tolist(),
            "reprojection_rmse_px": item["reprojection_rmse_px"],
            "frame_number": item["frame_number"], "timestamp_ms": item["timestamp_ms"],
            "serial": item["serial"], "K": item["K"].tolist(),
        }
    return metadata


def _load_pair(data_dir: Path, metadata: dict) -> dict:
    pair = {"T_sdk_flange": np.asarray(metadata["T_sdk_flange"])}
    for key in ("wrist", "third"):
        item = metadata[key]
        with np.load(data_dir / item["rgbd_file"]) as data:
            pair[key] = {"rgb": data["rgb"], "depth_m": data["depth_m"],
                         "K": data["K"], "T_camera_board": np.asarray(item["T_camera_board"])}
    return pair


def _show_overlay(data_dir: Path, sample: dict, T_sdk_third: np.ndarray,
                  T_flange_wrist: np.ndarray, check: dict) -> None:
    import open3d as o3d

    pair = _load_pair(data_dir, sample)
    transforms = {
        "wrist": pair["T_sdk_flange"] @ T_flange_wrist,
        "third": T_sdk_third,
    }
    geometries = [o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15)]
    for key in ("wrist", "third"):
        item = pair[key]
        rgb, depth, K = item["rgb"], item["depth_m"], item["K"]
        h, w = depth.shape
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            w, h, float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2]))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(rgb), o3d.geometry.Image(depth.astype(np.float32)),
            depth_scale=1.0, depth_trunc=2.5, convert_rgb_to_intensity=False)
        cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
        cloud = cloud.voxel_down_sample(0.005)
        cloud.transform(transforms[key])
        if key == "third":
            cloud.paint_uniform_color([0.05, 0.82, 0.88])  # 青色凸显与手腕 RGB 点云的错位。
        geometries.append(cloud)
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.06)
        axis.transform(transforms[key])
        geometries.append(axis)
    for key in ("T_sdk_board_from_wrist", "T_sdk_board_from_third"):
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.13 if key.endswith("wrist") else 0.09)
        axis.transform(check[key])
        geometries.append(axis)
    print("Open3D：手腕保留 RGB、第三视角点云为青色，均在 SDK base；大板轴来自手腕，小板轴来自第三视角。")
    print("请检查移动后的棋盘格和静态场景是否重合，不能仅看两组板轴。")
    o3d.visualization.draw_geometries(geometries, window_name="third-view cross-check in SDK base")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="交互式标定固定第三视角相机并与手腕相机交叉检查；只读机器人")
    parser.add_argument("--handeye", required=True, type=Path, help="已有手腕标定 handeye.json")
    parser.add_argument("--ip", required=True, help="RealMan SDK IP")
    parser.add_argument("--group", choices=("left", "right"), default="right")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--wrist-endpoint", default="tcp://127.0.0.1:18082")
    parser.add_argument("--third-endpoint", required=True, help="固定第三视角相机的独立 camera server endpoint")
    parser.add_argument("--third-serial", help="可选：固定相机 serial；程序仍会检查与手腕 serial 不同")
    parser.add_argument("--validation-poses", type=int, default=3,
                        help="第二板位绕至少两个轴采样，默认 3、最少 2")
    parser.add_argument("--max-reprojection-px", type=float, default=2.0)
    parser.add_argument("--max-joint-change-deg", type=float, default=0.2)
    parser.add_argument("--max-pair-gap-ms", type=float, default=250.0,
                        help="两相机新帧主机接收时间差上限，默认 250 ms")
    parser.add_argument("--max-validation-mm", type=float, default=10.0)
    parser.add_argument("--max-validation-deg", type=float, default=3.0)
    parser.add_argument("--connect-retries", type=int, default=30)
    parser.add_argument("--connect-retry-interval", type=float, default=0.5)
    parser.add_argument("--data-dir", type=Path, help="输出目录；省略则自动创建带时间戳的新目录")
    parser.add_argument("--resume", action="store_true", help="从本向导保存的 session.json 继续采集")
    parser.add_argument("--no-preview", action="store_true", help="不用 OpenCV 窗口，以终端回车采样")
    parser.add_argument("--visualize", action="store_true", help="数值通过后用 Open3D 叠加两相机点云")
    return parser


def _prompt_pair(cameras, robot, args, *, stage: str, pattern_size, square_size,
                 wrist_serial, third_serial, data_dir):
    import cv2

    while True:
        if args.no_preview:
            answer = input(f"{stage}：板与机械臂停稳，两相机都看全板；回车采样，q 退出：").strip().lower()
            if answer == "q":
                return None
        else:
            print(f"{stage}：观察双相机实时画面，停稳后按空格采样；Q 退出。", flush=True)
            if not _wait_for_capture(cameras, stage=stage, pattern_size=pattern_size,
                                     square_size_m=square_size, wrist_serial=wrist_serial,
                                     third_serial=third_serial,
                                     max_reprojection_px=args.max_reprojection_px):
                return None
        try:
            pair = capture_pair(
                cameras["wrist"], cameras["third"], robot,
                pattern_size=pattern_size, square_size_m=square_size,
                wrist_serial=wrist_serial, third_serial=third_serial,
                max_joint_change_deg=args.max_joint_change_deg,
                max_reprojection_px=args.max_reprojection_px,
                max_pair_gap_ms=args.max_pair_gap_ms,
            )
        except (ValueError, TimeoutError, RuntimeError) as exc:
            print(f"采样被拒绝：{exc}；请调整后重试。")
            continue
        candidate = data_dir / "candidate_corners.png"
        if not cv2.imwrite(str(candidate), _montage(pair, stage)):
            raise RuntimeError(f"无法保存候选画面：{candidate}")
        print(f"候选画面：{candidate}；手腕/固定相机 PnP "
              f"{pair['wrist']['reprojection_rmse_px']:.2f} / "
              f"{pair['third']['reprojection_rmse_px']:.2f} px；关节变化 "
              f"{pair['joint_change_deg']:.3f}°")
        if args.no_preview:
            answer = input("确认两画面红色 0 号角点是同一个物理角点？[y/N/q]：").strip().lower()
        else:
            print("冻结画面：松开脚踏板后再按空格确认；N 重拍，Q 退出。", flush=True)
            answer = _confirm_candidate_gui(candidate)
        if answer == "q":
            return None
        if answer == "y":
            return pair


def crosscheck_main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    resolve_cli_paths(args, "handeye", "data_dir")
    if args.validation_poses < 2:
        raise ValueError("第二板位至少需要两个绕不同轴的验证姿态")
    if args.wrist_endpoint == args.third_endpoint:
        raise ValueError("两个相机必须使用不同的 camera server endpoint")
    if args.visualize and args.no_preview and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        raise RuntimeError("Open3D 可视化需要图形桌面；请去掉 --visualize")
    handeye = json.loads(args.handeye.read_text(encoding="utf-8"))
    if handeye.get("calibration_type") != "eye_in_hand_fixed_board":
        raise ValueError("--handeye 必须是 guided_wrist_handeye 输出的 handeye.json")
    if handeye.get("robot_group") != args.group:
        raise ValueError("机械臂 group 与手腕标定记录不一致")
    T_flange_wrist = validate_pose(np.asarray(handeye["T_flange_camera"], dtype=float),
                                    field_name="T_flange_camera")
    pattern_size = tuple(handeye["pattern_size_inner_corners"])
    square_size = float(handeye["square_size_m"])
    wrist_serial = handeye["camera_serial"]
    if args.third_serial == wrist_serial:
        raise ValueError("固定相机 serial 不能与手腕相机相同")
    args.ip = normalize_realman_ip(args.ip)
    source_samples = args.handeye.parent / handeye.get("source_samples", "samples.json")
    if source_samples.is_file():
        original_run = json.loads(source_samples.read_text(encoding="utf-8"))
        original_ip = original_run.get("robot_sdk_ip")
        if original_ip and normalize_realman_ip(original_ip) != args.ip:
            raise ValueError("当前机器人 IP 与手腕标定使用的机器人 IP 不一致")
    if args.data_dir is None:
        args.data_dir = default_run_dir("crosscheck")
    session_path = args.data_dir / "session.json"
    config = {
        "schema_version": 1, "calibration_type": "third_view_from_wrist_crosscheck",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "handeye_file": str(args.handeye.resolve()),
        "T_flange_wrist": T_flange_wrist.tolist(),
        "robot_sdk_ip": args.ip, "robot_group": args.group,
        "wrist_endpoint": args.wrist_endpoint, "third_endpoint": args.third_endpoint,
        "wrist_serial": wrist_serial, "third_serial": args.third_serial,
        "pattern_size_inner_corners": list(pattern_size), "square_size_m": square_size,
        "pose_convention": "T_A_B maps B coordinates into A; translation in meter",
        "stage_a": None, "validation_samples": [],
    }
    if args.resume:
        if not session_path.exists():
            raise FileNotFoundError(f"没有可续采记录：{session_path}")
        session = json.loads(session_path.read_text(encoding="utf-8"))
        for field in ("handeye_file", "T_flange_wrist", "robot_sdk_ip", "robot_group", "wrist_endpoint",
                      "third_endpoint", "wrist_serial", "pattern_size_inner_corners", "square_size_m"):
            if not same_recorded_config({field: session.get(field)}, {field: config[field]}, ("handeye_file",)):
                raise ValueError(f"续采参数 {field} 与已有记录不一致")
        if args.third_serial and session.get("third_serial") != args.third_serial:
            raise ValueError("续采指定的第三视角相机 serial 与已有记录不一致")
        if (args.data_dir / "third_view_extrinsic.json").exists():
            raise FileExistsError("此目录已有正式外参；新一轮请使用新目录")
    else:
        if args.data_dir.exists() and any(args.data_dir.iterdir()):
            raise FileExistsError(f"输出目录非空；续采请加 --resume：{args.data_dir}")
        session = config
    print("\n第三视角相机交叉检查。标定板必须同时被手腕相机和固定相机看到。")
    print("第一板位用来求固定相机外参；之后移动板至少 3 cm 或旋转 5°，第二板位用于独立验证。")
    print("第二板位的手腕姿态须相对第一板位至少旋转 15°，并覆盖至少两个旋转轴。采帧时板与手腕都须静止。")
    print("两个画面的红色 0 号角点都须是板上的同一个物理内角点。机械臂由人移动，程序只读状态。")
    print(f"计划输出目录：{args.data_dir}；相机和机器人连接成功后才创建。")
    with ExitStack() as stack:
        cameras = {}
        for key, endpoint in (("wrist", args.wrist_endpoint), ("third", args.third_endpoint)):
            camera = CameraClient(endpoint)
            stack.callback(camera.close)
            try:
                _check_camera_health(
                    camera, name="手腕" if key == "wrist" else "第三视角",
                    endpoint=endpoint,
                )
            except RuntimeError as exc:
                print(f"相机连接失败：{exc}", file=sys.stderr, flush=True)
                return 2
            cameras[key] = camera
        robot = stack.enter_context(connect_realman_read_only(
            args.ip, group=args.group, port=args.port,
            connect_retries=args.connect_retries,
            connect_retry_interval_s=args.connect_retry_interval,
        ))
        if not args.no_preview:
            if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
                raise RuntimeError("未检测到图形桌面；请在桌面运行或加 --no-preview")
            if "QT_QPA_FONTS_DIR" not in os.environ and os.path.isdir("/usr/share/fonts"):
                os.environ["QT_QPA_FONTS_DIR"] = "/usr/share/fonts"
            import cv2
            cv2.namedWindow(_WINDOW, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(_WINDOW, 1400, 700)
            stack.callback(cv2.destroyAllWindows)
        args.data_dir.mkdir(parents=True, exist_ok=True)
        print(f"输出目录已创建：{args.data_dir}；中断后加 --resume 续采。")
        if not args.resume:
            _save_json(session_path, session)
        if session["stage_a"] is None:
            pair = _prompt_pair(cameras, robot, args, stage="板位 A", pattern_size=pattern_size,
                                square_size=square_size, wrist_serial=wrist_serial,
                                third_serial=args.third_serial, data_dir=args.data_dir)
            if pair is None:
                print("已退出；可用 --resume 从本目录继续。")
                return 1
            session["stage_a"] = _record_pair(args.data_dir, "board_a", pair)
            session["third_serial"] = pair["third"]["serial"]
            _save_json(session_path, session)
        source_a = session["stage_a"]
        stage_a = solve_fixed_camera_from_pair(
            np.asarray(source_a["T_sdk_flange"]), T_flange_wrist,
            np.asarray(source_a["wrist"]["T_camera_board"]),
            np.asarray(source_a["third"]["T_camera_board"]),
        )
        third_serial = session["third_serial"]
        print(f"板位 A 已求出候选 T_sdk_third；固定相机 serial {third_serial}。")
        print("现在请移动棋盘格到另一处并固定；两相机仍需看完整棋盘格。请人工改变手腕方向并停稳。")
        while len(session["validation_samples"]) < args.validation_poses:
            index = len(session["validation_samples"]) + 1
            pair = _prompt_pair(cameras, robot, args, stage=f"板位 B {index}/{args.validation_poses}",
                                pattern_size=pattern_size, square_size=square_size,
                                wrist_serial=wrist_serial, third_serial=third_serial,
                                data_dir=args.data_dir)
            if pair is None:
                print("已退出；可用 --resume 从本目录继续。")
                return 1
            try:
                angle_from_a = _rotation_deg(np.asarray(source_a["T_sdk_flange"]),
                                             pair["T_sdk_flange"])
                if angle_from_a < 15.0:
                    raise ValueError(f"手腕相对板位 A 只旋转 {angle_from_a:.1f}°；"
                                     "请至少旋转 15°，避免手眼误差抵消")
                check = compare_board_poses(
                    stage_a["T_sdk_third"], pair["T_sdk_flange"], T_flange_wrist,
                    pair["wrist"]["T_camera_board"], pair["third"]["T_camera_board"],
                    reference_board=stage_a["T_sdk_board"],
                )
                if index == args.validation_poses:
                    check_flange_pose_diversity([
                        np.asarray(source_a["T_sdk_flange"]),
                        *[np.asarray(item["T_sdk_flange"])
                          for item in session["validation_samples"]],
                        pair["T_sdk_flange"],
                    ])
            except ValueError as exc:
                print(f"此验证样本未收录：{exc}")
                continue
            print(f"两条路径的板位姿偏差：{check['translation_m'] * 1000:.2f} mm / "
                  f"{check['rotation_deg']:.2f}°")
            metadata = _record_pair(args.data_dir, f"board_b_{index:03d}", pair)
            metadata["crosscheck"] = _jsonable(check)
            session["validation_samples"].append(metadata)
            _save_json(session_path, session)
        checks = [entry["crosscheck"] for entry in session["validation_samples"]]
        try:
            diversity = check_flange_pose_diversity([
                np.asarray(source_a["T_sdk_flange"]),
                *[np.asarray(item["T_sdk_flange"]) for item in session["validation_samples"]],
            ])
            diversity_error = None
        except ValueError as exc:
            diversity = None
            diversity_error = str(exc)
        worst_mm = max(item["translation_m"] for item in checks) * 1000
        worst_deg = max(item["rotation_deg"] for item in checks)
        # 第二板位须保持静止；两条路径一致并不足以证明板没有在采样之间移动。
        b0 = np.asarray(checks[0]["T_sdk_board_from_wrist"])
        spread_mm = max(float(np.linalg.norm(
            np.asarray(item["T_sdk_board_from_wrist"])[:3, 3] - b0[:3, 3]))
            for item in checks) * 1000
        spread_deg = max(_rotation_deg(b0, np.asarray(item["T_sdk_board_from_wrist"]))
                         for item in checks)
        quality_passed = (worst_mm <= args.max_validation_mm
                          and worst_deg <= args.max_validation_deg
                          and spread_mm <= args.max_validation_mm
                          and spread_deg <= args.max_validation_deg
                          and diversity_error is None)
        report = {
            "T_sdk_third_candidate": stage_a["T_sdk_third"].tolist(),
            "T_sdk_board_stage_a": stage_a["T_sdk_board"].tolist(),
            "T_third_board_stage_a": stage_a["T_third_board"].tolist(),
            "validation_count": len(checks),
            "validation_translation_max_mm": worst_mm,
            "validation_rotation_max_deg": worst_deg,
            "stage_b_board_translation_spread_mm": spread_mm,
            "stage_b_board_rotation_spread_deg": spread_deg,
            "flange_rotation_diversity": diversity,
            "flange_rotation_diversity_error": diversity_error,
            "thresholds": {"translation_mm": args.max_validation_mm,
                           "rotation_deg": args.max_validation_deg},
            "quality_passed": quality_passed,
            "checks": checks,
        }
        _save_json(args.data_dir / "quality_report.json", report)
        print(f"\n第二板位最大双链偏差 {worst_mm:.2f} mm / {worst_deg:.2f}°；"
              f"板位稳定性 {spread_mm:.2f} mm / {spread_deg:.2f}°。"
              f"质量报告：{args.data_dir / 'quality_report.json'}")
        if not quality_passed:
            if diversity_error:
                print(f"姿态多样性不足：{diversity_error}")
            print("未通过数值检查；保留原始样本，不生成正式外参。请检查角点、板固定和相机安装。")
            return 2
        if args.visualize:
            if not args.no_preview:
                import cv2
                cv2.destroyAllWindows()
            first = session["validation_samples"][0]
            check = {key: np.asarray(first["crosscheck"][key])
                     for key in ("T_sdk_board_from_wrist", "T_sdk_board_from_third")}
            _show_overlay(args.data_dir, first, stage_a["T_sdk_third"], T_flange_wrist, check)
        prompt = ("确认第二板位两相机点云与板位姿可接受"
                  if args.visualize else "确认数值和角点预览可接受（本次未显示点云）")
        answer = input(f"{prompt}，输入 ACCEPT 保存外参；其他输入取消：").strip()
        if answer != "ACCEPT":
            print("未生成正式外参；原始样本与质量报告已保存。")
            return 1
        output = {
            "schema_version": 1, "calibration_type": "fixed_third_view_from_wrist_crosscheck",
            "T_sdk_third": stage_a["T_sdk_third"].tolist(),
            "T_third_sdk": np.linalg.inv(stage_a["T_sdk_third"]).tolist(),
            "T_third_board_stage_a": stage_a["T_third_board"].tolist(),
            "T_board_third_stage_a": np.linalg.inv(stage_a["T_third_board"]).tolist(),
            "camera_serial": third_serial, "robot_group": args.group,
            "handeye_file": str(args.handeye.resolve()),
            "source_session": "session.json", "quality_report": "quality_report.json",
            "validation_translation_max_mm": worst_mm,
            "validation_rotation_max_deg": worst_deg,
            "open3d_visual_confirmed": bool(args.visualize),
            "pose_convention": "T_A_B maps B coordinates into A; translation in meter",
        }
        target = args.data_dir / "third_view_extrinsic.json"
        _save_json(target, standard_camera_parameters(args.data_dir, session["stage_a"]["third"], output))
        publish_result(args.data_dir, "third_camera")
        print(f"已保存固定第三视角相机外参：{target}")
        return 0


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if "--crosscheck" in arguments:
        arguments.remove("--crosscheck")
        return crosscheck_main(arguments)
    from gene_agent.perception_modules.camera.calibration.guided_board_camera import main as fixed_main
    return fixed_main(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
