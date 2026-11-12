"""人工摆姿态、自动配对 SDK 法兰 FK 和手腕 RealSense 棋盘格观测。"""

from __future__ import annotations

from .storage import publish_result, default_run_dir, resolve_cli_paths

import argparse
import json
import os
import time
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from gene_agent.perception_modules.camera import CameraClient
from gene_agent.perception_modules.camera.calibration.wrist_handeye import (
    check_stationary, detect_board_pose, solve_handeye,
)
from gene_agent.realworld.devices.arms.realman.connection import connect_realman_read_only, normalize_realman_ip

_PREVIEW_WINDOW = "Wrist RealSense hand-eye calibration"


def _save_json(path: Path, content: dict) -> None:
    path.write_text(json.dumps(content, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _draw_corners(rgb: np.ndarray, corners: np.ndarray) -> np.ndarray:
    import cv2

    display = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    for point in corners:
        cv2.circle(display, tuple(np.round(point).astype(int)), 2, (0, 210, 0), -1)
    first = tuple(np.round(corners[0]).astype(int))
    cv2.circle(display, first, 13, (0, 0, 255), 3)
    cv2.putText(display, "corner 0", (first[0] + 15, first[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    return display


def _preview(rgb: np.ndarray, corners: np.ndarray, path: Path) -> None:
    import cv2

    display = _draw_corners(rgb, corners)
    if not cv2.imwrite(str(path), display):
        raise RuntimeError(f"无法保存角点预览：{path}")


def _wait_for_live_capture(camera: CameraClient, args, *, index: int, total: int,
                           first_serial: str | None, first_K: np.ndarray | None) -> str:
    """持续更新 RGB 棋盘格预览；只有有效画面按空格才进入配对采样。"""
    import cv2

    last_error = None
    while True:
        try:
            packet = camera.get_frame()
        except (TimeoutError, RuntimeError) as exc:
            if str(exc) != last_error:
                print(f"实时预览取帧失败，继续等待：{exc}", flush=True)
                last_error = str(exc)
            key = cv2.waitKey(50) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                return "quit"
            continue
        last_error = None
        display = cv2.cvtColor(packet.rgb, cv2.COLOR_RGB2BGR)
        ready = False
        status = "BOARD NOT FOUND"
        try:
            if args.serial and packet.serial != args.serial:
                status = f"WRONG CAMERA SERIAL: {packet.serial}"
            elif first_serial and packet.serial != first_serial:
                status = f"CAMERA SERIAL CHANGED: {packet.serial}"
            elif first_K is not None and not np.allclose(packet.intrinsic_matrix, first_K, atol=1e-6):
                status = "CAMERA INTRINSICS CHANGED"
            else:
                detected = detect_board_pose(packet.rgb, packet.intrinsic_matrix,
                                             tuple(args.pattern_size), args.square_size)
                if detected is not None:
                    display = _draw_corners(packet.rgb, detected["corners"])
                    rmse = detected["reprojection_rmse_px"]
                    ready = rmse <= args.max_reprojection_px
                    status = (f"BOARD FOUND | reprojection {rmse:.2f} px"
                              if ready else f"REPROJECTION HIGH: {rmse:.2f} px")
                    points = np.asarray(detected["corners"])
                    h, w = packet.rgb.shape[:2]
                    edge = min(float(points[:, 0].min()), float(points[:, 1].min()),
                               float(w - 1 - points[:, 0].max()),
                               float(h - 1 - points[:, 1].max()))
                    if ready and edge < 20:
                        status += " | EDGE CLOSE"
        except ValueError:
            status = "BOARD POSE INVALID"
        text_color = (60, 220, 60) if ready else (0, 180, 255)
        # 文本放在图像外的顶栏，避免遮住红色 0 号角点或板边缘。
        display = cv2.copyMakeBorder(display, 90, 0, 0, 0,
                                     cv2.BORDER_CONSTANT, value=(0, 0, 0))
        cv2.putText(display, f"Pose {index}/{total} | frame {packet.frame_number}",
                    (12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(display, status, (12, 54), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, text_color, 2)
        cv2.putText(display, "SPACE capture | Q quit" if ready else "Move wrist/board into view | Q quit",
                    (12, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow(_PREVIEW_WINDOW, display)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            return "quit"
        if key in (32, 13) and ready:
            return "capture"


def _confirm_candidate_gui(path: Path) -> str:
    """冻结本次实际采集帧，等待在窗口中确认红色 0 号角点。"""
    import cv2

    display = cv2.imread(str(path))
    if display is None:
        raise RuntimeError(f"无法打开本次采样预览：{path}")
    display = cv2.copyMakeBorder(display, 44, 0, 0, 0,
                                 cv2.BORDER_CONSTANT, value=(0, 0, 0))
    cv2.putText(display, "CORNER 0 OK?  Release, then SPACE confirm | N retry | Q quit", (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.imshow(_PREVIEW_WINDOW, display)
    space_released = False
    while True:
        raw_key = cv2.waitKey(20)
        if raw_key < 0:
            space_released = True
            continue
        key = raw_key & 0xFF
        # 过滤首次采样后仍在键盘队列中的空格；脚踏板应短踩并松开。
        if key == 32 and space_released:
            return "y"
        if key in (ord("y"), ord("Y")):
            return "y"
        if key in (ord("n"), ord("N"), 27):
            return "n"
        if key in (ord("q"), ord("Q")):
            return "q"


def _capture(camera: CameraClient, robot, args, candidate_path: Path) -> dict:
    # 先取得一帧作为序号基准，再夹取关节读数；第二帧必须更新。
    camera.get_frame()
    _T_before, joints_before = robot.read_flange_pose()
    host_before_ms = time.time_ns() / 1e6
    packet = camera.get_frame()
    host_after_ms = time.time_ns() / 1e6
    T_after, joints_after = robot.read_flange_pose()
    delta_deg = check_stationary(joints_before, joints_after,
                                 max_change_deg=args.max_joint_change_deg)
    if packet.frame_number < 0:
        raise ValueError("相机帧号无效")
    if args.serial and packet.serial != args.serial:
        raise ValueError(f"相机 serial 不匹配：期望 {args.serial}，收到 {packet.serial}")
    detected = detect_board_pose(packet.rgb, packet.intrinsic_matrix,
                                 tuple(args.pattern_size), args.square_size)
    if detected is None:
        raise ValueError("当前帧未检测到完整棋盘格，请调整手腕视角后重试")
    if detected["reprojection_rmse_px"] > args.max_reprojection_px:
        raise ValueError(f"PnP 重投影误差 {detected['reprojection_rmse_px']:.2f} px 超出阈值")
    _preview(packet.rgb, detected["corners"], candidate_path)
    return {
        "T_sdk_flange": T_after,
        "T_camera_board": detected["T_camera_board"],
        "joints_rad": joints_after,
        "joint_change_deg": delta_deg,
        "reprojection_rmse_px": detected["reprojection_rmse_px"],
        "frame_number": packet.frame_number,
        "sensor_timestamp_ms": packet.timestamp_ms,
        "host_request_before_ms": host_before_ms,
        "host_frame_received_ms": host_after_ms,
        "serial": packet.serial,
        "K": np.asarray(packet.intrinsic_matrix),
        "rgb": packet.rgb,
        "depth_m": packet.depth,
    }


def _record_sample(data_dir: Path, candidate: dict, index: int, manifest: dict) -> None:
    stem = f"sample_{index:03d}"
    np.savez_compressed(
        data_dir / f"{stem}.npz", rgb=candidate["rgb"],
        depth_m=candidate["depth_m"], K=candidate["K"],
    )
    import cv2

    preview = cv2.imread(str(data_dir / "candidate.png"))
    if preview is None or not cv2.imwrite(str(data_dir / f"{stem}_corners.png"), preview):
        raise RuntimeError("无法归档角点预览图")
    entry = {
        "index": index, "rgbd_file": f"{stem}.npz",
        "corner_preview_file": f"{stem}_corners.png",
        "T_sdk_flange": candidate["T_sdk_flange"].tolist(),
        "T_camera_board": candidate["T_camera_board"].tolist(),
        "joints_rad": candidate["joints_rad"].tolist(),
        "joint_change_deg": candidate["joint_change_deg"],
        "reprojection_rmse_px": candidate["reprojection_rmse_px"],
        "frame_number": candidate["frame_number"],
        "sensor_timestamp_ms": candidate["sensor_timestamp_ms"],
        "host_request_before_ms": candidate["host_request_before_ms"],
        "host_frame_received_ms": candidate["host_frame_received_ms"],
        "serial": candidate["serial"],
        "K": candidate["K"].tolist(),
    }
    manifest["samples"].append(entry)
    _save_json(data_dir / "samples.json", manifest)


def _visualize_sdk_pointclouds(data_dir: Path, samples: list[dict], result: dict) -> None:
    """挑三个相距较远的姿态，将点云叠到 SDK base 检查静态板是否重合。"""
    import open3d as o3d

    selected = sorted({0, len(samples) // 2, len(samples) - 1})
    geometries = []
    for index in selected:
        sample = samples[index]
        with np.load(data_dir / sample["rgbd_file"]) as data:
            rgb = data["rgb"]
            depth = data["depth_m"]
            K = data["K"]
        h, w = depth.shape
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            w, h, float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2]),
        )
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(rgb), o3d.geometry.Image(depth.astype(np.float32)),
            depth_scale=1.0, depth_trunc=2.5, convert_rgb_to_intensity=False,
        )
        cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
        cloud = cloud.voxel_down_sample(0.005)
        T_sdk_camera = np.asarray(sample["T_sdk_flange"]) @ result["T_flange_camera"]
        cloud.transform(T_sdk_camera)
        geometries.append(cloud)
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.06)
        axis.transform(T_sdk_camera)
        geometries.append(axis)
    board_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.12)
    board_axis.transform(result["T_sdk_board_estimate"])
    geometries.append(board_axis)
    print(f"Open3D 显示采样 {', '.join(str(i + 1) for i in selected)} 的 SDK base 点云；"
          "请检查静态棋盘格和桌面是否在多姿态下重合。")
    o3d.visualization.draw_geometries(geometries, window_name="wrist hand-eye in SDK base")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="手腕 RealSense eye-in-hand 标定终端向导；机械臂全程人工移动")
    parser.add_argument("--ip", required=True, help="RealMan SDK IPv4；兼容 http(s)://IP/ 写法")
    parser.add_argument("--group", choices=("left", "right"), default="right")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--connect-retries", type=int, default=30,
                        help="SDK 连接最多尝试次数，默认 30；0 表示持续重试直到 Ctrl+C")
    parser.add_argument("--connect-retry-interval", type=float, default=0.5,
                        help="连接失败后重试间隔，单位秒")
    parser.add_argument("--camera-endpoint", default="tcp://127.0.0.1:18082")
    parser.add_argument("--serial", help="可选：强制匹配手腕 RealSense serial")
    parser.add_argument("--pattern-size", nargs=2, type=int, default=(11, 8), metavar=("COLS", "ROWS"))
    parser.add_argument("--square-size", type=float, default=0.015, help="实际内角点间距，单位米")
    parser.add_argument("--poses", type=int, default=20, help="目标采样姿态数，至少 10")
    parser.add_argument("--data-dir", type=Path, help="输出目录，默认data/calibration/runs/handeye/时间目录")
    parser.add_argument("--resume", action="store_true", help="从已有 samples.json 续采或重新求解")
    parser.add_argument("--max-joint-change-deg", type=float, default=0.2)
    parser.add_argument("--max-reprojection-px", type=float, default=2.0)
    parser.add_argument("--max-holdout-mm", type=float, default=10.0)
    parser.add_argument("--max-holdout-deg", type=float, default=3.0)
    parser.add_argument("--visualize", action="store_true", help="数值通过后用 Open3D 显示多姿态点云并人工确认")
    parser.add_argument("--no-preview", action="store_true",
                        help="关闭实时 OpenCV 窗口，用原终端回车和 y/n/q 操作")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    resolve_cli_paths(args, "data_dir")
    if args.resume and args.data_dir is None:
        raise ValueError("--resume需要明确指定--data-dir")
    args.data_dir = args.data_dir or default_run_dir("handeye")
    if args.poses < 10 or args.square_size <= 0:
        raise ValueError("至少采集 10 个姿态，格宽必须大于零")
    raw_ip = args.ip
    args.ip = normalize_realman_ip(raw_ip)
    if raw_ip != args.ip:
        print(f"RealMan SDK 地址规范化：{raw_ip} → {args.ip}", flush=True)
    manifest_path = args.data_dir / "samples.json"
    existing_files = list(args.data_dir.iterdir()) if args.data_dir.exists() else []
    auto_reuse_empty = (not args.resume and len(existing_files) == 1
                        and existing_files[0].name == "samples.json")
    needs_initial_write = False
    if args.resume or auto_reuse_empty:
        if (args.data_dir / "handeye.json").exists():
            raise FileExistsError("目录已有正式 handeye.json，请新建目录进行新一轮标定")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if auto_reuse_empty and manifest.get("samples"):
            raise FileExistsError(f"输出目录已有样本；如需续采请添加 --resume：{args.data_dir}")
        expected = {
            "robot_group": args.group,
            "camera_endpoint": args.camera_endpoint,
            "pattern_size_inner_corners": list(args.pattern_size),
            "square_size_m": args.square_size,
        }
        for key, value in expected.items():
            if manifest.get(key) != value:
                raise ValueError(f"续采参数 {key} 与已有 samples.json 不一致")
        if normalize_realman_ip(manifest.get("robot_sdk_ip", "")) != args.ip:
            raise ValueError("续采机器人 IP 与已有 samples.json 不一致")
        if args.serial and manifest["samples"] and manifest["samples"][0]["serial"] != args.serial:
            raise ValueError("续采指定的相机 serial 与已有样本不同")
        print(f"从 {args.data_dir} 恢复 {len(manifest['samples'])} 个样本。")
    else:
        if existing_files:
            raise FileExistsError(f"输出目录非空；续采请添加 --resume，或新建目录：{args.data_dir}")
        manifest = {
            "schema_version": 1, "calibration_type": "eye_in_hand_fixed_board",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "robot_group": args.group, "robot_sdk_ip": args.ip,
            "camera_endpoint": args.camera_endpoint,
            "pattern_size_inner_corners": list(args.pattern_size),
            "square_size_m": args.square_size,
            "pose_convention": "T_A_B maps B coordinates into A; T_sdk_flange and T_camera_board",
            "samples": [],
        }
        needs_initial_write = True
    print("\n手腕相机标定：将棋盘格固定在工作台，全程不要移动棋盘格。")
    print("每次人工移动并停稳机械臂，确保整块棋盘格可见，依次改变位置和至少两个旋转轴。")
    print("程序只读取机器人状态，不发送运动指令。预览红圈 0 号角点须始终对应棋盘格同一个物理角点。")
    with ExitStack() as stack:
        camera = CameraClient(args.camera_endpoint)
        stack.callback(camera.close)
        health = camera.health()
        if not health.get("ok"):
            raise RuntimeError(f"相机服务未就绪：{health}")
        robot = stack.enter_context(connect_realman_read_only(
            args.ip, group=args.group, port=args.port,
            connect_retries=args.connect_retries,
            connect_retry_interval_s=args.connect_retry_interval,
        ))
        if not args.no_preview:
            if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
                raise RuntimeError("未检测到图形桌面；请在本地图形桌面运行，或加 --no-preview")
            if "QT_QPA_FONTS_DIR" not in os.environ and os.path.isdir("/usr/share/fonts"):
                os.environ["QT_QPA_FONTS_DIR"] = "/usr/share/fonts"
            import cv2

            cv2.namedWindow(_PREVIEW_WINDOW, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(_PREVIEW_WINDOW, 1100, 760)
            stack.callback(cv2.destroyAllWindows)
            print("OpenCV 实时窗口已打开：移动手腕观察棋盘格；按空格采样，按 Q 退出。", flush=True)
        if needs_initial_write:
            args.data_dir.mkdir(parents=True, exist_ok=True)
            _save_json(manifest_path, manifest)
        first_serial = manifest["samples"][0]["serial"] if manifest["samples"] else None
        first_K = np.asarray(manifest["samples"][0]["K"]) if manifest["samples"] else None
        while len(manifest["samples"]) < args.poses:
            index = len(manifest["samples"]) + 1
            if args.no_preview:
                action = input(f"\n[{index}/{args.poses}] 人工移动到新姿态并停稳；回车采样，q 退出：").strip().lower()
                quit_requested = action == "q"
            else:
                print(f"\n[{index}/{args.poses}] 移动并停稳机械臂；在 OpenCV 窗口按空格采样、Q 退出。", flush=True)
                quit_requested = _wait_for_live_capture(
                    camera, args, index=index, total=args.poses,
                    first_serial=first_serial, first_K=first_K,
                ) == "quit"
            if quit_requested:
                print(f"已保存 {index - 1} 个原始样本；未输出手眼结果。")
                return 1
            try:
                candidate = _capture(camera, robot, args, args.data_dir / "candidate.png")
                if first_serial is not None and candidate["serial"] != first_serial:
                    raise ValueError("本次相机 serial 与此前样本不同")
                if first_K is not None and not np.allclose(candidate["K"], first_K, atol=1e-6):
                    raise ValueError("本次相机内参与此前样本不同，请保持相同分辨率和设备")
            except (ValueError, TimeoutError, RuntimeError) as exc:
                print(f"本次采样未收录：{exc}")
                continue
            print(f"预览：{args.data_dir / 'candidate.png'}")
            print(f"帧号 {candidate['frame_number']}；重投影误差 {candidate['reprojection_rmse_px']:.2f} px；"
                  f"采帧时最大关节变化 {candidate['joint_change_deg']:.3f}°")
            if args.no_preview:
                answer = input("检查红圈 0 是否始终对应同一个物理内角点；确认收录？[y/N/q]：").strip().lower()
            else:
                print("检查冻结画面的红圈 0：松开脚踏板后再按一次空格确认收录；N 重拍、Q 退出。", flush=True)
                answer = _confirm_candidate_gui(args.data_dir / "candidate.png")
            if answer == "q":
                print("已保存此前样本；未输出手眼结果。")
                return 1
            if answer != "y":
                print("已丢弃本次样本。")
                continue
            _record_sample(args.data_dir, candidate, index, manifest)
            first_serial = candidate["serial"]
            first_K = candidate["K"].copy()
            print(f"已收录 {index} 个姿态。")
        if not args.no_preview:
            import cv2
            cv2.destroyAllWindows()
        result = solve_handeye(manifest["samples"])
        translation_mm = result["holdout_translation_max_m"] * 1000
        rotation_deg = result["holdout_rotation_max_deg"]
        quality_passed = (translation_mm <= args.max_holdout_mm
                          and rotation_deg <= args.max_holdout_deg)
        thresholds = {
            "max_joint_change_deg": args.max_joint_change_deg,
            "max_reprojection_px": args.max_reprojection_px,
            "max_holdout_translation_mm": args.max_holdout_mm,
            "max_holdout_rotation_deg": args.max_holdout_deg,
        }
        _save_json(args.data_dir / "quality_report.json", {
            "candidate_T_flange_camera": result["T_flange_camera"].tolist(),
            "T_sdk_board_estimate": result["T_sdk_board_estimate"].tolist(),
            "train_count": result["train_count"], "holdout_count": result["holdout_count"],
            "holdout_translation_max_m": result["holdout_translation_max_m"],
            "holdout_rotation_max_deg": rotation_deg,
            "thresholds": thresholds, "quality_passed": quality_passed,
            "residuals": result["residuals"],
        })
        print(f"\n留出样本 {result['holdout_count']} 个；板位姿最大残差 "
              f"{translation_mm:.2f} mm / {rotation_deg:.2f}°")
        if not quality_passed:
            print("质量检查未通过；保留原始样本，请检查棋盘格是否移动、0 号角点、机器人静止及内参。")
            return 2
        if args.visualize:
            _visualize_sdk_pointclouds(args.data_dir, manifest["samples"], result)
            if input("若静态棋盘格和桌面重合，输入 ACCEPT 保存标定；其他输入取消：").strip() != "ACCEPT":
                print("已保留原始样本；未输出手眼结果。")
                return 1
        output = {
            "schema_version": 1, "calibration_type": "eye_in_hand_fixed_board",
            "T_flange_camera": result["T_flange_camera"].tolist(),
            "T_sdk_board_estimate": result["T_sdk_board_estimate"].tolist(),
            "camera_serial": first_serial, "intrinsic_matrix": first_K.tolist(),
            "robot_group": args.group, "sdk_base_frame": "RealMan SDK base",
            "units": {"translation": "meter", "rotation_residual": "degree"},
            "pattern_size_inner_corners": list(args.pattern_size),
            "square_size_m": args.square_size,
            "train_count": result["train_count"], "holdout_count": result["holdout_count"],
            "holdout_translation_max_m": result["holdout_translation_max_m"],
            "holdout_rotation_max_deg": rotation_deg,
            "thresholds": thresholds,
            "open3d_visual_confirmed": bool(args.visualize),
            "residuals": result["residuals"],
            "source_samples": "samples.json",
            "runtime_formula": "T_sdk_camera(q) = T_sdk_flange(q) @ T_flange_camera",
        }
        path = args.data_dir / "handeye.json"
        _save_json(path, output)
        publish_result(args.data_dir, "handeye")
        print(f"标定通过，结果：{path}")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
