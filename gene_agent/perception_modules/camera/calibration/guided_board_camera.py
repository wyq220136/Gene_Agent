"""固定棋盘格到单台相机的 PnP 向导；无需机器人连接。"""

from __future__ import annotations

from .storage import standard_camera_parameters, publish_result, default_run_dir, resolve_cli_paths, same_recorded_config

import argparse
import json
import os
import sys
import time
from contextlib import ExitStack
from pathlib import Path

import numpy as np

from gene_agent.knowledge.geometry.transforms import validate_pose
from gene_agent.perception_modules.camera import CameraClient
from .guided_third_view_crosscheck import (
    _check_camera_health, _confirm_candidate_gui, _detect, _jsonable, _panel, _save_json,
)
from .third_view_crosscheck import _rotation_deg
from .wrist_handeye import _mean_pose, board_object_points

_WINDOW = "Fixed board to camera calibration"


def load_board_pose(path: Path, *, key: str = "T_sdk_board") -> np.ndarray:
    """仅载入用户明确指定的板→SDK base；不猜测旧手眼记录中的板位。"""
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(value, dict):
        if key not in value:
            raise ValueError(f"板位文件缺少 {key}；请明确指定 --board-pose-key，不能自动推断变换方向")
        value = value[key]
    return validate_pose(np.asarray(value, dtype=float), field_name="T_sdk_board")


def summarize_samples(poses: list[np.ndarray], *, T_sdk_board=None,
                      max_spread_mm: float = 5.0, max_spread_deg: float = 2.0) -> dict:
    """静止多帧重复性；以首帧为参考检查移动，避免平均后掩盖变化。"""
    if not poses:
        raise ValueError("至少需要一组棋盘格观测")
    if not np.isfinite([max_spread_mm, max_spread_deg]).all() or min(max_spread_mm, max_spread_deg) <= 0:
        raise ValueError("静止重复性阈值须为有限正数")
    validated = [validate_pose(np.asarray(p, dtype=float), field_name="T_camera_board") for p in poses]
    mean = validate_pose(_mean_pose(validated), field_name="T_camera_board_mean")
    mm = max(float(np.linalg.norm(p[:3, 3] - validated[0][:3, 3])) for p in validated) * 1000
    deg = max(_rotation_deg(validated[0], p) for p in validated)
    result = {
        "T_camera_board": mean, "T_board_camera": np.linalg.inv(mean),
        "sample_count": len(poses), "repeatability_checked": len(poses) > 1,
        "translation_spread_max_mm": mm, "rotation_spread_max_deg": deg,
        "thresholds": {"translation_mm": max_spread_mm, "rotation_deg": max_spread_deg},
        "quality_passed": mm <= max_spread_mm and deg <= max_spread_deg,
    }
    if T_sdk_board is not None:
        board = validate_pose(np.asarray(T_sdk_board, dtype=float), field_name="T_sdk_board")
        camera = validate_pose(board @ np.linalg.inv(mean), field_name="T_sdk_camera")
        result.update(T_sdk_board=board, T_sdk_camera=camera, T_camera_sdk=np.linalg.inv(camera))
    return result


def _validated_detection(packet, args):
    if args.reference_K is not None and not np.allclose(
        packet.intrinsic_matrix, args.reference_K, atol=1e-6, rtol=0,
    ):
        raise ValueError("相机内参已变化，请恢复分辨率/服务配置或新建标定目录")
    return _detect(packet, args.pattern_size, args.square_size, args.max_reprojection_px, args.serial)


def _preview(packet, found, index: int, count: int, *, frozen: bool = False):
    import cv2

    picture = _panel(packet, found, "FIXED CAMERA",
                     f"PnP {found['reprojection_rmse_px']:.2f} px | corner 0 in red"
                     if found is not None else "Show complete board to camera")
    picture = cv2.copyMakeBorder(picture, 42, 0, 0, 0, cv2.BORDER_CONSTANT)
    action = "release pedal, SPACE confirm | N retry | Q quit" if frozen else "SPACE capture | Q quit"
    cv2.putText(picture, f"{index}/{count} | {action}", (8, 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
    return picture


def prompt_sample(camera, args, index: int):
    """预览与冻结确认均使用空格，保持采集的就是用户看到的那一帧。"""
    import cv2

    last_error = None
    while True:
        if args.no_preview:
            if input(f"[{index}/{args.samples}] 板和相机保持固定；回车取样，q 退出：").strip().lower() == "q":
                return None
        try:
            packet = camera.get_frame()
            try:
                found = _validated_detection(packet, args)
                last_error = None
            except ValueError as exc:
                found = None
                if str(exc) != last_error:
                    print(f"画面未就绪：{exc}", flush=True)
                    last_error = str(exc)
                if args.no_preview:
                    continue
            if not args.no_preview:
                cv2.imshow(_WINDOW, _preview(packet, found, index, args.samples))
                key = cv2.waitKey(20) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    return None
                if key not in (32, 13):
                    continue
                if found is None:
                    print("无法采样：棋盘格、serial、内参或重投影误差不符合要求。")
                    continue
        except (TimeoutError, RuntimeError) as exc:
            print(f"相机取帧失败，可重试：{exc}", flush=True)
            if not args.no_preview and cv2.waitKey(50) & 0xFF in (ord("q"), ord("Q"), 27):
                return None
            continue
        candidate = args.data_dir / "candidate_corners.png"
        if not cv2.imwrite(str(candidate), _preview(packet, found, index, args.samples, frozen=True)):
            raise RuntimeError(f"预览保存失败：{candidate}")
        print(f"候选：{candidate}；PnP {found['reprojection_rmse_px']:.2f} px。"
              "确认红色 0 是指定物理原点，且板与相机未移动。", flush=True)
        answer = (input("接受该帧？[y/N/q]：").strip().lower() if args.no_preview
                  else _confirm_candidate_gui(candidate, window_name=_WINDOW))
        if answer == "q":
            return None
        if answer == "y":
            return packet, found


def _parser():
    parser = argparse.ArgumentParser(description=(
        "固定棋盘格到第三视角相机的交互式标定；仅需一台相机，不连接机器人。"
        "旧双相机移板验证须在原入口加 --crosscheck。"))
    parser.add_argument("--third-endpoint", "--camera-endpoint", dest="endpoint", default="tcp://127.0.0.1:18083")
    parser.add_argument("--third-serial", "--serial", dest="serial")
    parser.add_argument("--pattern-size", nargs=2, type=int, help="内角点列数 行数，默认 11 8")
    parser.add_argument("--square-size", type=float, help="相邻内角点间距，米，默认 0.015")
    parser.add_argument("--handeye", type=Path, help="可选：仅读取棋盘格尺寸，不读取旧板位或连接机器人")
    parser.add_argument("--samples", type=int, default=3, help="静止重复采样次数；可设 1")
    parser.add_argument("--max-reprojection-px", type=float, default=2.0)
    parser.add_argument("--max-spread-mm", type=float, default=5.0)
    parser.add_argument("--max-spread-deg", type=float, default=2.0)
    parser.add_argument("--board-pose", type=Path, help="可选：已有 T_sdk_board 的 JSON 文件，平移单位米")
    parser.add_argument("--board-pose-key", default="T_sdk_board", help="JSON 对象中的矩阵字段；纯 4x4 数组不需要")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-preview", action="store_true", help="以终端取样、确认，并保存角点 PNG")
    parser.add_argument("--visualize", action="store_true", help="Open3D 显示相机点云和板坐标系")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    resolve_cli_paths(args, "handeye", "data_dir", "board_pose")
    if args.samples < 1 or not np.isfinite(args.max_reprojection_px) or args.max_reprojection_px <= 0:
        raise ValueError("采样次数和重投影阈值必须为正")
    if (not args.no_preview or args.visualize) and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        raise RuntimeError("未检测到图形桌面；请加 --no-preview 并去掉 --visualize")
    source = json.loads(args.handeye.read_text(encoding="utf-8")) if args.handeye else {}
    args.pattern_size = tuple(args.pattern_size or source.get("pattern_size_inner_corners", (11, 8)))
    args.square_size = args.square_size if args.square_size is not None else source.get("square_size_m", 0.015)
    board_object_points(args.pattern_size, args.square_size)
    board = load_board_pose(args.board_pose, key=args.board_pose_key) if args.board_pose else None
    summarize_samples([np.eye(4)], max_spread_mm=args.max_spread_mm, max_spread_deg=args.max_spread_deg)
    args.data_dir = args.data_dir or default_run_dir("board_camera")
    path = args.data_dir / "session.json"
    config = {
        "schema_version": 1, "calibration_type": "fixed_board_camera",
        "camera_endpoint": args.endpoint, "camera_serial": args.serial,
        "pattern_size_inner_corners": list(args.pattern_size), "square_size_m": args.square_size,
        "T_sdk_board": board.tolist() if board is not None else None,
        "board_pose_source": str(args.board_pose.resolve()) if args.board_pose else None,
        "board_pose_key": args.board_pose_key, "corner_detector": "sb_aligned_to_classic_origin",
        "max_reprojection_px": args.max_reprojection_px,
        "samples": [],
    }
    if args.resume:
        session = json.loads(path.read_text(encoding="utf-8"))
        for key in ("calibration_type", "camera_endpoint", "pattern_size_inner_corners", "square_size_m",
                    "T_sdk_board", "corner_detector", "max_reprojection_px"):
            if not same_recorded_config({key: session.get(key)}, {key: config[key]}, ("handeye_file", "replay_samples", "board_pose_source")):
                raise ValueError(f"续采参数 {key} 与已有记录不一致；旧双相机 session 不能用于固定板模式")
        if args.serial and session.get("camera_serial") not in (None, args.serial):
            raise ValueError("续采相机 serial 不一致")
        if len(session["samples"]) > args.samples:
            raise ValueError("--samples 小于已保存样本数")
    else:
        if args.data_dir.exists() and any(args.data_dir.iterdir()):
            raise FileExistsError(f"目录非空：{args.data_dir}；请使用新目录或 --resume")
        session = config
    if (args.data_dir / "camera_to_board.json").exists():
        raise FileExistsError("此目录已有正式标定结果，请使用新目录")
    args.serial = args.serial or session.get("camera_serial")
    args.reference_K = np.asarray(session["samples"][0]["K"]) if session["samples"] else None
    print("\n固定板→相机标定：棋盘格与第三视角相机全程保持固定，机械臂无需移动。")
    print(f"内角点 {args.pattern_size}，间距 {args.square_size} m；采 {args.samples} 组静止画面。")
    if board is not None:
        print("将用已有 T_sdk_board 合并相机到 SDK base 的变换。板的位置、原点和轴向必须与该矩阵完全一致。")
        if input("确认已有矩阵仍适用，输入 BOARD_UNCHANGED 继续：").strip() != "BOARD_UNCHANGED":
            print("已取消。")
            return 1
    with ExitStack() as stack:
        camera = CameraClient(args.endpoint)
        stack.callback(camera.close)
        try:
            _check_camera_health(camera, name="固定第三视角", endpoint=args.endpoint)
        except RuntimeError as exc:
            print(f"相机连接失败：{exc}", file=sys.stderr)
            return 2
        if not args.no_preview:
            import cv2
            os.environ.setdefault("QT_QPA_FONTS_DIR", "/usr/share/fonts")
            cv2.namedWindow(_WINDOW, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(_WINDOW, 1100, 750)
            stack.callback(cv2.destroyAllWindows)
        args.data_dir.mkdir(parents=True, exist_ok=True)
        _save_json(path, session)
        print(f"输出目录：{args.data_dir}；空格取样，松开脚踏板约 1 秒，再按空格确认。")
        while len(session["samples"]) < args.samples:
            index = len(session["samples"]) + 1
            candidate = prompt_sample(camera, args, index)
            if candidate is None:
                print("已退出；原始样本保留，可指定相同目录加 --resume 续采。")
                return 1
            packet, found = candidate
            if args.serial and packet.serial != args.serial:
                raise ValueError("相机 serial 在采样中变化")
            if args.reference_K is not None and not np.allclose(packet.intrinsic_matrix, args.reference_K, atol=1e-6, rtol=0):
                raise ValueError("相机内参在采样中变化")
            poses = [np.asarray(s["T_camera_board"]) for s in session["samples"]] + [found["T_camera_board"]]
            summary = summarize_samples(poses, max_spread_mm=args.max_spread_mm, max_spread_deg=args.max_spread_deg)
            if not summary["quality_passed"]:
                print(f"此帧未收录：相对首帧变化 {summary['translation_spread_max_mm']:.2f} mm / "
                      f"{summary['rotation_spread_max_deg']:.2f}°。请恢复固定板位和原点，或新建目录重采。")
                continue
            import cv2
            name = f"sample_{index:03d}"
            preview = args.data_dir / f"{name}_corners.png"
            if not cv2.imwrite(str(preview), _preview(packet, found, index, args.samples, frozen=True)):
                raise RuntimeError(f"预览保存失败：{preview}")
            np.savez_compressed(args.data_dir / f"{name}.npz", rgb=packet.rgb, depth_m=packet.depth,
                                K=packet.intrinsic_matrix, corners=found["corners"])
            args.serial = packet.serial
            args.reference_K = np.asarray(packet.intrinsic_matrix)
            session["camera_serial"] = packet.serial
            session["samples"].append({
                "T_camera_board": found["T_camera_board"].tolist(),
                "reprojection_rmse_px": found["reprojection_rmse_px"],
                "serial": packet.serial, "K": args.reference_K.tolist(),
                "frame_number": packet.frame_number, "timestamp_ms": packet.timestamp_ms,
                "captured_at_ns": time.time_ns(), "rgbd_file": f"{name}.npz",
                "corner_preview_file": preview.name,
            })
            _save_json(path, session)
            print(f"已收录 {index}/{args.samples}；继续保持板和相机固定。")
        summary = summarize_samples([np.asarray(s["T_camera_board"]) for s in session["samples"]],
                                    T_sdk_board=board, max_spread_mm=args.max_spread_mm,
                                    max_spread_deg=args.max_spread_deg)
        _save_json(args.data_dir / "quality_report.json", _jsonable(summary))
        print(f"静止重复性：{summary['translation_spread_max_mm']:.2f} mm / "
              f"{summary['rotation_spread_max_deg']:.2f}°；仅表示重复性，不是绝对标定误差。")
        if not summary["quality_passed"]:
            print("重复性未通过，保留原始样本，不生成正式矩阵。")
            return 2
        if not summary["repeatability_checked"]:
            print("本次仅单帧求解，未进行多帧重复性检查。")
        if args.visualize:
            # 工具缺少依赖时不得把结果标成已目视确认。
            import open3d  # noqa: F401
            from .camera_to_board import _visualize_pointcloud_and_frames
            if not args.no_preview:
                import cv2
                cv2.destroyAllWindows()
            with np.load(args.data_dir / session["samples"][0]["rgbd_file"]) as raw:
                _visualize_pointcloud_and_frames(raw["rgb"], raw["depth_m"], raw["K"],
                                                summary["T_camera_board"], args.pattern_size, args.square_size)
        if input("确认角点原点与标定结果可接受，输入 ACCEPT 保存；其他输入取消：").strip() != "ACCEPT":
            print("已取消；原始样本与质量报告已保存，未生成正式矩阵。")
            return 1
        result = {
            "schema_version": 1, "calibration_type": "fixed_board_camera",
            "pose_convention": "T_A_B maps B coordinates into A; translation in meter",
            "T_camera_board": summary["T_camera_board"].tolist(),
            "T_board_camera": summary["T_board_camera"].tolist(),
            "board_to_camera": summary["T_camera_board"].tolist(),
            "camera_to_board": summary["T_board_camera"].tolist(),
            "camera_serial": args.serial, "pattern_size_inner_corners": list(args.pattern_size),
            "square_size_m": args.square_size, "origin_confirmed": True,
            "captured_at_ns": session["samples"][-1]["captured_at_ns"],
            "source_session": "session.json", "quality_report": "quality_report.json",
            "open3d_visual_confirmed": bool(args.visualize),
        }
        _save_json(args.data_dir / "camera_to_board.json", result)
        if board is not None:
            _save_json(args.data_dir / "third_view_extrinsic.json", standard_camera_parameters(args.data_dir, session["samples"][0], {
                "T_sdk_board": board.tolist(), "T_sdk_third": summary["T_sdk_camera"].tolist(),
                "T_third_sdk": summary["T_camera_sdk"].tolist(),
                "camera_serial": args.serial, "board_pose_source": config["board_pose_source"],
                "board_pose_key": args.board_pose_key, "board_unchanged_confirmed": True,
                "pose_convention": result["pose_convention"],
                "source_session": "session.json", "source_board_camera": "camera_to_board.json",
            }))
        publish_result(args.data_dir, "board_camera")
        if board is not None:
            publish_result(args.data_dir, "third_camera")
        print(f"已保存板→相机 T_camera_board 及其逆：{args.data_dir / 'camera_to_board.json'}")
        if board is not None:
            print(f"已合并固定相机到 SDK base：{args.data_dir / 'third_view_extrinsic.json'}")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
