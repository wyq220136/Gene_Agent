"""用单臂 SDK TCP 实测位姿合并相机外参；不连接或运动机械臂。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from gene_agent.knowledge.geometry.transforms import validate_pose
from .board_to_robot import _board_to_robot_from_tcp
from .compute_extrinsic import compute_extrinsic


def validate_tcp_for_aligned_board(tcp_pose: np.ndarray) -> tuple[np.ndarray, float]:
    """确认 SDK TCP 满足平放标定板所需的向下 0° 姿态。"""
    pose = validate_pose(tcp_pose, field_name="robot_tcp_pose")
    target_rotation = np.diag([1.0, -1.0, -1.0])
    error_deg = float(np.rad2deg(Rotation.from_matrix(
        target_rotation.T @ pose[:3, :3]
    ).magnitude()))
    if error_deg > 2.0:
        raise ValueError(f"TCP 姿态与要求的向下 0° 姿态相差 {error_deg:.2f}°，超过 2°")
    return pose, error_deg


def calibrate_single_arm(
    data_dir: str | Path,
    side: str,
    tcp_pose: np.ndarray,
    board_tcp_position: list[float] | np.ndarray,
    *,
    board_unchanged_confirmed: bool,
    task_camera_parameters: str | Path | None = None,
) -> np.ndarray:
    """输出 T_SDK_CAMERA；输入 TCP pose 必须已在该臂 SDK base 下，单位米。"""
    if side not in {"left", "right"}:
        raise ValueError("side 必须是 left 或 right")
    if not board_unchanged_confirmed:
        raise ValueError("必须确认标定板轴与 SDK base 对齐且读取 TCP 后未移动标定板")
    directory = Path(data_dir)
    pose, error_deg = validate_tcp_for_aligned_board(tcp_pose)
    position = np.asarray(board_tcp_position, dtype=float)
    if position.shape != (3,) or not np.isfinite(position).all():
        raise ValueError("board_tcp_position 必须是有限的 3D 米坐标")
    target_rotation = np.diag([1.0, -1.0, -1.0])

    camera_record = json.loads((directory / "camera_to_board.json").read_text(encoding="utf-8"))
    if not camera_record.get("origin_confirmed"):
        raise ValueError("相机记录未确认棋盘格物理角点 0")
    capture_ns = int(camera_record.get("captured_at_ns", 0))
    if capture_ns <= 0:
        raise ValueError("相机记录缺少有效 captured_at_ns")
    board = _board_to_robot_from_tcp(pose, position)
    board_record = {
        "board_to_robot": board.tolist(),
        "robot_to_board": np.linalg.inv(board).tolist(),
        "robot_tcp_position": pose[:3, 3].tolist(),
        "board_tcp_position": position.tolist(),
        "robot_tcp_pose": pose.tolist(),
        "board_tcp_rotation": target_rotation.tolist(),
        "orientation_error_deg": error_deg,
        "camera_board_captured_at_ns": capture_ns,
        "side": side,
    }
    return _finish_calibration(directory, side, board_record, task_camera_parameters)


def calibrate_single_arm_from_positions(
    data_dir: str | Path,
    side: str,
    sdk_tcp_position: list[float] | np.ndarray,
    board_tcp_position: list[float] | np.ndarray,
    *,
    board_unchanged_confirmed: bool,
    task_camera_parameters: str | Path | None = None,
) -> np.ndarray:
    """板轴已与 SDK 对齐时，只用两坐标系中的同一个 TCP 点求外参。"""
    if side not in {"left", "right"}:
        raise ValueError("side 必须是 left 或 right")
    if not board_unchanged_confirmed:
        raise ValueError("必须确认标定板轴与 SDK base 对齐且板未移动")
    sdk = np.asarray(sdk_tcp_position, dtype=float)
    board_point = np.asarray(board_tcp_position, dtype=float)
    if sdk.shape != (3,) or board_point.shape != (3,) or not np.isfinite(sdk).all() or not np.isfinite(board_point).all():
        raise ValueError("SDK 和标定板中的 TCP 位置都必须是有限的 3D 米坐标")
    directory = Path(data_dir)
    camera_record = json.loads((directory / "camera_to_board.json").read_text(encoding="utf-8"))
    if not camera_record.get("origin_confirmed"):
        raise ValueError("相机记录未确认棋盘格物理角点 0")
    capture_ns = int(camera_record.get("captured_at_ns", 0))
    if capture_ns <= 0:
        raise ValueError("相机记录缺少有效 captured_at_ns")
    board = np.eye(4)
    board[:3, 3] = sdk - board_point
    board_record = {
        "board_to_robot": board.tolist(),
        "robot_to_board": np.linalg.inv(board).tolist(),
        "robot_tcp_position": sdk.tolist(),
        "board_tcp_position": board_point.tolist(),
        "board_axes_aligned_to_sdk": True,
        "camera_board_captured_at_ns": capture_ns,
        "side": side,
    }
    return _finish_calibration(directory, side, board_record, task_camera_parameters)


def _finish_calibration(
    directory: Path,
    side: str,
    board_record: dict,
    task_camera_parameters: str | Path | None,
) -> np.ndarray:
    camera_path = directory / "cam_param.json"
    if not camera_path.is_file():
        raise FileNotFoundError(f"缺少相机内参文件: {camera_path}")
    target_path = Path(task_camera_parameters) if task_camera_parameters else None
    if target_path is not None and not target_path.is_file():
        raise FileNotFoundError(f"缺少任务相机参数文件: {target_path}")
    (directory / "board_to_robot.json").write_text(
        json.dumps(board_record, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    result = np.asarray(compute_extrinsic(str(directory))["camera_to_robot"], dtype=float)
    camera = json.loads(camera_path.read_text(encoding="utf-8"))
    for stale_key in (
        "extrinsic_matrix_left", "extrinsic_matrix_right",
        "left_sdk_to_right_sdk", "left_curobo_to_right_curobo", "installation_name",
        f"{'left' if side == 'right' else 'right'}_extrinsic_matrix",
    ):
        camera.pop(stale_key, None)
    camera[f"{side}_extrinsic_matrix"] = result.tolist()
    camera["calibrated_side"] = side
    camera_path.write_text(json.dumps(camera, ensure_ascii=False, indent=2), encoding="utf-8")
    if target_path is not None:
        task_camera = json.loads(target_path.read_text(encoding="utf-8"))
        task_camera[f"{side}_extrinsic_matrix"] = result.tolist()
        target_path.write_text(json.dumps(task_camera, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="单臂 SDK base 相机外参离线合并")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--side", choices=("left", "right"), required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--tcp-pose-json", type=Path,
                        help="可选完整 4x4 T_SDK_TCP JSON 数组；平移单位米")
    source.add_argument("--sdk-tcp-position", type=float, nargs=3,
                        metavar=("X", "Y", "Z"), help="TCP 在该臂 SDK base 中的位置，米")
    parser.add_argument("--board-tcp-position", type=float, nargs=3, required=True,
                        metavar=("X", "Y", "Z"), help="TCP 在棋盘格坐标系的位置，米")
    parser.add_argument("--confirm-board-aligned-and-unchanged", action="store_true",
                        help="确认板轴与 SDK base 对齐，且 TCP 测量后至相机采集期间板未移动")
    parser.add_argument("--task-camera-parameters", type=Path,
                        help="可选：同时更新现有 camera_parameters_0.json 的单臂外参字段")
    args = parser.parse_args()
    if args.tcp_pose_json is not None:
        pose = json.loads(args.tcp_pose_json.read_text(encoding="utf-8"))
        result = calibrate_single_arm(
            args.data_dir, args.side, pose, args.board_tcp_position,
            board_unchanged_confirmed=args.confirm_board_aligned_and_unchanged,
            task_camera_parameters=args.task_camera_parameters,
        )
    else:
        result = calibrate_single_arm_from_positions(
            args.data_dir, args.side, args.sdk_tcp_position, args.board_tcp_position,
            board_unchanged_confirmed=args.confirm_board_aligned_and_unchanged,
            task_camera_parameters=args.task_camera_parameters,
        )
    print(f"已保存 T_S{args.side}_CAMERA：\n{result}")


if __name__ == "__main__":
    main()
