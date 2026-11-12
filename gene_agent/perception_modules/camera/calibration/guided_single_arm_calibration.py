"""单臂标定终端向导：人工摆板/测量，复用相机采集与 Open3D 检查。"""

from __future__ import annotations

from .storage import publish_result, protect_latest_result, default_run_dir, resolve_cli_paths

import argparse
import json
from pathlib import Path
import time
from typing import Callable

import numpy as np

from gene_agent.realworld.devices.arms.realman.connection import read_realman_tcp_pose
from .compute_extrinsic import visualize_extrinsic
from .single_arm_calibration import (
    calibrate_single_arm, calibrate_single_arm_from_positions,
    validate_tcp_for_aligned_board,
)


def calibrate_camera_to_board(**kwargs):
    """只在实际采集时加载 OpenCV，相机离线测试无需该依赖。"""
    from .camera_to_board import calibrate_camera_to_board as capture
    return capture(**kwargs)


def _require(input_fn: Callable[[str], str], token: str, prompt: str) -> None:
    if input_fn(f"{prompt}\n输入 {token} 继续，其他输入中止：").strip() != token:
        raise RuntimeError(f"未确认 {token}，流程中止")


def _read_xyz(input_fn: Callable[[str], str], prompt: str) -> list[float]:
    raw = input_fn(prompt).strip()
    try:
        values = [float(value) for value in raw.replace(",", " ").split()]
    except ValueError as exc:
        raise ValueError("请输入三个以米为单位的数字") from exc
    if len(values) != 3 or not np.isfinite(values).all():
        raise ValueError("请输入三个有限的米数")
    return values


def run_guided_calibration(
    data_dir: str | Path,
    side: str,
    tcp_pose_json: str | Path | None = None,
    *,
    robot_ip: str | None = None,
    robot_port: int = 8080,
    sdk_tcp_position: list[float] | None = None,
    camera_endpoint: str = "tcp://127.0.0.1:18082",
    pattern_size: tuple[int, int] = (11, 8),
    square_size: float = 0.015,
    num_frames: int = 50,
    task_camera_parameters: str | Path | None = None,
    input_fn: Callable[[str], str] = input,
) -> np.ndarray | None:
    """完成采集、合并和复核；只有最终 ACCEPT 才向任务文件发布外参。"""
    directory = Path(data_dir)
    protect_latest_result(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if side not in {"left", "right"}:
        raise ValueError("side 必须是 left 或 right")
    if tcp_pose_json is not None and sdk_tcp_position is not None:
        raise ValueError("离线 TCP pose 文件与离线 TCP 位置不能同时提供")
    if tcp_pose_json is None and sdk_tcp_position is None and not robot_ip:
        raise ValueError("自动读取 TCP 需要 --robot-ip；离线复现可提供 --tcp-pose-json 或 --sdk-tcp-position")
    task_path = Path(task_camera_parameters) if task_camera_parameters else None
    if task_path is not None and not task_path.is_file():
        raise FileNotFoundError(f"任务相机文件不存在: {task_path}")
    print("此向导不会发送机械臂运动命令。请确认机器人静止、TCP/夹爪安装配置与实物一致。")
    _require(input_fn, "BOARD_ALIGNED",
             "摆好标定板：物理标记的角点为原点；板 +X/+Y 对齐 SDK base +X/+Y，板 +Z 向上。"
             "板在后续采集期间不能移动；默认位置输入不要求夹爪具有特定朝向。")

    pose = None
    if tcp_pose_json is not None:
        tcp = json.loads(Path(tcp_pose_json).read_text(encoding="utf-8"))
        pose, error_deg = validate_tcp_for_aligned_board(tcp)
        sdk_tcp_position = pose[:3, 3].tolist()
        print(f"离线 TCP 在 {side} SDK base 中的位置（米）: {sdk_tcp_position}；姿态误差 {error_deg:.2f}°")
    elif sdk_tcp_position is not None:
        values = np.asarray(sdk_tcp_position, dtype=float)
        if values.shape != (3,) or not np.isfinite(values).all():
            raise ValueError("离线 SDK TCP 位置必须是三个有限的米数")
        sdk_tcp_position = values.tolist()
        print(f"离线 TCP 在 {side} SDK base 中的位置（米）: {sdk_tcp_position}")
    else:
        print(f"正在只读连接 {side} 臂 {robot_ip}:{robot_port}，读取关节并通过 SDK FK 求 TCP...")
        live_pose, joints = read_realman_tcp_pose(
            robot_ip, group=side, port=robot_port,
        )
        sdk_tcp_position = live_pose[:3, 3].tolist()
        record = {
            "source": "realman_sdk_fk",
            "robot_ip": robot_ip,
            "side": side,
            "captured_at_ns": time.time_ns(),
            "joints_rad": np.asarray(joints, dtype=float).tolist(),
            "tcp_pose_sdk": np.asarray(live_pose, dtype=float).tolist(),
        }
        (directory / "sdk_tcp_reading.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"当前 TCP 在 SDK base 中的位置（米）: {sdk_tcp_position}")
        print(f"完整读数已保存: {directory / 'sdk_tcp_reading.json'}")

    while True:
        print("将采集相机→板；画面红圈 0 须落在物理标记角点。采集完成后会打开 Open3D。")
        captured = calibrate_camera_to_board(
            camera_endpoint=camera_endpoint,
            pattern_size=pattern_size,
            square_size=square_size,
            data_dir=str(directory),
            num_frames=num_frames,
        )
        if captured is None:
            raise RuntimeError("相机采集被取消")
        answer = input_fn("检查 Open3D 中棋盘格坐标轴是否贴合板面。输入 CAMERA_OK 继续、RETRY 重拍：").strip()
        if answer == "CAMERA_OK":
            break
        if answer != "RETRY":
            raise RuntimeError("未确认相机到板标定，流程中止")

    board_tcp_position = _read_xyz(input_fn,
        "用格尺测量 TCP 在板坐标系中的 x y z（米），例如 0 0 0.15：")
    _require(input_fn, "BOARD_UNCHANGED",
             "核对测量值和正负号；确认从 TCP 位姿记录到相机采集期间板始终未移动。")
    if pose is None:
        result = calibrate_single_arm_from_positions(
            directory, side, sdk_tcp_position, board_tcp_position,
            board_unchanged_confirmed=True,
        )
    else:
        result = calibrate_single_arm(
            directory, side, pose, board_tcp_position,
            board_unchanged_confirmed=True,
        )
    print(f"候选 T_SDK_CAMERA 已写入 {directory / 'cam_param.json'}。")
    _require(input_fn, "TCP_VISIBLE",
             "请让夹爪 TCP 回到记录的已知位置并进入相机视野，随后检查叠加点云。")
    visualize_extrinsic(str(directory), camera_endpoint=camera_endpoint)
    answer = input_fn("检查黄色 TCP 球体与真实夹爪及板/桌面位置。输入 ACCEPT 发布到任务文件，其他输入保留候选并结束：").strip()
    if answer != "ACCEPT":
        print("结果保留在标定目录；未更新任务相机参数。")
        return None
    if task_path is not None:
        task_data = json.loads(task_path.read_text(encoding="utf-8"))
        task_data[f"{side}_extrinsic_matrix"] = result.tolist()
        task_path.write_text(json.dumps(task_data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"已更新任务相机外参: {task_path}")
    publish_result(directory, "manual")
    print("单臂标定已由操作者确认。")
    return result


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="终端引导的单臂相机外参标定")
    parser.add_argument("--data-dir", type=Path, help="默认data/calibration/runs/manual/时间目录")
    parser.add_argument("--side", choices=("left", "right"), required=True)
    parser.add_argument("--robot-ip", help="默认在线流程必须提供机械臂 IP；只读连接，不发运动命令")
    parser.add_argument("--robot-port", type=int, default=8080)
    parser.add_argument("--tcp-pose-json", type=Path,
                        help="仅离线复现：已保存的完整 4x4 T_SDK_TCP")
    parser.add_argument("--sdk-tcp-position", type=float, nargs=3,
                        metavar=("X", "Y", "Z"), help="仅离线复现：SDK TCP 位置，米")
    parser.add_argument("--camera-endpoint", default="tcp://127.0.0.1:18082")
    parser.add_argument("--pattern-size", type=int, nargs=2, default=(11, 8), metavar=("COLS", "ROWS"))
    parser.add_argument("--square-size", type=float, default=0.015)
    parser.add_argument("--num-frames", type=int, default=50)
    parser.add_argument("--task-camera-parameters", type=Path)
    args = parser.parse_args(argv)
    resolve_cli_paths(args, "data_dir", "tcp_pose_json")
    args.data_dir = args.data_dir or default_run_dir("manual")
    run_guided_calibration(
        args.data_dir, args.side, args.tcp_pose_json,
        robot_ip=args.robot_ip,
        robot_port=args.robot_port,
        sdk_tcp_position=args.sdk_tcp_position,
        camera_endpoint=args.camera_endpoint,
        pattern_size=tuple(args.pattern_size),
        square_size=args.square_size,
        num_frames=args.num_frames,
        task_camera_parameters=args.task_camera_parameters,
    )


if __name__ == "__main__":
    main()
