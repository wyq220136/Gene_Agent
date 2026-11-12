"""
标定步骤 3：计算相机 → 机械臂 base 外参矩阵。

将步骤 1 的 camera_to_board 和步骤 2 的 board_to_robot 合并，
生成相机到机械臂 base 的变换矩阵，写入 cam_param.json 的 extrinsic_matrix 字段。

用法:
    python -m gene_agent.perception_modules.camera.calibration.compute_extrinsic \
        --data_dir data/calibration

    计算完成后获取当前 RGB-D 帧并可视化：
    python -m gene_agent.perception_modules.camera.calibration.compute_extrinsic \
        --data_dir data/calibration \
        --visualize \
        --camera_endpoint tcp://127.0.0.1:18082
"""

from __future__ import annotations

from gene_agent.perception_modules.paths import PROJECT_ROOT

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = PROJECT_ROOT
for _path in (_REPO_ROOT, _SCRIPT_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


def _depth_to_points(
    depth: np.ndarray,
    rgb: np.ndarray,
    intrinsic: np.ndarray,
    *,
    max_depth: float,
    max_points: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """将有效 RGB-D 像素反投影为相机坐标系点云。"""
    depth = np.asarray(depth)
    rgb = np.asarray(rgb)
    intrinsic = np.asarray(intrinsic, dtype=float)
    if depth.ndim != 2:
        raise ValueError("depth 必须是 HxW 数组")
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("rgb 必须是 HxWx3 数组")
    if rgb.shape[:2] != depth.shape:
        raise ValueError("rgb 和 depth 的尺寸必须一致")
    if intrinsic.shape != (3, 3):
        raise ValueError("intrinsic 必须是 3x3 矩阵")
    if max_depth <= 0:
        raise ValueError("max_depth 必须大于 0")
    if max_points is not None and max_points <= 0:
        raise ValueError("max_points 必须为正数或 None")

    rows, cols = np.nonzero(
        np.isfinite(depth) & (depth > 0.0) & (depth < float(max_depth))
    )
    if len(rows) == 0:
        return (
            np.empty((0, 3), dtype=np.float64),
            np.empty((0, 3), dtype=np.uint8),
        )

    if max_points is not None and len(rows) > max_points:
        indices = np.linspace(0, len(rows) - 1, max_points, dtype=np.int64)
        rows = rows[indices]
        cols = cols[indices]

    z = depth[rows, cols].astype(np.float64)
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    if fx <= 0 or fy <= 0:
        raise ValueError("intrinsic 的 fx 和 fy 必须大于 0")
    x = (cols - cx) * z / fx
    y = (rows - cy) * z / fy
    points = np.column_stack((x, y, z))
    colors = np.ascontiguousarray(rgb[rows, cols])
    return points, colors


def _transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """使用 T_robot_camera 将相机坐标系点云变换到机器人坐标系。"""
    points = np.asarray(points, dtype=float)
    transform = np.asarray(transform, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points 必须是 Nx3 数组")
    if transform.shape != (4, 4):
        raise ValueError("transform 必须是 4x4 矩阵")
    return (
        transform[:3, :3] @ points.T + transform[:3, 3:4]
    ).T


def _load_recorded_tcp_position(data_dir: str) -> np.ndarray:
    """读取步骤 2 保存的机器人 SDK base 中 TCP 位置。"""
    path = os.path.join(data_dir, "board_to_robot.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"找不到 TCP 标定记录: {path}")
    with open(path, "r", encoding="utf-8") as stream:
        data = json.load(stream)
    position = np.asarray(data.get("robot_tcp_position"), dtype=float)
    if position.shape != (3,) or not np.all(np.isfinite(position)):
        raise ValueError(
            "board_to_robot.json 中的 robot_tcp_position 必须是有限的 3D 坐标"
        )
    return position


def _recompute_board_to_robot(
    old_board_to_robot: np.ndarray,
    new_board_tcp_position: np.ndarray,
    *,
    robot_tcp_position: np.ndarray | None = None,
    robot_tcp_pose: np.ndarray | None = None,
    board_tcp_rotation: np.ndarray | None = None,
    board_axes_aligned_to_sdk: bool = False,
) -> np.ndarray:
    """从保存的完整 TCP pose 与板中 TCP 旋转重算 T_SDK_BOARD。"""
    old_board_to_robot = np.asarray(old_board_to_robot, dtype=float)
    new_board_tcp_position = np.asarray(new_board_tcp_position, dtype=float)
    if old_board_to_robot.shape != (4, 4):
        raise ValueError("board_to_robot 必须是 4x4 矩阵")
    if (
        new_board_tcp_position.shape != (3,)
        or not np.all(np.isfinite(new_board_tcp_position))
    ):
        raise ValueError("new_board_tcp_position 必须是有限的 3D 坐标")
    if board_axes_aligned_to_sdk:
        point = np.asarray(robot_tcp_position, dtype=float)
        if point.shape != (3,) or not np.isfinite(point).all():
            raise ValueError("robot_tcp_position 必须是有限的 3D SDK 坐标")
        result = np.eye(4)
        result[:3, 3] = point - new_board_tcp_position
        return result
    if robot_tcp_pose is None or board_tcp_rotation is None:
        raise ValueError(
            "旧标定缺少 robot_tcp_pose 或 board_tcp_rotation；"
            "不能从单点记录判断 90°/180° 朝向，请重新执行步骤 2"
        )
    robot_tcp_pose = np.asarray(robot_tcp_pose, dtype=float)
    board_tcp_rotation = np.asarray(board_tcp_rotation, dtype=float)
    if robot_tcp_pose.shape != (4, 4) or board_tcp_rotation.shape != (3, 3):
        raise ValueError("robot_tcp_pose 必须为 4x4，board_tcp_rotation 必须为 3x3")
    if not np.isfinite(robot_tcp_pose).all() or not np.isfinite(board_tcp_rotation).all():
        raise ValueError("TCP 标定 pose 必须是有限数")
    if robot_tcp_position is not None and not np.allclose(
        robot_tcp_pose[:3, 3], np.asarray(robot_tcp_position, dtype=float), atol=1e-6
    ):
        raise ValueError("robot_tcp_pose 与 robot_tcp_position 不一致")
    robot_tcp_in_board = np.eye(4, dtype=float)
    robot_tcp_in_board[:3, :3] = board_tcp_rotation
    robot_tcp_in_board[:3, 3] = new_board_tcp_position
    return robot_tcp_pose @ np.linalg.inv(robot_tcp_in_board)


def visualize_extrinsic(
    data_dir: str,
    *,
    camera_endpoint: str = "tcp://127.0.0.1:18082",
    camera_timeout_ms: int = 6000,
    max_depth: float = 3.0,
    max_points: int = 50000,
) -> None:
    """显示当前 RGB-D 点云和相机到机器人 base 的外参结果。"""
    try:
        import open3d as o3d
    except ImportError as exc:
        raise RuntimeError(
            "--visualize 需要安装 open3d，请先在相应环境中安装"
        ) from exc

    cam_param_path = os.path.join(data_dir, "cam_param.json")
    if not os.path.isfile(cam_param_path):
        raise FileNotFoundError(f"找不到相机参数文件: {cam_param_path}")
    with open(cam_param_path, "r", encoding="utf-8") as stream:
        cam_param = json.load(stream)

    camera_to_robot = np.asarray(cam_param["extrinsic_matrix"], dtype=float)
    recorded_tcp_position = _load_recorded_tcp_position(data_dir)
    calibration_intrinsic = np.asarray(
        cam_param["intrinsic_matrix"], dtype=float
    )

    try:
        from gene_agent.perception_modules.camera import CameraClient
    except ModuleNotFoundError:
        # 直接执行脚本时，perception_modules 目录可能是 sys.path 根目录。
        from gene_agent.perception_modules.camera import CameraClient

    print(f"\n连接相机: {camera_endpoint}")
    client = CameraClient(camera_endpoint, timeout_ms=camera_timeout_ms)
    try:
        packet = client.get_frame()
    finally:
        client.close()

    live_intrinsic = np.asarray(packet.intrinsic_matrix, dtype=float)
    if not np.allclose(live_intrinsic, calibration_intrinsic, atol=1e-6):
        print("⚠ 当前帧内参与标定时内参不同，将使用当前帧内参进行点云反投影")

    points_camera, colors = _depth_to_points(
        packet.depth,
        packet.rgb,
        live_intrinsic,
        max_depth=max_depth,
        max_points=max_points,
    )
    if len(points_camera) == 0:
        raise RuntimeError("当前 RGB-D 帧没有有效深度点，无法可视化")
    points_robot = _transform_points(points_camera, camera_to_robot)

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points_robot)
    point_cloud.colors = o3d.utility.Vector3dVector(
        colors.astype(np.float64) / 255.0
    )

    robot_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15)
    camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.12)
    camera_frame.transform(camera_to_robot)
    tcp_marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
    tcp_marker.paint_uniform_color([1.0, 0.65, 0.0])
    tcp_marker.translate(recorded_tcp_position)

    print(f"点云顶点数: {len(points_robot)}")
    print("显示坐标系: 机器人 SDK base")
    print("红/绿/蓝坐标轴分别是 X/Y/Z；较小坐标轴表示相机坐标系")
    print(
        "黄色球体表示步骤 2 记录的 TCP 位置: "
        f"[{recorded_tcp_position[0]:.4f}, "
        f"{recorded_tcp_position[1]:.4f}, "
        f"{recorded_tcp_position[2]:.4f}] m"
    )
    o3d.visualization.draw_geometries(
        [point_cloud, robot_frame, camera_frame, tcp_marker],
        window_name="相机外参验证 - 机器人 SDK base",
        width=1280,
        height=720,
    )


def compute_extrinsic(data_dir: str) -> dict:
    """合并 camera→board 和 board→robot，输出 camera→robot 外参。

    棋盘格坐标系定义（和 camera_to_board 一致）:
        原点=棋盘格左上角，x right, y down, z out（右手系）

    坐标变换：步骤 2 已要求实际板轴与 SDK base 对齐，直接组合
    T_SDK_CAMERA = T_SDK_BOARD @ T_BOARD_CAMERA，不额外加 90°/180° 补偿。
    """
    # ---- 加载输入 ----
    ctb_path = os.path.join(data_dir, "camera_to_board.json")
    btr_path = os.path.join(data_dir, "board_to_robot.json")
    cam_param_path = os.path.join(data_dir, "cam_param.json")

    for path in [ctb_path, btr_path, cam_param_path]:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"找不到输入文件: {path}")

    with open(ctb_path, "r") as f:
        ctb_data = json.load(f)
    with open(btr_path, "r") as f:
        btr_data = json.load(f)
    with open(cam_param_path, "r") as f:
        cam_param = json.load(f)

    expected_capture = btr_data.get("camera_board_captured_at_ns")
    actual_capture = ctb_data.get("captured_at_ns")
    if expected_capture is None or actual_capture != expected_capture:
        raise ValueError(
            "camera_to_board 与 board_to_robot 不是同一次板放置的记录；"
            "请保持板固定并重新完成交互标定"
        )
    if not ctb_data.get("origin_confirmed"):
        raise ValueError("camera_to_board 未确认物理角点 0，不能合并外参")

    camera_to_board = np.asarray(ctb_data["camera_to_board"], dtype=float)
    old_board_to_robot = np.asarray(btr_data["board_to_robot"], dtype=float)
    new_board_tcp_position = np.asarray(
        btr_data["board_tcp_position"], dtype=float
    )
    robot_tcp_position = np.asarray(btr_data["robot_tcp_position"], dtype=float)

    # 允许直接编辑 board_to_robot.json 中的 board_tcp_position，重新推导
    # board_to_robot；机器人 TCP 在 SDK base 中的位置保持不变。
    board_to_robot = _recompute_board_to_robot(
        old_board_to_robot,
        new_board_tcp_position,
        robot_tcp_position=robot_tcp_position,
        robot_tcp_pose=btr_data.get("robot_tcp_pose"),
        board_tcp_rotation=btr_data.get("board_tcp_rotation"),
        board_axes_aligned_to_sdk=btr_data.get("board_axes_aligned_to_sdk") is True,
    )
    btr_data["board_to_robot"] = board_to_robot.tolist()
    btr_data["robot_to_board"] = np.linalg.inv(board_to_robot).tolist()
    btr_data["board_tcp_position"] = new_board_tcp_position.tolist()
    with open(btr_path, "w", encoding="utf-8") as stream:
        json.dump(btr_data, stream, indent=2)
    print(f"已根据 board_tcp_position 更新: {btr_path}")

    print(f"camera_to_board:\n{camera_to_board}")
    print(f"board_to_robot:\n{board_to_robot}")

    # 两个变换都是列向量约定，无隐藏的板轴翻转。
    camera_to_robot = board_to_robot @ camera_to_board
    print(f"\ncamera_to_robot:\n{camera_to_robot}")

    # ---- 验证：检查变换矩阵是否有效 ----
    rotation = camera_to_robot[:3, :3]
    det = np.linalg.det(rotation)
    print(f"旋转矩阵行列式: {det:.6f} (应接近 1.0)")

    orthogonality_error = float(np.linalg.norm(rotation.T @ rotation - np.eye(3)))
    is_valid = bool(np.isfinite(rotation).all() and abs(det - 1.0) < 0.01 and orthogonality_error < 0.01)
    if not is_valid:
        raise ValueError("旋转矩阵的行列式或正交性不符合刚体变换要求，不写入外参")

    # ---- 保存 ----
    # 更新 cam_param.json，写入外参
    cam_param["extrinsic_matrix"] = camera_to_robot.tolist()
    cam_param["camera_to_board_original"] = camera_to_board.tolist()
    cam_param["board_to_robot"] = board_to_robot.tolist()
    cam_param["note"] = (
        "extrinsic_matrix = board_to_robot @ camera_to_board; "
        "board orientation was confirmed interactively in step 2"
    )

    with open(cam_param_path, "w") as f:
        json.dump(cam_param, f, indent=2)
    print(f"\n✓ 外参矩阵已写入: {cam_param_path}")
    print(f"  extrinsic_matrix = camera → robot base 变换")
    print(f"  刚体矩阵结构检查: {'通过' if is_valid else '不通过'}；"
          "此检查不能证明外参角度或平移准确")

    return {
        "camera_to_robot": camera_to_robot.tolist(),
        "valid": bool(is_valid),
        "cam_param_path": cam_param_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="合并 camera→board 和 board→robot，计算 camera→robot 外参（步骤 3/3）"
    )
    parser.add_argument(
        "--data_dir", default="data/calibration",
        help="包含 camera_to_board.json 和 board_to_robot.json 的目录",
    )
    parser.add_argument(
        "--visualize", action="store_true",
        help="计算完成后获取当前 RGB-D 帧并显示机器人坐标系下的点云",
    )
    parser.add_argument(
        "--camera_endpoint", default="tcp://127.0.0.1:18082",
        help="camera server 地址（--visualize 时使用）",
    )
    parser.add_argument(
        "--camera_timeout_ms", type=int, default=6000,
        help="获取可视化帧的超时时间（毫秒）",
    )
    parser.add_argument(
        "--max_depth", type=float, default=3.0,
        help="可视化点云的最大深度（米）",
    )
    parser.add_argument(
        "--max_points", type=int, default=50000,
        help="可视化点云的最大点数",
    )
    args = parser.parse_args()

    compute_extrinsic(args.data_dir)
    if args.visualize:
        visualize_extrinsic(
            args.data_dir,
            camera_endpoint=args.camera_endpoint,
            camera_timeout_ms=args.camera_timeout_ms,
            max_depth=args.max_depth,
            max_points=args.max_points,
        )


if __name__ == "__main__":
    main()
