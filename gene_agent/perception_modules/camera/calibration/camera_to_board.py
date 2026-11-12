"""
标定步骤 1：相机 → 棋盘格变换。

从 camera server 获取 RGB-D 帧，实时显示棋盘格角点检测，
用户按 SPACE 开始采集多帧 PnP 求解相机→棋盘格变换。
完成后用 Open3D 显示点云 + 棋盘格坐标系验证。

用法:
    python -m gene_agent.perception_modules.camera.calibration.camera_to_board \
        --camera_endpoint tcp://127.0.0.1:18082 \
        --pattern_size 11 8 --square_size 0.015 \
        --data_dir data/calibration

    加 --headless 跳过所有可视化窗口（自动化模式）。
"""

from __future__ import annotations

from gene_agent.perception_modules.paths import PROJECT_ROOT

import argparse
import json
import os
import sys
import time
from pathlib import Path

# 修复 conda 环境下 OpenCV Qt5 后端找不到字体的问题
if "QT_QPA_FONTS_DIR" not in os.environ and os.path.isdir("/usr/share/fonts"):
    os.environ["QT_QPA_FONTS_DIR"] = "/usr/share/fonts"

import cv2
import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = PROJECT_ROOT
for _path in (_REPO_ROOT, _SCRIPT_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from gene_agent.perception_modules.camera import CameraClient
from gene_agent.perception_modules.camera.calibration.rotation_average import mean_rodrigues_rotation


# ---- OpenCV 实时窗口显示（和原项目 c2b.py 一致的写法）----


# ---- Open3D 点云 + 坐标系可视化 ----

def _make_board_plane(
    pattern_size: tuple[int, int],
    square_size: float,
    alpha: float = 1.0,
) -> "o3d.geometry.TriangleMesh":
    """生成棋盘格平面 mesh，和原项目 visualize_fixed_camera_board.py 一致。"""
    import open3d as o3d

    cols, rows = pattern_size
    size = float(square_size)
    width = (int(cols) - 1) * size
    height = (int(rows) - 1) * size
    vertices = np.asarray([
        [0.0, 0.0, 0.0],
        [width, 0.0, 0.0],
        [width, height, 0.0],
        [0.0, height, 0.0],
    ], dtype=np.float64)
    triangles = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices),
        o3d.utility.Vector3iVector(triangles),
    )
    mesh.paint_uniform_color([0.05, 0.65, 0.95])
    mesh.compute_vertex_normals()
    if alpha < 1.0:
        pass  # Open3D 旧版本不稳定支持透明材质
    return mesh


def _visualize_pointcloud_and_frames(
    rgb: np.ndarray,
    depth: np.ndarray,
    K: np.ndarray,
    board_to_camera: np.ndarray,
    pattern_size: tuple[int, int],
    square_size: float,
    *,
    max_depth: float = 2.5,
    voxel_size: float = 0.003,
    axis_size: float = 0.12,
) -> None:
    """用 Open3D 显示相机坐标系下的点云 + 棋盘格坐标系。

    和原项目 visualize_fixed_camera_board.py 保持一致的写法和默认参数。
    """
    try:
        import open3d as o3d
    except ImportError:
        print("请先安装 open3d: pip install open3d")
        return

    # ---- 生成点云（和原项目 load_rgbd_pointcloud 一致）----
    h, w = depth.shape
    if rgb.shape[:2] != depth.shape[:2]:
        rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)

    camera_matrix = np.asarray(K, dtype=np.float64).reshape(3, 3)
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width=w, height=h,
        fx=float(camera_matrix[0, 0]), fy=float(camera_matrix[1, 1]),
        cx=float(camera_matrix[0, 2]), cy=float(camera_matrix[1, 2]),
    )
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color=o3d.geometry.Image(rgb),
        depth=o3d.geometry.Image(depth.astype(np.float32)),
        depth_scale=1.0,
        depth_trunc=max_depth,
        convert_rgb_to_intensity=False,
    )
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
    if voxel_size > 0 and len(pcd.points) > 0:
        pcd = pcd.voxel_down_sample(voxel_size)

    # ---- 坐标系变换（和原项目 main() frame="camera" 分支一致）----
    # 相机坐标系原点
    camera_frame_tf = np.eye(4, dtype=np.float64)
    # 棋盘格坐标系在相机坐标系中的位姿（board2camera = T_board→cam）
    board_frame_tf = board_to_camera.copy()

    # 相机坐标轴
    camera_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_size)
    camera_axis.transform(camera_frame_tf)
    # 棋盘格坐标轴
    board_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_size)
    board_axis.transform(board_frame_tf)
    # 棋盘格平面（和原项目 make_board_plane 一致）
    board_plane = _make_board_plane(pattern_size, square_size, alpha=1.0)
    board_plane.transform(board_frame_tf)

    # ---- 显示顺序和原项目完全一致 ----
    geometries = [
        o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_size * 1.3),
        pcd,
        board_plane,
        board_axis,
        camera_axis,
    ]
    print(f"\n=== Open3D 点云验证 ===")
    print("显示坐标系: 相机坐标系")
    print("红/绿/蓝坐标轴分别是 X/Y/Z；标定正确时，标定板坐标轴和平面应贴在点云里的棋盘格上。")

    o3d.visualization.draw_geometries(
        geometries,
        window_name="fixed camera board check - camera",
    )


# ---- 主标定流程 ----

def calibrate_camera_to_board(
    camera_endpoint: str,
    pattern_size: tuple[int, int],
    square_size: float,
    data_dir: str,
    *,
    num_frames: int = 50,
    camera_timeout_ms: int = 6000,
    headless: bool = False,
) -> dict | None:
    """从 camera server 获取帧，实时显示角点，用户按 SPACE 开始采集。

    参数:
        camera_endpoint:  camera server 地址
        pattern_size:     棋盘格内角点数量 (cols, rows)
        square_size:      棋盘格方格边长（米）
        data_dir:         输出目录
        num_frames:       采集帧数（取平均）
        camera_timeout_ms: 相机取帧超时
        headless:         跳过所有可视化窗口

    返回:
        {"camera_to_board": 4x4矩阵, "board_to_camera": 4x4矩阵, "intrinsics": {...}}
        若用户按 Q 退出则返回 None
    """
    os.makedirs(data_dir, exist_ok=True)
    cols, rows = pattern_size

    # 构建 3D 棋盘格点（以棋盘格左上角为原点，x right, y down, z out=right-hand）
    objp = np.zeros((rows * cols, 3), np.float32)
    for i in range(rows):
        for j in range(cols):
            objp[i * cols + j] = [j * square_size, i * square_size, 0.0]

    # ---- 连接相机 ----
    client = CameraClient(camera_endpoint, timeout_ms=camera_timeout_ms)
    try:
        health = client.health()
        if not health.get("ok"):
            raise RuntimeError(f"camera server 未就绪: {health}")
        print(f"camera server 已连接: {health}")
        print(f"棋盘格: {cols}x{rows}, 方格边长={square_size}m")
        if not headless:
            print("请在板的目标原点内角点做可见标记；红圈 0 必须落在该角点。")
            print("确认后按 SPACE 开始采集，按 Q 退出")

        # ---- 采集状态 ----
        rvec_list: list[np.ndarray] = []
        tvec_list: list[np.ndarray] = []
        first_intrinsics: dict | None = None
        last_rgb: np.ndarray | None = None
        last_depth: np.ndarray | None = None
        last_corners: np.ndarray | None = None
        first_corner_pixel: np.ndarray | None = None
        K_for_vis: np.ndarray | None = None
        collecting = False
        origin_confirmed = False

        if headless:
            # headless 模式：直接采集 num_frames 帧
            collecting = True

        while True:
            packet = client.get_frame()
            rgb = packet.rgb  # HxWx3, uint8
            depth = packet.depth  # HxW, float32, 米
            K = np.asarray(packet.intrinsic_matrix, dtype=np.float64)
            dist = np.zeros(5)

            if first_intrinsics is None:
                first_intrinsics = {
                    "width": rgb.shape[1],
                    "height": rgb.shape[0],
                    "intrinsic_matrix": K.tolist(),
                    "serial": packet.serial,
                }
                if not headless:
                    h, w = rgb.shape[:2]
                    cv2.namedWindow("Camera → Board 标定", cv2.WINDOW_NORMAL)
                    cv2.resizeWindow("Camera → Board 标定", min(w, 1280), min(h, 720))
            K_for_vis = K

            # 转换为 BGR 用于 OpenCV 显示（camera server 输出 RGB）
            display = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            # 棋盘格检测
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, pattern_size)

            # 绘制状态文字（和原项目一致）
            if not collecting:
                cv2.putText(display, "Press SPACE to start collecting", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(display, f"Collecting... {len(rvec_list)}/{num_frames}", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # 绘制棋盘格角点（用原始 corners，和原项目完全一致）
            if ret:
                cv2.drawChessboardCorners(display, pattern_size, corners, ret)
                corner_zero = tuple(map(int, np.rint(corners[0, 0])))
                cv2.circle(display, corner_zero, 12, (0, 0, 255), 3)
                cv2.putText(display, "0: marked origin?", (corner_zero[0] + 14, corner_zero[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                # 采集模式：亚像素优化 + PnP
                if collecting:
                    corners_sub = cv2.cornerSubPix(
                        gray, corners, (11, 11), (-1, -1),
                        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
                    )
                    # 普通棋盘格可能被 OpenCV 从对角角点重新编号。板在采集期间
                    # 应保持静止；角点 0 大幅跳动意味着 180° 起点歧义，不能平均 pose。
                    corner_zero = corners_sub[0, 0].astype(np.float64)
                    board_diagonal_px = float(np.linalg.norm(corners_sub[-1, 0] - corner_zero))
                    if first_corner_pixel is not None and np.linalg.norm(
                        corner_zero - first_corner_pixel
                    ) > max(10.0, board_diagonal_px * 0.25):
                        raise RuntimeError(
                            "采集期间角点 0 跳到另一端，棋盘格 180° 编号不稳定；"
                            "请检查物理原点标记并重新采集"
                        )
                    success, rvec, tvec = cv2.solvePnP(objp, corners_sub, K, dist)
                    if success:
                        if first_corner_pixel is None:
                            first_corner_pixel = corner_zero.copy()
                        rvec_list.append(rvec)
                        tvec_list.append(tvec)
                        last_rgb = rgb.copy()
                        last_depth = depth.copy()
                        last_corners = corners_sub.copy()
                        print(f"  采集进度: {len(rvec_list)}/{num_frames}")

            if not headless:
                cv2.imshow("Camera → Board 标定", display)

            # 按键处理
            if headless:
                if len(rvec_list) >= num_frames:
                    break
            else:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("用户按 Q 退出")
                    cv2.destroyAllWindows()
                    return None
                if key == ord(' ') and not collecting:
                    if not ret:
                        print("当前帧没有检测到棋盘格，不能确认角点 0")
                        continue
                    collecting = True
                    origin_confirmed = True
                    rvec_list.clear()
                    tvec_list.clear()
                    print(f"开始采集 {num_frames} 帧...")

                # 采集完成自动停止
                if collecting and len(rvec_list) >= num_frames:
                    print("采集完成，正在计算平均值...")
                    break

    finally:
        client.close()
        if not headless:
            cv2.destroyAllWindows()

    if len(rvec_list) == 0:
        raise RuntimeError("没有成功检测到棋盘格的帧，请检查棋盘格摆放和光照")

    # ---- 计算平均位姿 ----
    print(f"\n采集完成，共 {len(rvec_list)} 帧，计算平均值...")
    tvec_avg = np.mean(tvec_list, axis=0)

    R_board2cam = mean_rodrigues_rotation(rvec_list)
    board_to_camera = np.eye(4)
    board_to_camera[:3, :3] = R_board2cam
    board_to_camera[:3, 3] = tvec_avg.flatten()

    camera_to_board = np.linalg.inv(board_to_camera)

    # ---- 保存 ----
    cam_param_path = os.path.join(data_dir, "cam_param.json")
    with open(cam_param_path, "w") as f:
        json.dump(first_intrinsics, f, indent=2)
    print(f"相机内参已保存: {cam_param_path}")

    ctb_path = os.path.join(data_dir, "camera_to_board.json")
    ctb_data = {
        "camera_to_board": camera_to_board.tolist(),
        "board_to_camera": board_to_camera.tolist(),
        "pattern_size": list(pattern_size),
        "square_size": square_size,
        "origin_confirmed": origin_confirmed,
        "captured_at_ns": time.time_ns(),
    }
    with open(ctb_path, "w") as f:
        json.dump(ctb_data, f, indent=2)
    print(f"相机→棋盘格 变换已保存: {ctb_path}")

    ctb_trans = camera_to_board[:3, 3] * 1000
    btc_trans = board_to_camera[:3, 3] * 1000
    print(f"\n=== 相机 → 棋盘格 标定完成 ===")
    print(f"成功帧数: {len(rvec_list)}")
    print(f"Camera→Board 平移: [{ctb_trans[0]:.1f}, {ctb_trans[1]:.1f}, {ctb_trans[2]:.1f}] mm")
    print(f"Board→Camera 平移: [{btc_trans[0]:.1f}, {btc_trans[1]:.1f}, {btc_trans[2]:.1f}] mm")

    # ---- Open3D 点云 + 坐标系可视化 ----
    if not headless and last_rgb is not None and last_depth is not None and K_for_vis is not None:
        _visualize_pointcloud_and_frames(
            rgb=last_rgb,
            depth=last_depth,
            K=K_for_vis,
            board_to_camera=board_to_camera,
            pattern_size=pattern_size,
            square_size=square_size,
        )

    return {
        "camera_to_board": camera_to_board,
        "board_to_camera": board_to_camera,
        "intrinsics": first_intrinsics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="相机 → 棋盘格变换采集（须在标定板最终放置后执行）"
    )
    parser.add_argument(
        "--camera_endpoint", default="tcp://127.0.0.1:18082",
        help="camera server 地址",
    )
    parser.add_argument(
        "--pattern_size", type=int, nargs=2, default=[11, 8],
        help="棋盘格内角点数量 (cols rows)",
    )
    parser.add_argument(
        "--square_size", type=float, default=0.015,
        help="棋盘格方格边长（米）",
    )
    parser.add_argument(
        "--data_dir", default="data/calibration",
        help="输出目录",
    )
    parser.add_argument(
        "--num_frames", type=int, default=50,
        help="采集帧数",
    )
    parser.add_argument(
        "--camera_timeout_ms", type=int, default=6000,
    )
    parser.add_argument(
        "--headless", action="store_true",
        help="跳过所有可视化窗口（自动化模式）",
    )
    args = parser.parse_args()

    calibrate_camera_to_board(
        camera_endpoint=args.camera_endpoint,
        pattern_size=tuple(args.pattern_size),
        square_size=args.square_size,
        data_dir=args.data_dir,
        num_frames=args.num_frames,
        camera_timeout_ms=args.camera_timeout_ms,
        headless=args.headless,
    )


if __name__ == "__main__":
    main()
