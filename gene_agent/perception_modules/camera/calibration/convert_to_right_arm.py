"""
标定步骤 4：左臂外参 → 右臂外参 + 点云可视化验证。

利用 RealmanInstallation 的完整变换链（和 curobo_demo.py 一致），
从只标定左臂的结果推导右臂外参，并用场景点云可视化核对。

变换链（以右臂 cuRobo 为 reference，和 curobo_demo.py 一致）:
    左 SDK → body → 右 SDK     (sdk_from_sdk, 标定外参转换用)
    左 cuRobo → 右 cuRobo      (reference_from_curobo, 规划帧转换用)

用法:
    # 仅计算并保存右臂外参
    python -m gene_agent.perception_modules.camera.calibration.convert_to_right_arm \
        --data_dir data/calibration

    # 计算 + 点云可视化验证
    python -m gene_agent.perception_modules.camera.calibration.convert_to_right_arm \
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

from gene_agent.robots.installation import InstallationTransforms


load_installation_transforms = InstallationTransforms.from_file


def _load_matrix(data: dict | list) -> np.ndarray:
    return np.asarray(data, dtype=float)


def _is_valid_transform(matrix: np.ndarray) -> bool:
    if matrix.shape != (4, 4):
        return False
    if not np.allclose(matrix[3], [0, 0, 0, 1]):
        return False
    det = np.linalg.det(matrix[:3, :3])
    return abs(det - 1.0) < 0.01


def _right_from_left_sdk(
    camera_to_left: np.ndarray,
    camera_to_right: np.ndarray,
) -> np.ndarray:
    """计算左臂 SDK base 到右臂 SDK base 的变换。"""
    camera_to_left = np.asarray(camera_to_left, dtype=float)
    camera_to_right = np.asarray(camera_to_right, dtype=float)
    if camera_to_left.shape != (4, 4) or camera_to_right.shape != (4, 4):
        raise ValueError("相机到左右 SDK base 的外参都必须是 4x4 矩阵")
    # T_Sright_Sleft = T_Sright_C @ inverse(T_Sleft_C)
    return camera_to_right @ np.linalg.inv(camera_to_left)


def _override_right_from_left_x(
    transform: np.ndarray,
    x_translation_mm: float,
) -> np.ndarray:
    """只覆盖左 SDK base 到右 SDK base 的 x 平移，单位为毫米。"""
    value = np.asarray(transform, dtype=float)
    if value.shape != (4, 4) or not np.all(np.isfinite(value)):
        raise ValueError("左右 SDK base 变换必须是有限的 4x4 矩阵")
    if not np.isfinite(x_translation_mm):
        raise ValueError("x_translation_mm 必须是有限数值")
    adjusted = value.copy()
    adjusted[0, 3] = float(x_translation_mm) / 1000.0
    return adjusted


def convert_left_to_right(
    data_dir: str,
    installation_path: str = "assets/robots/realman/installation.json",
    installation_name: str = "mounted",
    right_from_left_x_mm: float | None = None,
) -> dict:
    """从左臂标定结果推导右臂外参。

    使用 RealmanInstallation（和 curobo_demo.py 一致的 API）计算:
    - left_sdk_to_right_sdk: 左臂 SDK → 右臂 SDK（用于标定外参转换）
    - left_curobo_to_right_curobo: 左臂 cuRobo 规划帧 → 右臂 cuRobo 规划帧

    参数:
        data_dir:           左臂标定输出目录（含 cam_param.json）
        installation_path:  installation.json 路径
        installation_name:  installation 配置名 ("mounted" 或 "unrotated")
        right_from_left_x_mm: 可选，覆盖右 SDK 相对左 SDK 的 x 平移（毫米）

    返回:
        {"cam_to_right_base": 4x4矩阵, "left_sdk_to_right_sdk": 4x4矩阵, ...}
    """
    # ---- 加载输入 ----
    cam_param_path = os.path.join(data_dir, "cam_param.json")
    if not os.path.isfile(cam_param_path):
        raise FileNotFoundError(f"找不到左臂标定结果: {cam_param_path}")
    resolved_installation_path = (_REPO_ROOT / installation_path).resolve()
    if not resolved_installation_path.is_file():
        raise FileNotFoundError(f"找不到 installation.json: {resolved_installation_path}")

    with open(cam_param_path, "r") as f:
        cam_param = json.load(f)

    cam_to_left_sdk = _load_matrix(cam_param["extrinsic_matrix"])
    if not _is_valid_transform(cam_to_left_sdk):
        raise ValueError("cam_param.json 中 extrinsic_matrix 不是有效的 4x4 变换矩阵")

    # ---- 使用 RealmanInstallation（和 curobo_demo.py 一致）----
    installation = load_installation_transforms(resolved_installation_path, installation_name)

    # 左臂 SDK → 右臂 SDK（标定外参用）
    # 推导: left_sdk → body → right_sdk = inv(body_from_sdk["right"]) @ body_from_sdk["left"]
    left_sdk_to_right_sdk = installation.sdk_from_sdk("left", "right")
    if right_from_left_x_mm is not None:
        left_sdk_to_right_sdk = _override_right_from_left_x(
            left_sdk_to_right_sdk,
            right_from_left_x_mm,
        )
        print(
            "\n已覆盖右 SDK 相对左 SDK 的 x 平移: "
            f"{right_from_left_x_mm:.3f} mm"
        )

    # 左臂 cuRobo → 右臂 cuRobo（规划帧用，以右臂为 reference）
    # 这是 curobo_demo.py 中 _reference_from_curobo("left") 的结果
    left_curobo_to_right_curobo = installation.reference_from_curobo(
        "left", reference_side="right"
    )

    # ---- 相机 → 右臂 SDK ----
    # cam → right_sdk = left_sdk_to_right_sdk @ cam_to_left_sdk
    cam_to_right_sdk = left_sdk_to_right_sdk @ cam_to_left_sdk

    # ---- 验证 ----
    print("=== 左臂外参 → 右臂外参 ===")
    print(f"\n安装配置: {installation_name}")
    print(f"基于: {installation_path}")

    print(f"\n--- 左臂 SDK → 右臂 SDK（标定外参转换用）---")
    print(left_sdk_to_right_sdk)
    trans_mm = left_sdk_to_right_sdk[:3, 3] * 1000
    print(f"平移: x={trans_mm[0]:.1f}mm, y={trans_mm[1]:.1f}mm, z={trans_mm[2]:.1f}mm")
    print(f"距离: {np.linalg.norm(left_sdk_to_right_sdk[:3, 3]) * 1000:.1f}mm")

    print(f"\n--- 左臂 cuRobo → 右臂 cuRobo（规划帧转换用）---")
    print(left_curobo_to_right_curobo)
    trans_cb = left_curobo_to_right_curobo[:3, 3] * 1000
    print(f"平移: x={trans_cb[0]:.1f}mm, y={trans_cb[1]:.1f}mm, z={trans_cb[2]:.1f}mm")
    print(f"距离: {np.linalg.norm(left_curobo_to_right_curobo[:3, 3]) * 1000:.1f}mm")

    print(f"\n--- body_from_sdk（用于理解 SDK 原点在 body 中的位置）---")
    print(f"右臂 SDK 在 body 坐标系中: {installation.body_from_sdk['right'][:3, 3] * 1000} mm")
    print(f"左臂 SDK 在 body 坐标系中: {installation.body_from_sdk['left'][:3, 3] * 1000} mm")

    print(f"\n--- sdk_to_curobo（SDK → cuRobo 规划帧的旋转）---")
    print(f"右臂 sdk_to_curobo (绕 y 轴): {np.degrees(np.arcsin(installation.sdk_to_curobo['right'][0, 2])):.1f}°")
    print(f"左臂 sdk_to_curobo (绕 y 轴): {np.degrees(np.arcsin(installation.sdk_to_curobo['left'][2, 0])):.1f}°")

    print(f"\n相机 → 左臂 SDK (标定值):")
    print(cam_to_left_sdk)
    print(f"\n相机 → 右臂 SDK (推导值):")
    print(cam_to_right_sdk)

    if not _is_valid_transform(cam_to_right_sdk):
        print("⚠ 警告: 推导的右臂外参不是有效变换矩阵")
    else:
        print("✓ 变换矩阵有效")

    # ---- 保存 ----
    # 更新 cam_param.json，追加右臂外参和完整变换信息
    cam_param["extrinsic_matrix_right"] = cam_to_right_sdk.tolist()
    cam_param["extrinsic_matrix_left"] = cam_to_left_sdk.tolist()
    cam_param["left_sdk_to_right_sdk"] = left_sdk_to_right_sdk.tolist()
    cam_param["left_curobo_to_right_curobo"] = left_curobo_to_right_curobo.tolist()
    cam_param["installation_name"] = installation_name
    cam_param["note"] = cam_param.get("note", "") + (
        " | extrinsic_matrix_right = left_sdk_to_right_sdk @ extrinsic_matrix_left, "
        "使用 RealmanInstallation.sdk_from_sdk('left','right') 计算"
    )

    # 同时保存独立的右臂 cam_param
    right_cam_param = dict(cam_param)
    right_cam_param["extrinsic_matrix"] = cam_to_right_sdk.tolist()
    right_cam_param_path = os.path.join(data_dir, "cam_param_right.json")
    with open(right_cam_param_path, "w") as f:
        json.dump(right_cam_param, f, indent=2)

    # 更新原始 cam_param.json
    with open(cam_param_path, "w") as f:
        json.dump(cam_param, f, indent=2)

    print(f"\n✓ 右臂外参已保存:")
    print(f"  {cam_param_path} (新增 extrinsic_matrix_right 字段)")
    print(f"  {right_cam_param_path} (独立右臂外参文件)")

    return {
        "cam_to_left_sdk": cam_to_left_sdk,
        "cam_to_right_sdk": cam_to_right_sdk,
        "left_sdk_to_right_sdk": left_sdk_to_right_sdk,
        "left_curobo_to_right_curobo": left_curobo_to_right_curobo,
        "body_from_sdk_left": installation.body_from_sdk["left"],
        "body_from_sdk_right": installation.body_from_sdk["right"],
    }


def visualize_with_pointcloud(
    data_dir: str,
    camera_endpoint: str = "tcp://127.0.0.1:18082",
    camera_timeout_ms: int = 6000,
    robot_ip: str = "192.168.1.18",
    robot_port: int = 8080,
    robot_level: int = 3,
    robot_mode: int = 2,
) -> None:
    """获取当前场景点云和右臂 TCP，并统一显示在右臂 SDK base 中。

    需要安装 open3d: pip install open3d
    """
    try:
        import open3d as o3d
    except ImportError:
        print("请先安装 open3d: pip install open3d")
        return

    # 加载外参
    cam_param_path = os.path.join(data_dir, "cam_param.json")
    with open(cam_param_path, "r") as f:
        cam_param = json.load(f)

    cam_to_left = _load_matrix(cam_param["extrinsic_matrix"])
    cam_to_right = _load_matrix(cam_param.get(
        "extrinsic_matrix_right",
        cam_param.get("extrinsic_matrix", [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]),
    ))
    K = _load_matrix(cam_param["intrinsic_matrix"])

    # 获取当前帧
    from gene_agent.perception_modules.camera import CameraClient

    print(f"\n连接相机: {camera_endpoint}")
    client = CameraClient(camera_endpoint, timeout_ms=camera_timeout_ms)
    try:
        packet = client.get_frame()
    finally:
        client.close()

    # 只在可视化分支连接真机，并读取一次当前右臂 TCP pose。
    from robots.realman.realman_arm import RealmanArm

    print(f"连接右臂: {robot_ip}:{robot_port}")
    arm = RealmanArm(
        ip=robot_ip,
        port=robot_port,
        side="right",
        level=robot_level,
        mode=robot_mode,
    )
    arm.connect()
    try:
        right_tcp_pose = arm.get_tcp_pose()
    finally:
        arm.disconnect()
    print("已读取当前右臂 TCP pose（右臂 SDK base）:")
    print(right_tcp_pose)

    rgb = packet.rgb
    depth = packet.depth  # meters, HxW
    h, w = depth.shape
    print(f"帧尺寸: {w}x{h}")

    # 生成点云（在相机坐标系中）
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    z = depth
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    valid = (z > 0) & (z < 3.0)  # 只取 0~3m 范围
    points_cam = np.stack([x[valid], y[valid], z[valid]], axis=-1)  # (N, 3)
    colors_cam = rgb[valid].astype(float) / 255.0  # (N, 3)

    # 下采样（避免点太多）
    if len(points_cam) > 50000:
        idx = np.random.choice(len(points_cam), 50000, replace=False)
        points_cam = points_cam[idx]
        colors_cam = colors_cam[idx]

    # 转换到右臂 base 坐标系
    points_right = (cam_to_right[:3, :3] @ points_cam.T + cam_to_right[:3, 3:4]).T

    # 构建 open3d 对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_right)
    pcd.colors = o3d.utility.Vector3dVector(colors_cam)

    # 右臂 base 是显示坐标系原点，左臂 base 转换到右臂坐标系中显示。
    right_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15)
    right_from_left = _right_from_left_sdk(cam_to_left, cam_to_right)
    left_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15)
    left_frame.transform(right_from_left)

    # 黄色球体和坐标轴表示实时读取到的右臂 TCP pose。
    tcp_marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
    tcp_marker.paint_uniform_color([1.0, 0.65, 0.0])
    tcp_marker.translate(right_tcp_pose[:3, 3])
    tcp_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.12)
    tcp_frame.transform(right_tcp_pose)

    print(f"\n点云顶点数: {len(points_cam)}")
    print("显示坐标系: 右臂 SDK base（红色=x, 绿色=y, 蓝色=z）")
    print("左臂 SDK base 在右臂坐标系中的变换:")
    print(right_from_left)
    tcp_trans = right_tcp_pose[:3, 3] * 1000
    print(
        "右臂 TCP 位置（右臂 SDK base）: "
        f"x={tcp_trans[0]:.1f}mm, y={tcp_trans[1]:.1f}mm, z={tcp_trans[2]:.1f}mm"
    )

    o3d.visualization.draw_geometries(
        [pcd, left_frame, right_frame, tcp_marker, tcp_frame],
        window_name="点云 + 右臂实时 TCP 验证",
        width=1280,
        height=720,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="左臂外参 → 右臂外参 + 点云可视化验证（步骤 4）"
    )
    parser.add_argument(
        "--data_dir", default="data/calibration",
        help="左臂标定输出目录",
    )
    parser.add_argument(
        "--installation_path",
        default="assets/robots/realman/installation.json",
        help="installation.json 路径",
    )
    parser.add_argument(
        "--installation_name",
        default="mounted",
        choices=["mounted", "unrotated"],
        help="installation 配置名",
    )
    parser.add_argument(
        "--visualize", action="store_true",
        help="读取一次右臂 TCP，并可视化右臂坐标系和 RGB-D 点云",
    )
    parser.add_argument(
        "--camera_endpoint", default="tcp://127.0.0.1:18082",
        help="camera server 地址（--visualize 时需要）",
    )
    parser.add_argument("--robot_ip", default="192.168.1.18", help="右臂 IP")
    parser.add_argument("--robot_port", type=int, default=8080, help="右臂端口")
    parser.add_argument("--robot_level", type=int, default=3)
    parser.add_argument("--robot_mode", type=int, default=2)
    parser.add_argument(
        "--right_from_left_x_mm",
        type=float,
        default=None,
        help="覆盖右 SDK 相对左 SDK 的 x 平移（毫米），不修改 installation.json",
    )
    args = parser.parse_args()

    convert_left_to_right(
        data_dir=args.data_dir,
        installation_path=args.installation_path,
        installation_name=args.installation_name,
        right_from_left_x_mm=args.right_from_left_x_mm,
    )

    if args.visualize:
        visualize_with_pointcloud(
            data_dir=args.data_dir,
            camera_endpoint=args.camera_endpoint,
            robot_ip=args.robot_ip,
            robot_port=args.robot_port,
            robot_level=args.robot_level,
            robot_mode=args.robot_mode,
        )


if __name__ == "__main__":
    main()
