"""
标定步骤 2：棋盘格 → 机械臂 base 变换（交互式）。

交互流程:
    1. 连接 Realman 机械臂
    2. 用户手动把机械臂 TCP 移动到棋盘格坐标系中的已知位置
    3. 用户明确确认低速运动，脚本将 TCP 调到唯一的 0° 姿态；
       若只差约 90° 腕部角，优先转末关节，其余情况发 TCP pose 目标
    4. 用户转动标定板并确认板 +X/+Y 与 SDK base +X/+Y 对齐
    5. 重新读取 TCP 位姿，输入其在板坐标系中的位置（米）
    6. 使用显式 T_BOARD_TCP 计算并确认 board_to_robot 变换

棋盘格坐标系的定义（和 camera_to_board 一致）:
    原点在棋盘格左上角内角点，x right, y down, z out（右手系）；
    此步骤要求将板平放，+Z 向上，+X/+Y 分别沿 SDK base +X/+Y。

用法:
    python -m gene_agent.perception_modules.camera.calibration.board_to_robot \
        --robot_ip 192.168.1.18 --robot_port 8080 \
        --side left \
        --data_dir data/calibration

    常用的参考测量:
      TCP 对准棋盘格原点，高度 z = 夹具到棋盘面的距离
      → board_pos: [0, 0, z]
      TCP 对准棋盘格中心
      → board_pos: [cols*square_size/2, rows*square_size/2, z]
"""

from __future__ import annotations

from gene_agent.perception_modules.paths import PROJECT_ROOT

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = PROJECT_ROOT
for _path in (_REPO_ROOT, _SCRIPT_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


def _interactive_input(prompt: str, default: str = "") -> str:
    """带默认值的交互式输入。"""
    if default:
        result = input(f"{prompt} [{default}]: ").strip()
        return result if result else default
    return input(f"{prompt}: ").strip()


def _standard_downward_rotation(current_rotation: np.ndarray) -> np.ndarray:
    """返回唯一的 0° 目标：TCP +X 沿 SDK +X，+Y 沿 SDK -Y，+Z 向下。"""
    current = np.asarray(current_rotation, dtype=float)
    if current.shape != (3, 3) or not np.all(np.isfinite(current)):
        raise ValueError("current_rotation 必须是有限的 3x3 旋转矩阵")
    return np.diag([1.0, -1.0, -1.0])


def _board_to_robot_from_tcp(
    sdk_tcp: np.ndarray, board_tcp_position: np.ndarray,
) -> np.ndarray:
    """已确认板轴与 SDK base 对齐时，由同一 TCP pose 求 T_SDK_BOARD。

    板平放且 +Z 向上；向下 TCP 的 +Z 向下，因此 T_BOARD_TCP 的旋转
    为绕板 +X 轴 180°，不能设为单位阵。平移是实测 TCP 在板坐标系中的米数。
    """
    sdk_tcp = np.asarray(sdk_tcp, dtype=float)
    board_tcp_position = np.asarray(board_tcp_position, dtype=float)
    if sdk_tcp.shape != (4, 4) or board_tcp_position.shape != (3,):
        raise ValueError("需要 4x4 TCP pose 和 3D 板坐标")
    if not np.isfinite(sdk_tcp).all() or not np.isfinite(board_tcp_position).all():
        raise ValueError("标定输入必须是有限数")
    board_tcp = np.eye(4)
    board_tcp[:3, :3] = np.diag([1.0, -1.0, -1.0])
    board_tcp[:3, 3] = board_tcp_position
    return sdk_tcp @ np.linalg.inv(board_tcp)


def _last_joint_alignment_delta(
    current_rotation: np.ndarray, target_rotation: np.ndarray,
) -> float | None:
    """仅当姿态差是局部 Z 轴约 90° 时，返回末关节应转的弧度数。"""
    relative = np.asarray(current_rotation, dtype=float).T @ np.asarray(target_rotation, dtype=float)
    vector = Rotation.from_matrix(relative).as_rotvec()
    if np.linalg.norm(vector[:2]) > np.deg2rad(2.0):
        return None
    if abs(vector[2]) > np.deg2rad(95.0):
        return None
    return float(vector[2])


def _motion_debug_values(
    current_tcp: np.ndarray,
    target_tcp: np.ndarray,
    flange_tcp: np.ndarray,
) -> dict[str, np.ndarray | float]:
    """计算姿态调整诊断数据，不连接机器人也可以单元测试。"""
    current_tcp = np.asarray(current_tcp, dtype=float)
    target_tcp = np.asarray(target_tcp, dtype=float)
    flange_tcp = np.asarray(flange_tcp, dtype=float)
    current_flange = current_tcp @ np.linalg.inv(flange_tcp)
    target_flange = target_tcp @ np.linalg.inv(flange_tcp)
    relative_rotation = (
        current_flange[:3, :3].T @ target_flange[:3, :3]
    )
    return {
        "current_flange_pose": current_flange,
        "target_flange_pose": target_flange,
        "tcp_position_delta_m": target_tcp[:3, 3] - current_tcp[:3, 3],
        "flange_position_delta_m": target_flange[:3, 3] - current_flange[:3, 3],
        "flange_displacement_m": float(
            np.linalg.norm(target_flange[:3, 3] - current_flange[:3, 3])
        ),
        "flange_rotation_delta_deg": float(
            np.rad2deg(Rotation.from_matrix(relative_rotation).magnitude())
        ),
    }


def _print_motion_debug(arm, current_tcp: np.ndarray, target_tcp: np.ndarray) -> None:
    """打印运动前的 pose 转换、位移和 SDK 可达性诊断。"""
    values = _motion_debug_values(current_tcp, target_tcp, arm.flange_tcp)
    target_flange = values["target_flange_pose"]
    assert isinstance(target_flange, np.ndarray)

    print("\n[debug] ===== 姿态调整诊断 =====")
    print("[debug] 当前 TCP pose（SDK base）:")
    print(np.array2string(current_tcp, precision=6, suppress_small=True))
    print("[debug] 目标 TCP pose（SDK base）:")
    print(np.array2string(target_tcp, precision=6, suppress_small=True))
    print("[debug] 当前 flange pose（SDK base）:")
    print(
        np.array2string(
            values["current_flange_pose"], precision=6, suppress_small=True
        )
    )
    print("[debug] 实际发送的目标 flange pose（SDK base）:")
    print(np.array2string(target_flange, precision=6, suppress_small=True))
    print(
        "[debug] TCP 位移: "
        f"{np.linalg.norm(values['tcp_position_delta_m']):.6f} m；"
        "flange 位移: "
        f"{values['flange_displacement_m']:.6f} m；"
        "flange 姿态变化: "
        f"{values['flange_rotation_delta_deg']:.3f}°"
    )
    sdk_pose = arm._pose_to_6d(target_flange)
    print(f"[debug] 目标 flange 对应的 SDK 6D pose: {sdk_pose}")
    print(
        "[debug] SDK 姿态运动速度参数: "
        f"{arm._speed(0.1, arm.cartesian_speed)}"
    )
    try:
        print(f"[debug] SDK IK 可达性: {arm.is_pose_reachable(target_flange)}")
    except Exception as error:
        print(f"[debug] SDK IK 可达性检查失败: {type(error).__name__}: {error}")
    print("[debug] =========================\n")


def _print_realman_state_debug(arm) -> None:
    """运动失败后通过 FK 和独立接口读取当前状态。"""
    try:
        flange_pose = arm.get_flange_pose()
        tcp_pose = flange_pose @ arm.flange_tcp
        print("[debug] 当前 flange pose（关节角 + SDK FK）:")
        print(np.array2string(flange_pose, precision=6, suppress_small=True))
        print("[debug] 当前 TCP pose（flange @ T_F_TCP）:")
        print(np.array2string(tcp_pose, precision=6, suppress_small=True))
    except Exception as error:
        print(f"[debug] 读取当前 FK 位姿失败: {type(error).__name__}: {error}")

    state_queries = (
        ("控制器状态", "rm_get_controller_state"),
        ("关节错误和抱闸状态", "rm_get_joint_err_flag"),
        ("关节使能状态", "rm_get_joint_en_state"),
        ("机械臂电源状态", "rm_get_arm_power_state"),
    )
    for label, method_name in state_queries:
        try:
            state = getattr(arm.robot, method_name)()
            print(f"[debug] {label}: {state}")
        except Exception as error:
            print(f"[debug] 读取{label}失败: {type(error).__name__}: {error}")


def calibrate_board_to_robot(
    robot_ip: str,
    robot_port: int,
    side: str,
    data_dir: str,
    *,
    robot_level: int = 3,
    robot_mode: int = 2,
    debug: bool = False,
    arm_factory=None,
) -> dict:
    """交互式测量棋盘格到机械臂 base 的变换。

    返回:
        {"board_to_robot": 4x4矩阵, "robot_to_board": 4x4矩阵}
    """
    os.makedirs(data_dir, exist_ok=True)

    # ---- 连接机械臂 ----
    if arm_factory is None:
        try:
            from robots.realman.realman_arm import RealmanArm
        except ImportError as exc:
            raise RuntimeError(
                "当前 Gene_Agent 未集成 Realman 实机控制后端；"
                "需在受控真机环境提供 RealmanArm，离线数值验证不连接机械臂"
            ) from exc
        arm_factory = RealmanArm

    print(f"\n连接 Realman: {robot_ip}:{robot_port} (side={side})")
    arm = arm_factory(
        ip=robot_ip,
        port=robot_port,
        side=side,
        level=robot_level,
        mode=robot_mode,
    )
    arm.connect()
    print("✓ 机械臂已连接")

    try:
        # ---- 步骤 1: 用户移动机械臂 ----
        print("\n" + "=" * 60)
        print("  步骤 1: 移动机械臂到目标位置")
        print("=" * 60)
        print("请手动将机械臂 TCP 移动到棋盘格上的已知位置，并保证周围净空。")
        print("推荐对准棋盘格左上角内角点上方；后续会固定到唯一的 0° 姿态。")
        input("\n移动完成后按 Enter 继续...")

        # ---- 步骤 2: 姿态标准化（夹爪垂直向下，轴对齐世界坐标）----
        print("\n" + "=" * 60)
        print("  步骤 2: 姿态标准化")
        print("=" * 60)
        print("将 TCP 姿态调整为标准姿态：")
        print("  夹爪垂直向下（TCP z → 世界 -z）")
        print("  TCP +X 沿 SDK base +X，TCP +Y 沿 SDK base -Y；不允许 x/y 互换")
        print()
        tcp_pose = arm.get_tcp_pose()
        position = tcp_pose[:3, 3].copy()
        print(f"当前 TCP 位置: [{position[0]:.4f}, {position[1]:.4f}, {position[2]:.4f}] m")
        print(f"当前 TCP 姿态:\n{tcp_pose[:3, :3]}")

        standard_rot = _standard_downward_rotation(tcp_pose[:3, :3])
        standard_rotation_delta_deg = np.rad2deg(
            Rotation.from_matrix(tcp_pose[:3, :3].T @ standard_rot).magnitude()
        )
        standard_pose = np.eye(4)
        standard_pose[:3, :3] = standard_rot
        standard_pose[:3, 3] = position
        print("唯一的 0° 目标 TCP 姿态:")
        print(standard_rot)
        print(f"需要旋转: {standard_rotation_delta_deg:.3f}°")
        wrist_delta = _last_joint_alignment_delta(tcp_pose[:3, :3], standard_rot)
        use_wrist = wrist_delta is not None and hasattr(arm, "rotate_last_joint")
        if use_wrist:
            print(f"检测到仅需末关节转 {np.rad2deg(wrist_delta):+.2f}°；"
                  "将保持前面关节不变，以 speed_scale=0.1 执行。")
        else:
            print("将以 speed_scale=0.1 发送 TCP 姿态目标；请观察整臂路径及净空。")
        if _interactive_input("确认安全后输入 MOVE 执行低速运动") != "MOVE":
            raise RuntimeError("未确认低速运动，标定中止")
        if debug:
            if use_wrist:
                print(f"[debug] 末关节相对角: {np.rad2deg(wrist_delta):+.3f}°")
            else:
                _print_motion_debug(arm, tcp_pose, standard_pose)
        try:
            if use_wrist:
                arm.rotate_last_joint(wrist_delta, speed_scale=0.1, blocking=True)
            else:
                arm.move_to_tcp_pose(standard_pose, speed_scale=0.1, blocking=True)
        except Exception:
            if debug:
                _print_realman_state_debug(arm)
            raise
        print("✓ 低速姿态调整命令已完成")

        # 棋盘格的外观有 90° 对称性；必须由人确认真实角点与 +X/+Y 方向。
        print("\n现在请转动标定板（不是改写计算结果）：")
        print("  板 +X（左上角内角点向右）沿 SDK base +X；")
        print("  板 +Y（同一角点向下）沿 SDK base +Y；板 +Z 向上。")
        print("  如夹爪与板差 90°，先用示教器低速调整最后一个关节，")
        print("  然后重新执行本步骤，使最终 TCP 仍回到上面打印的 0° 姿态。")
        if _interactive_input("目视确认板的角点和轴向后输入 BOARD_ALIGNED") != "BOARD_ALIGNED":
            raise RuntimeError("未确认标定板轴向，标定中止")
        board_aligned_ns = time.time_ns()

        # ---- 步骤 3: 确认并读取位姿 ----
        print("\n" + "=" * 60)
        print("  步骤 3: 读取 TCP 位姿")
        print("=" * 60)
        tcp_pose = arm.get_tcp_pose()
        robot_position = tcp_pose[:3, 3].copy()
        robot_rotation = tcp_pose[:3, :3].copy()
        orientation_error_deg = np.rad2deg(
            Rotation.from_matrix(standard_rot.T @ robot_rotation).magnitude()
        )
        if orientation_error_deg > 2.0:
            raise RuntimeError(
                f"最终 TCP 与 0° 姿态相差 {orientation_error_deg:.2f}°；"
                "请重新调整最后关节及标定板后重试"
            )
        print(f"TCP 在机械臂 base 坐标系中的位置:")
        print(f"  x={robot_position[0]:.4f} m")
        print(f"  y={robot_position[1]:.4f} m")
        print(f"  z={robot_position[2]:.4f} m")
        print(f"旋转矩阵:\n{robot_rotation}")

        # ---- 步骤 4: 用户输入棋盘格坐标 ----
        print("\n" + "=" * 60)
        print("  步骤 4: 输入 TCP 在棋盘格坐标系中的位置")
        print("=" * 60)
        print("请测量/计算 TCP 在棋盘格坐标系中的位置（单位：米）。")
        print("棋盘格坐标系: 原点=棋盘格左上角, x right, y down, z out")
        print()
        print("常用参考值:")
        print("  TCP 对准棋盘格原点, z=夹具到棋盘面距离 → 0, 0, z")
        print("  TCP 对准棋盘格中心 → cols*size/2, rows*size/2, z")

        bx = float(_interactive_input("board x (m)", default="0.0"))
        by = float(_interactive_input("board y (m)", default="0.0"))
        bz = float(_interactive_input("board z (m, 夹具到棋盘面的高度)", default="0.15"))
        board_position = np.array([bx, by, bz])
        print(f"\nTCP 在棋盘格坐标系中的位置: [{bx:.4f}, {by:.4f}, {bz:.4f}] m")

        # ---- 步骤 5: 计算变换 ----
        print("\n" + "=" * 60)
        print("  步骤 5: 计算 board → robot 变换")
        print("=" * 60)

        board_to_robot = _board_to_robot_from_tcp(tcp_pose, board_position)

        robot_to_board = np.linalg.inv(board_to_robot)

        # ---- 步骤 6: 验证 ----
        print("\n" + "=" * 60)
        print("  步骤 6: 验证")
        print("=" * 60)
        # 用 board→robot 变换把 TCP 在棋盘格中的坐标映射回机械臂坐标系
        tcp_in_robot_from_transform = board_to_robot @ np.append(board_position, 1.0)
        predicted = tcp_in_robot_from_transform[:3]
        actual = robot_position
        error = np.linalg.norm(predicted - actual) * 1000  # mm

        print(f"机械臂读取的 TCP 位置:       {actual}")
        print(f"通过变换反算的 TCP 位置:     {predicted}")
        print(f"代回误差: {error:.2f} mm（恒等检查，不能证明外参准确）")
        print(f"最终 TCP 与 0° 目标的角度误差: {orientation_error_deg:.3f}°")
        print("板 +X/+Y/+Z 在 SDK base 中的预测方向（列）:")
        print(board_to_robot[:3, :3])
        print("\n请保持标定板完全不动；此时可将机械臂移出相机视野。")
        print("在另一终端重新运行 camera_to_board，确认角点 0 的物理标记，")
        print("完成后回到本终端；此前保存的 camera_to_board 已因转板而过期。")
        if _interactive_input("重新采集完成后输入 CAMERA_CAPTURED") != "CAMERA_CAPTURED":
            raise RuntimeError("相机到标定板变换未重新采集，标定中止")
        camera_board_path = Path(data_dir) / "camera_to_board.json"
        if not camera_board_path.is_file():
            raise RuntimeError("camera_to_board.json 早于本次板轴确认，请保持板固定并重新采集")
        camera_board_record = json.loads(camera_board_path.read_text(encoding="utf-8"))
        if int(camera_board_record.get("captured_at_ns", 0)) <= board_aligned_ns:
            raise RuntimeError("camera_to_board.json 早于本次板轴确认，请保持板固定并重新采集")
        if not camera_board_record.get("origin_confirmed"):
            raise RuntimeError("相机标定未确认角点 0 对应物理标记，不保存板到机器人变换")
        if _interactive_input("核对板原点及轴向后输入 SAVE 保存") != "SAVE":
            raise RuntimeError("未确认轴向，标定结果未保存")

        # ---- 保存 ----
        result = {
            "board_to_robot": board_to_robot.tolist(),
            "robot_to_board": robot_to_board.tolist(),
            "robot_tcp_position": robot_position.tolist(),
            "board_tcp_position": board_position.tolist(),
            "robot_tcp_pose": tcp_pose.tolist(),
            "board_tcp_rotation": np.diag([1.0, -1.0, -1.0]).tolist(),
            "orientation_error_deg": float(orientation_error_deg),
            "camera_board_captured_at_ns": int(camera_board_record["captured_at_ns"]),
        }
        output_path = os.path.join(data_dir, "board_to_robot.json")
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n✓ 已保存: {output_path}")

    finally:
        arm.disconnect()
        print("机械臂已断开")

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="夹爪与棋盘格对齐、刷新相机板位姿并记录板到机械臂 base 变换"
    )
    parser.add_argument("--robot_ip", default="192.168.1.18", help="机械臂 IP")
    parser.add_argument("--robot_port", type=int, default=8080, help="机械臂端口")
    parser.add_argument(
        "--side", default="left", choices=["left", "right"],
        help="机械臂左右手",
    )
    parser.add_argument(
        "--data_dir", default="data/calibration",
        help="输出目录",
    )
    parser.add_argument("--robot_level", type=int, default=3)
    parser.add_argument("--robot_mode", type=int, default=2)
    parser.add_argument(
        "--debug",
        action="store_true",
        help="打印姿态转换、SDK 6D pose、可达性和控制器状态诊断",
    )
    args = parser.parse_args()

    calibrate_board_to_robot(
        robot_ip=args.robot_ip,
        robot_port=args.robot_port,
        side=args.side,
        data_dir=args.data_dir,
        robot_level=args.robot_level,
        robot_mode=args.robot_mode,
        debug=args.debug,
    )


if __name__ == "__main__":
    main()
