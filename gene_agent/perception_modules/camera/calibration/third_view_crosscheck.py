"""用同一块棋盘格把手腕相机标定传递到固定第三视角相机。"""

from __future__ import annotations

import math

import numpy as np

from gene_agent.knowledge.geometry.transforms import validate_pose


def _pose(value: np.ndarray, name: str) -> np.ndarray:
    return validate_pose(np.asarray(value, dtype=float), field_name=name)


def _rotation_deg(a: np.ndarray, b: np.ndarray) -> float:
    relative = a[:3, :3].T @ b[:3, :3]
    skew = np.array([relative[2, 1] - relative[1, 2],
                     relative[0, 2] - relative[2, 0],
                     relative[1, 0] - relative[0, 1]])
    return math.degrees(math.atan2(float(np.linalg.norm(skew)) / 2.0,
                                   float(np.trace(relative) - 1.0) / 2.0))


def check_flange_pose_diversity(
    poses: list[np.ndarray], *, min_from_first_deg: float = 15.0,
    min_secondary_span_deg: float = 10.0,
) -> dict:
    """板位 B 的手腕须相对 A 旋转，且验证姿态覆盖至少两个旋转轴。"""
    import cv2

    if len(poses) < 3:
        raise ValueError("至少需要两个绕不同轴的验证姿态")
    validated = [_pose(item, f"T_sdk_flange_{index}") for index, item in enumerate(poses)]
    reference = validated[0]
    vectors = np.stack([
        cv2.Rodrigues(reference[:3, :3].T @ item[:3, :3])[0].reshape(3)
        for item in validated[1:]
    ])
    angles = np.rad2deg(np.linalg.norm(vectors, axis=1))
    singular = np.rad2deg(np.linalg.svd(vectors, compute_uv=False))
    if float(angles.min()) < min_from_first_deg or singular[1] < min_secondary_span_deg:
        raise ValueError("板位 B 手腕姿态须相对 A 至少旋转 15°，并绕至少两个轴明显旋转")
    return {"min_rotation_from_stage_a_deg": float(angles.min()),
            "secondary_rotation_span_deg": float(singular[1])}


def solve_fixed_camera_from_pair(
    T_sdk_flange: np.ndarray, T_flange_wrist: np.ndarray,
    T_wrist_board: np.ndarray, T_third_board: np.ndarray,
) -> dict:
    """首个板位：T_sdk_third = T_sdk_flange @ T_flange_wrist @ T_wrist_board @ inv(T_third_board)。"""
    flange = _pose(T_sdk_flange, "T_sdk_flange")
    wrist = _pose(T_flange_wrist, "T_flange_wrist")
    wrist_board = _pose(T_wrist_board, "T_wrist_board")
    third_board = _pose(T_third_board, "T_third_board")
    sdk_board = _pose(flange @ wrist @ wrist_board, "T_sdk_board")
    sdk_third = _pose(sdk_board @ np.linalg.inv(third_board), "T_sdk_third")
    return {
        "T_sdk_board": sdk_board,
        "T_sdk_third": sdk_third,
        "T_third_board": third_board,
    }


def compare_board_poses(
    T_sdk_third: np.ndarray, T_sdk_flange: np.ndarray,
    T_flange_wrist: np.ndarray, T_wrist_board: np.ndarray,
    T_third_board: np.ndarray, *, reference_board: np.ndarray,
    min_board_move_m: float = 0.03, min_board_rotation_deg: float = 5.0,
) -> dict:
    """移板后分别由手腕链和第三视角链求板位姿，并计算相互偏差。"""
    if min_board_move_m < 0 or min_board_rotation_deg < 0:
        raise ValueError("最小移板量不能为负")
    base = _pose(T_sdk_third, "T_sdk_third")
    board_wrist = _pose(
        _pose(T_sdk_flange, "T_sdk_flange")
        @ _pose(T_flange_wrist, "T_flange_wrist")
        @ _pose(T_wrist_board, "T_wrist_board"),
        "T_sdk_board_from_wrist",
    )
    board_third = _pose(base @ _pose(T_third_board, "T_third_board"),
                        "T_sdk_board_from_third")
    original = _pose(reference_board, "reference_board")
    moved_m = float(np.linalg.norm(board_wrist[:3, 3] - original[:3, 3]))
    moved_deg = _rotation_deg(original, board_wrist)
    if moved_m < min_board_move_m and moved_deg < min_board_rotation_deg:
        raise ValueError("请先移动标定板到不同的位置或方向；原位置不能独立验证")
    return {
        "T_sdk_board_from_wrist": board_wrist,
        "T_sdk_board_from_third": board_third,
        "translation_m": float(np.linalg.norm(board_wrist[:3, 3] - board_third[:3, 3])),
        "rotation_deg": _rotation_deg(board_wrist, board_third),
        "board_move_translation_m": moved_m,
        "board_move_rotation_deg": moved_deg,
    }
