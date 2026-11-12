"""固定机器人与双相机、多板位外参估计；T_A_B将B中的点转换到A，单位米。"""
from __future__ import annotations

import numpy as np

from gene_agent.knowledge.geometry.transforms import validate_pose
from .wrist_handeye import _mean_pose, _rotation_deg


def _pose(value):
    return validate_pose(np.asarray(value, dtype=float))


def _difference(first, second):
    return (float(np.linalg.norm(first[:3, 3] - second[:3, 3]) * 1000),
            _rotation_deg(first[:3, :3].T @ second[:3, :3]))


def check_fixed_robot(sample, reference_flange, reference_joints):
    """相机固定的假设必须以实际法兰与关节检查；不能仅比较TCP目标。"""
    current = _pose(sample['T_sdk_flange'])
    reference = _pose(reference_flange)
    joints = np.asarray(sample['joints_rad'], dtype=float)
    baseline = np.asarray(reference_joints, dtype=float)
    if (joints.shape != (7,) or baseline.shape != (7,)
            or not np.isfinite(joints).all() or not np.isfinite(baseline).all()):
        raise ValueError('机械臂状态必须包含七个有限弧度关节角')
    mm, deg = _difference(reference, current)
    delta = float(np.rad2deg(np.max(np.abs(joints - baseline))))
    if mm > 1.0 or deg > 0.2 or delta > 0.2:
        raise ValueError(f'机械臂相对初始姿态移动：{mm:.2f} mm / {deg:.2f}°，最大关节变化{delta:.3f}°；请恢复初始姿态或新建采集')


def check_new_board(sample, previous):
    """每个板位相对所有已收录板位至少移动30mm或旋转5度，不以重复图像充数。"""
    board = _pose(sample['wrist']['T_camera_board'])
    for item in previous:
        mm, deg = _difference(board, _pose(item['wrist']['T_camera_board']))
        if mm < 30 and deg < 5:
            raise ValueError(f'板位与已有样本过近：{mm:.1f} mm / {deg:.1f}°；请移动板至少3cm或旋转5°')


def evaluate_moving_board(samples, reference_flange, T_flange_wrist, *, validation_poses=2,
                          max_mm=10.0, max_deg=3.0):
    """等权平均训练观测的相对相机SE(3)变换；最后若干板位从不参与拟合。"""
    if (not isinstance(validation_poses, int) or validation_poses < 1
            or len(samples) - validation_poses < 3):
        raise ValueError('至少三个估计板位和一个独立验证板位')
    if not np.isfinite([max_mm, max_deg]).all() or min(max_mm, max_deg) <= 0:
        raise ValueError('误差阈值必须是有限正数')
    flange, wrist = _pose(reference_flange), _pose(T_flange_wrist)
    base_wrist = flange @ wrist
    relative = []
    for i, sample in enumerate(samples):
        check_fixed_robot(sample, flange, samples[0]['joints_rad'])
        check_new_board(sample, samples[:i])
        relative.append(_pose(sample['wrist']['T_camera_board']) @
                        np.linalg.inv(_pose(sample['third']['T_camera_board'])))
    train_count = len(samples) - validation_poses
    # 多板位估计相对相机变换，机器人基座关系只在拟合后组合一次。
    wrist_third = _mean_pose(relative[:train_count])
    base_third = base_wrist @ wrist_third
    checks, boards = [], []
    for i, sample in enumerate(samples):
        via_wrist = base_wrist @ _pose(sample['wrist']['T_camera_board'])
        via_third = base_third @ _pose(sample['third']['T_camera_board'])
        mm, deg = _difference(via_wrist, via_third)
        checks.append({'index': i + 1, 'holdout': i >= train_count,
                       'translation_mm': mm, 'rotation_deg': deg})
        boards.append({'index': i + 1, 'T_sdk_board': via_wrist.tolist(),
                       'T_board_sdk': np.linalg.inv(via_wrist).tolist(),
                       'T_wrist_board': _pose(sample['wrist']['T_camera_board']).tolist(),
                       'T_third_board': _pose(sample['third']['T_camera_board']).tolist(),
                       'T_board_third': np.linalg.inv(_pose(sample['third']['T_camera_board'])).tolist()})
    def maximum(items, key):
        return max(c[key] for c in items)
    held = checks[train_count:]
    return {'T_sdk_third': base_third.tolist(), 'T_third_sdk': np.linalg.inv(base_third).tolist(),
            'T_wrist_third': wrist_third.tolist(), 'T_sdk_wrist': base_wrist.tolist(),
            'train_count': train_count, 'validation_count': validation_poses, 'checks': checks, 'boards': boards,
            'validation_translation_max_mm': maximum(held, 'translation_mm'),
            'validation_rotation_max_deg': maximum(held, 'rotation_deg'),
            'thresholds': {'translation_mm': max_mm, 'rotation_deg': max_deg},
            'quality_passed': all(c['translation_mm'] <= max_mm and c['rotation_deg'] <= max_deg for c in checks),
            'method': 'equal_weight_translation_and_rotation_mean_on_training_pairs',
            'note': '留出板位不参与拟合；固定手腕的内部一致性无法检验已有手眼变换的绝对精度。'}
