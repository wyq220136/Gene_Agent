"""保留原杯把顶抓几何，供候选与场景诊断共用。"""
import numpy as np

def top_down_cup_handle_grasp(cup_pose: np.ndarray,
                              handle_pose_in_cup: np.ndarray) -> np.ndarray:
    """TCP 局部 -X 指向机械臂本体，故世界 TCP +X 朝下，夹爪从上方进入。

    局部 Y 是双指闭合轴，对齐杯子局部 X 以夹住杯把上缘的厚度。
    TCP 比内侧指垫接触点低约 13.5 mm，符合 Pika 实际 site 几何。
    """
    cup = np.asarray(cup_pose, dtype=float)
    handle = np.asarray(handle_pose_in_cup, dtype=float)
    if cup.shape != (4, 4) or handle.shape != (4, 4):
        raise ValueError("杯子和杯把位姿必须为 4×4 变换")
    target = np.eye(4)
    target[:3, 3] = (cup @ np.r_[handle[:3, 3] + [0., .006, .0105], 1.])[:3]
    local_rotation = np.array([[0., 1., 0.],
                               [0., 0., -1.],
                               [-1., 0., 0.]])
    target[:3, :3] = cup[:3, :3] @ local_rotation
    return target

