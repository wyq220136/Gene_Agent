"""以当前碗和真实指垫支点生成炉内取碗目标，不依赖旧放置TCP。"""
from __future__ import annotations

import numpy as np

from gene_agent.knowledge import bowl_grasp_candidates


def actual_bowl_extract_targets(
    bowl_world, tcp_world, pad_center_world, inward_world, *, tilt_deg=-45.,
) -> tuple[dict[str, np.ndarray], dict]:
    """从炉口侧斜抓碗缘；X开口、Z接近，倾斜绕指垫中心进行。

    bowl pose来自在线感知/GT；pad支点来自机器人几何，不使用成功轨迹。
    方位按接近轴与进炉方向的一致程度排序，避免掌部进入炉腔。
    """
    bowl = np.asarray(bowl_world, dtype=float)
    tcp = np.asarray(tcp_world, dtype=float)
    inward = np.asarray(inward_world, dtype=float)
    inward /= np.linalg.norm(inward)
    pivot = tcp[:3, :3].T @ (np.asarray(pad_center_world) - tcp[:3, 3])
    candidates = bowl_grasp_candidates(bowl, parameters={
        "tip_y_rad": [float(np.deg2rad(tilt_deg))],
        "radial_offset_m": .006,
        "tcp_world_offset_m": [0., 0., -.012],
        "tilt_pivot_tcp_m": pivot.tolist(),
    })
    candidates.sort(key=lambda row: float(
        np.asarray(row["T_world_tcp"])[:3, 2] @ inward), reverse=True)
    chosen = candidates[0]
    grasp = np.asarray(chosen["T_world_tcp"])
    pre2 = grasp.copy()
    pre2[:3, 3] -= grasp[:3, 2] * .045
    pre1 = pre2.copy()
    pre1[:3, 3] -= inward * .15
    return {"pre_extract_1": pre1, "pre_extract_2": pre2, "grasp": grasp}, {
        "target_policy": "actual_bowl_rim_contact_pivot",
        "selected_candidate": chosen,
        "inward_world": inward.tolist(),
        "pad_pivot_tcp_m": pivot.tolist(),
        "pose_source": "online_actual_bowl_pose",
    }


def actual_hold_retreat_targets(tcp_world, inward_world) -> dict[str, np.ndarray]:
    """先小幅抬离转盘，再保持实际夹持姿态完全退到炉口外。"""
    start = np.asarray(tcp_world, dtype=float)
    inward = np.asarray(inward_world, dtype=float)
    inward /= np.linalg.norm(inward)
    lift = start.copy()
    lift[2, 3] += .018
    outside = lift.copy()
    outside[:3, 3] -= inward * .19
    return {"post_extract_1": lift, "post_extract_2": outside}
