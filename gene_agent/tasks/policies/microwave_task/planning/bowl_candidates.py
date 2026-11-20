"""仿真与真机共用的 Task1 自动抓碗候选输入，保持参考仿真的数值和顺序。"""

from __future__ import annotations

import numpy as np

BOWL_GRASP_TABLE_ANGLE_RANGE_DEG = (45.0, 70.0)
BOWL_GRASP_PREFERRED_TABLE_ANGLE_DEG = 55.0
BOWL_AUTO_GRASP_PROFILE = "pick_bowl_legacy_visual"
BOWL_AUTO_GRASP_DEPTH_M = 0.015


def auto_bowl_grasp_parameters(model, data, robot_metadata, *, depth_m=BOWL_AUTO_GRASP_DEPTH_M, group="right"):
    """从当前 FK 计算 TCP 中的指垫支点，不推进动力学、不修改模型状态。

    世界坐标只用于测得指垫中心；倾斜支点用 T_world_tcp 的逆旋转转回 TCP。
    抓取深度和世界 Z 偏移继承 Gene_Agent_new 的 auto 路线；显式深度优先。
    """
    if group != "right":
        raise NotImplementedError("自动抓碗候选筛选目前只支持右臂")
    tcp = data.site(robot_metadata["tcp_sites"][group])
    pad_ids = [model.geom(name).id for name in
               robot_metadata["grippers"][group]["pad_geom_names"]]
    if len(pad_ids) != 2:
        raise ValueError("自动抓碗需要两个指垫中心以确定倾斜支点")
    pad_center = np.asarray(data.geom_xpos[pad_ids], dtype=float).mean(axis=0)
    pivot = np.asarray(tcp.xmat, dtype=float).reshape(3, 3).T @ (pad_center - tcp.xpos)
    depth = float(depth_m)
    if not np.isfinite(depth) or not np.isfinite(pivot).all():
        raise ValueError("抓碗深度和 TCP 指垫支点必须有限")
    return {
        "depth_m": depth,
        # 支点落在碗沿厚度中心；两侧指垫跨过内外壁。
        "radial_offset_m": 0.0,
        # 负倾角让掌部留在碗外，避免保持碗直立入炉时腕部撞炉顶。
        "tip_y_rad": np.deg2rad([-35.0, -40.0, -30.0, -45.0, -25.0, -21.0]).tolist(),
        "max_abs_tip_y_rad": np.deg2rad(50.0),
        # 相对旧 -15 mm 偏移上移 3 mm，为内侧指杆留出抬升净空。
        "tcp_world_offset_m": [0.0, 0.0, -0.012],
        "tilt_pivot_tcp_m": pivot.tolist(),
    }
