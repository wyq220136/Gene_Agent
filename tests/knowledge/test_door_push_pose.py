"""门板推压 Knowledge 只由几何参数决定，不依赖可视化或机器人。"""

import numpy as np

from gene_agent.knowledge.operation.door_push import door_push_pose


def test_push_contact_moves_with_hinge_fraction_top_distance_and_standoff():
    panel = np.eye(4)
    # 门宽沿世界 -Y，正面朝世界 -X，门轴位于 +Y 一侧。
    panel[:3, :3] = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
    panel[:3, 3] = [.5, .55, .16]
    common = dict(panel_pose_world=panel, panel_width_m=.36, panel_height_m=.22,
                  panel_thickness_m=.016, surface="outside")
    a = door_push_pose(**common, hinge_fraction=.5, top_down_distance_m=.05,
                       standoff_m=.03, tip_x_deg=0, tip_y_deg=0)
    b = door_push_pose(**common, hinge_fraction=.75, top_down_distance_m=.07,
                       standoff_m=.04, tip_x_deg=0, tip_y_deg=0)
    assert np.allclose(b[:3, 3] - a[:3, 3], [-.01, -.09, -.02])
    assert np.allclose(a[:3, 2], [0, 0, -1])


def test_push_rejects_invalid_contact_fraction():
    panel = np.eye(4)
    try:
        door_push_pose(panel, panel_width_m=.36, panel_height_m=.22,
                       panel_thickness_m=.016, hinge_fraction=1.2,
                       top_down_distance_m=.05, standoff_m=.03,
                       tip_x_deg=0, tip_y_deg=0, surface="outside")
    except ValueError as error:
        assert "hinge_fraction" in str(error)
    else:
        raise AssertionError("必须拒绝超出门板宽度的接触点")
