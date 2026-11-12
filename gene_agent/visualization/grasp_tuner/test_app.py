"""Viser 抓取代理的局部坐标约定测试。"""

from __future__ import annotations

import numpy as np

from .app import build_proxy_geometry


def test_proxy_extends_from_negative_z_palm_toward_positive_z_fingertips() -> None:
    """防止代理夹爪相对 TCP 翻转，而误导对 Knowledge pose 的判断。"""
    geometry = build_proxy_geometry(closing_width_m=0.040)

    np.testing.assert_allclose(geometry.palm_position, [0.0, 0.0, -0.025])
    np.testing.assert_allclose(geometry.left_finger_position, [0.024, 0.0, 0.012])
    np.testing.assert_allclose(geometry.right_finger_position, [-0.024, 0.0, 0.012])

    palm_tip_z = geometry.palm_position[2] + geometry.palm_extents[2] / 2.0
    finger_base_z = geometry.left_finger_position[2] - geometry.finger_extents[2] / 2.0
    fingertip_z = geometry.left_finger_position[2] + geometry.finger_extents[2] / 2.0
    assert palm_tip_z < 0.0
    assert finger_base_z < 0.0 < fingertip_z
    assert fingertip_z > palm_tip_z


def test_proxy_opening_is_symmetric_about_tcp_x_axis() -> None:
    """两指中心须沿 TCP ±X 对称，内表面间距等于 closing width。"""
    geometry = build_proxy_geometry(closing_width_m=0.040)

    assert geometry.left_finger_position[0] == -geometry.right_finger_position[0]
    inside_gap = (
        geometry.left_finger_position[0]
        - geometry.finger_extents[0] / 2.0
        - (geometry.right_finger_position[0] + geometry.finger_extents[0] / 2.0)
    )
    assert inside_gap == 0.040
