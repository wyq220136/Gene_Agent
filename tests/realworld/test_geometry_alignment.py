"""实际平面与对称把手的几何契约；不运行求解器、仿真或设备。"""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from gene_agent.realworld.microwave.geometry_alignment import (
    fitted_table_pose, handle_halfturn, canonical_handle_pose, table_top_z_at_xy,
)


def test_observed_plane_preserves_tilt_and_thickness_not_center_z():
    x, y = np.meshgrid(np.linspace(-.3, .3, 35), np.linspace(-.2, .2, 30))
    points = np.column_stack((x.ravel(), y.ravel(), (.10*x-.06*y+.4).ravel()))
    result = fitted_table_pose(points, [0, 0, 1.2], [0, 0], [.6, .4], .02)
    pose = np.array(result["table_pose_world"])
    top = pose[:3, 3]+pose[:3, 2]*.01
    assert top[2] == pytest.approx(.4)
    np.testing.assert_allclose((points-top)@pose[:3, 2], 0, atol=1e-10)
    assert table_top_z_at_xy(pose, .02, [.2, -.1]) == pytest.approx(.426)
    assert result["sdk_up_consistent"] is True


def test_inverted_sdk_axis_is_recorded_not_repaired_by_rotating_robot():
    x, y = np.meshgrid(np.linspace(-.2, .2, 20), np.linspace(-.2, .2, 20))
    points = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
    result = fitted_table_pose(points, [0, 0, -.8], [0, 0], [.4, .4], .02)
    assert result["sdk_up_consistent"] is False
    pose = np.array(result["table_pose_world"])
    np.testing.assert_allclose(pose[:3, 2], [0, 0, -1], atol=1e-10)
    np.testing.assert_allclose(pose[:3, 3], [0, 0, .01], atol=1e-10)
    with pytest.raises(ValueError, match="camera_sdk_axis_inconsistent"):
        table_top_z_at_xy(pose, .02, [0, 0])


def test_equivalent_halfturn_resolves_cabinet_without_moving_handle_center():
    vertices = np.array([[x, y, z] for x in [-.01,.01] for y in [-.07,.07] for z in [-.02,.01]])
    center = vertices.mean(0)
    turn, residual = handle_halfturn(vertices, center)
    assert residual < 1e-10
    # 柜体 Z 沿原把手 Y；原分支使柜体朝下，几何相同的另一分支直立。
    alignment = np.eye(4)
    alignment[:3, :3] = Rotation.from_euler("x", -90, degrees=True).as_matrix()
    raw = np.eye(4); raw[:3, :3] = Rotation.from_euler("x", 90, degrees=True).as_matrix()@turn[:3,:3]
    raw[:3, 3] = [.3, .2, .4]
    config = dict(T_mesh_equivalent_halfturn=turn.tolist(), T_mesh_asset=alignment.tolist(), observed_up_sdk=[0,0,1])
    chosen, report = canonical_handle_pose(raw, config)
    assert report["selected_equivalent_branch"] == 1
    assert (chosen@alignment)[2, 2] > .999
    np.testing.assert_allclose(chosen[:3,:3]@center+chosen[:3,3],raw[:3,:3]@center+raw[:3,3],atol=1e-12)
    stable, second = canonical_handle_pose(chosen, config)
    assert second["selected_equivalent_branch"] == 0
    np.testing.assert_allclose(stable, chosen)


def test_non_symmetric_handle_is_not_rotated():
    with pytest.raises(ValueError, match="不满足"):
        handle_halfturn([[0,0,0],[.1,0,0],[0,.3,.2]], [0,0,0])
