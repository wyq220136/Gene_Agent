"""Task1 把手/门面候选集合与 pre-contact 的公共契约。"""

import hashlib
import json
import numpy as np
from scipy.spatial.transform import Rotation

from gene_agent.knowledge.operation import (
    AxisSpec,
    generate_handle_pull_candidates,
    generate_handle_pull_path,
    generate_mode_door_push_candidate_sets,
    make_door_push_precontact,
    select_door_push_path,
)
from gene_agent.knowledge.operation.door_push import (
    generate_profile_door_push_candidate_sets,
)


def test_explicit_push_profile_keeps_verified_waypoints_and_actual_start_angle():
    """物理推门可复用已验证逐角参数，首点跟随实测门角。"""
    panel = np.eye(4)
    panel[:3, 3] = [.4, .3, .2]
    axis = AxisSpec([.4, .48, .2], [0, 0, -1], "world", "test:door_hinge")
    profile = (
        {"angle_deg": 20., "hinge_fraction": 1., "top_down_distance_m": .03,
         "standoff_m": 0., "tip_x_deg": -30., "tip_y_deg": -90.,
         "panel_tilt_degrees": 0.},
        {"angle_deg": 30., "hinge_fraction": 1., "top_down_distance_m": .03,
         "standoff_m": 0., "tip_x_deg": -60., "tip_y_deg": -90.,
         "panel_tilt_degrees": 0.},
    )

    sets = generate_profile_door_push_candidate_sets(
        panel, panel_width_m=.36, panel_height_m=.22,
        panel_thickness_m=.016, axis=axis, mode="right_open",
        start_angle_rad=np.deg2rad(19.34),
        target_angle_rad=np.deg2rad(30.),
        surface="inside", waypoints=profile,
        source="test:verified_profile",
    )

    np.testing.assert_allclose(np.rad2deg([item.door_angle_rad for item in sets]),
                               [19.34, 30.])
    assert [len(item.candidates) for item in sets] == [1, 1]
    assert [item.candidates[0].parameters["tip_y_deg"] for item in sets] == [-90., -90.]
    assert sets[0].candidates[0].parameters["tip_x_deg"] == -30.
    assert sets[1].candidates[0].parameters["tip_x_deg"] == -60.
    assert all(item.candidates[0].source == "test:verified_profile" for item in sets)


def test_task1_handle_pull_candidates_keep_order_parameters_and_position() -> None:
    base = np.eye(4)
    base[:3, 3] = [.31, .22, .27]
    toward_panel = np.array([.35, -.25, .9])
    candidates = generate_handle_pull_candidates(base, toward_panel)

    assert candidates[0].candidate_id == "handle_pull:candidate_0000"
    assert candidates[0].parameters == {
        "local_x_rotation_deg": 0.,
        "local_y_rotation_deg": 0.,
    }
    assert [item.parameters for item in candidates[:5]] == [
        {"local_x_rotation_deg": 0., "local_y_rotation_deg": value}
        for value in (0., 5., -5., 10., -10.)
    ]
    assert len(candidates) < 49  # 朝门板更深的局部 X 旋转必须被剔除。
    for rank, candidate in enumerate(candidates):
        assert candidate.preference_rank == rank
        np.testing.assert_allclose(candidate.grasp_tcp_pose_world[:3, 3], base[:3, 3])
        x_deg = candidate.parameters["local_x_rotation_deg"]
        y_deg = candidate.parameters["local_y_rotation_deg"]
        y_pose = base[:3, :3] @ Rotation.from_euler("y", y_deg, degrees=True).as_matrix()
        if x_deg:
            assert candidate.grasp_tcp_pose_world[:3, 2] @ (toward_panel / np.linalg.norm(toward_panel)) <= (
                y_pose[:, 2] @ (toward_panel / np.linalg.norm(toward_panel)) + 1e-9
            )


def test_task1_handle_pull_path_uses_three_equal_steps_and_local_x_compensation() -> None:
    """冻结 Task1 阶段一实际使用的 3 点门轴路径与指尖局部 X 补偿。"""
    grasp = np.eye(4)
    grasp[:3, 3] = [.34, .12, .26]
    axis = AxisSpec([.2, .3, .1], [0., 0., -1.], "world", "test:door_hinge")

    path, labels, widths = generate_handle_pull_path(
        grasp,
        axis,
        start_angle_rad=0.,
        target_angle_rad=np.deg2rad(25.),
        rotation_steps=3,
        tip_rotation_x_scale=1.,
        pregrasp_distance_m=.03,
        retreat_distance_m=.015,
        open_width_m=.08,
        close_width_m=.015,
    )

    assert labels == (
        "pregrasp", "grasp", "close", "pull_08", "pull_17", "pull_25",
        "release", "retreat",
    )
    assert len(path.waypoints) == len(widths) == 8
    expected_angles = np.deg2rad([25. / 3., 50. / 3., 25.])
    np.testing.assert_allclose(
        [point.angle_rad for point in path.waypoints[3:6]],
        expected_angles,
    )
    for point, angle in zip(path.waypoints[3:6], expected_angles, strict=True):
        world_rotation = Rotation.from_rotvec(axis.direction * angle).as_matrix()
        expected = grasp.copy()
        expected[:3, :3] = (
            world_rotation
            @ grasp[:3, :3]
            @ Rotation.from_euler("x", -angle).as_matrix()
        )
        expected[:3, 3] = axis.point + world_rotation @ (grasp[:3, 3] - axis.point)
        np.testing.assert_allclose(point.tcp_pose_world, expected, atol=1e-12)


def _candidate_digest(value) -> str:
    return hashlib.sha256(json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()).hexdigest()


def test_complete_task1_candidate_collections_keep_verified_order_and_geometry() -> None:
    """全集合快照防止中心角、关门 10° 偏好终点或后续顺序静默漂移。"""
    base = np.eye(4)
    base[:3, 3] = [.31, .22, .27]
    handle_candidates = generate_handle_pull_candidates(
        base,
        np.array([.35, -.25, .9]),
    )
    handle_payload = [
        {
            "pose": np.round(item.grasp_tcp_pose_world, 12).tolist(),
            "parameters": item.parameters,
        }
        for item in handle_candidates
    ]
    assert _candidate_digest(handle_payload) == (
        "447a22797b35534af5dc398f86f4980eae400bbcbd6b1bf4cf71a5e1863a4d61"
    )

    expected = {
        ("right_open", 30, 85): "2fdfe6308fd6f9e79428ecdb423fd29e76772acd651271eed2196e58ef199516",
        ("right_close", 85, 0): "7b15a23cea2fb1ad5b7afa140576d4994cab2c6583b4c7d65f0c4a46150d5a1c",
    }
    for arguments, digest in expected.items():
        candidate_sets = _candidate_sets(
            mode=arguments[0],
            start_deg=arguments[1],
            target_deg=arguments[2],
        )
        payload = [
            {
                "angle": round(candidate_set.door_angle_rad, 12),
                "candidates": [
                    {
                        "pose": np.round(item.tcp_pose_world, 12).tolist(),
                        "normal": np.round(item.surface_normal_world, 12).tolist(),
                        "parameters": item.parameters,
                    }
                    for item in candidate_set.candidates
                ],
            }
            for candidate_set in candidate_sets
        ]
        assert _candidate_digest(payload) == digest


def _candidate_sets(*, mode: str, start_deg: float, target_deg: float):
    panel = np.eye(4)
    panel[:3, 3] = [.4, .3, .2]
    axis = AxisSpec([.4, .48, .2], [0, 0, -1], "world", "test:door_hinge")
    return generate_mode_door_push_candidate_sets(
        panel,
        panel_width_m=.36,
        panel_height_m=.22,
        panel_thickness_m=.016,
        axis=axis,
        mode=mode,
        start_angle_rad=np.deg2rad(start_deg),
        target_angle_rad=np.deg2rad(target_deg),
        step_rad=np.deg2rad(20),
        reference_origin_world=np.zeros(3),
    )


def test_task1_right_open_returns_ordered_candidates_with_stable_ids():
    """把生成器退化成每角单 pose 时，本测试必须失败。"""
    candidate_sets = _candidate_sets(mode="right_open", start_deg=30, target_deg=85)

    np.testing.assert_allclose(
        np.rad2deg([item.door_angle_rad for item in candidate_sets]),
        [30, 50, 70, 85],
    )
    first = candidate_sets[0]
    assert len(first.candidates) > 100
    assert first.candidates[0].candidate_id == "right_open:angle_030.000:candidate_0000"
    assert first.candidates[1].candidate_id == "right_open:angle_030.000:candidate_0001"
    assert [item.parameters["hinge_fraction"] for item in first.candidates[:3]] == [
        .75,
        .875,
        1.,
    ]
    assert len({item.parameters["tip_x_deg"] for item in first.candidates[:3]}) == 1
    assert first.candidates[3].parameters["hinge_fraction"] == .75
    assert first.candidates[3].parameters["tip_x_deg"] != first.candidates[0].parameters["tip_x_deg"]
    assert all(item.parameters["surface"] == "inside" for item in first.candidates)
    assert [item.preference_rank for item in first.candidates[:4]] == [0, 1, 2, 3]


def test_precontact_uses_selected_candidate_surface_normal_and_keeps_orientation():
    """pre-contact 若仍沿 TCP 局部 Y 或固定取首候选，本测试必须失败。"""
    candidate_set = _candidate_sets(
        mode="right_open", start_deg=30, target_deg=50
    )[0]
    selected = candidate_set.candidates[1]
    precontact = make_door_push_precontact(selected, distance_m=.05)

    expected_inside_normal = np.array([.5, np.sqrt(3.) / 2., 0.])
    np.testing.assert_allclose(selected.surface_normal_world, expected_inside_normal, atol=1e-9)
    np.testing.assert_allclose(
        precontact[:3, 3] - selected.tcp_pose_world[:3, 3],
        .05 * expected_inside_normal,
        atol=1e-9,
    )
    np.testing.assert_allclose(precontact[:3, :3], selected.tcp_pose_world[:3, :3])
    assert np.isclose(np.linalg.norm(precontact[:3, 3] - selected.tcp_pose_world[:3, 3]), .05)

    first_precontact = make_door_push_precontact(candidate_set.candidates[0], distance_m=.05)
    assert not np.allclose(first_precontact[:3, 3], precontact[:3, 3])


def test_right_close_uses_outside_normal_and_selection_preserves_candidate_ids():
    """关门必须从 outside 面生成，且显式候选选择要进入接触路径。"""
    candidate_sets = _candidate_sets(mode="right_close", start_deg=85, target_deg=5)
    selected_ids = tuple(item.candidates[1].candidate_id for item in candidate_sets)
    path, parameters = select_door_push_path(
        candidate_sets,
        selected_candidate_ids=selected_ids,
    )

    assert len(path.waypoints) == len(candidate_sets) == 5
    np.testing.assert_allclose(path.axis_point_world, [.4, .48, .2])
    np.testing.assert_allclose(path.axis_direction_world, [0., 0., -1.])
    assert path.axis_source == "test:door_hinge"
    assert [item["candidate_id"] for item in parameters] == list(selected_ids)
    assert all(item["surface"] == "outside" for item in parameters)
    assert np.isclose(parameters[0]["standoff_m"], .0375)
    assert np.isclose(parameters[0]["top_down_distance_m"], .03125)
    for candidate_set, waypoint, parameter in zip(
        candidate_sets, path.waypoints, parameters, strict=True
    ):
        selected = candidate_set.candidates[1]
        np.testing.assert_allclose(waypoint.tcp_pose_world, selected.tcp_pose_world)
        np.testing.assert_allclose(parameter["surface_normal_world"], selected.surface_normal_world)
        assert np.isclose(np.linalg.norm(selected.surface_normal_world), 1.)


def test_candidate_geometry_is_equivariant_under_scene_pose_change():
    """移动/旋转微波炉时，候选和法向必须应用一次相同 world 变换。"""
    base = _candidate_sets(mode="right_open", start_deg=30, target_deg=50)[0].candidates[0]

    theta = np.deg2rad(37.)
    rotation = np.array([
        [np.cos(theta), -np.sin(theta), 0.],
        [np.sin(theta), np.cos(theta), 0.],
        [0., 0., 1.],
    ])
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = [.12, -.08, .04]
    panel = np.eye(4)
    panel[:3, 3] = [.4, .3, .2]
    moved_panel = transform @ panel
    moved_axis_point = transform[:3, :3] @ np.array([.4, .48, .2]) + transform[:3, 3]
    moved_reference = transform[:3, 3]
    moved = generate_mode_door_push_candidate_sets(
        moved_panel,
        panel_width_m=.36,
        panel_height_m=.22,
        panel_thickness_m=.016,
        axis=AxisSpec(moved_axis_point, rotation @ [0., 0., -1.], "world", "test:moved"),
        mode="right_open",
        start_angle_rad=np.deg2rad(30),
        target_angle_rad=np.deg2rad(50),
        step_rad=np.deg2rad(20),
        reference_origin_world=moved_reference,
    )[0].candidates[0]

    np.testing.assert_allclose(moved.tcp_pose_world, transform @ base.tcp_pose_world, atol=1e-9)
    np.testing.assert_allclose(moved.surface_normal_world, rotation @ base.surface_normal_world, atol=1e-9)
