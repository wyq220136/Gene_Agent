import json

import numpy as np

from gene_agent.visualization.curobo_failure import (
    FORK_PROGRESS_COLORS, TRAJECTORY_VIEW_ANGLES, _fork_progress_colors,
    _fork_segments, _path_sample_indices, _trajectory_view_path,
    export_failure_bundle, read_colored_ply, sample_scene_points, write_colored_ply,
)


def test_trajectory_waypoint_sampling_is_sparse_and_keeps_endpoints():
    points = np.column_stack((np.linspace(0.0, 0.5, 101), np.zeros((101, 2))))
    indices = _path_sample_indices(points, maximum=6, minimum_spacing_m=0.055)
    assert indices[0] == 0
    assert indices[-1] == 100
    assert len(indices) <= 6
    assert indices == sorted(set(indices))


def test_tuning_fork_has_handle_bridge_and_two_parallel_tines():
    pose = np.eye(4)
    pose[:3, 3] = [0.3, -0.2, 0.1]
    segments = _fork_segments(pose)
    assert len(segments) == 4
    handle, bridge, positive, negative = segments
    np.testing.assert_allclose(handle[0], [0.3, -0.2, 0.048])
    np.testing.assert_allclose(bridge[0], [0.272, -0.2, 0.088])
    np.testing.assert_allclose(bridge[1], [0.328, -0.2, 0.088])
    assert handle[0][2] < handle[1][2]  # 手柄沿 TCP -Z 指向腕部/末端关节。
    np.testing.assert_allclose(positive[1] - positive[0], negative[1] - negative[0])


def test_tuning_fork_progress_colors_follow_rainbow_order():
    colors = _fork_progress_colors(7)
    assert [item[0] for item in colors] == [item[0] for item in FORK_PROGRESS_COLORS]
    for (_, expected), (_, rgb, rgba) in zip(FORK_PROGRESS_COLORS, colors):
        np.testing.assert_array_equal(rgb, expected)
        np.testing.assert_allclose(rgba[:3], np.asarray(expected) / 255.0)
        assert rgba[3] == np.float32(0.92)


def test_tuning_fork_progress_colors_span_start_to_end_for_sparse_paths():
    colors = _fork_progress_colors(3)
    assert [item[0] for item in colors] == ["red", "green", "purple"]


def test_trajectory_views_are_standalone_and_cover_multiple_directions(tmp_path):
    labels = [item[0] for item in TRAJECTORY_VIEW_ANGLES]
    angles = [(item[1], item[2]) for item in TRAJECTORY_VIEW_ANGLES]
    assert labels == [
        "task", "front", "front_high", "oblique_left",
        "oblique_left_high", "left", "top",
    ]
    assert len(set(angles)) == len(angles)
    assert _trajectory_view_path(tmp_path / "trajectory.png", "task") == (
        tmp_path / "trajectory_task.png"
    )
    assert _trajectory_view_path(tmp_path / "trajectory.png", "oblique_left") == (
        tmp_path / "trajectory_oblique_left.png"
    )


def test_colored_ply_can_store_triangle_mesh(tmp_path):
    path = tmp_path / "mesh.ply"
    points = np.asarray([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
    colors = np.asarray([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8)
    write_colored_ply(path, points, colors, np.asarray([[0, 1, 2]]))
    text = path.read_text(encoding="ascii")
    assert "element vertex 3" in text
    assert "element face 1" in text
    assert text.rstrip().endswith("3 0 1 2")
    loaded_points, loaded_colors = read_colored_ply(path)
    np.testing.assert_allclose(loaded_points, points)
    np.testing.assert_array_equal(loaded_colors, colors)


def test_failure_bundle_keeps_each_trajectory_separate(tmp_path):
    scene = {
        "cuboid": {
            "table": {
                "dims": [0.8, 0.8, 0.04],
                "pose": [0.4, 0.2, -0.02, 1.0, 0.0, 0.0, 0.0],
            }
        },
        "mesh": {},
    }
    traces = []
    for index in range(2):
        points = np.array([[0.0, index * 0.1, 0.2], [0.2, index * 0.1, 0.3]])
        traces.append({
            "trace_id": f"seed_{index}",
            "source": "test",
            "synthetic": False,
            "joint_positions_rad": [[0.0] * 7, [0.1] * 7],
            "tcp_path_world_m": points.tolist(),
            "robot_sphere_centers_world_m": points.tolist(),
            "first_invalid_waypoint": None,
            "collision_check": None,
        })
    target = np.eye(4)
    target[:3, 3] = [0.25, 0.05, 0.3]
    result = export_failure_bundle(
        tmp_path / "bundle", scene_model=scene, target_pose_world=target,
        traces=traces, metadata={"failure_category": "path_not_found"},
        optimizer_history={"iteration_0000": np.zeros((2, 4, 7))},
    )
    assert result["trace_count"] == 2
    manifest = json.loads((tmp_path / "bundle/manifest.json").read_text())
    assert len(manifest["traces"]) == 2
    for index in range(2):
        root = tmp_path / f"bundle/trajectories/seed_{index}"
        assert (root / "trajectory.png").is_file()
        assert (root / "scene_and_trajectory.ply").is_file()
        assert (root / "trajectory.json").is_file()
    assert not (tmp_path / "bundle/all_trajectories_overview.png").exists()
    assert not (tmp_path / "bundle/scene_and_all_trajectories.ply").exists()


def test_scene_cloud_uses_planner_base_to_world_transform():
    scene = {"cuboid": {"box": {
        "dims": [.2, .3, .4],
        "pose": [0., 0., 0., 1., 0., 0., 0.],
    }}, "mesh": {}}
    base_points, _ = sample_scene_points(scene)
    transform = np.eye(4)
    transform[:3, 3] = [1.2, -.4, .7]
    world_points, _ = sample_scene_points(scene, world_from_scene=transform)
    np.testing.assert_allclose(
        world_points - base_points,
        np.repeat(transform[None, :3, 3], len(base_points), axis=0),
        atol=1e-12,
    )


def test_colliding_curobo_obstacle_is_red_in_per_trace_point_cloud(tmp_path):
    scene = {"cuboid": {"microwave": {
        "dims": [.3, .2, .4],
        "pose": [.2, .1, .2, 1., 0., 0., 0.],
    }}, "mesh": {}}
    target = np.eye(4)
    trace = {
        "trace_id": "collision",
        "source": "test",
        "synthetic": False,
        "joint_positions_rad": [[0.] * 7],
        "tcp_path_world_m": [[0., 0., .2]],
        "robot_sphere_centers_world_m": [[0., 0., .2]],
        "first_invalid_waypoint": 0,
        "collision_check": {
            "scene_collision": True,
            "self_collision": False,
            "joint_limit_violation": False,
            "colliding_obstacles": ["microwave"],
        },
    }
    export_failure_bundle(
        tmp_path / "bundle", scene_model=scene, target_pose_world=target,
        traces=(trace,), metadata={"failure_category": "path_collision"},
    )
    _, colors = read_colored_ply(
        tmp_path / "bundle/trajectories/collision/scene_and_trajectory.ply",
    )
    assert np.any(np.all(colors == [255, 8, 8], axis=1))
