"""旧 Viser 的微波炉固定 keypose 数值迁移检查。"""

import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
import pytest

from gene_agent.scene.microwave_frame import estimate_legacy_microwave_frame
from gene_agent.tasks.policies.microwave_task.operations.bowl import (
    fixed_extract_targets,
    fixed_place_targets,
    load_object_operation_profile,
)
from gene_agent.tasks.policies.microwave_task.planning.grasp import (
    base_side_bowl_proposals,
    legacy_bowl_proposals,
)
from gene_agent.knowledge import bowl_grasp_candidates
from gene_agent.simulation import build_scene
from gene_agent.visualization.microwave_object_keypose_app import projected_keyposes


TASK_DIR = Path(__file__).resolve().parents[4] / "data/task_1"
SOURCE_SHA256 = "8a4f205b874b76e37d8f21f15c45b842fc82f2991d03583399be15c9d5a8d1c5"


def test_old_task_profile_is_copied_without_numeric_change():
    payload = (TASK_DIR / "knowledge/microwave_object_manipulation.json").read_bytes()
    assert hashlib.sha256(payload).hexdigest() == SOURCE_SHA256
    bowl = load_object_operation_profile(TASK_DIR, "Bowl_0")
    plate = load_object_operation_profile(TASK_DIR, "Plate_0")
    assert bowl["grasp_opening_width_m"] == pytest.approx(0.007)
    assert bowl["grasp_profile_name"] == "task_1_bowl_0"
    assert plate["grasp_profile_name"] == "task_1_plate_0"


def test_fixed_bowl_place_targets_keep_old_order_and_attachment():
    profile = load_object_operation_profile(TASK_DIR, "Bowl_0")
    # 这里传的是旧 profile 所属的微波炉 frame；当前校准 URDF 的 root
    # 不能直接代入。使用平移验证相对 pose 只组合一次。
    T_world_profile_microwave = np.eye(4)
    T_world_profile_microwave[:3, 3] = [0.4, -0.2, 0.3]
    T_tcp_object = np.eye(4)
    T_tcp_object[:3, 3] = [0.02, 0.0, 0.05]
    targets = fixed_place_targets(T_world_profile_microwave, profile, T_tcp_object)
    np.testing.assert_allclose(
        targets.pre_place_tcp,
        T_world_profile_microwave @ profile["microwave_from_pre_place_tcp"],
    )
    np.testing.assert_allclose(
        targets.place_tcp,
        T_world_profile_microwave @ profile["microwave_from_place_tcp"],
    )
    np.testing.assert_allclose(
        targets.post_place_1_tcp,
        T_world_profile_microwave @ profile["microwave_from_post_place_1_tcp"],
    )
    np.testing.assert_allclose(
        targets.post_place_2_tcp,
        T_world_profile_microwave @ profile["microwave_from_post_place_2_tcp"],
    )
    np.testing.assert_allclose(targets.placed_object, targets.place_tcp @ T_tcp_object)


def test_fixed_bowl_extract_targets_keep_five_legacy_waypoints():
    profile = load_object_operation_profile(TASK_DIR, "Bowl_0")
    world = np.eye(4)
    targets = fixed_extract_targets(world, profile)
    expected_names = (
        "microwave_from_pre_extract_1_tcp",
        "microwave_from_pre_extract_2_tcp",
        "microwave_from_extract_tcp",
        "microwave_from_post_extract_1_tcp",
        "microwave_from_post_extract_2_tcp",
    )
    for target, name in zip(targets, expected_names, strict=True):
        np.testing.assert_allclose(target, profile[name])


def test_uncalibrated_fixed_targets_fail_instead_of_falling_back():
    profile = dict(load_object_operation_profile(TASK_DIR, "Bowl_0"))
    profile["calibration"] = dict(profile["calibration"], place=False)
    with pytest.raises(ValueError, match="place"):
        fixed_place_targets(np.eye(4), profile, np.eye(4))


def test_old_door_and_handle_define_a_rigid_display_alignment(tmp_path):
    model, data, _ = build_scene(tmp_path / "scene")
    mujoco.mj_forward(model, data)
    alignment = estimate_legacy_microwave_frame(model, data, TASK_DIR)
    expected_rotation = np.array([[-1., 0., 0.], [0., 0., 1.], [0., 1., 0.]])
    np.testing.assert_allclose(alignment.T_calibrated_legacy[:3, :3], expected_rotation)
    assert alignment.handle_center_residual_m < 0.006
    assert alignment.door_size_max_residual_m < 0.015
    # 这只是门/把手几何的叠图初值，不把它误标为当前资产已标定的动作目标。
    assert alignment.accepted_for_planning is False


def test_projected_targets_follow_edited_microwave_without_changing_relative_pose(tmp_path):
    model, data, _ = build_scene(tmp_path / "scene")
    mujoco.mj_forward(model, data)
    alignment = estimate_legacy_microwave_frame(model, data, TASK_DIR)
    profile = load_object_operation_profile(TASK_DIR, "Bowl_0")
    original = np.eye(4)
    original[:3, :3] = data.body("microwave").xmat.reshape(3, 3)
    original[:3, 3] = data.body("microwave").xpos
    translated = original.copy()
    translated[:3, 3] += [0.03, -0.02, 0.01]
    first, first_bowl = projected_keyposes(original, alignment, profile)
    second, second_bowl = projected_keyposes(translated, alignment, profile)
    assert tuple(first) == tuple(second)
    for name in first:
        np.testing.assert_allclose(second[name][:3, 3] - first[name][:3, 3],
                                   [0.03, -0.02, 0.01], atol=1e-9)
    np.testing.assert_allclose(second_bowl[:3, 3] - first_bowl[:3, 3],
                               [0.03, -0.02, 0.01], atol=1e-9)


def test_bowl_grasp_order_and_all_poses_match_old_generator_at_same_object_pose():
    # 此 pose 固定为旧/新候选对照时的相同输入。哈希由旧版
    # build_object_ring_grasps 产生的 40 个有序 4x4 pose 计算。
    bowl_pose = np.array([
        [.3603849916451323, 0., -.9328036544723322, .3],
        [-.9328036544723322, 0., -.3603849916451323, .2],
        [0., 1., 0., -.078],
        [0., 0., 0., 1.],
    ])
    candidates = bowl_grasp_candidates(bowl_pose, pose_source="test_gt")
    proposals = legacy_bowl_proposals(candidates, object_pose_world=bowl_pose)
    assert len(proposals) == 40
    assert [item.candidate_id for item in proposals[:5]] == [
        f"bowl_ring_02_{index:02d}" for index in range(5)
    ]
    poses = np.asarray([item.tcp_pose_world for item in proposals])
    rounded = np.round(poses, 9)
    # 新旧矩阵只在 1e-16 量级和 IEEE 正/负零编码上有差异；哈希前
    # 统一为 1 nm/约 1e-9 旋转矩阵元素，保证比较的是动作数值。
    rounded[rounded == 0] = 0.0
    digest = hashlib.sha256(rounded.astype("<f8").tobytes()).hexdigest()
    assert digest == "9d3496c39fc1683c1bae9a908ffb330856023835e291747ee9e10b8a818c6030"


def test_bowl_rim_grasps_shift_outward_and_keep_palm_away_from_bowl():
    bowl_pose = np.eye(4)
    concept = json.loads((TASK_DIR.parents[1] / "assets/objects/bowl_concept.json").read_text())
    concept_pose = np.asarray(concept["conceptualization"][0]["parameters"]["pose_matrix"])
    candidates = bowl_grasp_candidates(
        bowl_pose,
        parameters={
            "radial_offset_m": 0.006,
            "tip_y_rad": np.deg2rad([-45.0, -55.0, -35.0]).tolist(),
        },
    )

    assert len(candidates) == 8 * 3
    for candidate in candidates:
        angle = candidate["parameters"]["azimuth_rad"]
        radial = concept_pose[:3, :3] @ np.array([np.cos(angle), np.sin(angle), 0.0])
        tcp_object = np.asarray(candidate["T_object_tcp"])
        baseline = bowl_grasp_candidates(bowl_pose, parameters={
            "azimuths_rad": [angle], "tip_y_rad": [candidate["parameters"]["tip_y_rad_value"]],
        })[0]
        baseline_object = np.asarray(baseline["T_object_tcp"])
        assert np.dot(tcp_object[:3, 3] - baseline_object[:3, 3], radial) == pytest.approx(.006)
        assert np.dot(tcp_object[:3, 2], radial) < -.3


def test_bowl_tilt_rotates_around_pad_contact_center_instead_of_tcp_tip():
    pivot = np.array([0., 0., -.010451])
    parameters = {'azimuths_rad': [.7], 'tip_y_rad': [0., -.7853981634],
                  'radial_offset_m': .006, 'tilt_pivot_tcp_m': pivot.tolist()}
    upright, tilted = bowl_grasp_candidates(np.eye(4), parameters=parameters)
    poses = [np.asarray(c['T_world_tcp']) for c in (upright, tilted)]
    centers = [p[:3, :3] @ pivot + p[:3, 3] for p in poses]
    np.testing.assert_allclose(centers[0], centers[1], atol=1e-12)
    assert not np.allclose(poses[0][:3, 3], poses[1][:3, 3])


def test_pad_pivot_compensation_does_not_reorder_same_rim_tilt_preferences():
    from scipy.spatial.transform import Rotation
    bowl = np.eye(4)
    bowl[:3, :3] = Rotation.from_euler('xyz', [90., .02, -68.], degrees=True).as_matrix()
    candidates = bowl_grasp_candidates(bowl, parameters={
        'tip_y_rad': np.deg2rad([-45., -55., -35.]).tolist(),
        'radial_offset_m': .006, 'tilt_pivot_tcp_m': [0, 0, -.010451]})
    proposals = legacy_bowl_proposals(candidates, object_pose_world=bowl)
    for index in range(0, len(proposals), 3):
        group = proposals[index:index+3]
        assert len({p.parameters['azimuth_rad'] for p in group}) == 1
        np.testing.assert_allclose([p.parameters['tip_y_rad_value'] for p in group],
                                   np.deg2rad([-45., -55., -35.]))


def test_base_side_order_prefers_right_arm_side_external_rim_and_55_degree_table_angle():
    bowl = np.eye(4)
    bowl[:3, :3] = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
    concept = json.loads((TASK_DIR.parents[1] / "assets/objects/bowl_concept.json").read_text())
    ring = np.asarray(concept["conceptualization"][0]["parameters"]["pose_matrix"])
    base = ring[:3, 3] + ring[:3, :3] @ np.array([1.0, 0.0, 0.0])
    candidates = bowl_grasp_candidates(bowl, parameters={
        "tip_y_rad": np.deg2rad([-35., -40., -30., -45., -25., -21.]).tolist(),
        "max_abs_tip_y_rad": np.deg2rad(50.),
        "radial_offset_m": 0.,
        "tilt_pivot_tcp_m": [0., 0., -.010451],
    })

    proposals = base_side_bowl_proposals(
        candidates, object_pose_world=bowl, base_position_world=base,
    )

    assert proposals[0].candidate_id == "bowl_ring_00_00"
    assert proposals[0].parameters["tip_y_rad_value"] == pytest.approx(np.deg2rad(-35.))
    assert all(45.0 - 1e-9 <= np.rad2deg(np.arcsin(abs(item.tcp_pose_world[2, 2])))
               <= 70.0 + 1e-9
               for item in proposals)


def test_base_side_order_rejects_opposite_half_horizontal_and_oververtical_angles():
    bowl = np.eye(4)
    bowl[:3, :3] = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
    concept = json.loads((TASK_DIR.parents[1] / "assets/objects/bowl_concept.json").read_text())
    ring = np.asarray(concept["conceptualization"][0]["parameters"]["pose_matrix"])
    base = ring[:3, 3] + ring[:3, :3] @ np.array([1.0, 0.0, 0.0])
    candidates = bowl_grasp_candidates(bowl, parameters={
        "tip_y_rad": np.deg2rad([-75., -45., -35., -21., -10.]).tolist(),
        "max_abs_tip_y_rad": np.deg2rad(80.),
        "radial_offset_m": 0.,
        "tilt_pivot_tcp_m": [0., 0., -.010451],
    })

    proposals = base_side_bowl_proposals(
        candidates, object_pose_world=bowl, base_position_world=base,
    )

    angles = [np.rad2deg(np.arcsin(abs(item.tcp_pose_world[2, 2]))) for item in proposals]
    assert min(angles) >= 45.0 - 1e-9
    assert max(angles) <= 70.0 + 1e-9
    assert {round(np.rad2deg(item.parameters["azimuth_rad"])) % 360
            for item in proposals}.isdisjoint({135, 180, 225})


def test_base_side_order_rejects_contact_pivot_outside_rim():
    bowl = np.eye(4)
    bowl[:3, :3] = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
    concept = json.loads((TASK_DIR.parents[1] / "assets/objects/bowl_concept.json").read_text())
    ring = np.asarray(concept["conceptualization"][0]["parameters"]["pose_matrix"])
    base = ring[:3, 3] + ring[:3, :3] @ np.array([1.0, 0.0, 0.0])
    candidates = bowl_grasp_candidates(bowl, parameters={
        "tip_y_rad": [np.deg2rad(-35.)],
        "radial_offset_m": .006,
        "tilt_pivot_tcp_m": [0., 0., -.010451],
    })

    with pytest.raises(ValueError, match="外侧碗沿"):
        base_side_bowl_proposals(
            candidates, object_pose_world=bowl, base_position_world=base,
        )


def test_base_side_order_rejects_approach_from_bowl_inside():
    bowl = np.eye(4)
    bowl[:3, :3] = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
    concept = json.loads((TASK_DIR.parents[1] / "assets/objects/bowl_concept.json").read_text())
    ring = np.asarray(concept["conceptualization"][0]["parameters"]["pose_matrix"])
    base = (bowl @ ring)[:3, 3] + (bowl @ ring)[:3, :3] @ np.array([1., 0., 0.])
    candidate = bowl_grasp_candidates(bowl, parameters={
        "azimuths_rad": [0.],
        "tip_y_rad": [np.deg2rad(-35.)],
        "radial_offset_m": 0.,
    })[0]
    inside = {**candidate, "T_world_tcp": np.asarray(candidate["T_world_tcp"]).copy()}
    # 同时翻转 TCP +X/+Z 保持右手坐标系和接触点不变，只把接近方向改成从碗内向外。
    inside["T_world_tcp"][:3, (0, 2)] *= -1.
    inside["T_world_tcp"] = inside["T_world_tcp"].tolist()

    with pytest.raises(ValueError, match="外侧碗沿"):
        base_side_bowl_proposals(
            [inside], object_pose_world=bowl, base_position_world=base,
        )
