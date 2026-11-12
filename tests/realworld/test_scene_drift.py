"""真机发令前，重新观测的障碍状态必须仍属于本次规划条件。"""

from copy import deepcopy

import numpy as np

from gene_agent.realworld.scene_drift import SceneDriftLimits, compare_scene_drift


def _scene(frame):
    pose = np.eye(4).tolist()
    return {
        "snapshot": {
            "active_group": "right", "world_frame": "world",
            "snapshot_sha256": f"hash-{frame}",
            "records": [{"geom_name": "bowl_bottom", "body_name": "bowl",
                         "export_kind": "mesh", "mesh_sha256": "shape-v1",
                         "geom_size": [.1, .1, .1], "contact_margin_m": 0.,
                         "contype": 1, "conaffinity": 1}],
        },
        "provenance": {
            "physics_stepped": False, "pose_frame_verified": True,
            "collision_snapshot_sha256": f"hash-{frame}",
            "source_frame_number": frame, "source_timestamp_ms": 1000. + frame,
            "source_camera_id": "0", "scene_config_sha256": "scene-v1",
            "aligned_object_poses_world": {
                "Bowl/Bowl_0": pose, "Microwave/Microwave_0": pose},
            "table_pose_world": pose, "table_size_xy_m": [1., .7],
            "accepted_table_thickness_m": .04,
            "accepted_support_size_xyz_m": [.4, .3, .2],
            "accepted_support_center_xy_local_m": [0., 0.],
            "joint_positions_by_group_rad": {"left": [0.] * 7,
                                              "right": [0.] * 7},
            "gripper_widths_m": {"left": .08, "right": .04},
            "door_angle_rad": .3, "knob_angle_rad": .1,
        },
    }


LIMITS = SceneDriftLimits(
    max_object_translation_m=.01, max_object_rotation_rad=.05,
    max_robot_joint_error_rad=.02, max_gripper_width_error_m=.005,
    max_articulated_angle_error_rad=.03,
)


def test_new_frame_with_small_pose_noise_keeps_plan_valid():
    """微小感知抖动不应让同一可执行场景总是被 hash 差异拒绝。"""
    planned, current = _scene(10), _scene(11)
    current["provenance"]["aligned_object_poses_world"]["Bowl/Bowl_0"][0][3] = .003
    result = compare_scene_drift(planned, current, limits=LIMITS)
    assert result.status == "pass"
    assert result.evidence["object_translation_m"]["Bowl/Bowl_0"] == .003


def test_moved_bowl_rejects_plan_even_if_active_arm_is_unchanged():
    """规划后碗移动，活动臂起点检查单独无法发现。"""
    planned, current = _scene(10), _scene(11)
    current["provenance"]["aligned_object_poses_world"]["Bowl/Bowl_0"][0][3] = .04
    result = compare_scene_drift(planned, current, limits=LIMITS)
    assert result.status == "fail"
    assert "Bowl/Bowl_0" in result.reason


def test_other_arm_motion_rejects_plan():
    """冻结在碰撞中的另一臂改变时，不能仍按旧障碍发令。"""
    planned, current = _scene(10), _scene(11)
    current["provenance"]["joint_positions_by_group_rad"]["left"][2] = .2
    result = compare_scene_drift(planned, current, limits=LIMITS)
    assert result.status == "fail"
    assert "left" in result.reason


def test_same_camera_frame_cannot_be_called_reobservation():
    """即便状态 hash 一样，缓存帧也不能证明发令前场景没变。"""
    planned = _scene(10)
    current = deepcopy(planned)
    result = compare_scene_drift(planned, current, limits=LIMITS)
    assert result.status == "fail"
    assert "新相机帧" in result.reason


def test_changed_collision_mesh_rejects_plan_without_pose_change():
    """碰撞资产变了时，位姿容差不能掩盖代理几何变化。"""
    planned, current = _scene(10), _scene(11)
    current["snapshot"]["records"][0]["mesh_sha256"] = "shape-v2"
    result = compare_scene_drift(planned, current, limits=LIMITS)
    assert result.status == "fail"
    assert "碰撞资产" in result.reason


def test_raw_door_angle_change_is_not_hidden_by_planning_grid():
    """两次规划网格角相同，现场实测门角不同仍必须发现。"""
    planned, current = _scene(10), _scene(11)
    planned["provenance"]["live_state"] = {
        "door": {"angle_rad": .3, "measured_angle_rad": .31},
        "knob": {"angle_rad": .1, "measured_angle_rad": .1},
    }
    current["provenance"]["live_state"] = {
        "door": {"angle_rad": .3, "measured_angle_rad": .37},
        "knob": {"angle_rad": .1, "measured_angle_rad": .1},
    }
    result = compare_scene_drift(planned, current, limits=LIMITS)
    assert result.status == "fail"
    assert result.evidence["articulated_angle_error_rad"]["door_angle_rad"] > .05
