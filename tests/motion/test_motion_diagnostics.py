import pytest

from gene_agent.feedback.motion_diagnostics import summarize_motion_failure
from gene_agent.motion.curobo import classify_failure_detail


def _push_close_failure():
    return {
        "success": False,
        "status": "goal_ik_no_solution",
        "failure_category": "goal_ik_no_solution",
        "message": "cuRobo did not find a successful IK seed",
        "start_state_check": {"valid": True},
        "ik_diagnostic": {
            "seed_count": 16,
            "successful_seed_count": 0,
            "feasible_seed_count": 0,
            "pose_converged_seed_count": 0,
            "position_tolerance_m": 0.005,
            "rotation_tolerance_rad": 0.05,
            "selected_seed_index": 6,
            "seeds": [{
                "index": 6,
                "position_error_m": 0.105401866,
                "rotation_error_rad": 0.105401948,
            }],
            "collision_checked": True,
            "collision_confirmed": False,
            "collision_check": {
                "joint_limit_violation": False,
                "self_collision": False,
                "scene_collision": False,
                "colliding_obstacles": [],
            },
            "world_obstruction_probe": {
                "world_collision_disabled_for_probe": True,
                "relaxed_ik_success": False,
                "reason": "unresolved_without_world",
                "blocking_obstacles": [],
            },
        },
        "target_selection_attempts": {
            "candidate_count": 24,
            "ik_success_count": 0,
            "planned_candidates": [],
        },
    }


def test_push_close_unreachable_summary_does_not_blame_collision():
    summary = summarize_motion_failure(_push_close_failure())
    payload = summary.to_dict()
    assert payload["kind"] == "goal_ik_not_converged"
    assert payload["start_state_valid"] is True
    assert payload["ik_seed_count"] == 16
    assert payload["ik_success_count"] == 0
    assert payload["local_candidate_count"] == 24
    assert payload["local_candidate_success_count"] == 0
    assert payload["best_position_error_m"] == pytest.approx(0.105401866)
    assert payload["best_rotation_error_rad"] == pytest.approx(0.105401948)
    assert payload["scene_collision"] is False
    assert payload["self_collision"] is False
    assert payload["world_disabled_still_failed"] is True
    assert payload["collision_groups"] == []
    assert "没有证据指向场景碰撞" in payload["conclusion"]
    markdown = summary.to_markdown()
    assert "105.4 mm" in markdown
    assert "0/16" in markdown
    assert "0/24" in markdown


def test_collision_summary_groups_raw_obstacles_by_scene_object():
    result = {
        "success": False,
        "failure_category": "start_collision",
        "message": "起始姿态碰撞",
        "start_state_check": {
            "valid": False,
            "collision_diagnostic": {
                "scene_collision": True,
                "self_collision": False,
                "joint_limit_violation": False,
                "colliding_obstacles": [
                    "mj_geom_0_scene_table",
                    "mj_geom_260_microwave_door_panel_collision",
                ],
            },
        },
    }
    snapshot = {
        "records": [
            {"obstacle_name": "mj_geom_0_scene_table", "body_name": "world",
             "geom_name": "scene_table"},
            {"obstacle_name": "mj_geom_260_microwave_door_panel_collision",
             "body_name": "microwave_door", "geom_name": "microwave_door_panel_collision"},
        ]
    }
    summary = summarize_motion_failure(result, snapshot)
    assert summary.kind == "start_scene_collision"
    assert summary.collision_groups == ("桌面", "微波炉门")
    assert "mj_geom_260_microwave_door_panel_collision" in summary.colliding_obstacles
    assert "桌面、微波炉门" in summary.to_markdown()


def test_relaxed_world_probe_reports_the_obstacle_that_blocks_goal_ik():
    result = _push_close_failure()
    result["ik_diagnostic"]["world_obstruction_probe"] = {
        "reason": "scene_collision_at_relaxed_solution",
        "world_collision_disabled_for_probe": True,
        "relaxed_ik_success": True,
        "blocking_obstacles": ["mj_geom_260_microwave_door_panel_collision"],
        "strict_collision_check": {
            "scene_collision": True,
            "self_collision": False,
            "joint_limit_violation": False,
            "colliding_obstacles": ["mj_geom_260_microwave_door_panel_collision"],
        },
    }
    snapshot = {"records": [{
        "obstacle_name": "mj_geom_260_microwave_door_panel_collision",
        "body_name": "microwave_door",
        "geom_name": "microwave_door_panel_collision",
    }]}
    summary = summarize_motion_failure(result, snapshot)
    assert summary.scene_collision is True
    assert summary.world_disabled_still_failed is False
    assert summary.collision_groups == ("微波炉门",)
    assert summary.collision_evidence_source == "world_relaxed_ik_solution"
    assert "关闭场景碰撞后找到 IK" in summary.conclusion
    assert "微波炉门" in summary.to_markdown()


def test_path_failure_without_collision_is_reported_as_search_failure():
    summary = summarize_motion_failure({
        "success": False,
        "failure_category": "path_not_found",
        "message": "目标 IK 已找到种子，但轨迹优化未返回可行路径",
        "start_state_check": {"valid": True},
        "solver_diagnostic": {
            "collision_confirmed": False,
            "collision_check": {"scene_collision": False, "self_collision": False,
                                "colliding_obstacles": []},
        },
    })
    assert summary.kind == "path_not_found"
    assert "未发现碰撞证据" in summary.conclusion


def test_failure_detail_separates_goal_collision_joint_limit_and_finite_search():
    scene = classify_failure_detail(
        "goal_ik",
        {"collision_check": {"scene_collision": False, "self_collision": False,
                             "joint_limit_violation": False}},
        world_obstruction_probe={
            "world_collision_disabled_for_probe": True,
            "relaxed_ik_success": True,
            "strict_collision_check": {
                "scene_collision": True, "self_collision": False,
                "joint_limit_violation": False,
                "colliding_obstacles": ["microwave_body_collision"],
            },
        },
    )
    assert scene["reason_code"] == "goal_scene_collision"
    assert scene["evidence_source"] == "world_relaxed_ik_solution"

    joint = classify_failure_detail("trajectory", {
        "first_invalid_waypoint": 17,
        "collision_check": {
            "scene_collision": False, "self_collision": False,
            "joint_limit_violation": True,
        },
    })
    assert joint["reason_code"] == "path_joint_limit_violation"
    assert joint["first_invalid_waypoint"] == 17
    assert "waypoint=17" in joint["human_summary"]

    finite = classify_failure_detail(
        "goal_ik", {"selected_seed_reason": "pose_not_converged"},
        world_obstruction_probe={
            "world_collision_disabled_for_probe": True,
            "relaxed_ik_success": False,
        },
    )
    assert finite["reason_code"] == "goal_pose_unreachable_within_search"
    assert finite["finite_search_not_global_proof"] is True
    assert "不构成全局不可达证明" in finite["human_summary"]

    far_colliding_seed = classify_failure_detail("goal_ik", {
        "pose_converged": False,
        "collision_check": {
            "scene_collision": False, "self_collision": True,
            "joint_limit_violation": False,
        },
    })
    assert far_colliding_seed["reason_code"] == "goal_pose_not_converged"
    assert far_colliding_seed["constraint_flags"] == []
    assert far_colliding_seed["observed_candidate_constraint_flags"] == [
        "self_collision"
    ]


def test_human_summary_uses_detailed_path_joint_limit_reason():
    summary = summarize_motion_failure({
        "success": False,
        "failure_category": "path_not_found",
        "message": "候选路径违反关节限位",
        "failure_detail": {
            "reason_code": "path_joint_limit_violation",
            "collision_check": {
                "scene_collision": False,
                "self_collision": False,
                "joint_limit_violation": True,
                "colliding_obstacles": [],
            },
        },
    })
    assert summary.kind == "path_joint_limit_violation"
    assert summary.detailed_reason_code == "path_joint_limit_violation"
    markdown = summary.to_markdown()
    assert "关节限位 违反" in markdown
    assert "`path_joint_limit_violation`" in markdown
