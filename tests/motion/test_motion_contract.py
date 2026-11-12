"""不依赖 GPU 的具名关节输入和冻结碰撞配置回归。"""
import json
from pathlib import Path
import numpy as np
import pytest
import yaml
from gene_agent.motion.curobo import (
    ordered_positions,
    load_realman_motion_config,
    scene_collision_cache,
    empty_solver_result,
    summarize_ik_failure,
)
from gene_agent.robots import build_robot


def test_joint_order_is_named_and_generic():
    np.testing.assert_array_equal(ordered_positions(["b", "a", "c"], [2., 1., 3.], ["c", "a", "b"]), [3., 1., 2.])


@pytest.mark.parametrize("names,values", [(["a", "a"], [1, 2]), (["a"], [1]), (["a", "b"], [1, np.nan]), (["a", "b"], [[1, 2]])])
def test_invalid_joint_mapping_rejected(names, values):
    with pytest.raises(ValueError):
        ordered_positions(names, values, ["a", "b"])


def test_realman_config_preserves_collision_and_frames(tmp_path):
    metadata = build_robot(tmp_path)
    for group in metadata["joint_groups"]:
        loaded = load_realman_motion_config(group, metadata)
        source = yaml.safe_load((Path(__file__).resolve().parents[2] / f"assets/robots/realman/config/realman_{group}.yml").read_text())
        for key in ["collision_spheres", "collision_link_names", "self_collision_ignore", "self_collision_buffer"]:
            assert loaded["robot_config"]["kinematics"][key] == source["kinematics"][key]
        assert Path(loaded["robot_config"]["kinematics"]["urdf_path"]).is_file()
        assert "AffordToolAct" not in json.dumps(loaded)
        np.testing.assert_array_equal(loaded["world_from_base"], metadata["transforms"][group]["world_from_urdf_base"])


def test_scene_collision_cache_covers_every_obstacle_without_fixed_limit():
    scene = {
        "cuboid": {f"box_{index}": {} for index in range(156)},
        "mesh": {f"mesh_{index}": {} for index in range(12)},
    }

    assert scene_collision_cache(scene) == {"cuboid": 156, "mesh": 12}


def test_scene_collision_cache_keeps_empty_types_allocatable():
    assert scene_collision_cache({"cuboid": {}, "mesh": {}}) == {
        "cuboid": 1,
        "mesh": 1,
    }


def test_empty_curobo_result_is_an_explicit_no_solution_failure():
    result = empty_solver_result("plan")
    assert result == {
        "success": False,
        "status": "goal_ik_no_solution",
        "raw_result": "None",
        "raw_error": None,
        "message": "cuRobo did not find a successful IK seed",
        "failure_category": "goal_ik_no_solution",
    }


def test_ik_failure_summary_separates_pose_error_from_constraint_rejection():
    from types import SimpleNamespace
    raw = SimpleNamespace(
        success=np.array([[False, False, False]]),
        feasible=np.array([[True, False, False]]),
        position_error=np.array([[.030, .001, .020]]),
        rotation_error=np.array([[.010, .001, .020]]),
        solution=np.array([[[1., 0.], [2., 0.], [3., 0.]]]),
    )
    solver = SimpleNamespace(config=SimpleNamespace(
        position_tolerance=.005, orientation_tolerance=.005,
    ))

    diagnosis = summarize_ik_failure(raw, solver)

    assert diagnosis["seed_count"] == 3
    assert diagnosis["feasible_seed_count"] == 1
    assert diagnosis["pose_converged_seed_count"] == 1
    assert diagnosis["successful_seed_count"] == 0
    assert diagnosis["selected_seed_index"] == 0
    assert diagnosis["selected_seed_reason"] == "pose_not_converged"
    assert diagnosis["selected_joint_positions_rad"] == [1., 0.]
    assert diagnosis["seeds"][1]["pose_converged"] is True
    assert diagnosis["seeds"][1]["constraint_feasible"] is False
    assert diagnosis["seeds"][2]["joint_positions_rad"] == [3., 0.]
    assert diagnosis["collision_confirmed"] is False


def test_relaxed_ik_diagnostic_reports_observed_collision_without_claiming_execution():
    from gene_agent.motion.curobo import classify_world_obstruction
    result = classify_world_obstruction(
        True,
        {"scene_collision": True, "self_collision": False,
         "joint_limit_violation": False,
         "colliding_obstacles": ["bowl_wall"]},
    )
    assert result["reason"] == "scene_collision_at_relaxed_solution"
    assert result["blocking_obstacles"] == ["bowl_wall"]
    assert result["diagnostic_only"] is True
    assert classify_world_obstruction(False, None)["reason"] == "unresolved_without_world"
