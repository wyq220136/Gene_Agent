import json

import numpy as np

from gene_agent.visualization.realworld_pick_debug import (
    load_pick_debug,
    pick_debug_summary,
)


def _pose(x):
    value = np.eye(4)
    value[0, 3] = x
    return value.tolist()


def test_loads_candidate_failures_clamps_and_saved_trajectory(tmp_path):
    root = tmp_path / "run"
    selection = root / "grasp/selection/selection.json"
    selection.parent.mkdir(parents=True)
    selection.write_text(json.dumps({
        "input_state_id": "scene-snapshot",
        "candidate_order": ["a", "b"],
        "candidate_parameters": {"candidate_targets": [
            {"candidate_id": "a", "T_world_tcp": _pose(.1),
             "T_sdk_tcp_before_table_clamp": _pose(.08),
             "table_clamp_delta_sdk_z_m": .02},
            {"candidate_id": "b", "T_world_tcp": _pose(.2),
             "T_sdk_tcp_before_table_clamp": _pose(.17),
             "table_clamp_delta_sdk_z_m": .03},
        ]},
        "plan": {
            "selected_candidate_id": "b",
            "attempts": [{
                "candidate_id": "a", "stage": "final_grasp_ik",
                "request": {"target": {"matrix": _pose(.1)}},
                "result": {"success": False, "failure": {
                    "kind": "backend_error", "backend_status": "AttributeError",
                    "summary": "robot_spheres missing"}},
            }],
            "stages": [{"name": "approach", "result": {"trajectory": {
                "joint_names": ["right_joint_1"], "times_s": [0., .1],
                "positions_rad": [[0.], [.1]],
            }}}],
        },
    }), encoding="utf-8")

    debug = load_pick_debug(root, scene_snapshot_sha256="scene-snapshot")

    assert debug["candidate_count"] == 2
    assert debug["status_counts"] == {"rejected": 1, "selected": 1}
    assert debug["failure_stage_counts"] == {"final_grasp_ik": 1}
    assert debug["top_failures"][0]["reason"] == "robot_spheres missing"
    assert debug["trajectory_count"] == 1
    assert debug["trajectories"][0]["positions_rad"].shape == (2, 1)
    assert debug["clamp_summary"]["mean_m"] == .025
    assert debug["snapshot_match"] is True


def test_explicit_selection_reports_scene_mismatch_and_no_trajectory(tmp_path):
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({
        "input_state_id": "other-scene", "candidate_order": ["a"],
        "candidate_parameters": {"candidate_targets": [
            {"candidate_id": "a", "T_world_tcp": _pose(.1)}]},
        "plan": {"attempts": []},
    }), encoding="utf-8")

    debug = load_pick_debug(tmp_path / "unrelated_run", selection=selection,
                            scene_snapshot_sha256="scene-snapshot")
    summary = pick_debug_summary(debug)

    assert summary["snapshot_match"] is False
    assert summary["trajectory_count"] == 0
    assert "没有生成关节轨迹" in summary["no_trajectory_reason"]


def test_missing_selection_is_reported_without_fabricating_candidates(tmp_path):
    assert load_pick_debug(tmp_path) is None
    assert pick_debug_summary(None)["available"] is False
