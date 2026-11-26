import json

import numpy as np
import pytest

from gene_agent.visualization.realworld_pick_debug import (
    _add_line_segments_compat,
    load_pick_debug,
    pick_debug_summary,
)


def _pose(x):
    value = np.eye(4)
    value[0, 3] = x
    return value.tolist()


class _LineScene:
    def __init__(self, *, reject_width=False, error=None):
        self.reject_width = reject_width
        self.error = error
        self.calls = []

    def add_line_segments(self, name, **kwargs):
        self.calls.append((name, kwargs))
        if self.error is not None:
            raise self.error
        if self.reject_width and "thickness" in kwargs:
            raise TypeError(
                "SceneApi.add_line_segments() got an unexpected keyword argument 'thickness'")
        return "line-handle"


def test_line_segments_keeps_width_for_new_viser():
    scene = _LineScene()

    handle = _add_line_segments_compat(
        scene, "/path", points=np.zeros((1, 2, 3)), colors=(1, 2, 3),
        thickness=.004, thickness_units="world")

    assert handle == "line-handle"
    assert len(scene.calls) == 1
    assert scene.calls[0][1]["thickness"] == .004
    assert scene.calls[0][1]["thickness_units"] == "world"


def test_line_segments_retries_without_width_for_old_viser():
    scene = _LineScene(reject_width=True)

    handle = _add_line_segments_compat(
        scene, "/path", points=np.zeros((1, 2, 3)), colors=(1, 2, 3),
        thickness=.004, thickness_units="world")

    assert handle == "line-handle"
    assert len(scene.calls) == 2
    assert "thickness" not in scene.calls[1][1]
    assert "thickness_units" not in scene.calls[1][1]
    assert scene.calls[1][1]["colors"] == (1, 2, 3)


def test_line_segments_does_not_hide_unrelated_type_error():
    scene = _LineScene(error=TypeError("points must have shape (N, 2, 3)"))

    with pytest.raises(TypeError, match="points must have shape"):
        _add_line_segments_compat(
            scene, "/path", points=np.zeros((1, 3)), colors=(1, 2, 3),
            thickness=.004, thickness_units="world")

    assert len(scene.calls) == 1


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
