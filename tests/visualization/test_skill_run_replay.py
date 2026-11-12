"""运行记录的事实边界：实例、场景快照和规划/实际回放不可串用。"""

import json

import pytest

from gene_agent.feedback.skill_run import load_skill_run
from gene_agent.visualization.replay import ReplayTimeline


def _write(root, relative, data):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def _planned_run(root):
    _write(root, "collision/collision_snapshot.json", {"snapshot_sha256": "scene-a"})
    _write(root, "skill_pick_bowl_result.json", {
        "schema_version": 1, "skill": "pick", "task": "pick_bowl_skill_planning",
        "instance_id": "bowl_2", "snapshot_sha256": "scene-a",
        "selected_candidate_id": "ring-2", "stages": [
            {"name": "pregrasp", "status": "planned",
             "request": {"operation": "plan", "scene": {"snapshot_sha256": "scene-a"},
                         "start": {"joint_names": ["j1"], "positions_rad": [0]}},
             "result": {"operation": "plan", "trajectory": {
                 "joint_names": ["j1"], "times_s": [0, .5, 1.0],
                 "positions_rad": [[0], [.5], [1.0]]}}},
            {"name": "approach", "status": "failed",
             "request": {"operation": "linear_tcp", "scene": {"snapshot_sha256": "scene-a"},
                         "start": {"joint_names": ["j1"], "positions_rad": [1.0]}},
             "result": {"operation": "linear_tcp", "success": False, "trajectory": None}},
        ]})


def test_planning_run_preserves_instance_and_disallows_actual(tmp_path):
    _planned_run(tmp_path)
    view = load_skill_run(tmp_path)
    assert view.instance_id == "bowl_2"
    assert [stage.path_kind for stage in view.stages] == ["plan", "linear_tcp"]
    assert all(stage.execution_status == "not_executed" for stage in view.stages)
    timeline = ReplayTimeline.load(tmp_path)
    assert timeline.seek(.75).joint_positions_rad == (.5,)
    assert timeline.seek(1.0).stage_id == "01_approach"
    assert timeline.seek(1.0).sample_quality == "stage_marker"
    with pytest.raises(ValueError, match="没有实际执行"):
        timeline.select(mode="actual")
    timeline.select("00_pregrasp", mode="planned")
    assert timeline.duration_s == 1.0
    assert timeline.seek(.5).local_time_s == .5


def test_planning_non_motion_stages_keep_operation_without_request(tmp_path):
    _planned_run(tmp_path)
    path = tmp_path / "skill_pick_bowl_result.json"
    report = json.loads(path.read_text())
    report["stage_contract"] = [{"name": "open", "operation": "gripper_command"}]
    report["stages"].insert(0, {"name": "open", "status": "planned"})
    report["stages"].append({"name": "hold", "status": "planned"})
    path.write_text(json.dumps(report), encoding="utf-8")
    view = load_skill_run(tmp_path)
    assert view.stages[0].operation == "gripper_command"
    assert view.stages[-1].operation == "hold"
    assert view.stages[-1].path_kind == "hold"


def test_scene_mismatch_and_missing_samples_are_rejected(tmp_path):
    _planned_run(tmp_path)
    path = tmp_path / "collision/collision_snapshot.json"
    path.write_text(json.dumps({"snapshot_sha256": "other"}), encoding="utf-8")
    with pytest.raises(ValueError, match="哈希不一致"):
        load_skill_run(tmp_path)
    path.write_text(json.dumps({"snapshot_sha256": "scene-a"}), encoding="utf-8")
    report_path = tmp_path / "skill_pick_bowl_result.json"
    report = json.loads(report_path.read_text())
    report["stages"][0]["result"]["trajectory"]["times_s"] = [0, .5]
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="采样数不一致"):
        ReplayTimeline.load(tmp_path)


def _observation(state_id, sim_time_s, joint):
    return {"state_id": state_id, "sim_time_s": sim_time_s,
            "objects": {"bowl_2": {"T_world_object": [[joint]]}},
            "robots": {"arm": {"joint_names": ["j1"], "joint_positions_rad": [joint],
                               "gripper": {"width_m": .08}}}}


def _executed_run(root):
    states = {"initial": ("s0", 0, 0), "opened": ("s1", .2, 0),
              "final": ("s2", 1.2, 1)}
    for name, (state_id, time_s, joint) in states.items():
        _write(root, f"stages/{name}/observation.json", _observation(state_id, time_s, joint))
    _write(root, "collision/final/collision_snapshot.json", {"snapshot_sha256": "scene-b"})
    _write(root, "plans/final.json", {"operation": "plan", "success": True,
                                      "scene_snapshot_sha256": "scene-b",
                                      "joint_names": ["j1"], "interpolation_dt_s": .5,
                                      "positions_rad": [[0], [.5], [1]]})
    _write(root, "executions/final.json", {
        "success": True, "start_state_id": "s1", "end_state_id": "s2",
        "joint_names": ["j1"], "samples": [
            {"time_s": .5, "actual_positions_rad": [.4]},
        ]})
    _write(root, "visual/manifest.json", {"target_body": "bowl_2"})
    _write(root, "result.json", {
        "mode": "temporary_pick_bowl_operation_interface_probe",
        "stage_state_ids": {name: row[0] for name, row in states.items()},
        "stages": {
            "open": {"success": True, "start_state_id": "s0", "end_state_id": "s1"},
            "final": {"success": True, "start_state_id": "s1", "end_state_id": "s2"},
        }})


def test_legacy_actual_replay_marks_partial_sample_unknown_object_pose(tmp_path):
    _executed_run(tmp_path)
    timeline = ReplayTimeline.load(tmp_path)
    assert timeline.view.instance_id == "bowl_2"
    assert timeline.duration_s == pytest.approx(1.2)
    midpoint = timeline.seek(.7)
    assert midpoint.stage_id == "01_final"
    assert midpoint.joint_positions_rad == (.4,)
    assert midpoint.sample_quality == "partial_actual_sample"
    assert midpoint.object_poses_world is None
    assert midpoint.state_id is None
    assert timeline.seek(1.2).state_id == "s2"
    timeline.select("01_final", mode="planned")
    assert timeline.seek(.5).joint_positions_rad == (.5,)


def test_complete_actual_replay_uses_recorded_state_without_interpolation(tmp_path):
    _executed_run(tmp_path)
    path = tmp_path / "executions/final.json"
    report = json.loads(path.read_text())
    report["replay_samples"] = [
        {"state_id": "middle", "sim_time_s": .7, "scene_revision": 3,
         "qpos": [.4, 1.2], "objects": {"bowl_2": {"T_world_object": [[.4]]}},
         "robots": {"arm": {"joint_names": ["j1"], "joint_positions_rad": [.4],
                            "T_world_tcp": [[.6]], "gripper": {"width_m": .03}}},
         "attachment": {"object_instance_id": "bowl_2"}},
        {"state_id": "s2", "sim_time_s": 1.2, "scene_revision": 3,
         "qpos": [1., 1.3], "objects": {"bowl_2": {"T_world_object": [[1.]]}},
         "robots": {"arm": {"joint_names": ["j1"], "joint_positions_rad": [1.],
                            "T_world_tcp": [[1.]], "gripper": {"width_m": .02}}}},
    ]
    path.write_text(json.dumps(report), encoding="utf-8")
    timeline = ReplayTimeline.load(tmp_path)
    assert timeline.view.stages[1].actual_sample_format == "complete_actual_state"
    frame = timeline.seek(.71)
    assert frame.sample_quality == "complete_actual_sample"
    assert frame.state_id == "middle"
    assert frame.sample_time_s == pytest.approx(.5)
    assert frame.qpos == (.4, 1.2)
    assert frame.object_poses_world == {"bowl_2": [[.4]]}
    assert frame.tcp_poses_world == {"arm": [[.6]]}
    assert frame.scene_revision == 3
    assert frame.sim_time_s == .7
    assert frame.attachment == {"object_instance_id": "bowl_2"}
    assert timeline.seek(1.2).state_id == "s2"

    report["replay_samples"][-1]["state_id"] = "wrong"
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="完整采样终点"):
        ReplayTimeline.load(tmp_path)


def test_new_execution_report_keeps_instance_and_physical_result(tmp_path):
    _executed_run(tmp_path)
    path = tmp_path / "result.json"
    report = json.loads(path.read_text())
    report.update(mode="synchronous_pick_skill", instance_id="bowl_2",
                  physical_pick_success=False)
    path.write_text(json.dumps(report), encoding="utf-8")
    timeline = ReplayTimeline.load(tmp_path)
    assert timeline.view.source_format == "skill_execution"
    assert timeline.view.instance_id == "bowl_2"
    assert timeline.view.physical_success is False
    assert timeline.mode == "actual"


def test_legacy_state_chain_mismatch_rejected(tmp_path):
    _executed_run(tmp_path)
    path = tmp_path / "result.json"
    report = json.loads(path.read_text())
    report["stages"]["final"]["start_state_id"] = "other"
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="起点状态ID"):
        load_skill_run(tmp_path)


def test_failed_legacy_stage_opens_without_claiming_execution(tmp_path):
    _executed_run(tmp_path)
    (tmp_path / "executions/final.json").unlink()
    (tmp_path / "stages/final/observation.json").unlink()
    plan_path = tmp_path / "plans/final.json"
    plan = json.loads(plan_path.read_text())
    plan["success"] = False
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    report_path = tmp_path / "result.json"
    report = json.loads(report_path.read_text())
    del report["stage_state_ids"]["final"]
    report["stages"]["final"] = {"success": False, "plan": plan, "execution": None}
    report_path.write_text(json.dumps(report), encoding="utf-8")
    timeline = ReplayTimeline.load(tmp_path)
    assert timeline.mode == "planned"
    timeline.select("01_final", mode="planned")
    assert timeline.seek(0).sample_quality == "failed_stage_marker"
    with pytest.raises(ValueError, match="没有实际执行"):
        timeline.select("01_final", mode="actual")
