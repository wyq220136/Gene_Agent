"""终态失败才输出 cuRobo 详情；CPU 替身，不启动求解器或设备。"""
import json

import numpy as np
import pytest

from gene_agent.feedback.curobo_log_policy import failed_in_curobo, write_selection_report
from gene_agent.feedback.realworld_diagnostics import RealworldRunDiagnostics, write_candidate_inventory
from gene_agent.motion.curobo.adapter import CuroboPlannerAdapter
from gene_agent.motion.model import (JointState, MotionOperation, MotionRequest, PlannerMode,
                                    PlanningSceneRef, PoseTarget)


def request(name):
    return MotionRequest(name, MotionOperation.IK, "right", JointState(("q",), (0.,)),
                         PoseTarget("world_tcp", np.eye(4)),
                         PlanningSceneRef(0, "snapshot", "scene"), seed=123)


class Planner:
    joint_names = ("q",)
    scene_snapshot_sha256 = "snapshot"
    scene_scope = "scene"
    planner_mode = PlannerMode.CUDA_GRAPH_BENCHMARK
    random_seed = 123
    diagnostic_snapshot = {"snapshot_sha256": "snapshot", "records": []}

    def solve_ik_batch(self, names, start, targets, **kwargs):
        return tuple({"success": index == 0, "positions_rad": [[.1]], "duration_s": .1,
                      "status": "ok" if index == 0 else "solver_failed",
                      "failure_category": "goal_ik_no_solution",
                      "solver_diagnostic": {"joint_positions_rad": [.1],
                                            "position_error_m": .02}}
                     for index, _ in enumerate(targets))


@pytest.fixture
def hooks(monkeypatch):
    from gene_agent.motion.curobo import diagnostics as module
    calls = []
    def capture(planner, req, snapshot):
        calls.append("context")
        return {"schema_version": 1, "request": req.to_dict(), "snapshot": snapshot,
                "replay_config": {}, "effective_robot_config": {"sentinel": "X" * 100_000}}
    def detail(planner, req, raw, snapshot, options, *, context=None):
        calls.append("detail")
        assert context == {}  # 公共上下文不逐失败复制。
        return {"status": "collected", "candidates": {"checked": []}, "raw_failure": raw}
    monkeypatch.setattr(module, "capture_request_context", capture)
    monkeypatch.setattr(module, "collect_failure_diagnostics", detail)
    return calls


def evaluate(recorder):
    planner = Planner()
    results = CuroboPlannerAdapter(planner, group="right", scene_revision=0).evaluate_candidates(
        (request("a:final_ik"), request("b:final_ik")))
    assert [result.success for result in results] == [True, False]
    return results


@pytest.mark.parametrize("code", [None, "not_executed", "grasp_not_confirmed", "connection_failed",
                                   "perception_failed", "motion_execution_failed", "collision_postcheck_rejected"])
def test_recovered_candidates_and_non_curobo_failures_do_not_dump_details(tmp_path, hooks, code):
    report = {"status": "failed" if code else "validated_prefix",
              "first_failure": None if code is None else {"code": code}}
    with RealworldRunDiagnostics(tmp_path, failure_only=True) as recorder:
        results = evaluate(recorder)
        assert "diagnostic" not in results[1].failure.details
        assert not (tmp_path / "motion_results").exists()
        assert not (tmp_path / "initial_scene/motion_requests").exists()
        assert recorder.finish(report) is report
    assert hooks == ["context"]  # 没有追加逐候选 FK/碰撞诊断。
    assert not (tmp_path / "events.jsonl").exists()
    assert not (tmp_path / "diagnostics").exists()
    assert not (tmp_path / "motion_results").exists()
    manifest = json.loads((tmp_path / "diagnostic_manifest.json").read_text())
    assert manifest["curobo_logging"]["failed_requests"] == 1
    assert manifest["curobo_logging"]["status"] == "suppressed_no_terminal_curobo_failure"


def test_terminal_curobo_failure_saves_one_batch_and_results_with_references(tmp_path, hooks):
    report = {"first_failure": {"code": "grasp_selection_failed", "failure_counts": [
        {"kind": "goal_ik_no_solution", "count": 1}]}}
    with RealworldRunDiagnostics(tmp_path, failure_only=True) as recorder:
        evaluate(recorder)
        recorder.finish(report)
        recorder.finish(report)  # 不重复落盘。
    assert hooks == ["context", "detail"]
    bundles = list((tmp_path / "initial_scene/motion_requests").glob("*.json"))
    assert len(bundles) == 1
    assert not (tmp_path / "motion_requests").exists()  # 可携带现场中就是唯一的一份。
    bundle = json.loads(bundles[0].read_text())
    assert len(bundle["requests"]) == 2 and bundle["source_batch_size"] == 2
    assert "results" not in bundle and len(bundle["result_refs"]) == 2
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert [event["success"] for event in events] == [False]
    assert events[0]["batch_index"] == 1 and events[0]["batch_size"] == 2
    assert bundle["result_refs"][0] is None  # 成功项保留输入，不再写逐项结果。
    failed = json.loads((tmp_path / events[0]["result_ref"]).read_text())
    diagnostic = failed["failure"]["details"]["diagnostic"]
    assert "context_ref" in diagnostic and "effective_robot_config" not in diagnostic
    assert "raw_failure" not in diagnostic
    index = json.loads((tmp_path / events[0]["diagnostic_ref"]).read_text())
    assert index["result_ref"] == events[0]["result_ref"] and "planning" not in index
    summary = json.loads((tmp_path / "failure_analysis.json").read_text())
    assert summary["events"] == [] and summary["events_ref"] == "events.jsonl"
    manifest = json.loads((tmp_path / "initial_scene/manifest.json").read_text())
    assert len(manifest["requests"]) == 1


def test_selection_does_not_reembed_attempt_details_or_duplicate_the_plan(tmp_path, hooks):
    with RealworldRunDiagnostics(tmp_path, failure_only=True) as recorder:
        results = evaluate(recorder)
        attempts = [{"candidate_id": name, "stage": "final_grasp_ik",
                     "request": request(name + ":final_ik").to_dict(), "result": result.to_dict()}
                    for name, result in zip(("a", "b"), results)]
        # 即使上游塞入大的诊断，selection/索引也不能复制它。
        attempts[1]["result"]["failure"]["details"]["diagnostic"] = {"noise": "N" * 1_000_000}
        plan = {"attempts": attempts, "selected_candidate_id": "a",
                "stages": [{"name": name, "status": "planned"}
                           for name in ("open", "pregrasp", "approach", "close")]}
        value = {"selected_candidate_id": "a", "plan": plan, "candidate_order": ["a", "b"]}
        folder = tmp_path / "selection"
        write_selection_report(folder / "selection.json", value)
        write_selection_report(folder / "candidate_selection.json", value)
        candidates = [{"candidate_id": name, "T_world_tcp": np.eye(4).tolist(), "parameters": {}}
                      for name in ("a", "b")]
        write_candidate_inventory(folder, candidates, plan)
        assert len(value["plan"]["attempts"]) == 2  # 内存执行/失败分析契约不变。
        assert all(path.stat().st_size < 20_000 for path in folder.glob("*.json"))
        disk = json.loads((folder / "selection.json").read_text())
        assert disk["plan"]["attempts"] == [] and disk["plan"]["attempt_count"] == 2
        assert json.loads((folder / "candidate_selection.json").read_text())["selection_ref"] == "selection.json"
        recorder.finish({})
    summary = json.loads((tmp_path / "failure_analysis.json").read_text())
    assert all("candidates" not in row for row in summary["inventories"])


def test_recording_failure_never_replaces_operation_failure(tmp_path, hooks, monkeypatch):
    import gene_agent.feedback.realworld_diagnostics as module
    report = {"first_failure": {"code": "motion_planning_failed"}}
    with RealworldRunDiagnostics(tmp_path, failure_only=True) as recorder:
        evaluate(recorder)
        original = module.write_json
        def broken(path, value):
            if "motion_requests" in str(path):
                raise OSError("disk full")
            return original(path, value)
        monkeypatch.setattr(module, "write_json", broken)
        assert recorder.finish(report) is report
    assert report["first_failure"]["code"] == "motion_planning_failed"
    assert json.loads((tmp_path / "diagnostic_errors.json").read_text())


def test_posture_only_rejection_and_plan_only_failure_classification():
    assert not failed_in_curobo({"first_failure": {"code": "grasp_selection_failed",
        "failure_counts": [{"kind": "pregrasp_joint_travel_exceeded"}]}})
    assert failed_in_curobo({"first_failure": {"code": "grasp_selection_failed", "details": {
        "plan": {"attempts": [{"result": {"backend": "curobo", "success": False}}]}}}})


def test_compact_viewer_uses_candidate_index_without_marking_checked_candidates_unknown(tmp_path, hooks):
    from gene_agent.visualization.realworld_pick_debug import load_pick_debug
    with RealworldRunDiagnostics(tmp_path, failure_only=True) as recorder:
        results = evaluate(recorder)
        plan = {"selected_candidate_id": "a", "attempts": [
            {"candidate_id": name, "stage": "final_grasp_ik", "request": request(name).to_dict(),
             "result": result.to_dict()} for name, result in zip(("a", "b"), results)],
            "stages": [{"name": name, "status": "planned"}
                       for name in ("open", "pregrasp", "approach", "close")]}
        candidates = [{"candidate_id": name, "T_world_tcp": np.eye(4).tolist(), "parameters": {}}
                      for name in ("a", "b")]
        folder = tmp_path / "selection"
        write_candidate_inventory(folder, candidates, plan)
        write_selection_report(folder / "selection.json", {"plan": plan,
            "candidate_order": ["a", "b"], "candidate_parameters": {"candidate_targets": candidates}})
        recorder.finish({})
    debug = load_pick_debug(tmp_path)
    assert debug["status_counts"] == {"selected": 1, "rejected": 1}
    assert debug["candidate_count"] == 2
    assert debug["top_failures"][0]["count"] == 1


def test_summary_groups_repeated_solver_reason_and_wrapper_failure_once(tmp_path, hooks):
    from gene_agent.feedback.realworld_diagnostics import write_json
    report = {"execute": False, "status": "failed",
              "first_failure": {"code": "motion_planning_failed", "message": "无可用轨迹"}}
    with RealworldRunDiagnostics(tmp_path, failure_only=True) as recorder:
        planner = Planner()
        adapter = CuroboPlannerAdapter(planner, group="right", scene_revision=0)
        for _ in range(3):
            adapter.evaluate_candidates((request("a:final_ik"), request("b:final_ik")))
        write_json(tmp_path / "pick_result.json", report)
        write_json(tmp_path / "grasp/grasp_result.json", report)
        recorder.finish(report)
    value = json.loads((tmp_path / "failure_analysis.json").read_text())
    assert len(value["operation_failures"]) == 1
    assert len(value["operation_failures"][0]["report_refs"]) == 2
    assert value["event_count"] == 3
    markdown = (tmp_path / "failure_analysis.md").read_text()
    assert markdown.count("|final_grasp_ik（3 次）|") == 1
    assert hooks.count("context") == 1 and hooks.count("detail") == 3


def test_successful_grasp_discards_candidate_noise_before_later_planning_failure(tmp_path, hooks):
    with RealworldRunDiagnostics(tmp_path, failure_only=True) as recorder:
        evaluate(recorder)
        write_selection_report(tmp_path / "selection.json", {"plan": {"stages": [
            {"name": name, "status": "planned"} for name in ("open", "pregrasp", "approach", "close")],
            "attempts": []}})
        assert not recorder.curobo_logs.pending
        CuroboPlannerAdapter(Planner(), group="right", scene_revision=0).evaluate_candidates(
            (request("lift_a:final_ik"), request("lift_b:final_ik")))
        recorder.finish({"first_failure": {"code": "lift_planning_failed"}})
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert [event["request_id"] for event in events] == ["lift_b:final_ik"]
    manifest = json.loads((tmp_path / "diagnostic_manifest.json").read_text())
    assert manifest["curobo_logging"]["failed_requests"] == 2
    assert manifest["curobo_logging"]["saved_failed_requests"] == 1
