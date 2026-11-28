"""真机按操作终态保存 cuRobo 详情；正常候选淘汰不触发落盘。"""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import json
from uuid import uuid4
from gene_agent.feedback.runtime_timing import timed_runtime_call, current_stage_timing_event_id



_MOTION_FAILURES = {
    "start_collision", "goal_ik_no_solution", "goal_collision", "path_not_found",
    "path_collision", "constraint_violation", "backend_error", "invalid_request",
    "motion_planning_failed", "planning_failed", "lift_planning_failed",
    "postpick_planning_failed", "timeout", "ik_pose_not_converged",
    "ik_constraint_rejected", "solver_rejected", "door_planning_failed",
}


def failed_in_curobo(report):
    """只查看操作终态的失败，不遍历候选 attempts 或历史失败分支。"""
    if not isinstance(report, dict):
        return False
    failure = report.get("first_failure") or report.get("failure") or {}
    if not isinstance(failure, dict):
        return False
    code = failure.get("code") or failure.get("kind")
    if code in _MOTION_FAILURES:
        return True
    if code == "grasp_selection_failed":
        # 全部只因姿态/接触独立后检淘汰时，不归因给 cuRobo。
        counts = failure.get("failure_counts")
        if counts is not None:
            return any(row.get("kind") in _MOTION_FAILURES for row in counts)
        details = failure.get("details") or {}
        attempts = details.get("plan", {}).get("attempts", ())
        return any(row.get("result", {}).get("backend") == "curobo"
                   and row["result"].get("success") is False for row in attempts)
    if code in {"insert_prefix_failed", "extract_prefix_failed", "place_prefix_failed"}:
        details = failure.get("details") or {}
        return any(row.get("result", {}).get("backend") == "curobo"
                   and row["result"].get("success") is False
                   for row in details.get("stages", ()))
    # Task 包装层可能把子任务失败原样嵌入；不扫描其他已完成子任务。
    nested = failure.get("first_failure") or failure.get("failure")
    return failed_in_curobo({"first_failure": nested}) if isinstance(nested, dict) else False


@timed_runtime_call("diagnostics")
def write_selection_report(path, report):
    """内存仍交付完整计划供执行；磁盘 selection 只保留一份必要阶段。"""
    from gene_agent.motion.recording import current_motion_recording
    from gene_agent.feedback.realworld_diagnostics import write_json
    recording = current_motion_recording()
    policy = getattr(recording, "curobo_logs", None)
    if policy is None:
        return write_json(path, report)
    value = {key: val for key, val in report.items() if key != "plan"}
    value["runtime_timing_event_id"] = current_stage_timing_event_id()
    value["timing_scope"] = "planning_wall_s is local preparation/planning; full stage including output uses runtime_timing_event_id"
    plan = report.get("plan", {})
    policy.resolve_successful_selection(plan)
    value["plan"] = {key: val for key, val in plan.items() if key != "attempts"}
    value["plan"]["attempts"] = []
    value["plan"]["attempt_count"] = len(plan.get("attempts", ()))
    value["plan"]["attempt_logging"] = "terminal_curobo_failure_only"
    if Path(path).name == "candidate_selection.json":
        value = {"schema_version": 2, "selection_ref": "selection.json",
                 "selected_candidate_id": report.get("selected_candidate_id"),
                 "pregrasp_search_summary": report.get("pregrasp_search_summary")}
    return write_json(path, value)


class FailureOnlyCuroboLogs:
    def __init__(self, recorder):
        self.recorder = recorder
        self.raw = {}
        self.contexts = {}
        self.pending = []
        self.finished = False
        self.batches = self.requests = self.failed_requests = 0
        self.status = "pending"
        self.saved_failed_requests = 0

    def defer_diagnostic(self, planner, request, raw, snapshot, options):
        self.raw[id(request)] = (deepcopy(raw), options)

    @timed_runtime_call("diagnostics")
    def record(self, planner, requests, results, *, invocation=None):
        requests, results = tuple(requests), tuple(results)
        for request, result in zip(requests, results):
            self.recorder.record_trajectory(request, result)
        self.batches += 1
        self.requests += len(requests)
        self.failed_requests += sum(not result.success for result in results)
        if not any(not result.success for result in results):
            return results
        snapshot = deepcopy(getattr(planner, "diagnostic_snapshot", {}))
        key = (id(planner), requests[0].scene, requests[0].seed, requests[0].planner_mode)
        if key not in self.contexts:
            try:
                from gene_agent.motion.curobo.diagnostics import capture_request_context
                self.contexts[key] = deepcopy(capture_request_context(planner, requests[0], snapshot))
            except Exception as error:
                self.contexts[key] = {"context_capture_error": f"{type(error).__name__}: {error}",
                                      "snapshot": snapshot}
                self.recorder.record_error("deferred_curobo_context", error)
        raws = tuple(self.raw.pop(id(request), None) for request in requests)
        self.pending.append((planner, requests, results, snapshot, key, raws, invocation))
        return results

    def resolve_successful_selection(self, plan):
        """抓取前缀已经通过时，淘汰的 IK/路径不带入后续执行或抬升故障。"""
        stages = {row["name"]: row["status"] for row in plan.get("stages", ())}
        if ((plan.get("success") is True and plan.get("screen_contract")=="full_insert_release_retreat")
                or all(stages.get(name) == "planned" for name in ("open", "pregrasp", "approach", "close"))):
            self.pending.clear()
            self.raw.clear()
            self.contexts.clear()

    def finish(self, report):
        if self.finished:
            return report
        self.finished = True
        trigger = failed_in_curobo(report) and bool(self.pending)
        self.status = "saved_on_curobo_failure" if trigger else "suppressed_no_terminal_curobo_failure"
        try:
            if trigger:
                self.saved_failed_requests = sum(not result.success for row in self.pending for result in row[2])
                self._flush()
                print(f"[diagnostics] cuRobo 终态失败：保存 {self.saved_failed_requests} 条失败请求的详情；"
                      "汇总见 failure_analysis.md，逐请求见 events.jsonl", flush=True)
        except Exception as error:
            self.status = "partial_recording_error"
            self.recorder.record_error("deferred_curobo_flush", error)
        finally:
            self.pending.clear()
            self.raw.clear()
            self.contexts.clear()
            self.recorder.manifest_curobo_logging = dict(
                policy="terminal_failure_only", status=self.status,
                batches=self.batches, requests=self.requests, failed_requests=self.failed_requests,
                saved_failed_requests=self.saved_failed_requests, event_scope="failed_requests_only",
                note="候选失败数不是操作失败数；仅 cuRobo 导致操作终态失败时保存详细批次")
        return report

    @timed_runtime_call("diagnostics")
    def _flush(self):
        from gene_agent.feedback.realworld_diagnostics import write_json
        from gene_agent.motion.curobo.diagnostics import collect_failure_diagnostics, digest
        for planner, requests, results, snapshot, key, raws, invocation in self.pending:
            path = self.recorder.request_output_dir / (uuid4().hex + ".json")
            context = self.contexts[key]
            first = requests[0].to_dict()
            payload = {**context, "request": first, "request_sha256": digest(first),
                       "mode": "frozen_motion_request", "source_batch_size": len(requests),
                       "invocation": invocation or ("solve_ik" if first["operation"] == "ik" else "plan"),
                       "requests": [r.to_dict() for r in requests]}
            payload["requests_sha256"] = digest(payload["requests"])
            write_json(path, payload)
            recorded = []
            for request, result, source in zip(requests, results, raws):
                if not result.success and source is not None:
                    raw, options = source
                    try:
                        # 句柄可能复用到别的场景；不拿新现场补写旧碰撞证据。
                        if planner.scene_snapshot_sha256 != request.scene.snapshot_sha256:
                            raise RuntimeError("planner 场景已变化，保留原始失败和冻结输入，不补查碰撞")
                        diagnostic = collect_failure_diagnostics(
                            planner, request, raw, snapshot, options, context={})
                        # raw_failure 中的 seeds/状态已经在原结果 details 中，避免再嵌一遍。
                        diagnostic.pop("raw_failure", None)
                        diagnostic["context_ref"] = str(path.relative_to(self.recorder.root))
                        diagnostic = json.loads(json.dumps(diagnostic, allow_nan=False,
                                                           default=lambda value: value.tolist()))
                        result = replace(result, failure=replace(result.failure,
                            details={**result.failure.details, "diagnostic": diagnostic}))
                    except Exception as error:
                        result = replace(result, failure=replace(result.failure,
                            details={**result.failure.details,
                                     "diagnostic_error": f"{type(error).__name__}: {error}"}))
                result = replace(result, metrics={**result.metrics, "replay_request_ref": str(path)})
                recorded.append(result)
            failed_indices = [index for index, result in enumerate(recorded) if not result.success]
            refs = self.recorder.record(
                tuple(requests[index] for index in failed_indices),
                tuple(recorded[index] for index in failed_indices), snapshot=snapshot,
                replay_ref=path, invocation=invocation,
                source_batch_indices=failed_indices, source_batch_size=len(requests))
            payload["result_refs"] = [None] * len(requests)
            for index, ref in zip(failed_indices, refs):
                payload["result_refs"][index] = ref
            write_json(path, payload)
