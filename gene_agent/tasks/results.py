"""Task 结果范围与失败记录；不推断机器人或物体状态。"""
from __future__ import annotations
import traceback


def annotate_task_result(report: dict, *, scope: str, full_scope: str,
                         actual_final: dict | None = None, report_ref: str = "task_result.json") -> dict:
    report["requested_scope"] = "full" if scope == full_scope else scope
    report["operation_success"] = bool(report.get("success"))
    report["task_success"] = report["operation_success"] if scope == full_scope else None
    if actual_final is not None:
        report["actual_final"] = actual_final
    if not report["operation_success"]:
        stage = report.get("failure_stage")
        row = report.get("stages", {}).get(stage, {})
        plan = row.get("plan") or row.get("planning") or {}
        execution = row.get("execution") or {}
        report["failure"] = {
            "stage": stage,
            "code": (plan.get("failure_category") or execution.get("failure_type")
                     or row.get("failure_type") or "operation_failed"),
            "message": str(report.get("error") or plan.get("raw_error")
                           or plan.get("status") or execution.get("message") or stage),
            "report_ref": report_ref,
        }
    return report


def record_task_exception(report: dict, error: Exception) -> None:
    report.update(status="failed", success=False, failure_stage="exception",
                  error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())


def finish_task_report(report: dict, *, success: bool, scope: str, full_scope: str,
                       observe_final, save) -> dict:
    report.update(success=bool(success), status="completed" if success else "failed",
                  until_skill=scope)
    try:
        final = observe_final()
        report["final"] = final
    except Exception as error:
        # 已执行动作的失败原因优先保留；不能编造一次新的现场观测。
        report["final_observation_error"] = f"{type(error).__name__}: {error}"
        final = {"state_unavailable": True}
        if success:
            report.update(success=False, status="unknown", failure_stage="final_observation")
    annotate_task_result(report, scope=scope, full_scope=full_scope, actual_final=final)
    save()
    return report
