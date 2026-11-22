"""全流程预览的预测状态协议；不属于真机后置证据。"""
from __future__ import annotations
from collections.abc import Mapping

SOURCE = "microwave_plan_only"

def is_prediction(scene):
    source = scene.get("provenance", {})
    return (source.get("state_source") == SOURCE
            and source.get("physics_stepped") is False
            and bool(source.get("prediction_origin_sha256"))
            and isinstance(source.get("prediction_step"), int))

def predicted_report(scene, report, kind):
    """预测报告只能供同一预览链使用，绝不替代真机实测报告。"""
    if not isinstance(report, Mapping):
        return False
    preview = report.get("source_format") == f"{SOURCE}_{kind}"
    if preview and (not is_prediction(scene)
            or report.get("execute") is not False
            or report.get("prediction_origin_sha256") !=
                scene["provenance"]["prediction_origin_sha256"]
            or not isinstance(report.get("prediction_step"), int)
            or report["prediction_step"] > scene["provenance"]["prediction_step"]):
        raise ValueError("预测持物/释放报告只能用于同一 plan_only 预测场景")
    return preview

def report_basis(scene, kind):
    if not is_prediction(scene):
        raise ValueError("只有 plan_only 场景可生成预测状态报告")
    return {"source_format": f"{SOURCE}_{kind}", "execute": False,
            "predicted": True, "status": "planned",
            "prediction_origin_sha256": scene["provenance"]["prediction_origin_sha256"],
            "prediction_step": scene["provenance"]["prediction_step"]}

def assumption_check(scene):
    from gene_agent.tasks.checks import CheckResult
    if not is_prediction(scene):
        raise ValueError("规划假设不能绑定到真实现场")
    p = scene["provenance"]
    return CheckResult("pass", "planning_assumption", {
        "predicted": True,
        "collision_snapshot_sha256": scene["snapshot"]["snapshot_sha256"],
        "source_frame_number": p["source_frame_number"],
        "source_timestamp_ms": p["source_timestamp_ms"],
        "prediction_origin_sha256": p["prediction_origin_sha256"],
        "prediction_step": p["prediction_step"],
    }, "仅为继续预览假定接触/持物/释放成立；真实执行仍需现场验证")

def reject_assumption_on_live_scene(scene, check):
    if check.get("source") == "planning_assumption" and not is_prediction(scene):
        raise ValueError("规划假设不能作为真机持物或释放证据")
