"""抓碗筛选工作量统计与严格同请求复用；不跨进程信任旧规划。"""
from collections import Counter
from dataclasses import replace
from gene_agent.motion.model import motion_request_from_dict, motion_result_from_dict
from gene_agent.skills.library.pick.pregrasp_search import PregraspRequestCache


def summarize_selection(plan):
    stages = {}
    seen = set()
    for row in plan.get("attempts", ()):
        request, result = row.get("request"), row.get("result")
        if not request or not isinstance(result, dict) or "operation" not in result:
            continue
        group = stages.setdefault(row["stage"], Counter())
        key = PregraspRequestCache.key(motion_request_from_dict(request))
        reused = bool((result.get("metrics") or {}).get("request_reuse"))
        group["requests"] += 1
        group["cache_hits"] += int(reused)
        group["backend_calls_or_batch_items"] += int(not reused)
        group["success"] += int(bool(result.get("success")))
        group["failure"] += int(not result.get("success"))
        group["repeated_exact_requests"] += int(key in seen)
        seen.add(key)
    return {"stages": {k: dict(v) for k, v in stages.items()},
            "unique_request_count": len(seen),
            "claim": "批IK计数为真实目标条目，不等于GPU调用次数；失败不是重复的证据"}


def reuse_selected_motion(selection, stage_name, request, *, identity, strict_snapshot_sha256):
    """同次运行、相同模型配置/精确快照/起点/请求才复用成功结果。"""
    from os import getpid
    if (selection.get("reuse_process_id") != getpid()
            or not identity or selection.get("planning_reuse_identity") != identity):
        return None, {}, "planning_identity_changed_or_unavailable"
    rows = [r for r in selection.get("plan", {}).get("stages", ()) if r["name"] == stage_name]
    if len(rows) != 1 or rows[0].get("status") != "planned":
        return None, {}, "selected_stage_unavailable"
    row = rows[0]
    raw = row.get("request")
    composition = (row.get("details") or {}).get("approach_composition") or {}
    # 组合接近的 request=None；其各子段的起点/场景不同，不套用单请求缓存。
    if raw is None:
        return None, {}, "composite_requires_replanning"
    previous = motion_request_from_dict(raw)
    comparable = replace(request, scene=replace(request.scene, revision=previous.scene.revision))
    if PregraspRequestCache.key(previous) != PregraspRequestCache.key(comparable):
        return None, {}, "exact_request_changed"
    if selection.get("stage_collision_snapshot_sha256", {}).get("pregrasp") != strict_snapshot_sha256:
        return None, {}, "strict_scene_changed"
    result = motion_result_from_dict(row["result"])
    if (not result.success or result.trajectory is None or result.goal_state is None
            or result.request_id != previous.request_id or result.operation != previous.operation):
        return None, {}, "successful_result_unavailable"
    reused = PregraspRequestCache.reuse(request, result)
    reused = replace(reused, metrics={**reused.metrics, "request_reuse": {
        **reused.metrics["request_reuse"], "scope": "selection_to_execution_exact_state"}})
    return reused, composition, "exact_request_reused"
