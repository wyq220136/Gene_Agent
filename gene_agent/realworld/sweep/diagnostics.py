"""只规划记录的简短失败说明；兼容历史记录，不修改其事实。"""
from __future__ import annotations


def failed_stage(report):
    stages = list(report.get('pick', {}).get('stages', []))
    for candidate in report.get('candidates', []):
        stages.extend(candidate.get('stages', []))
    return next((stage for stage in stages if stage.get('success') is False), None)


def failure_summary(report):
    if report.get('planning_complete'):
        return '规划完整；未执行、未验证接触或任务成功'
    stage = failed_stage(report)
    if stage is None:
        return str(report.get('error') or report.get('status', '未形成完整规划'))
    check = stage.get('postcheck', stage.get('full_world_postcheck', {}))
    contacts = check.get('forbidden_contacts', [])
    if contacts:
        worst = max(contacts, key=lambda row: row['penetration_m'])
        roles = ' ↔ '.join(worst['roles'])
        return f"停止阶段 {stage['name']}：几何碰撞 {roles}，最大穿透 {worst['penetration_m']*1000:.1f} mm"
    result = stage.get('result', {})
    reason = result.get('failure') or result.get('failure_kind') or result.get('message') or '阶段失败，查看原始 planner 返回'
    return f"停止阶段 {stage['name']}：{reason}"
