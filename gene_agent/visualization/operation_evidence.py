"""共享操作记录的只读说明；viewer 不推断或改写操作成功。"""


def operation_evidence_markdown(report):
    pick = report.get('pick', {})
    profile = report.get('pick_profile') or pick.get('effective_profile')
    flow = report.get('pick_flow') or pick.get('shared_flow') or report.get('shared_flow')
    lines = []
    if profile:
        lines += ['### 抓取有效参数', '', '| 参数 | 最终值 | 来源 |', '|---|---|---|']
        for key, value in profile['values'].items():
            lines.append(f"| {key} | {value} | {profile['sources'].get(key, '历史记录未注明')} |")
        geometry = report.get('pick_geometry', {})
        if geometry.get('grasp_mode') == 'world_minus_z':
            lines += ['', '当前为直立长刷的世界 −Z 抓取，实际目标与距离见 pick_geometry；上表物体相对倾角不用于此模式。']
    if flow:
        lines += ['', '### 共享抓取阶段', '', ' → '.join(flow['stages'])]
        for row in flow.get('stage_outcomes', []):
            check = row.get('check')
            if check:
                lines += ['', f"{row['stage']}：{check['status']}；证据来源：{check['source']}；{check.get('reason', '')}"]
        if not flow.get('executed'):
            lines += ['', '**以上为预测流程，不构成实际夹持证据。**']
    return '\n'.join(lines)
