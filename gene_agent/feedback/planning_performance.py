"""只读规划统计；共享调用/lease去重，旧记录缺失项保持未知。"""
from __future__ import annotations

import json
from pathlib import Path


def _seconds(value):
    import math
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0 else None


def _json_lines(path, warnings):
    if not path.exists():
        return []
    rows = []
    for index, line in enumerate(path.read_text().splitlines()):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError('记录不是object')
            rows.append(row)
        except ValueError:
            warnings.append(f'{path.name}:{index + 1} 损坏/未写完，覆盖不完整')
    return rows


def summarize_planning_run(root, *, stage=None):
    """优先独立调用日志；兼容events与旧候选工件，不把父级等待加到子项。"""
    root = Path(root)
    warnings = []
    events = _json_lines(root / 'events.jsonl', warnings)
    calls = []
    modern = (root / 'planning_calls.jsonl').exists()
    if modern:
        for row in _json_lines(root / 'planning_calls.jsonl', warnings):
            if row.get('operation') in {'describe', 'fk'}:
                continue
            if stage and row.get('stage_id') != stage:
                continue
            calls.append(row)
    else:
        results = []
        for event in events:
            if event.get('kind') == 'planning' and (not stage or event.get('stage_id') == stage):
                result = event.get('planning', event.get('result'))
                if isinstance(result, dict):
                    results.append((result, 'events', event.get('stage_id')))
        for path in sorted(root.glob('candidate_screen_*.json')) if not stage else ():
            try:
                content = json.loads(path.read_text())
                results.extend((r, path.name, 'candidate') for r in content.get('results', ()))
                for record in content.get('path_screens', ()):
                    results.extend((r, path.name, 'candidate') for r in record.values()
                                   if isinstance(r, dict) and 'duration_s' in r)
            except (ValueError, TypeError):
                warnings.append(f'{path.name} 损坏，覆盖不完整')
        groups = {}
        for index, (result, source, stage_id) in enumerate(results):
            metrics = result.get('metrics') or {}
            timing = metrics.get('planning_timing') or {}
            call_id = timing.get('call_id')
            if not call_id and metrics.get('batch_size'):
                # 旧batch没有身份；只能按同工件内实际批次元数据识别，明确保留限制。
                call_id = source + json.dumps([metrics.get('planner_cache'), metrics.get('batch_total_duration_s'), metrics['batch_size']], sort_keys=True)
                warnings.append('旧batch缺调用ID，批次按同工件元数据分组；完全相同的重复调用无法严格区分')
            call_id = call_id or f'{source}:{index}'
            if call_id not in groups:
                groups[call_id] = dict(call_id=call_id, stage_id=stage_id, results=[])
            groups[call_id]['results'].append(result)
        calls = list(groups.values())

    totals = dict(backend_duration_s=0., pool_setup_s=0., request_roundtrip_s=0.)
    known = {key: False for key in totals}
    seen_leases = set()
    result_count = failed_count = errors = cache_hits = 0
    for row in calls:
        wall = _seconds(row.get('roundtrip_wall_s'))
        if wall is not None:
            totals['request_roundtrip_s'] += wall
            known['request_roundtrip_s'] = True
        errors += bool(row.get('error'))
        if row.get('error'):
            warnings.append('部分调用异常，只有往返时间；后端/准备组件覆盖不完整')
        rows = row.get('results') or []
        result_count += len(rows)
        for result in rows:
            failed_count += result.get('success') is False
            duration = _seconds(result.get('duration_s'))
            if duration is not None:
                totals['backend_duration_s'] += duration
                known['backend_duration_s'] = True
            else:
                warnings.append('结果缺少有效duration_s，后端累计不完整')
        # 同lease可在一个stage调用多次，或被复制进batch每项，只计算一次。
        cache = next((r.get('metrics', {}).get('planner_cache') for r in rows if r.get('metrics', {}).get('planner_cache')), None)
        if cache:
            lease_id = cache.get('acquisition_id') or row['call_id']
            if lease_id not in seen_leases:
                seen_leases.add(lease_id)
                setup = _seconds(cache.get('setup_duration_s'))
                if setup is not None:
                    totals['pool_setup_s'] += setup
                    known['pool_setup_s'] = True
                cache_hits += cache.get('cache_hit') is True
        elif rows:
            warnings.append('部分调用缺planner_cache，准备时间未知')
    answer = {key: value if known[key] else None for key, value in totals.items()}
    answer.update(schema_version=1, source='planning_calls' if modern else 'legacy_artifacts',
                  call_count=len(calls), result_count=result_count, failed_result_count=failed_count,
                  error_call_count=errors, cache_hit_acquisitions=cache_hits,
                  acquisition_count=len(seen_leases),
                  recorded_planning_s=(totals['backend_duration_s'] + totals['pool_setup_s'])
                      if known['backend_duration_s'] and known['pool_setup_s'] else None)
    for key, kind, field in (('action_planning_wait_s', 'planning', 'planning_wall_s'),
                             ('candidate_screen_wall_s', 'candidate_timing', 'wall_s')):
        values = [_seconds(e.get(field)) for e in events if e.get('kind') == kind and (not stage or e.get('stage_id') == stage)]
        values = [v for v in values if v is not None]
        answer[key] = sum(values) if values else None
    answer['phase_duration_s'] = {}
    for call in calls:
        for result in call.get('results') or []:
            for name, value in result.get('metrics', {}).get('timing', {}).items():
                seconds = _seconds(value)
                if seconds is not None:
                    answer['phase_duration_s'][name] = answer['phase_duration_s'].get(name, 0.) + seconds
    if not calls:
        warnings.append('没有可统计的规划调用')
    answer['warnings'] = list(dict.fromkeys(warnings))
    answer['scope'] = '组件累计，不含物理执行；roundtrip/阶段wall包含子项，不能与组件相加；候选IK不等于动作失败'
    answer['slow_calls'] = sorted(calls, key=lambda r: r.get('roundtrip_wall_s', 0.), reverse=True)[:5] if modern else []
    return answer


def performance_markdown(summary):
    def fmt(value):
        return '未知' if value is None else f'{value:.3f}s'
    return (f"规划：{summary['call_count']} 次调用，{summary['failed_result_count']} 个失败结果，"
            f"{summary['error_call_count']} 次异常；后端 {fmt(summary['backend_duration_s'])}，"
            f"pool准备 {fmt(summary['pool_setup_s'])}，组件合计 {fmt(summary['recorded_planning_s'])}；"
            f"请求等待 {fmt(summary['request_roundtrip_s'])}。\n"
            + '\n'.join('统计限制：' + text for text in summary['warnings']))
