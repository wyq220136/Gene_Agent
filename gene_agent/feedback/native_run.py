"""原生运行证据索引：保留评分真源、事件前缀和媒体原格式。"""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import traceback


def _hash(path):
    """源证据内容签名，不用文件名代替版本。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


class NativeRunRecorder:
    """独占新目录，每事件追加落盘，异常不会丢弃此前阶段。"""
    def __init__(self, output_dir, *, manifest):
        self.root=Path(output_dir).resolve()
        self.root.mkdir(parents=True,exist_ok=False)
        self.manifest=deepcopy(dict(manifest))
        (self.root/'manifest.json').write_text(json.dumps(self.manifest,ensure_ascii=False,indent=2,allow_nan=False))

    def append_event(self, event):
        """先验证/序列化再写，拒绝跨episode混入同一事实流。"""
        row=deepcopy(dict(event))
        if self.manifest.get('episode_id') and row.get('episode_id') != self.manifest['episode_id']:
            raise ValueError('事件episode与记录器不一致')
        if not row.get('stage_id'):raise ValueError('事件缺少stage_id')
        encoded=json.dumps(row,ensure_ascii=False,allow_nan=False)
        with (self.root/'events.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(encoded+'\n');stream.flush()

    def append_state(self, stage_id, state):
        """公共state携带episode时不重复传参，保留真实时间与来源。"""
        unavailable=bool(state.values.get('state_unavailable'))
        self.append_event({**state.values,'episode_id':state.values.get('episode_id',self.manifest.get('episode_id')) if unavailable else state.values.get('episode_id'),
                           'stage_id':stage_id,'kind':'state_unavailable' if unavailable else 'actual_sample',
                           'state_id':state.state_id,'source':state.source})

    def finalize(self, *, native_result_ref, error=None):
        """只从明确的原生评分读结果，缺失报告保持未评测。"""
        report=dict(schema_version=1,manifest_ref='manifest.json',events_ref='events.jsonl',
                    status='error' if error else 'unscored',task_success=None,native_result=None)
        if native_result_ref is not None:
            path=Path(native_result_ref).resolve()
            try:
                native=json.loads(path.read_text())
                success=native.get('task_success',native.get('success'))
                if type(success) is not bool and success is not None:
                    raise ValueError('原评分必须为bool或null')
                report.update(task_success=success,native_result={'path':str(path),'sha256':_hash(path)},
                              status='error' if error else 'scored' if success is not None else 'unscored')
            except (OSError,ValueError,TypeError,AttributeError) as exc:
                # 评分读取失败不能覆盖控制器/规划器原始失败，也不能推断成功。
                report.update(status='error',recording_errors=[f'native_score: {type(exc).__name__}: {exc}'],
                              native_result_unavailable=str(path))
        if error is not None:
            report['error']=f'{type(error).__name__}: {error}'
            report['traceback']=''.join(traceback.format_exception(type(error),error,error.__traceback__))
        try:
            report['development_summary_ref']=str(write_development_summary(self.root))
        except Exception as exc:
            report.setdefault('recording_errors',[]).append(f'development_summary: {type(exc).__name__}: {exc}')
        target=self.root/'native_run.json'
        temporary=self.root/'native_run.pending.json'
        temporary.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False))
        temporary.replace(target)
        return target


def load_native_run(path):
    """只读索引并校验评分引用，诊断工具不改写原报告。"""
    path=Path(path)
    if path.is_dir():path=path/'native_run.json'
    result=json.loads(path.read_text())
    raw=result.get('native_result')
    if raw and _hash(Path(raw['path']))!=raw['sha256']:
        raise ValueError('原生评分报告hash不一致')
    result['manifest']=json.loads((path.parent/result['manifest_ref']).read_text())
    event_path=path.parent/result['events_ref']
    result['events']=[json.loads(row) for row in event_path.read_text().splitlines() if row.strip()] if event_path.exists() else []
    return result


def compare_stage_runs(baseline,current,*,stage,hypothesis,require_executed=False):
    """只比较同案例的阶段事实；单次改善不会自动证明输入的因果猜想。"""
    if not stage or not hypothesis.strip():raise ValueError('比较需要目标阶段与待验证猜想')
    reports=[load_native_run(path) for path in (baseline,current)]
    identities=[]
    for report in reports:
        manifest=report['manifest'];native=manifest.get('native_identity') or {}
        identities.append(dict(benchmark=manifest.get('benchmark'),embodiment=manifest.get('embodiment'),native_seed=native.get('native_seed')))
    known=all(all(value is not None for value in identity.values()) for identity in identities)
    case_match=identities[0]==identities[1] if known else None
    # 已知题目/机器人/seed不同，不能因另一字段缺失而掩盖不匹配。
    if any(a is not None and b is not None and a!=b for a,b in zip(identities[0].values(),identities[1].values())):case_match=False
    stages=[]
    for report in reports:
        events=[e for e in report['events'] if e.get('stage_id')==stage]
        outcomes=[e for e in events if e.get('kind')=='stage_outcome']
        last=outcomes[-1] if outcomes else {}
        execution=last.get('execution_status','unknown')
        # 单张actual_sample也可能只是静态观测；执行事实由阶段结果声明。
        executed=execution in {'executed','partial'}
        stages.append(dict(reached=bool(events),executed=executed,execution_status=execution,
            completed=last.get('success') is True,failure=last.get('failure'),
            task_success=report.get('task_success'),event_count=len(events),ik_policies=[
                e['ik_policy'] for e in events if 'ik_policy' in e]))
    before,after=stages
    failure_changed=None
    if isinstance(before['failure'],dict) and isinstance(after['failure'],dict):
        failure_changed=any(before['failure'].get(key)!=after['failure'].get(key) for key in ('code','source'))
    if case_match is not True:verdict='not_comparable'
    elif not before['reached'] or not after['reached'] or (require_executed and (not before['executed'] or not after['executed'])):verdict='not_tested'
    elif after['completed']:verdict='stage_passed'
    elif after['task_success'] is True and after['execution_status']=='partial':verdict='native_task_succeeded_stage_partial'
    elif after['task_success'] is True and (after['failure'] or {}).get('code')=='native_terminated':verdict='native_task_succeeded_stage_terminated'
    elif after['task_success'] is True and after['execution_status']=='executed':verdict='native_task_succeeded_stage_failed'
    elif after['failure'] is not None:verdict='failure_changed' if failure_changed else 'failure_persists'
    else:verdict='unknown'
    return dict(schema_version=1,stage=stage,hypothesis=hypothesis,require_executed=require_executed,
        baseline_ref=str(Path(baseline).resolve()),current_ref=str(Path(current).resolve()),
        case_match=case_match,identities=identities,identity_scope='benchmark/embodiment/native_seed；不替代场景、资产与有效参数核对',
        before=before,after=after,verdict=verdict,failure_changed=failure_changed,hypothesis_verdict='requires_observable_review',
        evaluation_scope='开发回归证据；被用于调参的旧验收案例不再作为独立验收样本')


def index_native_evaluation(output_dir):
    """原evaluator退出后建立引用索引；启动器退出码不推断任务得分。"""
    root=Path(output_dir).resolve()
    kinds={'_result.txt':'native_result','_metadata.json':'native_metadata',
           'eval_config.yaml':'resolved_native_config'}
    artifacts=[]
    for path in sorted((root/'native').rglob('*')):
        if not path.is_file():continue
        kind='video' if path.suffix=='.mp4' else kinds.get(path.name)
        if kind:artifacts.append(dict(kind=kind,path=str(path),sha256=_hash(path)))
    episodes=[str(p.resolve()) for p in sorted((root/'episodes').glob('*/native_run.json'))]
    links=[dict(report_ref=p,native_identity=json.loads((Path(p).parent/'manifest.json').read_text()).get('native_identity',{})) for p in episodes]
    report=dict(schema_version=1,task_success=None,scope='native_evaluation_index',episodes=episodes,episode_links=links,
                native_artifacts=artifacts,media_status='available' if any(a['kind']=='video' for a in artifacts) else 'unknown')
    path=root/'evaluation_index.json'
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False))
    return path


def read_planning_failures(path, *, stage=None):
    """流式读取单episode的规划失败；不加载实际采样，更不启动GPU。"""
    from .motion_diagnostics import summarize_motion_result
    root = Path(path)
    if root.is_file():
        root = root.parent
    if not (root/'events.jsonl').exists():
        return
    with (root / 'events.jsonl').open() as stream:
        for line in stream:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                import warnings
                warnings.warn(f'{root}/events.jsonl 含不完整或损坏记录；摘要覆盖不完整', RuntimeWarning)
                continue
            if event.get('kind') != 'planning' or (stage and event.get('stage_id') != stage):
                continue
            result = event.get('planning', event.get('result'))
            if not isinstance(result,dict):
                import warnings
                warnings.warn('规划事件缺少结果，归因未知',RuntimeWarning)
                continue
            if result.get('success'):
                continue
            digest = result.get('metrics', {}).get('scene_snapshot_sha256')
            snapshot_path = root / 'collision' / f'{digest}.json'
            snapshot = json.loads(snapshot_path.read_text()) if snapshot_path.exists() else None
            summary = summarize_motion_result(result, snapshot)
            yield dict(episode_id=event.get('episode_id'), stage_id=event.get('stage_id'),
                       state_id=event.get('state_id'), summary=summary.to_dict(),
                       markdown=summary.to_markdown(), planning=result,
                       target_world_tcp=event.get('target_world_tcp'),
                       events_ref=str(root / 'events.jsonl'),
                       snapshot_ref=str(snapshot_path) if snapshot is not None else None)


def write_development_summary(root):
    """结束时给下一轮Agent直接入口；只汇总已有事实，不影响评分。"""
    import shlex
    from .planning_performance import summarize_planning_run, performance_markdown
    root=Path(root)
    summary=summarize_planning_run(root)
    (root/'planning_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False))
    lines=[performance_markdown(summary),summary['scope']]
    events_path=root/'events.jsonl'
    policies={}
    if events_path.exists():
        for row in events_path.read_text().splitlines():
            event=json.loads(row)
            if 'ik_policy' in event:policies[event['stage_id']]=event['ik_policy']
    lines.append('阶段IK策略：'+(json.dumps(policies,ensure_ascii=False) if policies else '旧记录未显式记录；核对实际请求，不能由求解器默认容差推断已启用近似IK'))
    failures=list(read_planning_failures(root))
    for failure in failures[:3]:
        lines.extend([f"阶段 {failure['stage_id']} / state {failure['state_id']}",failure['markdown']])
        replay=failure['planning'].get('metrics',{}).get('replay_request_ref')
        if replay:
            lines.append('重规划：python tools/diagnostics/replay_motion_request.py --bundle '+shlex.quote(replay)+' --output <新目录>')
    if len(failures)>3:lines.append(f'另有{len(failures)-3}个规划失败，见详细检查命令')
    lines.append('检查：python tools/diagnostics/inspect_native_run.py --run '+shlex.quote(str(root)))
    lines.append('迭代先复测原失败：python tools/diagnostics/inspect_native_run.py --run '+shlex.quote(str(root))+
        ' --compare-with <原失败episode目录> --stage <目标阶段> --hypothesis <本轮猜想> --require-executed --format json；'
        '物理假设必须实际执行目标阶段，规划假设可不加--require-executed。先核对原失败，再小范围回归，最后新样本验收。')
    lines.append('规划结果不等于物理成功；任务评分见native_run.json引用的原评分。')
    target=root/'development_summary.md';target.write_text('\n\n'.join(lines)+'\n')
    return target
