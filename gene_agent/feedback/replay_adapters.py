"""桌面任务原生记录接入共享实际回放；失败前缀同样可读。"""
import hashlib
import json
import os
from pathlib import Path

from gene_agent.feedback.skill_run import SkillRunView, StageView


TASKS = {'cup_pour_balls_into_bowl_and_return', 'scoop_balls_into_other_bowl',
         'sweep_cube_into_dustpan'}


def load_tabletop_actual_run(root, *, skill=None):
    """只索引原始 replay_samples；不要求成功、不补帧、不写转换产物。"""
    root = Path(root).resolve()
    primary = next((p for p in (root / 'task_result.json', root / 'result.json')
                    if p.is_file()), None)
    if primary is None:
        return None
    report = json.loads(primary.read_text(encoding='utf-8'))
    # 独立 Pick 的 task 是原场景名（如 scoop_orange），不能只认完整 Task 名。
    standalone_pick = (report.get('execution_backend') == 'mujoco_physical'
                       and bool(report.get('skill')) and bool(report.get('pick_result_path'))
                       and (root / 'pick/result.json').is_file())
    if report.get('task') not in TASKS and not standalone_pick:
        return None
    if skill is not None:
        raise ValueError('桌面原生回放当前支持完整实际前缀，请省略 --skill')
    model = Path(report['scene_xml_path'])
    if not model.is_absolute():
        model = root / model
    if hashlib.sha256(model.read_bytes()).hexdigest() != report['scene_xml_sha256']:
        raise ValueError('实际回放场景哈希与运行记录不一致')
    # 复用既有完整 qpos/时间校验；该收集器主动排除 planned_* 分支。
    from gene_agent.feedback.microwave_actual_run import _sample_groups
    indexed = {}
    pick_report = {}
    for path in (root / 'pick/result.json', primary):
        if not path.is_file():
            continue
        value = json.loads(path.read_text(encoding='utf-8'))
        if path == root / 'pick/result.json':
            pick_report = value
        for pointer, samples, success in _sample_groups(value.get('stages', {})):
            label = '/'.join((str(path.relative_to(root).with_suffix('')), *pointer))
            for sample in samples:
                indexed[round(float(sample['sim_time_s']), 9)] = (label, path, sample, success)
    # 中断发生在 execute 内部时，该阶段的返回值还没落盘；只追加已保存的
    # 实际流尾部，标签明确表示阶段结论未知，不把缺失阶段补成成功。
    from gene_agent.feedback.live_record import JsonlTail
    cutoff = max(indexed, default=-float('inf')) + 1e-8
    tail = [row for row in JsonlTail(root / 'live/frames.jsonl').read()
            if float(row['sim_time_s']) > cutoff]
    if tail:
        import mujoco
        original = mujoco.MjModel.from_xml_path(str(model))
        for name in {row['model'] for row in tail}:
            if Path(name).name != name:
                raise ValueError('实际尾部冻结模型名称无效')
            frozen = mujoco.MjModel.from_binary_path(str(root / 'live' / name))
            if (original.nq != frozen.nq or original.nv != frozen.nv
                    or [original.joint(i).name for i in range(original.njnt)]
                    != [frozen.joint(i).name for i in range(frozen.njnt)]):
                raise ValueError('实际尾部模型关节与场景不一致')
        for _, samples, _ in _sample_groups({'replay_samples': tail}):
            for sample in samples:
                if len(sample['qpos']) != original.nq:
                    raise ValueError('实际尾部 qpos 维度无效')
                indexed[round(float(sample['sim_time_s']), 9)] = (
                    'live_tail/unreported_stage', root / 'live/frames.jsonl', sample, None)
    if not indexed:
        raise ValueError('桌面记录缺少实际 MuJoCo 完整采样')
    groups = []
    for key in sorted(indexed):
        label, path, sample, success = indexed[key]
        if not groups or groups[-1][0] != label:
            groups.append((label, path, [], success))
        groups[-1][2].append(sample)
    stages, recorded = [], {}
    for index, (label, path, samples, success) in enumerate(groups):
        stage_id = f'{index:02d}_{label}'
        stages.append(StageView(stage_id, label, None, 'not_recorded',
            'failed' if success is False else 'unknown' if success is None else 'completed', None, None, None, None,
            samples[0].get('state_id'), samples[-1].get('state_id'),
            samples_ref=str(path.relative_to(root)), actual_sample_format='complete_actual_state'))
        recorded[stage_id] = samples
    instance = ((pick_report.get('instance_id') if standalone_pick else None)
                or report.get('cup_name') or report.get('tool_name') or 'brush')
    view = SkillRunView(report['task'], instance, None, 'tabletop_actual_record', None,
        tuple(stages), model_ref=os.path.relpath(model, root),
        physical_success=(pick_report.get('physical_pick_success') if standalone_pick
                          else report.get('fully_physical_success')))
    return view, recorded
