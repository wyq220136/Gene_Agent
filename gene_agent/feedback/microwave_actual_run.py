"""微波炉原始 MuJoCo 记录的只读索引；只收集 Runtime 的完整 qpos 采样。"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

from gene_agent.feedback.skill_run import SkillRunView, StageView


_REPORT_NAMES = ('summary.json', 'report.json', 'result.json', 'insert_result.json')
_SKILL_FILTERS = {
    'handle_pull': ('handle_pull',),
    'push_open': ('push_open',),
    'push_close': ('push_close',),
    'pick': ('pick_insert',),
    'insert': ('physical_insert', 'continuation'),
    'place': ('table_place',),
    'extract': ('extract_grasp', 'extract_retreat'),
    'rotate_knob_0': ('knob_execution',),
    'open_door': ('open_first', 'open_second'),
    'close_door': ('close_first', 'close_second'),
    'object_insert': ('pick_insert',),
    'object_extract': ('extract_grasp', 'extract_retreat', 'table_place'),
    'door_cycle': ('open_first', 'close_first', 'open_second', 'close_second'),
}


def _read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def _reports(root):
    return [(root / name, _read(root / name)) for name in _REPORT_NAMES
            if (root / name).is_file()]


def is_microwave_actual_run(root):
    """只识别已有物理/混合微波炉格式；规划目录继续走原规划 loader。"""
    root = Path(root).resolve()
    for path, report in _reports(root):
        if not isinstance(report, dict) or isinstance(report.get('execute'), bool):
            continue
        if (report.get('task') == 'microwave_task1'
                or report.get('source_format', '').startswith('mujoco_physical')
                or report.get('source_format', '').startswith('mujoco_robot_execution')
                or report.get('mode') == 'physical_assisted'
                or (report.get('mode') == 'physical'
                    and str(report.get('skill', '')).startswith('extract_bowl'))
                or report.get('continuation')
                or path.name == 'insert_result.json'):
            return True
    return False


def _model(root, reports):
    for _, report in reports:
        raw = report.get('scene_ref') if isinstance(report, dict) else None
        if raw:
            path = Path(raw)
            path = path if path.is_absolute() else root / path
            if path.is_file():
                return os.path.relpath(path, root)
    # 子 Skill 与完整 Task1 共享保存模型；只沿本轮 run 的父目录找模型。
    for directory in (root, *root.parents[:3]):
        for relative in ('model/scene.xml', 'source_scene/scene.xml', 'scene/source_scene/scene.xml'):
            path = directory / relative
            if path.is_file():
                return os.path.relpath(path, root)
    raise FileNotFoundError('实际微波炉记录缺少保存的 MuJoCo scene.xml')


def _sample_groups(value, path=()):
    if isinstance(value, dict):
        samples = value.get('replay_samples')
        if samples:
            if not isinstance(samples, list):
                raise ValueError('实际 replay_samples 必须是数组')
            previous = -math.inf
            for row in samples:
                if (not isinstance(row, dict) or not row.get('qpos')
                        or 'sim_time_s' not in row
                        or not all(math.isfinite(float(v)) for v in row['qpos'])):
                    raise ValueError('实际采样缺少有限完整 qpos 或 sim_time_s')
                time_s = float(row['sim_time_s'])
                if not math.isfinite(time_s) or time_s < previous:
                    raise ValueError('实际采样时间无效或倒退')
                previous = time_s
            yield path, samples, value.get('success')
        for key, child in value.items():
            if key != 'replay_samples' and not key.startswith('planned'):
                yield from _sample_groups(child, (*path, key))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            label = child.get('stage', child.get('name', index)) if isinstance(child, dict) else index
            yield from _sample_groups(child, (*path, str(label)))


def _matches_scope(root, report, skill, label):
    if skill is None:
        return True
    if skill not in _SKILL_FILTERS:
        raise ValueError(f'未知实际微波炉 Skill：{skill}')
    if skill == 'pick' and any(token in label for token in ('continuation', 'physical_insert')):
        return False
    context = root.name + '/' + label
    if any(token in context for token in _SKILL_FILTERS[skill]):
        return True
    # 独立运行的文件夹可以任意命名；范围以报告声明为准。
    declared = str(report.get('selection') or report.get('skill') or '')
    declared_scope = {
        'microwave_door_open': 'open_door', 'microwave_door_close': 'close_door',
        'place_extracted_bowl_on_table': 'place',
    }.get(declared)
    if declared.startswith('extract_bowl'):
        declared_scope = 'extract'
    elif declared.startswith('rotate_knob'):
        declared_scope = 'rotate_knob_0'
    elif report.get('mode') == 'synchronous_pick_skill':
        declared_scope = 'pick'
    elif (root / 'insert_result.json').is_file():
        declared_scope = 'insert'
    return declared_scope == skill or (
        skill == 'object_insert' and declared_scope == 'pick') or (
        skill == 'object_extract' and declared_scope in {'extract', 'place'}) or (
        skill == 'door_cycle' and declared_scope in {'open_door', 'close_door'})


def load_microwave_actual_run(root, *, skill=None):
    """返回 view 与各阶段原始采样；不写转换文件、不补造缺失采样。"""
    root = Path(root).resolve()
    if not is_microwave_actual_run(root):
        return None
    reports = _reports(root)
    primary = reports[0][1]
    model_ref = _model(root, reports)
    files = [path for path in root.rglob('*.json') if (
        path.name in {*_REPORT_NAMES, 'stages.json'}
        or path.parent.name in {'executions', 'stages'})]
    # 汇总报告可能重复内嵌执行结果。专用 execution/stages 记录覆盖同一时刻
    # 的汇总副本；stages.json 最后读取，保留旋钮显式运动学提交的终态。
    files.sort(key=lambda p: (p.name == 'stages.json', p.parent.name in {'executions', 'stages'}, str(p)))
    indexed = {}
    for path in files:
        value = _read(path)
        for pointer, samples, success in _sample_groups(value):
            relative = str(path.relative_to(root))
            label = '/'.join((str(path.relative_to(root).with_suffix('')), *map(str, pointer)))
            if not _matches_scope(root, primary, skill, label):
                continue
            for sample in samples:
                indexed[round(float(sample['sim_time_s']), 9)] = (label, relative, sample, success)
    # execute 内部因碰撞或异常提前退出时，阶段返回值可能还未写入 JSON。
    # 在线帧流是在每个真实 MuJoCo 采样点追加的，因此只补入结构化报告
    # 尚未覆盖的尾部，保留“执行到实际停止点”的视频证据。
    from gene_agent.feedback.live_record import JsonlTail
    cutoff = max(indexed, default=-float('inf')) + 1e-8
    tail_path = root / 'live/frames.jsonl'
    tail = [row for row in JsonlTail(tail_path).read()
            if float(row['sim_time_s']) > cutoff]
    if tail:
        import mujoco
        original = mujoco.MjModel.from_xml_path(str(root / model_ref))
        for name in {row['model'] for row in tail}:
            if Path(name).name != name:
                raise ValueError('实际尾部冻结模型名称无效')
            frozen = mujoco.MjModel.from_binary_path(str(root / 'live' / name))
            if (original.nq != frozen.nq or original.nv != frozen.nv
                    or [original.joint(i).name for i in range(original.njnt)]
                    != [frozen.joint(i).name for i in range(frozen.njnt)]):
                raise ValueError('实际尾部模型关节与保存场景不一致')
        for _, samples, _ in _sample_groups({'replay_samples': tail}):
            for sample in samples:
                if len(sample['qpos']) != original.nq:
                    raise ValueError('实际尾部 qpos 维度无效')
                indexed[round(float(sample['sim_time_s']), 9)] = (
                    'live_tail/unreported_stage', 'live/frames.jsonl', sample, None)
    # 一些旧 release/settle 只保存 NPZ 边界，仍可精确显示开爪和落稳。
    # 这里只读积分存档中的 qpos/time；不恢复接触力，也不积分推进场景。
    boundaries = [path for path in root.rglob('*.npz') if path.parent.name == 'states']
    if boundaries:
        import mujoco
        import numpy as np
        model = mujoco.MjModel.from_xml_path(str(root / model_ref))
        data = mujoco.MjData(model)
        for path in boundaries:
            if not _matches_scope(root, primary, skill, str(path.relative_to(root))):
                continue
            with np.load(path, allow_pickle=False) as archive:
                spec = int(archive['spec'])
                if not np.array_equal(archive['dimensions'][:2], [model.nq, model.nv]):
                    raise ValueError('实际状态边界维度与保存模型不一致')
                if len(archive['state']) != mujoco.mj_stateSize(model, spec):
                    raise ValueError('实际状态边界积分格式与保存模型不一致')
                mujoco.mj_setState(model, data, archive['state'], spec)
                sample = {'qpos': data.qpos.tolist(), 'sim_time_s': float(data.time),
                    'state_id': hashlib.sha256(archive['state'].tobytes()).hexdigest(),
                    'sample_quality': 'saved_actual_boundary', 'objects': {}, 'robots': {}}
            key = round(float(data.time), 9)
            # 同时刻已有 Runtime 采样时保留其机器人、物体和 attachment 元数据。
            if key not in indexed:
                ref = str(path.relative_to(root))
                indexed[key] = (str(path.relative_to(root).with_suffix('')), ref, sample, None)
    rows = [indexed[key] for key in sorted(indexed)]
    if not rows:
        raise ValueError('记录缺少所选范围的实际 MuJoCo 完整采样')
    groups = []
    for label, ref, sample, success in rows:
        if not groups or groups[-1][0] != label:
            groups.append((label, ref, [], success))
        groups[-1][2].append(sample)
    stages, samples_by_stage = [], {}
    for index, (label, ref, samples, success) in enumerate(groups):
        stage_id = f'{index:02d}_{label}'
        stages.append(StageView(stage_id, label, None, 'not_recorded',
            'failed' if success is False else 'unknown' if success is None else 'completed',
            None, None, None, None,
            samples[0].get('state_id'), samples[-1].get('state_id'),
            samples_ref=ref, actual_sample_format='complete_actual_state'))
        samples_by_stage[stage_id] = samples
    physical = primary.get('fully_physical_success', primary.get('physical_pick_success'))
    selection = skill or primary.get('skill') or primary.get('selection') or primary.get('task') or 'microwave'
    target = 'microwave_door' if 'door' in selection else 'knob_0' if 'knob' in selection else 'bowl'
    view = SkillRunView(selection, target, None, 'microwave_actual_record', None,
                        tuple(stages), model_ref=model_ref, physical_success=physical)
    return view, samples_by_stage
