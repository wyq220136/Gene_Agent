"""现场 Skill/Task 保存报告的只读索引；不依赖相机、SDK 或规划器。"""
from __future__ import annotations

import json
from pathlib import Path


_NAMES = ('task_result.json', 'door_subtask_result.json', 'pick_result.json',
          'grasp_result.json', 'insert_result.json', 'extract_result.json',
          'table_place_result.json', 'skill_result.json', 'transition_result.json')
_SCOPES = {'pick': {'pick_bowl', 'grasp_bowl'}, 'insert': {'insert_bowl'},
           'handle_pull': {'handle_pull'}, 'push_open': {'push_open'}, 'push_close': {'push_close'},
           'extract': {'extract_bowl'}, 'place': {'place_bowl_on_table'},
           'rotate_knob_0': {'rotate_knob_0'},
           'open_door': {'handle_pull', 'push_open', 'open'}, 'close_door': {'push_close', 'close'},
           'object_insert': {'pick_bowl', 'grasp_bowl', 'insert_bowl'},
           'object_extract': {'extract_bowl', 'place_bowl_on_table'},
           'door_cycle': {'handle_pull', 'push_open', 'push_close', 'open', 'close'}}


def _targets(value, path):
    if isinstance(value, dict):
        for key, child in value.items():
            label = f'{path}/{key}'
            if key in {'keyposes_world_tcp', 'targets_world_tcp'} and isinstance(child, dict):
                yield from ((f'{label}/{name}', pose) for name, pose in child.items())
            elif key == 'target' and isinstance(child, dict) and child.get('frame') == 'world_tcp':
                if child.get('matrix') is not None:
                    yield label, child['matrix']
            elif key != 'replay_samples':
                yield from _targets(child, label)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            if isinstance(child, (dict, list)):
                yield from _targets(child, f'{path}/{index}')


def load_realworld_record(root, *, skill=None):
    root = Path(root).resolve()
    reports = {}
    for path in sorted(root.rglob('*_result.json')):
        if path.name not in _NAMES:
            continue
        row = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(row, dict) or not isinstance(row.get('execute'), bool):
            continue
        label = (row.get('skill_id') or row.get('subtask') or row.get('task')
                 or row.get('part') or row.get('stage_id') or path.stem)
        if label is None:
            continue
        if skill in _SCOPES and label not in _SCOPES[skill]:
            continue
        if skill == 'pick' and 'extract' in str(path.relative_to(root)):
            continue
        reports[str(path.relative_to(root))] = row
    if not reports:
        return None
    targets = {}
    for name, row in reports.items():
        targets.update(_targets(row, name))
    for path in sorted(root.rglob('*.json')):
        if path.name == 'targets.json' or path.name.endswith(('_targets.json', '_plan.json')):
            targets.update(_targets(json.loads(path.read_text(encoding='utf-8')), str(path.relative_to(root))))
    scenes, missing = {}, []
    for path in sorted(root.rglob('observed_scene_provenance.json')):
        provenance = json.loads(path.read_text(encoding='utf-8'))
        directory = path.parent
        xml = next((directory / relative for relative in (
            'scene/scene.xml', 'source_scene/scene.xml', 'scene.xml')
            if (directory / relative).is_file()), None)
        snapshot_path = directory / 'collision/collision_snapshot.json'
        if xml is None or not snapshot_path.is_file():
            missing.append(f'{path.relative_to(root)} 缺少 scene.xml 或 collision/collision_snapshot.json')
            continue
        snapshot = json.loads(snapshot_path.read_text(encoding='utf-8'))
        if (provenance.get('physics_stepped') is not False
                or provenance.get('collision_snapshot_sha256') != snapshot.get('snapshot_sha256')
                or not snapshot.get('qpos_snapshot')):
            missing.append(f'{path.relative_to(root)} 缺少一致来源与完整 qpos_snapshot')
            continue
        scenes[str(path.relative_to(root))] = {'xml_path': str(xml),
            'snapshot': snapshot, 'provenance': provenance}
    if not scenes:
        missing.append('未保存可用的静态 scene.xml、完整 qpos_snapshot 和 observed_scene_provenance.json；只能查看报告证据')
    missing.append('SDK 实测端点不含连续完整场景采样；没有可播放的 actual 轨迹')
    return {'root': str(root), 'reports': reports, 'scenes': scenes, 'targets': targets, 'missing': missing}
