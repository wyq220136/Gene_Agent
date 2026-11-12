"""已接入任务的场景输入适配；共用编辑/采样，保留各任务原生配置。"""
from copy import deepcopy
import json
from pathlib import Path
from gene_agent.scene.table import table_bounds_xy

from .catalog import TASKS,safe_path
from .records import read_json


def load_input(root,task,path=None):
    info=TASKS[task]
    source=safe_path(root,path or info.get('scene') or info['config'],exists=True)
    if task=='microwave':
        config=source/'task.json' if source.is_dir() else source
        data=read_json(config)
        if not all(n+'_pose_world' in data for n in ('microwave','bowl')):
            raise ValueError('不是微波炉场景输入')
        return source,data
    if not source.is_dir():raise ValueError('需要清扫场景目录')
    data=read_json(source/'scene.json')
    if task in {'sweep_long','cup_pour','spoon_scoop'}:
        if 'nominal_object_poses' not in data:raise ValueError('不是该 benchmark 的重建场景输入')
    elif 'objects' not in data:raise ValueError('不是清扫场景输入')
    return source,data


def suggested_rules(task,data):
    if task=='microwave':
        return {'schema_version':1,'source':'codex','note':'保守开发起点，尚未验证分布任务成功率；炉体固定、碗小范围随机。',
            'objects':{
                'microwave':{'mode':'fixed','frame':'world','translation_half_range_m':[.015,.015,0],'yaw_half_range_deg':3},
                'bowl':{'mode':'random','frame':'world','translation_half_range_m':[.02,.02,0],'yaw_half_range_deg':8}},
            'constraints':{'bounds_xy_m':table_bounds_xy(data['table']) if data.get('table') else [.05,.95,-.05,.85],
                           'min_distances':[{'objects':['microwave','bowl'],'distance_m':.22}]},
            'max_attempts':100}
    cfg=data.get('randomization') or {'ranges_xyyaw_rad':{},'table_bounds_xy_m':[-2,2,-2,2],
                                      'min_clearance_m':.01,'max_attempts':100}
    objects={}
    for name in data['objects']:
        limits=cfg['ranges_xyyaw_rad'].get(name,[0,0,0])
        objects[name]={'mode':'random' if any(limits) else 'fixed','frame':'world',
            'translation_half_range_m':[float(limits[0]),float(limits[1]),0],
            'yaw_half_range_deg':float(limits[2])*180/3.141592653589793}
    return {'schema_version':1,'source':'codex','note':'复用已声明清扫分布，未按 IK 静默筛选样本。',
        'objects':objects,'constraints':{'bounds_xy_m':cfg['table_bounds_xy_m'],
                                       'min_clearance_m':cfg['min_clearance_m']},
        'max_attempts':cfg['max_attempts']}


def description(root,task,path=None):
    root=Path(root)
    source,data=load_input(root,task,path)
    objects=[]
    if task in {'sweep_long','cup_pour','spoon_scoop'}:
        for name,pose in data['nominal_object_poses'].items():
            objects.append(dict(name=name,pose=pose,mesh=None,concept=None))
        return dict(task=task,editable=False,viewable=True,path=str(source.relative_to(root)),objects=objects,
            placement_rules={},suggested_rules={},rules_saved=True,concept_editable=False,
            message='可查看 benchmark 的正式初态；网页拖拽编辑和分布图尚未接入此任务。场景由原生构建入口生成。')
    rows=(data['objects'] if task=='sweep' else
          {n:{'T_world_object':data[n+'_pose_world']} for n in ('microwave','bowl')})
    for name,row in rows.items():
        asset_name='cube' if name.startswith('cube_') else name
        mesh=source/'assets'/f'{asset_name}.obj' if source.is_dir() else None
        concept=source/'concepts'/f'{asset_name}.json' if source.is_dir() else None
        objects.append({'name':name,'pose':row['T_world_object'],
            'mesh':str(mesh.relative_to(root)) if mesh and mesh.exists() else None,
            'concept':str(concept.relative_to(root)) if concept and concept.exists() else None})
    from gene_agent.scene.table import validate_table
    defaults=read_json(Path(__file__).resolve().parents[2]/'config/scenes/task_1.json')['table'] if task=='microwave' else data.get('table')
    return {'task':task,'editable':True,'path':str(source.relative_to(root)),'objects':objects,
            'table':validate_table(data.get('table',{}),defaults) if defaults else None,
            'placement_rules':deepcopy(data.get('placement_rules') or suggested_rules(task,data)),
            'suggested_rules':suggested_rules(task,data),'rules_saved':'placement_rules' in data,
            'concept_editable':task=='sweep'}
