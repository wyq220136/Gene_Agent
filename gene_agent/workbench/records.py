"""小范围读取运行报告；缺失信息保持未知，不重算任务成功。"""
import json
import copy
from functools import lru_cache
from pathlib import Path
from .catalog import safe_path

REPORTS=('summary.json','task_result.json','result.json','execution_result.json','report.json','live/plan.json','live/manifest.json')

def read_json(path):
    with Path(path).open(encoding='utf-8') as stream:return json.load(stream)


def describe_run(root, value):
    root=Path(root).resolve();run=safe_path(root,value,('runs',),exists=True)
    watched=[run/name for name in REPORTS]+[run/name for name in ('','video','videos','previews','evidence','screenshots','renders')]
    signature=tuple((str(p),p.stat().st_mtime_ns,p.stat().st_size) for p in watched if p.exists())
    return copy.deepcopy(_describe_cached(str(root),str(run),signature))


@lru_cache(maxsize=64)
def _describe_cached(root,value,signature):
    # 只缓存精简摘要；不在内存保留带完整轨迹的大报告。
    root=Path(root).resolve();run=safe_path(root,value,('runs',),exists=True)
    row={'path':str(run.relative_to(root)),'name':run.name,'success':None,'status':'unknown',
         'failure_stage':None,'planner_backend':None,'evidence_mode':'未记录','reports':[],
         'errors':[],'stages':[],'media':[],'updated_at':run.stat().st_mtime,
         'live_available':(run/'live/manifest.json').is_file()}
    for name in REPORTS:
        if name.startswith('live/') and row['reports']:continue
        path=run/name
        if not path.is_file():continue
        try:
            data=read_json(path)
            if not isinstance(data,dict):raise ValueError('报告必须为 JSON 对象')
        except (ValueError,OSError) as exc:
            row['errors'].append(f'{name}: {exc}');continue
        row['reports'].append(str(path.relative_to(root)))
        # 第一份有效主报告优先；仅读取显式任务成功字段。
        if len(row['reports'])==1:
            row.update(status=data.get('status','unknown'),
                success=data.get('goal_success',data.get('task_success',data.get('success'))),
                failure_stage=data.get('failure_stage',data.get('failure_subtask')),
                planner_backend=data.get('planner_backend',data.get('motion_backend')),
                evidence_mode=data.get('evidence_mode',data.get('source_format','未记录')),
                task=data.get('task'),failure_type=data.get('failure_type'),exception=data.get('exception'),
                planning_success=data.get('planning_success'),record_type=data.get('record_type'),
                simulation_assistance=data.get('simulation_assistance'))
            if data.get('simulation_assistance'):
                row['evidence_mode']='仿真辅助：'+str(data['simulation_assistance'])
            placement=data.get('placement') or {}
            row['parameters']={'seed':(placement.get('sample') or {}).get('seed',placement.get('seed',data.get('seed'))) if isinstance(placement,dict) else data.get('seed'),
                'grasp_tilt_deg':data.get('grasp_tilt_deg'),'scene_sha256':data.get('scene_xml_sha256')}
            stages=data.get('stages',data.get('subtasks',data.get('skills',{})))
            if isinstance(stages,dict):stages=[dict(v,name=k) if isinstance(v,dict) else {'name':k,'value':v} for k,v in stages.items()]
            if isinstance(stages,list):
                for i,s in enumerate(stages[:120]):
                    if not isinstance(s,dict):continue
                    row['stages'].append({k:s[k] for k in ('name','stage_id','status','success','failure_type','failure_stage','message','path_kind','planning_success','execution_status') if k in s})
    if not row['reports'] and not row['errors']:row['errors'].append('该目录尚无支持的运行报告')
    # 有限层级媒体查询，不递归扫描模型和逐帧数据。
    media=[]
    for folder in ('','video','videos','previews','evidence','screenshots','renders'):
        base=run/folder
        if not base.is_dir():continue
        for p in sorted(base.iterdir()):
            if p.is_file() and p.suffix.lower() in {'.png','.jpg','.jpeg','.mp4'}:
                resolved=safe_path(root,p,('runs',))
                media.append({'path':str(resolved.relative_to(root)),'type':'video' if p.suffix=='.mp4' else 'image','name':p.name})
    row['media']=media[:40]
    if not isinstance(row['success'],bool):row['success']=None
    return row


def evidence_context(root, run):
    row=describe_run(root,run)
    docs=[]
    for name in row['reports']:
        with (Path(root)/name).open(encoding='utf-8') as stream:text=stream.read(35001)
        docs.append({'path':name,'text':text[:35000],'truncated':len(text)>35000})
    run_path=Path(root)/row['path']
    for path in (run_path/'task_exception.txt',run_path.parent/'output.log'):
        if path.is_file():
            safe_path(root,path,('runs',))
            with path.open('rb') as stream:
                stream.seek(max(0,path.stat().st_size-10000));text=stream.read().decode('utf-8',errors='replace')
            docs.append({'path':str(path.relative_to(root)),'text':text,'truncated':path.stat().st_size>10000})
    return {'summary':row,'reports':docs}
