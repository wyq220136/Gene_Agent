"""浏览器与 CLI 共用的本地工作台操作。"""
import hashlib
import json
from pathlib import Path
import shutil
import socket
import sys
import threading
import time
import uuid
import xml.etree.ElementTree as ET
import urllib.request

from .catalog import TASKS,build_command,safe_path,task_info,video_command
from .jobs import JobManager
from .records import describe_run,read_json
from .assistant import prepare_context
from .codex_client import resolve_codex_home
from .attachments import context_manifest
from .scene_inputs import description,load_input,suggested_rules
from gene_agent.scene.table import validate_table,table_bounds_xy
from gene_agent.scene.placement import validate_rules
from gene_agent.knowledge.geometry import validate_pose


class Workbench:
    def __init__(self,root,python=None):
        self.root=Path(root).resolve();self.python=python or sys.executable
        self.jobs=JobManager(self.root);self.lock=threading.RLock()
        self.state_path=self.root/'runs/workbench/library.json'
        self.library=read_json(self.state_path) if self.state_path.exists() else {'runs':[],'drafts':[]}
        self.base_url=None

    def _save(self):
        tmp=self.state_path.with_suffix('.tmp');tmp.write_text(json.dumps(self.library,ensure_ascii=False,indent=2));tmp.replace(self.state_path)

    def bootstrap(self):
        from gene_agent.tasks.planning.knowledge import KNOWLEDGE, SKILLS
        samples=[]
        for task,path in (
            ('microwave','runs/shared_operation_refactor/20261029/remaining/full_task'),
            ('sweep','runs/sweep_cube/verified_contact_full_36_45_20261019/seed_36')):
            if (self.root/path).is_dir():samples.append({'task':task,'path':path})
        return {'tasks':list(TASKS.values()),'drafts':self.library['drafts'],'scenes':self.library.get('scenes',[]),'python':str(self.python),
                'assistant_available':bool(shutil.which('codex')),'assistant_sessions':True,'table_editing':True,
                'codex_home':resolve_codex_home(must_exist=False),'samples':samples,
                'visual_previews':{'knowledge':{k:KNOWLEDGE.get(k,{}) for k in TASKS},'skills':{k:SKILLS.get(k,{s['id']:s['name'] for s in v['skills']}) for k,v in TASKS.items()}}}

    def runs(self):
        paths=list(self.library['runs'])
        jobs=self.jobs.list()
        for j in jobs:
            value=j.get('inputs',{}).get('output_run')
            if value and (self.root/value).is_dir():paths.append({'path':value,'task':j['inputs']['task'],
                'job_state':j['state'],'job_id':j['id'],'options':j['inputs'].get('options',{})})
        rows=[];seen=set()
        for item in reversed(paths):
            if item['path'] in seen:continue
            seen.add(item['path'])
            try:
                row={**describe_run(self.root,item['path']),'task_id':item['task']}
                if item.get('job_id'):
                    row['name']=TASKS[item['task']]['name']+' · '+item['job_id'][:15]
                    row['job_id']=item['job_id'];row['options']=item.get('options',{})
                    if item['job_state'] in {'queued','running','canceled','interrupted'} and not row['reports']:
                        row['status']=item['job_state'];row['errors']=[]
                for job in jobs:
                    inputs=job.get('inputs',{})
                    source=inputs.get('output_run') if job['kind']=='run_video' else inputs.get('options',{}).get('run')
                    if job['kind'] not in {'video','run_video'} or source!=row['path']:continue
                    video=self.jobs.directory(job['id'])/('export' if job['kind']=='run_video' else 'run')/'video.mp4'
                    last_step=job.get('steps',[])[-1:]
                    if video.is_file() and last_step and last_step[0].get('returncode')==0:
                        row['media'].append({'path':str(video.relative_to(self.root)),'type':'video','name':'导出视频 '+job['id'][:15]})
                rows.append(row)
            except (ValueError,OSError) as exc:rows.append({**item,'errors':[str(exc)],'success':None,'status':'unavailable','task_id':item['task']})
        return sorted(rows,key=lambda r:r.get('updated_at',0),reverse=True)

    def import_run(self,request):
        task_info(request['task']);row=describe_run(self.root,request['run'])
        with self.lock:
            item={'path':row['path'],'task':request['task']}
            if item not in self.library['runs']:self.library['runs'].append(item)
            self._save()
        return row

    @staticmethod
    def port():
        with socket.socket() as sock:
            sock.bind(('127.0.0.1',0));return sock.getsockname()[1]

    def submit(self,request):
        task=request.get('task');task_info(task)
        action=request.get('action');options=dict(request.get('options') or {})
        if task_info(task)['editable_scene'] and action in {'run','run_video','distribution','visual_plan','run_preview'} and options.get('randomize',action=='distribution'):
            current=self.scene(task,options.get('scene'))
            if not current['rules_saved']:
                current=self.save_scene({'task':task,'scene':current['path'],'placement_rules':current['placement_rules']})
                options['scene']=current['path']
        if action=='sample_viewer':
            preview=self.jobs.get(options.get('distribution_job',''))
            if preview['kind']!='distribution' or preview['inputs']['task']!=task:
                raise ValueError('只能查看当前任务的已有随机样本')
            from .catalog import integer
            seed=integer(options.get('seed'),0,2**31-1)
            samples=self.jobs.directory(preview['id'])/'run'
            if not (samples/f'seed_{seed}'/'view_meta.json').is_file():
                raise ValueError('该旧预览没有保存三维快照，请点击“预览 6 种随机摆位”重新生成')
            options['sample_dir']=str(samples.relative_to(self.root))
        jid=time.strftime('%Y%m%d_%H%M%S_')+uuid.uuid4().hex[:8]
        folder=self.jobs.directory(jid);output=folder/'run'
        inputs={'task':task,'action':action,'options':options}
        background=action in {'viewer','skill','scene_viewer','sample_viewer','concept','knowledge','ask','knowledge_preview','visual_viewer','live_viewer','run_preview'}
        if action=='ask':
            model=str(options.get('model','gpt-6-astra'))
            if len(model)>100 or not all(c.isalnum() or c in '-._' for c in model):raise ValueError('模型名称格式错误')
            mode=options.get('thread_mode','resume')
            if mode not in {'resume','fork','new'}:raise ValueError('无效会话接入方式')
            if mode!='new' and not options.get('thread_id'):raise ValueError('请先选择本地对话，或选择新建对话')
            if options.get('effort') not in (None,'','low','medium','high','xhigh','max','ultra','minimal','none'):
                raise ValueError('无效思考强度')
            context=prepare_context(self.root,{**options,'task':task})
            options['codex_home']=resolve_codex_home(options.get('codex_home'))
            folder.mkdir(parents=True)
            (folder/'context.json').write_text(json.dumps(context,ensure_ascii=False,indent=2))
            inputs['context']=context_manifest(context)
            (folder/'request.json').write_text(json.dumps({**options,'task':task},ensure_ascii=False))
            commands=[[str(self.python),'-m','gene_agent.workbench.assistant','--root',str(self.root),'--request',str(folder/'request.json')]]
        else:
            if background:
                options['port']=self.port();inputs['viewer_url']=f"http://127.0.0.1:{options['port']}"
            cmd=build_command(self.root,self.python,task,action,options,output)
            if action=='scene_viewer' and task_info(task)['editable_scene']:
                if not self.base_url:raise ValueError('场景编辑器需要由工作台 HTTP 服务启动')
                cmd+=['--edit','--workbench-url',self.base_url]
            commands=[cmd]
            if action in {'run','run_video','visual_plan'}:
                inputs['output_run']=str(output.relative_to(self.root))
            folder.mkdir(parents=True,exist_ok=True)
            if action=='concept':output.mkdir(exist_ok=True)
        if action=='run_video':
            commands.append(video_command(self.root,self.python,task,output,folder/'export'))
        sources={}
        for command in commands:
            if '-m' in command:
                module=command[command.index('-m')+1]
                if module.startswith('gene_agent.'):
                    path=self.root/(module.replace('.','/')+'.py')
                    if path.is_file():sources[str(path.relative_to(self.root))]=hashlib.sha256(path.read_bytes()).hexdigest()
            for part in command:
                p=Path(part)
                if p.is_absolute() and p.suffix=='.py' and p.is_file():sources[str(p.relative_to(self.root))]=hashlib.sha256(p.read_bytes()).hexdigest()
        inputs['entry_hashes']=sources
        return self.jobs.submit(action,inputs,commands,background=background,
                                timeout=300 if action=='ask' else (86400 if background else 7200),job_id=jid)

    def job(self,jid):
        row=self.jobs.get(jid);row['log']=self.jobs.log(jid)
        folder=self.jobs.directory(jid)
        if (folder/'answer.txt').is_file():row['answer']=(folder/'answer.txt').read_text()
        if (folder/'session.json').is_file():row['session']=read_json(folder/'session.json')
        if (folder/'assistant_error.json').is_file():row['assistant_error']=read_json(folder/'assistant_error.json')
        media=[]
        for base in (folder/'run',folder/'export'):
            if base.is_dir():
                for p in base.iterdir():
                    if p.is_file() and p.suffix in {'.png','.mp4'}:media.append(str(p.relative_to(self.root)))
        row['media']=media
        draft=folder/'run/draft_state.json'
        if draft.is_file():row['scene_draft']=read_json(draft)
        distribution=folder/'run/distribution.json'
        row['samples']=[{k:s[k] for k in ('seed','valid','error') if k in s}
                        for s in read_json(distribution).get('samples',[])] if distribution.exists() else []
        row['saved_concept']=str((folder/'run/concept.json').relative_to(self.root)) if (folder/'run/concept.json').exists() else None
        row['saved_scene']=read_json(folder/'run/saved_scene.json') if (folder/'run/saved_scene.json').exists() else None
        row['viewer_ready']=False
        if row['state']=='running' and row.get('inputs',{}).get('viewer_url'):
            try:
                with urllib.request.urlopen(row['inputs']['viewer_url'],timeout=.3):row['viewer_ready']=True
            except OSError:pass
        return row

    def scene(self,task,path=None):
        task_info(task)
        return description(self.root,task,path)

    def save_scene(self,request):
        """复制成新输入版本；T_world_object 使用世界系、米和右手旋转矩阵。"""
        if not task_info(request.get('task','sweep'))['editable_scene']:raise ValueError('此 benchmark 尚未接入网页场景编辑，请使用其原生场景构建入口')
        if request.get('task')=='microwave':
            return self._save_microwave_scene(request)
        import math
        import numpy as np
        source=safe_path(self.root,request.get('scene') or TASKS['sweep']['scene'],exists=True)
        if not (source/'robot/robot_metadata.json').is_file():raise ValueError('需要清扫场景目录')
        data=read_json(source/'scene.json');edits=request.get('objects',[])
        for edit in edits:
            name=edit['name']
            if name not in data['objects']:raise ValueError('未知物体')
            if 'T_world_object' in edit:
                matrix=np.asarray(edit['T_world_object'],dtype=float)
                if (matrix.shape!=(4,4) or not np.isfinite(matrix).all()
                    or not np.allclose(matrix[3],[0,0,0,1],atol=1e-6)
                    or not np.allclose(matrix[:3,:3].T@matrix[:3,:3],np.eye(3),atol=1e-5)
                    or not np.isclose(np.linalg.det(matrix[:3,:3]),1.,atol=1e-5)
                    or np.max(np.abs(matrix[:3,3]))>2):
                    raise ValueError('物体位姿必须是有效的米制刚体变换，位置范围 ±2 m')
                data['objects'][name]['T_world_object']=matrix.tolist()
                data.setdefault('initial_pose_overrides',{})[name]=matrix.tolist()
                continue
            x,y,yaw=[float(edit[k]) for k in ('x','y','yaw')]
            if not all(math.isfinite(v) for v in (x,y,yaw)) or abs(x)>2 or abs(y)>2 or abs(yaw)>360:raise ValueError('摆位参数超出编辑范围')
            matrix=data['objects'][name]['T_world_object'];old=math.atan2(matrix[1][0],matrix[0][0]);a=math.radians(yaw)-old
            oldrows=[list(matrix[0][:3]),list(matrix[1][:3])]
            for j in range(3):
                matrix[0][j]=math.cos(a)*oldrows[0][j]-math.sin(a)*oldrows[1][j]
                matrix[1][j]=math.sin(a)*oldrows[0][j]+math.cos(a)*oldrows[1][j]
            matrix[0][3]=x;matrix[1][3]=y
            if name in data.get('initial_pose_overrides',{}):data['initial_pose_overrides'][name]=matrix
        if 'placement_rules' in request:
            data['placement_rules']=validate_rules(request['placement_rules'],tuple(data['objects']))
        if 'table' in request:
            data['table']=validate_table(request['table'],data.get('table'))
            rules=data.setdefault('placement_rules',suggested_rules('sweep',data))
            rules.setdefault('constraints',{})['bounds_xy_m']=table_bounds_xy(data['table'])
            if 'randomization' in data:data['randomization']['table_bounds_xy_m']=table_bounds_xy(data['table'])
        overrides=request.get('concepts',{})
        validated={}
        for name,value in overrides.items():
            if name not in {'brush','cube','dustpan'}:raise ValueError('不支持该 Concept')
            p=safe_path(self.root,value,exists=True);content=read_json(p)
            if not isinstance(content.get('conceptualization'),list):raise ValueError('不是 Concept 标注文件')
            validated[name]=p
        dest=self.root/'runs/workbench/scenes'/('scene_'+uuid.uuid4().hex[:12])
        tree=ET.parse(source/'scene.xml')
        if 'table' in request:
            geom=tree.find("./worldbody/geom[@name='scene_table']")
            if geom is None:raise ValueError('场景缺少 scene_table 桌面几何')
            table=data['table']
            geom.set('pos',' '.join(map(str,[*table['center_xy_m'],table['top_z_m']-table['thickness_m']/2])))
            geom.set('size',' '.join(map(str,[table['size_xy_m'][0]/2,table['size_xy_m'][1]/2,table['thickness_m']/2])))
        for node in tree.getroot().iter():
            if node.get('file'):node.set('file',str((source/node.get('file')).resolve()))
        # runner 从 JSON 设置 freejoint；固定物体必须同步 XML，避免页面与仿真摆位不一致。
        from scipy.spatial.transform import Rotation
        for edit in edits:
            body=tree.find(f"./worldbody/body[@name='{edit['name']}']")
            if body is None:raise ValueError('仅支持场景顶层物体编辑')
            if body.find('freejoint') is None and body.find("joint[@type='free']") is None:
                matrix=data['objects'][edit['name']]['T_world_object']
                body.set('pos',' '.join(str(matrix[i][3]) for i in range(3)))
                quat=Rotation.from_matrix([row[:3] for row in matrix[:3]]).as_quat()
                for attr in ('euler','axisangle','xyaxes','zaxis'):body.attrib.pop(attr,None)
                body.set('quat',' '.join(map(str,[quat[3],*quat[:3]])))
        shutil.copytree(source,dest)
        tree.write(dest/'scene.xml',encoding='unicode')
        data['scene_xml_path']=str(dest/'scene.xml')
        (dest/'scene.json').write_text(json.dumps(data,ensure_ascii=False,indent=2))
        for name,p in validated.items():shutil.copy2(p,dest/f'concepts/{name}.json')
        (dest/'workbench_source.json').write_text(json.dumps({'source':str(source),'edits':edits,'concepts':overrides},ensure_ascii=False,indent=2))
        with self.lock:
            self.library.setdefault('scenes',[]).append({'path':str(dest.relative_to(self.root)),
                'label':time.strftime('%m-%d %H:%M')+' · '+dest.name,'task':'sweep'})
            self._save()
        return self.scene('sweep',str(dest))

    def _save_microwave_scene(self,request):
        source,data=load_input(self.root,'microwave',request.get('scene'))
        for edit in request.get('objects',[]):
            name=edit['name']
            if name not in {'microwave','bowl'}:raise ValueError('微波炉场景只能移动整机和碗')
            pose=validate_pose(edit['T_world_object'],field_name='位姿')
            if abs(pose[:3,3]).max()>2:raise ValueError('位姿位置范围为 ±2 m')
            data[name+'_pose_world']=pose.tolist()
        if 'concepts' in request:raise ValueError('微波炉 Concept 应通过原标注入口修改')
        if 'placement_rules' in request:
            data['placement_rules']=validate_rules(request['placement_rules'],('microwave','bowl'))
        if 'table' in request:
            defaults=read_json(Path(__file__).resolve().parents[2]/'config/scenes/task_1.json')['table']
            data['table']=validate_table(request['table'],data.get('table',defaults))
            rules=data.setdefault('placement_rules',suggested_rules('microwave',data))
            rules.setdefault('constraints',{})['bounds_xy_m']=table_bounds_xy(data['table'])
        dest=self.root/'runs/workbench/scenes'/('microwave_'+uuid.uuid4().hex[:12])
        dest.mkdir(parents=True)
        (dest/'task.json').write_text(json.dumps(data,ensure_ascii=False,indent=2))
        (dest/'workbench_source.json').write_text(json.dumps({'source':str(source),'request':request},ensure_ascii=False,indent=2))
        with self.lock:
            self.library.setdefault('scenes',[]).append({'path':str(dest.relative_to(self.root)),
                'label':time.strftime('%m-%d %H:%M')+' · '+dest.name,'task':'microwave'})
            self._save()
        return self.scene('microwave',str(dest))

    def freeze_sample(self,request):
        job=self.jobs.get(request['job'])
        if job['kind']!='distribution':raise ValueError('只能固定已有分布预览样本')
        folder=self.jobs.directory(job['id'])/'run'
        document=read_json(folder/'distribution.json')
        matches=[s for s in document['samples'] if s['seed']==request['seed']]
        if len(matches)!=1 or not matches[0].get('valid'):raise ValueError('该 seed 不是有效的预览初态')
        sample=read_json(folder/f"seed_{request['seed']}"/'sample.json')
        scene=self.save_scene({'task':job['inputs']['task'],'scene':sample['source'],
            'objects':[{'name':n,'T_world_object':p} for n,p in sample['sampled_poses_world'].items()]})
        (self.root/scene['path']/'frozen_sample.json').write_text(json.dumps(
            {'source_job':job['id'],'seed':request['seed'],'sample':sample},ensure_ascii=False,indent=2))
        return scene

    def draft(self,request):
        title=str(request.get('title','')).strip();intent=str(request.get('intent','')).strip()
        if not title or not intent or len(title)>100 or len(intent)>4000:raise ValueError('请输入任务名称和操作目标')
        task_info(request['template'])
        row={'id':uuid.uuid4().hex[:12],'title':title,'intent':intent,'template':request['template'],'status':'待开发'}
        with self.lock:self.library['drafts'].append(row);self._save()
        return row
