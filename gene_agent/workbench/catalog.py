"""可运行任务与参数白名单。所有子进程使用 argv，不经过 shell。"""
from pathlib import Path
import math


# 保留历史网页 ID；场景默认路径与标题以 benchmark 定义为准。
from benchmarks.catalog import load_definition

_ADAPTERS={
 'sweep':dict(benchmark_id='sweep_cube_task',task='sweep_cube_into_dustpan',
     description='抓起普通刷，把三块方块扫进簸箕。',skills=[{'id':'pick_brush','name':'抓取刷子'},{'id':'sweep_cube','name':'刷子清扫'}],
     randomization=True,editable_scene=True,visual_planning=True,skill_preview=True,distribution_preview=True),
 'microwave':dict(benchmark_id='microwave_task',task='microwave_task1',
     description='开门、放碗、关门、旋钮、取碗并归位。',skills=[{'id':'extract_bowl','name':'取碗（查看记录）'}],
     randomization=True,editable_scene=True,visual_planning=True,skill_preview=True,distribution_preview=True),
 'sweep_long':dict(benchmark_id='sweep_long_task',task='sweep_long',description='竖立长刷顶抓、离架并清扫三方块。',
     skills=[{'id':'pick_brush_long','name':'长刷抓取（记录回放）'},{'id':'sweep_long_brush','name':'长刷清扫（记录回放）'}],
     randomization=True,editable_scene=False,visual_planning=False,skill_preview=False,distribution_preview=False),
 'cup_pour':dict(benchmark_id='cup_pour_task',task='cup_pour_into_bowl',description='抓住杯把、倒球入碗并放回杯子。',
     skills=[{'id':'cup_pick','name':'杯把抓取'},{'id':'cup_pour','name':'杯子倾倒'},{'id':'cup_place','name':'杯子放回'}],
     randomization=False,editable_scene=False,visual_planning=False,skill_preview=True,distribution_preview=False),
 'spoon_scoop':dict(benchmark_id='spoon_scoop_task',task='spoon_scoop_into_bowl',description='抓勺、舀球并倒入另一碗。',
     skills=[{'id':'spoon_pick','name':'抓勺'},{'id':'spoon_scoop','name':'盛球'},{'id':'spoon_pour','name':'倾倒'}],
     randomization=False,editable_scene=False,visual_planning=False,skill_preview=True,distribution_preview=False),
}
TASKS={}
for _id,_adapter in _ADAPTERS.items():
    _definition=load_definition(_adapter['benchmark_id'])
    TASKS[_id]=dict(_adapter,id=_id,name=_definition['title'],scene_viewer=True,
        benchmark_version=_definition['version'],evidence_mode=_definition['evidence_mode'],
        **{('config' if _id=='microwave' else 'scene'):_definition['scene']['input']})


def safe_path(root, value, prefixes=('runs','data','config'), *, exists=False):
    root=Path(root).resolve()
    path=(root/str(value)).resolve()
    if not any(path.is_relative_to(root/p) for p in prefixes):
        raise ValueError('路径必须位于允许的项目数据目录内')
    if exists and not path.exists():raise ValueError(f'找不到文件：{value}')
    return path


def integer(value, low, high):
    if isinstance(value,bool) or str(value).strip()!=str(int(value)):
        raise ValueError('需要整数参数')
    value=int(value)
    if not low<=value<=high:raise ValueError(f'整数范围为 {low}–{high}')
    return value


def task_info(task):
    if task not in TASKS:raise ValueError('未知任务')
    return TASKS[task]


def video_command(root,python,task,run,output):
    """按实际报告格式选导出器；新视频及清单全部写入新作业目录。"""
    root=Path(root);output=Path(output)
    if task=='sweep' or task=='sweep_long':
        return [str(python),str(root/'tools/viewers/recording/record_sweep_cube_video.py'),str(run),
                '--output',str(output/'video.mp4'),'--manifest',str(output/'video_manifest.json')]
    if task in {'cup_pour','spoon_scoop'}:
        return [str(python),str(root/'tools/viewers/recording/record_live_video.py'),'--run',str(run),'--output',str(output/'video.mp4')]
    return [str(python),str(root/'tools/viewers/recording/record_skill_run_video.py'),
            '--run',str(run),'--output',str(output/'video.mp4'),'--mode','actual','--profile','microwave_task1']


def build_command(root, python, task, action, options, output):
    """返回已验证的命令；输出路径由服务生成，用户不能传入任意命令。"""
    root=Path(root);output=Path(output);info=task_info(task)
    collision_mode=options.get('collision_mode','full')
    if collision_mode not in {'full','robot_only'}:raise ValueError('未知碰撞模式')
    if (action=='visual_plan' or options.get('execution_mode')=='visual') and not info['visual_planning']:
        raise ValueError('该 benchmark 尚未适配无 MuJoCo 的纯视觉规划；请选择 MuJoCo 执行')
    if action=='distribution' and not info['distribution_preview']:raise ValueError('此 benchmark 尚未接入网页分布预览')
    prefix=[str(python)]
    def cli(name,*args):return [*prefix,str(root/name),*map(str,args)]
    if action in {'visual_plan','knowledge_preview'} or (action=='run' and options.get('execution_mode')=='visual'):
        tilt=float(options.get('tilt',-20))
        if not math.isfinite(tilt) or abs(tilt)>45:raise ValueError('倾角范围为 -45 到 45 度')
    if action=='run_preview':
        source=safe_path(root,options.get('scene') or info.get('scene') or info['config'],exists=True)
        mode=options.get('execution_mode','physical')
        if mode not in {'visual','physical'}:raise ValueError('未知运行方式')
        if options.get('randomize',False) and not info['randomization']:raise ValueError('此 benchmark 尚未接入在线随机采样')
        cmd=[*prefix,'-m','gene_agent.workbench.run_scene_preview','--task',task,'--scene',str(source),
             '--output',str(output),'--port',str(integer(options.get('port',8094),1024,65535)),
             '--execution-mode',mode,'--collision-mode',collision_mode,
             '--seed',str(integer(options.get('seed',123),0,2**31-1))]
        if options.get('randomize',False):cmd.append('--randomize')
        return cmd
    if action=='live_viewer':
        run=safe_path(root,options.get('run',''),('runs',),exists=False)
        return [*prefix,'-m','gene_agent.visualization.live_run','--run',str(run),'--output',str(output),
                '--port',str(integer(options.get('port',8093),1024,65535))]
    if action=='visual_plan' or (action=='run' and options.get('execution_mode')=='visual'):
        scene=safe_path(root,options.get('scene') or info.get('scene') or info['config'],exists=True)
        cmd=cli('tools/run/plan_visual.py','--task',task,'--scene',scene,'--output',output,
                '--seed',integer(options.get('seed',123),0,2**31-7),'--tilt',float(options.get('tilt',-20)),
                '--collision-mode',collision_mode)
        if options.get('randomize',False):cmd.append('--randomize')
        if options.get('preview_skill'):
            from gene_agent.tasks.planning.knowledge import SKILLS
            if options['preview_skill'] not in SKILLS[task]:raise ValueError('该任务不支持此 Skill')
            cmd+=['--skill',options['preview_skill']]
        return cmd
    if action in {'knowledge_preview','visual_viewer'}:
        cmd=[*prefix,'-m','gene_agent.visualization.pure_planning.cli','--output',str(output),
             '--port',str(integer(options.get('port',8093),1024,65535))]
        if action=='visual_viewer':
            run=safe_path(root,options.get('run',''),('runs',),exists=True)
            import json
            if json.loads((run/'result.json').read_text(encoding='utf-8')).get('record_type')!='pure_visual_plan':
                raise ValueError('此入口只接受纯视觉规划记录')
            return [*cmd,'--run',str(run)]
        from gene_agent.tasks.planning.knowledge import KNOWLEDGE
        selection=options.get('preview_knowledge')
        if selection not in KNOWLEDGE[task]:raise ValueError('该任务不支持此 Knowledge')
        scene=safe_path(root,options.get('scene') or info.get('scene') or info['config'],exists=True)
        return [*cmd,'--task',task,'--scene',str(scene),'--knowledge',selection,'--tilt',str(float(options.get('tilt',-20)))]
    if action=='run_video' and options.get('execution_mode')=='visual':
        raise ValueError('纯视觉模式请先生成规划并三维预览；此按钮只导出 MuJoCo 实际视频')
    if action in {'run','run_video'} and task in {'sweep_long','cup_pour','spoon_scoop'}:
        scene=safe_path(root,options.get('scene') or info['scene'],exists=True)
        entry={'sweep_long':'run_sweep_long.py','cup_pour':'run_cup_pour_task.py','spoon_scoop':'run_spoon_scoop_task.py'}[task]
        cmd=cli('tools/run/'+entry,'--scene-dir' if task=='sweep_long' else '--scene',scene,
                '--output-dir' if task=='sweep_long' else '--output',output,'--collision-mode',collision_mode,'--live')
        if task=='sweep_long':
            if options.get('randomize',False):cmd+=['--layout-seed',str(integer(options.get('seed',123),0,2**31-1))]
        elif options.get('randomize',False):raise ValueError('此 benchmark 使用已生成的固定初态；网页尚未接入在线随机采样')
        if options.get('preview_skill'):
            mapping={'cup_pick':'pick','cup_pour':'pour','cup_place':'place','spoon_pick':'pick','spoon_scoop':'scoop','spoon_pour':'pour'}
            selection=options['preview_skill']
            if selection not in {s['id'] for s in info['skills']} or selection not in mapping:raise ValueError('该 benchmark 尚未接入独立 Skill 前缀运行')
            cmd+=['--until-skill',mapping[selection]]
        return cmd
    if action in {'run','run_video'}:
        if task=='sweep':
            seed=integer(options.get('seed',36),0,2**31-1)
            tilt=float(options.get('tilt',-20))
            if not math.isfinite(tilt) or abs(tilt)>45:raise ValueError('倾角范围为 -45 到 45 度')
            scene=safe_path(root,options.get('scene') or info['scene'],exists=True)
            cmd=cli('tools/run/run_sweep_cube.py','--scene-dir',scene,'--output-dir',output,
                    '--seed',seed,'--grasp-tilt-deg',tilt,'--collision-mode',collision_mode,'--live')
            if options.get('randomize',True):cmd.append('--randomize')
            return cmd
        config=safe_path(root,options.get('scene') or options.get('config') or info['config'],exists=True)
        if config.is_dir():config=config/'task.json'
        return cli('tools/run/run_microwave_task1_physical.py','--config',config,'--output',output,
                   '--seed',integer(options.get('seed',123),0,2**31-1),
                   '--randomize' if options.get('randomize',False) else '--fixed-placement',
                   '--collision-mode',collision_mode,'--live')
    if action in {'video','viewer','skill'}:
        run=safe_path(root,options.get('run',''),('runs',),exists=True)
        if not run.is_dir():raise ValueError('运行记录应为目录')
        if action=='video':
            return video_command(root,python,task,run,output)
        port=integer(options.get('port',8093),1024,65535)
        if action=='skill':
            skill=options.get('skill')
            if skill not in {s['id'] for s in info['skills']}:raise ValueError('该任务不支持此 Skill')
            return cli('tools/viewers/view_skill.py','--skill',skill,'--run',run,'--port',port)
        return cli('tools/viewers/view_task.py','--task',info['task'],'--run',run,'--port',port)
    if action=='scene_viewer' and task not in {'sweep','microwave'}:
        scene=safe_path(root,options.get('scene') or info['scene'],exists=True)
        return cli('tools/viewers/view_benchmark.py','--benchmark',info['benchmark_id'],'--input',scene,
                   '--output',output,'--mode','viser','--port',integer(options.get('port',8094),1024,65535))
    if action in {'distribution','scene_viewer'}:
        scene=safe_path(root,options.get('scene') or info.get('scene') or info['config'],exists=True)
        seed=integer(options.get('seed',36),0,2**31-7)
        cmd=[*prefix,'-m','gene_agent.workbench.scene_preview','--scene',str(scene),'--output',str(output),'--seed',str(seed),'--task',task]
        if action=='scene_viewer':
            cmd+=['--port',str(integer(options.get('port',8094),1024,65535))]
            if not options.get('randomize',True):cmd.append('--nominal')
        elif not options.get('randomize',True):cmd.append('--nominal')
        return cmd
    if action=='sample_viewer':
        folder=safe_path(root,options['sample_dir'],('runs',),exists=True)
        return [*prefix,'-m','gene_agent.workbench.sample_viewer','--samples',str(folder),
            '--seed',str(integer(options['seed'],0,2**31-1)),
            '--port',str(integer(options['port'],1024,65535))]
    if action=='knowledge':
        return cli('tools/editors/view_grasp_knowledge.py','--kind','cylinder','--port',integer(options.get('port',8094),1024,65535))
    if action=='concept':
        mesh=safe_path(root,options.get('mesh',''),exists=True)
        concept=safe_path(root,options.get('concept',''),exists=True)
        if mesh.suffix.lower() not in {'.obj','.stl','.ply','.glb'} or concept.suffix!='.json':
            raise ValueError('需要米制 mesh 与 Concept JSON')
        return cli('tools/editors/edit_concept.py','--mesh',mesh,'--concept',concept,
                   '--output',output/'concept.json','--port',integer(options.get('port',8094),1024,65535))
    raise ValueError('该任务尚未接入此操作')
