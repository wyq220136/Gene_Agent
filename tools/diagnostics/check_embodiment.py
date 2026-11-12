#!/usr/bin/env python3
"""在原生模型上核对FK/TCP、实际短轨迹及碰撞，结果不作为完整任务成绩。"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def known_collision_check(result):
    """只接受新增障碍的起点碰撞证据，配置/IK/路径失败不能冒充碰撞检查。"""
    if result.success or result.failure is None or result.failure.kind.value!='start_collision':return False
    diagnostic=result.failure.details.get('collision_verification',{}).get('start_check',{}).get('collision_diagnostic',{})
    return 'diagnostic_tcp_block' in diagnostic.get('colliding_obstacles',[])


def native_check(request,output):
    """FK样本显式为诊断运动学状态，恢复后才执行真实控制轨迹。"""
    import numpy as np
    from gene_agent.simulation.robotwin.session import native_episode,robot_metadata,capture_scene
    from gene_agent.simulation.robotwin.runtime import RobotwinRuntime
    from gene_agent.simulation.robotwin.scene import RobotwinSceneProvider
    from gene_agent.motion.curobo.process import ProcessPlanningSession
    from gene_agent.motion import JointState,PoseTarget,CollisionPolicy
    from gene_agent.knowledge.geometry.transforms import pose_delta
    from gene_agent.robots.aloha import digest
    group=request['group'];out=Path(output)
    report=dict(status='running',scope='embodiment_diagnostic',task_success=None,checks=[],fk_mode='native_kinematic_samples_not_physical_motion')
    try:
        with native_episode(request,seed=request['seed'],output_dir=out/'scene') as (env,args):
            robot=robot_metadata(env,args)
            runtime=RobotwinRuntime(env,episode_id='embodiment_check',robot_metadata=robot,allowed_observation_fields=('objects','robots'),object_bindings={'tool':'hammer','target':'nail','board':'board'})
            provider=RobotwinSceneProvider(env,runtime,robot_metadata=robot,output_dir=out/'collision')
            (out/'robot_metadata.json').write_text(json.dumps(robot,ensure_ascii=False,indent=2))
            capture_scene(env,out/'views')
            entity=getattr(env.robot,f'{group}_entity')
            saved=np.asarray(entity.get_qpos()).copy()
            indices={j.name:i for i,j in enumerate(entity.get_active_joints())}
            names=robot['joint_groups'][group]
            with ProcessPlanningSession(robot,python=request.get('planner_python') or sys.executable,
                    pythonpath=request.get('planner_pythonpath'),log_path=out/'planner-worker.log') as session:
                report['planner_dependency']=session.describe()
                try:
                    for index,scale in enumerate([0.,.03,-.03,.06,-.06]):
                        q=saved.copy()
                        for i,name in enumerate(names):q[indices[name]]+=scale*(1 if i%2==0 else -1)
                        entity.set_qpos(q)  # 仅数值FK诊断，不推进物理、不判任务成功。
                        state=runtime.observe(object_names=('tool','target','board'),robot_groups=('left','right'))
                        snap=provider.snapshot(state=state,group=group,collision_policy=CollisionPolicy())
                        start=JointState(tuple(names),tuple(state.values['robots'][group]['joint_positions_rad']))
                        with session.stage(snapshot=snap,start=start,group=group,scene_revision=0) as motion:actual=motion.fk()
                        native=np.asarray(state.values['robots'][group]['T_world_tcp'])
                        translation,rotation=pose_delta(native,actual,field='native_vs_curobo_tcp')
                        report['checks'].append(dict(kind='fk',sample=index,joints_rad=list(start.positions_rad),
                            translation_error_m=translation,rotation_error_rad=rotation,
                            passed=bool(translation<=.001 and rotation<=np.deg2rad(.5))))
                        (out/'fk_samples.json').write_text(json.dumps(report['checks'],ensure_ascii=False,indent=2))
                finally:entity.set_qpos(saved)
                state=runtime.observe(object_names=('tool','target','board'),robot_groups=('left','right'))
                if request['checks']=='all':
                    snap=provider.snapshot(state=state,group=group,collision_policy=CollisionPolicy())
                    start=JointState(tuple(names),tuple(state.values['robots'][group]['joint_positions_rad']))
                    target=np.asarray(state.values['robots'][group]['T_world_tcp']).copy();target[2,3]+=.01
                    with session.stage(snapshot=snap,start=start,group=group,scene_revision=0) as motion:
                        ordinary=motion.plan_to(PoseTarget('world_tcp',target),request_id='embodiment/ordinary')
                        linear=motion.plan_linear(PoseTarget('world_tcp',target),request_id='embodiment/linear')
                    for label,result in [('ordinary',ordinary),('linear',linear)]:
                        (out/f'{label}.json').write_text(json.dumps(result.to_dict(),ensure_ascii=False,indent=2))
                        report['checks'].append(dict(kind=label,passed=result.success,result_ref=f'{label}.json'))
                    if linear.success:
                        outcome=runtime.execute_trajectory(linear.trajectory,group=group,stage_id='linear_motion',expected_state_id=state.state_id)
                        (out/'execution.json').write_text(json.dumps({'success':outcome.success,'start':outcome.start.values,'end':outcome.end.values,'details':outcome.details,'failure':outcome.failure},ensure_ascii=False,indent=2))
                        report['checks'].append(dict(kind='physical_execution',passed=outcome.success,result_ref='execution.json'))
                    # 空手开合只在当前净空位置检查；每次锁关节变化仍由原pool判断重建。
                    for label,width in [('gripper_close',0.),('gripper_open',.08)]:
                        state=runtime.observe(object_names=('tool','target','board'),robot_groups=('left','right'))
                        outcome=runtime.execute_gripper(group=group,width_m=width,duration_s=2. if width>0. else 1.,stage_id=label,expected_state_id=state.state_id)
                        aperture=outcome.end.values['robots'][group]['gripper']['width_m']
                        good=outcome.success and (aperture<=.002 if width==0. else abs(aperture-width)<=.004)
                        (out/f'{label}.json').write_text(json.dumps({'success':bool(good),'actual_width_m':aperture,
                            'start':outcome.start.values,'end':outcome.end.values,'failure':outcome.failure},ensure_ascii=False,indent=2))
                        report['checks'].append(dict(kind=label,passed=bool(good),actual_width_m=aperture,result_ref=f'{label}.json'))
                        state=outcome.end
                        snap=provider.snapshot(state=state,group=group,collision_policy=CollisionPolicy())
                        start=JointState(tuple(names),tuple(state.values['robots'][group]['joint_positions_rad']))
                        target=np.asarray(state.values['robots'][group]['T_world_tcp']).copy();target[2,3]+=.005
                        with session.stage(snapshot=snap,start=start,group=group,scene_revision=0) as motion:
                            synchronized=motion.plan_to(PoseTarget('world_tcp',target),request_id=f'embodiment/{label}_geometry')
                        (out/f'{label}_planning.json').write_text(json.dumps(synchronized.to_dict(),ensure_ascii=False,indent=2))
                        report['checks'].append(dict(kind=label+'_geometry',passed=synchronized.success,result_ref=f'{label}_planning.json'))
                    # 人工障碍只在独立规划诊断中注入，不更改原bench物理。
                    state=runtime.observe(object_names=('tool','target','board'),robot_groups=('left','right'))
                    blocked=provider.snapshot(state=state,group=group,collision_policy=CollisionPolicy())
                    tcp=np.asarray(state.values['robots'][group]['T_world_tcp'])
                    base=np.asarray(blocked['base_from_world'])@tcp
                    from scipy.spatial.transform import Rotation
                    q=Rotation.from_matrix(base[:3,:3]).as_quat()
                    blocked['scene_model']['cuboid']['diagnostic_tcp_block']=dict(dims=[.1,.1,.1],pose=[*base[:3,3],*q[[3,0,1,2]]])
                    blocked['scope']='diagnostic_only_injected_tcp_obstacle';blocked['snapshot_sha256']=digest(blocked)
                    start=JointState(tuple(names),tuple(state.values['robots'][group]['joint_positions_rad']))
                    with session.stage(snapshot=blocked,start=start,group=group,scene_revision=0) as motion:
                        collision=motion.plan_to(PoseTarget('world_tcp',tcp),request_id='embodiment/known_collision')
                    (out/'known_collision.json').write_text(json.dumps(collision.to_dict(),ensure_ascii=False,indent=2))
                    report['checks'].append(dict(kind='known_collision',passed=known_collision_check(collision),diagnostic_modified_world=True,result_ref='known_collision.json'))
            report['status']='passed' if all(c['passed'] for c in report['checks']) else 'failed'
    except Exception as exc:
        import traceback
        report.update(status='error',error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
    (out/'embodiment_check.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    return report['status']=='passed'


def main():
    """父进程解析来源，子进程加载原bench；规划worker单独选依赖。"""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',choices=('robotwin',),default='robotwin')
    parser.add_argument('--external-root',type=Path)
    parser.add_argument('--group',choices=('left','right'),default='left')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--python',default=sys.executable)
    parser.add_argument('--planner-python');parser.add_argument('--planner-pythonpath')
    parser.add_argument('--checks',choices=('fk','all'),default='all')
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--native-request',type=Path,help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.native_request:
        request=json.loads(args.native_request.read_text());raise SystemExit(0 if native_check(request,args.output.resolve()) else 1)
    if args.external_root is None:parser.error('--external-root必需')
    from benchmarks.novel_tool_task.source import resolve_source,native_environment
    request=resolve_source(args.external_root,family='hit',instance='hammer_train')
    request.update(group=args.group,seed=args.seed,planner_python=args.planner_python,planner_pythonpath=args.planner_pythonpath,checks=args.checks)
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    path=out/'check_request.json';path.write_text(json.dumps(request,ensure_ascii=False,indent=2))
    command=[args.python,str(Path(__file__).resolve()),'--native-request',str(path),'--output',str(out)]
    with (out/'native.log').open('w') as log:
        proc=subprocess.run(command,cwd=Path(request['external_root'])/'env/RoboTwin',env=native_environment(args.external_root,out),stdout=log,stderr=subprocess.STDOUT)
    print(f'embodiment检查 code={proc.returncode}；结果 {out}')
    raise SystemExit(proc.returncode)


if __name__=='__main__':main()
