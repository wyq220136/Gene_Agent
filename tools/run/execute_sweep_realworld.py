#!/usr/bin/env python3
"""复用微波炉 Runtime 执行旧刷清扫；默认只检查文件，不连接电机。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    inputs=parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--run',type=Path,help='首帧预览记录；执行时要求新鲜且实测初态')
    inputs.add_argument('--camera-param',type=Path,help='自动重新采集/感知/预览，再接执行')
    parser.add_argument('--config',type=Path,default=ROOT/'config/realworld/sweep_execution.example.json')
    parser.add_argument('--collision-mode', choices=['full', 'robot_only'])
    parser.add_argument('--execute',action='store_true',help='显式允许使能Pika并执行所选阶段')
    parser.add_argument('--yes',action='store_true',help='跳过首轮执行确认；不跳过夹持/稳定/任务结果确认')
    parser.add_argument('--stop-after',choices=['pregrasp','approach','pick','task'],default='pregrasp')
    parser.add_argument('--gripper-port',help='覆盖执行配置中的Pika串口')
    parser.add_argument('--output',type=Path,default=ROOT/'runs/realworld_sweep'/time.strftime('execution_%Y%m%d_%H%M%S'))
    args=parser.parse_args(argv)
    from gene_agent.realworld.sweep.execution import read_execution_config,check_execution_input,preview_ready_for_stop
    from gene_agent.realworld.sweep.planning import write_json
    config=read_execution_config(args.config)
    if args.collision_mode is not None:
        config['motion'] = {**config.get('motion', {}), 'collision_mode': args.collision_mode}
    if args.gripper_port:config['gripper_port']=args.gripper_port
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    if args.camera_param:
        from tools.run.preview_sweep_from_camera import main as preview
        code=preview(['--camera-param',str(args.camera_param.resolve()),'--no-viewer',
                      *(['--collision-mode', config['motion']['collision_mode']] if config.get('motion', {}).get('collision_mode') else []),
                      '--stop-after',args.stop_after,
                      '--output',str(output/'first_frame')])
        source=output/'first_frame'
        if code!=0:
            write_json(output/'entry_result.json',{'status':'preview_failed','execution_enabled':False,'source_run':str(source)})
            print(f'首帧 {args.stop_after} 范围预览失败，不进入执行：{source}');return 2
    else:source=args.run.resolve()
    report=json.loads((source/'result.json').read_text(encoding='utf-8'))
    site=json.loads((source/'site_input.json').read_text(encoding='utf-8'))
    # 执行配置覆盖预览配置；未覆盖时继承同一规划模式，避免前后不一致。
    config.setdefault('motion', site.get('motion', {}))
    site['motion'] = config['motion']
    snapshot=json.loads((source/'observation.json').read_text(encoding='utf-8'))
    if report.get('task')!='sweep_realworld_plan_only' or not preview_ready_for_stop(report,args.stop_after) or report.get('brush_variant')!='old':
        raise ValueError(f'需要原蓝色旧刷、成功规划至 {args.stop_after} 的首帧记录')
    site['support_adjustment']=report.get('support_adjustment')
    metadata={'source_run':str(source),'execute_requested':args.execute,'stop_after':args.stop_after,
              'runtime':'gene_agent.realworld.runtime.RealWorldRuntime','config':config,
              'input_hashes':{name:hashlib.sha256((source/name).read_bytes()).hexdigest()
                              for name in ['result.json','site_input.json','observation.json']}}
    write_json(output/'entry_input.json',metadata)
    if not args.execute:
        write_json(output/'entry_result.json',{'status':'files_checked_only','execution_enabled':False,
            'planning_complete':report['planning_complete'],'requested_scope_complete':True,
            'preview_scope':args.stop_after,'task_success':None,'tcp_validation':'requires_live_model_sdk_fk_check',
            'claim':'未连接SDK或串口；未读取当前机器人，不等于真机轨迹预检通过'})
        print(f'输入文件检查通过；未连接电机、未执行；记录：{output}')
        return 0
    check_execution_input(snapshot,site,config)
    calibration_path=source/'camera_calibration.json'
    calibration=json.loads(calibration_path.read_text(encoding='utf-8')) if calibration_path.exists() else {}
    from gene_agent.realworld.sweep.observation import resolve_preview_robot_ip
    ip=resolve_preview_robot_ip(site,None,calibration=calibration)
    if not args.yes:
        answer=input(f'将使用 {ip} 与 Pika {config["gripper_port"]}，低速执行至 {args.stop_after}。确认场景未移动并执行？[y/N] ')
        if answer.strip().lower() not in {'y','yes'}:
            write_json(output/'entry_result.json',{'status':'cancelled','execution_enabled':False});return 0
    check_execution_input(snapshot,site,config)
    import mujoco
    import numpy as np
    from gene_agent.robots.pika_frames import load_frame_profile
    from gene_agent.realworld.sweep.execution import execute_observed_sweep
    from gene_agent.realworld.sweep.sdk_bridge import SDKBridge
    model=mujoco.MjModel.from_xml_path(report['scene_xml_path']);data=mujoco.MjData(model)
    data.qpos[:]=np.load(source/'scene/initial_state.npz')['qpos'];mujoco.mj_forward(model,data)
    robot=json.loads((source/'scene/robot/robot_metadata.json').read_text(encoding='utf-8'))
    site.update(execution_scene_xml=report['scene_xml_path'],execution_robot_ip=ip,
                execution_concept_dir=str(source/'scene/concepts'),execution_candidate_index=report.get('selected_index') or 0)
    profile=load_frame_profile(site['frame_profile'])
    table_top=float(data.geom('scene_table').xpos[2]+model.geom('scene_table').size[2])
    worker_config={**config,'execute_authorized':True,'robot_ip':ip,
        'frame_profile':site['frame_profile'],'T_sdk_urdf_base':site['T_sdk_urdf_base'],
        'T_sdk_flange_operation_tcp':profile['T_sdk_flange_operation_tcp'].tolist(),
        'min_tcp_z_m':table_top+config['min_tcp_above_table_m']}
    # 校验前后均有输出证据。只有这里创建SDK/Pika，且只在显式--execute分支。
    with SDKBridge(worker_config,output/'device') as backend:
        check_execution_input(snapshot,site,config)
        result=execute_observed_sweep(model,data,robot,site,snapshot,config,backend,output,
                                       stop_after=args.stop_after)
    print(f'状态：{result["status"]}；运动阶段完成={result["motion_complete"]}；任务成功={result["task_success"]}；{output}')
    return 0 if result['status'] in {'executed','stopped_as_requested'} else 2


if __name__=='__main__':raise SystemExit(main())
