"""仅由显式 --execute 入口启动的 Pika/Realman SDK 进程。"""
from __future__ import annotations

import argparse
import copy
from contextlib import redirect_stdout
import json
from pathlib import Path
import sys
import traceback

import numpy as np


def remap_stage_names(row):
    """单臂模型 joint_i 与共享 RealmanAdapter 的 left_joint_i 按 SDK 1..7 显式映射。"""
    from gene_agent.motion.model import motion_request_from_dict, motion_result_from_dict
    from gene_agent.skills.runtime.planning import SkillStage
    row = copy.deepcopy(row)
    model_names = [f'joint_{i}' for i in range(1,8)]
    sdk_names = [f'left_joint_{i}' for i in range(1,8)]
    req, result = row['request'], row['result']
    for state in (req['start'], result.get('goal_state'), result.get('trajectory')):
        if state is None:continue
        if state['joint_names'] != model_names:
            raise ValueError('模型关节顺序不是 SDK 1..7')
        state['joint_names'] = sdk_names
    return SkillStage(row['name'], 'planned', request=motion_request_from_dict(req),
                      result=motion_result_from_dict(result))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args=parser.parse_args()
    config=json.loads(args.config.read_text(encoding='utf-8'))
    if config.get('execute_authorized') is not True:
        raise ValueError('SDK worker 未获显式执行授权')
    protocol=sys.stdout
    def reply(value):
        protocol.write('GENE_RPC='+json.dumps(value,ensure_ascii=False,allow_nan=False)+'\n');protocol.flush()
    from gene_agent.realworld.devices.arms.realman.connection import connect_realman_read_only
    from gene_agent.realworld.devices.grippers.pika.adapter import PikaGripperAdapter
    from gene_agent.realworld.runtime import RealWorldRuntime
    from gene_agent.motion.model import JointState
    gripper=None
    # 共享连接只创建 SDK arm，不调用 Realman 内置夹爪 route/position 接口。
    with redirect_stdout(sys.stderr), connect_realman_read_only(config['robot_ip'],group='left',connect_retries=3) as connection:
        robot=connection.robot
        robot.T_flange_tcp=np.asarray(config['T_sdk_flange_operation_tcp'])
        world_from_sdk=np.linalg.inv(config['T_sdk_urdf_base'])
        def height(q):return float((world_from_sdk@robot.tcp_pose_from_sdk_fk(q))[2,3])
        def runtime(snapshot, expected_width=None):
            def guard():
                width=gripper.read_width_m()
                if expected_width is not None and abs(width-expected_width)>config['max_width_error_m']:
                    raise RuntimeError('Pika开口偏离本阶段规划/夹持状态，停止机械臂')
            return RealWorldRuntime(robot,gripper=gripper,scene_snapshot_sha256=snapshot,
                max_start_error_rad=config['max_start_error_rad'],tcp_height_fn=height,
                min_tcp_z_m=config['min_tcp_z_m'],motion_speed_scale=config['motion_speed_scale'],
                max_tcp_speed_mps=config['max_tcp_speed_mps'],canfd_frequency_hz=config['canfd_frequency_hz'],
                execution_guard=guard if expected_width is not None else None)
        def state():
            joints=robot.read_joint_state()
            return {'joint_positions_rad':list(joints.positions_rad), 'gripper_width_m':gripper.read_width_m(),
                    'T_sdk_tcp':robot.tcp_pose_from_sdk_fk(joints.positions_rad).tolist()}
        try:
            gripper=PikaGripperAdapter.connect(config['gripper_port'])
            reply({'ready':True,'state':state()})
            for line in sys.stdin:
                request=json.loads(line);identity=request['id'];cmd=request['command'];p=request.get('payload',{})
                try:
                    if cmd=='shutdown':reply({'id':identity,'ok':True,'result':{}});break
                    if cmd=='state':result=state()
                    elif cmd=='motion':
                        stage=remap_stage_names(p['stage'])
                        expected_width=float(p['stage']['observed_before']['gripper_width_m'])
                        result=runtime(stage.request.scene.snapshot_sha256,expected_width).run_motion_stage(stage,execute=True)
                    elif cmd=='gripper':
                        result=runtime(p['snapshot_sha256']).run_gripper_stage(p['name'],group='left',
                            target_width_m=p['target_width_m'],
                            planned_start=JointState(tuple(robot.joint_names),tuple(p['joint_positions_rad'])),
                            planned_start_width_m=p['gripper_width_m'],max_width_error_m=config['max_width_error_m'],
                            planned_scene_snapshot_sha256=p['snapshot_sha256'],execute=True)
                    elif cmd=='stop':robot.stop();result={}
                    else:raise ValueError('未知 SDK worker 操作')
                    reply({'id':identity,'ok':True,'result':result})
                except Exception as error:
                    reply({'id':identity,'ok':False,'error':str(error),'traceback':traceback.format_exc()})
        finally:
            if gripper is not None:gripper.close()


if __name__=='__main__':main()
