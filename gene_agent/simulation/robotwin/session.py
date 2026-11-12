"""只在原生独立进程装配RoboTwin场景，保留原初始化与资产来源。"""
from __future__ import annotations
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import sys


def bootstrap_native(external_root):
    """拒绝同一进程混装两套benchmarks包；主启动器必须使用子进程。"""
    root=Path(external_root).resolve();rw=root/'env/RoboTwin'
    if not rw.is_dir():raise FileNotFoundError(rw)
    existing=sys.modules.get('benchmarks')
    if existing is not None:
        locations=[Path(p).resolve() for p in getattr(existing,'__path__',())]
        if (rw/'benchmarks').resolve() not in locations:
            raise RuntimeError('已导入其他benchmarks命名空间；请用独立原生进程')
    sys.path[:0]=[str(rw),str(rw/'script'),str(root/'src')]
    os.chdir(rw)
    return rw


@contextmanager
def native_episode(request, *, seed, output_dir):
    """只做原reset与资源管理；不复制原判据，也不调用专家充当本项目策略。"""
    rw=bootstrap_native(request['external_root'])
    from workflows.eval.eval_robotwin_motion_plan import load_task_args,class_decorator
    from script.task_setup import resolve_task_setup,install_task_setup
    from benchmarks.novel_tool.hit_suite import load_suite
    suite=Path(request['suite'])
    if hashlib.sha256(suite.read_bytes()).hexdigest()!=request['suite_sha256']:
        raise ValueError('套件在启动后已变化')
    load_suite(suite)  # 使用原套件验证，不在本项目重写其模型/分组规则。
    task=request['entry']['task']
    options=dict(task_setup='shared',tool_model_id=request['entry']['model_id'])
    if request.get('family')=='hit':options['tool_placement']='fixed'
    args=resolve_task_setup(task,load_task_args(task,request.get('task_config','novel_tool_clean')),options)
    out=Path(output_dir).resolve();out.mkdir(parents=True,exist_ok=True)
    args.update(save_path=str(out),save_data=False,collect_data=False,save_video=False,eval_video_log=False,
                save_freq=None,render_freq=0,is_test=True,now_ep_num=0)
    (out/'resolved_inputs.json').write_text(json.dumps(args,ensure_ascii=False,indent=2,default=str))
    env=class_decorator(task);install_task_setup(env,args)
    try:
        env.setup_demo(**dict(args,seed=int(seed)))
        yield env,args
    finally:
        if hasattr(env,'scene'):env.close_env(clear_cache=True)


def robot_metadata(env,args):
    """从实际base link和生效配置建立ALOHA；不使用frame_bias估计世界摆位。"""
    from gene_agent.robots.aloha import load_aloha_metadata
    from .runtime import pose_matrix
    links={link.name:link for link in env.robot.left_entity.get_links()}
    configs=args['left_embodiment_config']
    bases={group:pose_matrix(links[f'{prefix}_base_link'].get_entity_pose()).tolist() for group,prefix in [('left','fl'),('right','fr')]}
    frames={group:pose_matrix(getattr(env.robot,f'{group}_ee').get_pose_in_child()).tolist() for group in ('left','right')}
    return load_aloha_metadata(embodiment_dir=Path(args['left_robot_file']),resolved_config=configs,world_from_bases=bases,link_from_ee_frames=frames)


def capture_scene(env,out):
    """直接保存原生RGB，保留相机标定，不将状态转换到MuJoCo。"""
    from PIL import Image
    import numpy as np
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    env._update_render();env.cameras.update_picture()
    frames=env.cameras.get_rgb()
    for name,row in frames.items():
        Image.fromarray(np.asarray(row['rgb'],dtype=np.uint8)).save(out/f'{name}.png')
    camera_rows=[]
    for camera in env.cameras.static_camera_list:
        camera_rows.append(dict(name=camera.name,width=camera.width,height=camera.height,
            intrinsic_cv=camera.get_intrinsic_matrix().tolist(),extrinsic_cv=camera.get_extrinsic_matrix().tolist()))
    (out/'cameras.json').write_text(json.dumps(camera_rows,indent=2))
    return sorted(frames)


def main():
    """场景检查只声明场景/模型，不产生任务成功率。"""
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();request=json.loads(args.request.read_text())['request']
    output=args.output.resolve()
    seeds=[request.get('seed',0)] if request.get('viewer') else [request.get('seed',0),request.get('seed',0)+7,request.get('seed',0)+21]
    for seed in seeds:
        folder=output/f'seed_{seed}'
        with native_episode(request,seed=seed,output_dir=folder) as (env,resolved):
            metadata=robot_metadata(env,resolved)
            (folder/'robot_metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2))
            capture_scene(env,folder/'views')
            if request.get('viewer'):
                from sapien.utils.viewer import Viewer
                viewer=Viewer(env.renderer);viewer.set_scene(env.scene)
                viewer.set_camera_xyz(1.,-.5,1.6);viewer.set_camera_rpy(0.,-.7,2.3)
                while not viewer.closed:
                    env._update_render();viewer.render()
    print('原生初态检查产物:',output)


if __name__=='__main__':main()
