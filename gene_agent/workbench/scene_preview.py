"""两类任务共用的初态预览与场景编辑；不求解机器人运动。"""
import argparse
import json
import os
from pathlib import Path
import shutil
import threading
import traceback
from types import SimpleNamespace

os.environ.setdefault('MUJOCO_GL','egl')


def build_case(task,source,output,seed,randomize):
    from .scene_inputs import load_input,suggested_rules
    root=Path(__file__).resolve().parents[2]
    source,data=load_input(root,task,source)
    if task=='sweep':
        from gene_agent.tasks.policies.sweep_cube_task.simulation import SweepCubeRun
        case=SweepCubeRun(source,output,randomize=randomize,seed=seed)
        policy=case.placement.get('policy') or {}
        sampled=policy.get('sampled_poses_world') or {}
        if not sampled:
            # 旧无规则基线从实际落稳记录取姿态，明确来源。
            import numpy as np
            for name in data['objects']:
                body=case.data.body(name);pose=np.eye(4)
                pose[:3,:3]=body.xmat.reshape(3,3);pose[:3,3]=body.xpos;sampled[name]=pose.tolist()
        sample={'task':task,'source':str(source),'seed':seed,
                'sampled_poses_world':sampled,'placement':case.placement,
                'valid':case.placement['valid_initial_state']}
        candidates=list(case.output_dir.rglob('initial*.png'))
        image=candidates[0] if candidates else None
    else:
        from benchmarks.microwave_task.scene import resolve_settings,build_initial_scene
        from PIL import Image
        import mujoco
        output.mkdir(parents=True)
        if randomize and 'placement_rules' not in data:data['placement_rules']=suggested_rules(task,data)
        settings=resolve_settings(data,seed=seed,randomize=randomize)
        model,state,metadata,record=build_initial_scene(settings,output/'scene')
        case=SimpleNamespace(model=model,data=state,scene={'objects':{n:{} for n in ('microwave','bowl')}},
            scene_dir=source,output_dir=output,metadata=metadata)
        case.visual_roots={'microwave':('microwave','microwave_support'),'bowl':('bowl',)}
        sample={'task':task,'source':str(source),'seed':seed,'sampled_poses_world':record['sampled_poses_world'],
                'placement':record,'valid':record['valid']}
        camera=mujoco.MjvCamera();camera.azimuth=-45;camera.elevation=-28;camera.distance=2.1
        camera.lookat[:]=[.40,.25,.22]
        option=mujoco.MjvOption();option.geomgroup[3]=0
        image=output/'initial.png'
        with mujoco.Renderer(model,height=480,width=640) as renderer:
            renderer.update_scene(state,camera=camera,scene_option=option)
            Image.fromarray(renderer.render()).save(image)
    case.task=task
    case.rules=data.get('placement_rules') or suggested_rules(task,data)
    from .sample_viewer import save_snapshot
    save_snapshot(case,output,sample)
    (output/'sample.json').write_text(json.dumps(sample,ensure_ascii=False,indent=2))
    return case,sample,image


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--task',choices=('sweep','microwave'),default='sweep')
    parser.add_argument('--seed',type=int,default=36)
    parser.add_argument('--port',type=int)
    parser.add_argument('--nominal',action='store_true')
    parser.add_argument('--edit',action='store_true')
    parser.add_argument('--workbench-url')
    args=parser.parse_args()
    if args.edit and (not args.port or not args.workbench_url):parser.error('编辑模式需要 --port 和 --workbench-url')
    args.output.mkdir(parents=True,exist_ok=True)
    rows=[];first=None
    for seed in range(args.seed,args.seed+(1 if args.port else 6)):
        output=args.output/f'seed_{seed}'
        try:
            case,sample,image=build_case(args.task,args.scene,output,seed,not (args.nominal or args.edit))
            first=first or case
            row={'seed':seed,'valid':sample['valid'],'placement':sample['placement']}
            if image:
                target=args.output/f'seed_{seed}.png';shutil.copy2(image,target);row['image']=str(target)
        except Exception as exc:
            output.mkdir(parents=True,exist_ok=True)
            (output/'error.txt').write_text(traceback.format_exc())
            row={'seed':seed,'valid':False,'error':f'{type(exc).__name__}: {exc}'}
            if args.port:raise
        rows.append(row)
        (args.output/'distribution.json').write_text(json.dumps({'task':args.task,
            'source':str(args.scene),'robot_motion_planned':False,'samples':rows},ensure_ascii=False,indent=2))
    if not args.port:return
    import viser
    from gene_agent.visualization.pick_bowl.scene import PickBowlScene
    server=viser.ViserServer(host='127.0.0.1',port=args.port)
    server.scene.set_up_direction('+z')
    camera_position=(-.9,1.3,1.0) if args.task=='sweep' else (-1.,1.6,1.05)
    camera_target=(.18,-.05,.25) if args.task=='sweep' else (.35,.25,.3)
    server.initial_camera.position=camera_position;server.initial_camera.look_at=camera_target
    server.initial_camera.up=(0,0,1);server.initial_camera.fov=.8
    scene=PickBowlScene(server,first.model,first.data,{}, {})
    if args.edit:
        from .scene_editor import SceneEditor
        editor=SceneEditor(server,first,scene,workbench_url=args.workbench_url,output=args.output)
    else:server.gui.add_markdown('## 场景初态\n尚未规划运动。')
    server.scene.add_frame('/world',axes_length=.1,axes_radius=.002)
    scene.update_visuals(first.data)
    @server.on_client_connect
    def connect(client):
        client.camera.fov=.8;client.camera.position=camera_position
        client.camera.look_at=camera_target;client.camera.up=(0,0,1)
    print(f'场景编辑器 http://127.0.0.1:{args.port}',flush=True)
    try:threading.Event().wait()
    finally:server.stop()


if __name__=='__main__':main()
