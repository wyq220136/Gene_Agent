"""本次运行的初态预览：使用正式初始化与同一 seed，不执行机器人动作。"""
import argparse
from pathlib import Path
import time
import sys
import numpy as np
from gene_agent.feedback.live_record import atomic_json


def build_physical(task,source,output,*,seed,randomize,collision_mode='full'):
    from gene_agent.motion.configuration import motion_configuration
    if task in {'sweep','microwave'}:
        from .scene_preview import build_case
        with motion_configuration({'collision_mode':collision_mode}):
            case,sample,_=build_case(task,source,output/'case',seed,randomize)
        return case.model,case.data,sample
    if task=='sweep_long':
        from gene_agent.tasks.policies.sweep_long_task.simulation import prepare_long_brush_scene
        from gene_agent.tasks.policies.sweep_cube_task.simulation import SweepCubeRun
        adapter=prepare_long_brush_scene(source,output/'case',layout_seed=seed if randomize else None)
        with motion_configuration({'collision_mode':collision_mode}):
            case=SweepCubeRun(adapter,output/'case',prepared_output_dir=True)
        return case.model,case.data,dict(placement=case.placement,layout_seed=seed if randomize else None)
    from tools.viewers.view_benchmark import prepare
    from .catalog import TASKS
    from tools.viewers.view_reconstructed_tabletop_scene import load_scene
    scene=prepare(TASKS[task]['benchmark_id'],source,output/'case',seed=seed,randomize=randomize)
    spec,model,data=load_scene(scene)
    return model,data,dict(source_scene=str(scene),placement=spec.get('placement'),layout_seed=seed if randomize else None)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task',required=True);p.add_argument('--scene',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--port',type=int,required=True)
    p.add_argument('--execution-mode',choices=('physical','visual'),default='physical')
    p.add_argument('--collision-mode',choices=('full','robot_only'),default='full')
    p.add_argument('--seed',type=int,default=123);p.add_argument('--randomize',action='store_true')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    manifest=dict(task=args.task,source=str(args.scene.resolve()),seed=args.seed,randomize=args.randomize,
                  execution_mode=args.execution_mode,robot_motion_executed=False)
    if args.execution_mode=='visual':
        from gene_agent.visualization.pure_planning.app import PurePlanningApp
        import viser
        server=viser.ViserServer(host='127.0.0.1',port=args.port)
        app=PurePlanningApp(server,task=args.task,source=args.scene,output=args.output/'inputs',
                            seed=args.seed,randomize=args.randomize,initial_only=True)
        manifest.update(physics_used=False,poses_world={k:v['T_world_object'] for k,v in app.inputs.objects.items()})
    else:
        model,data,record=build_physical(args.task,args.scene,args.output,seed=args.seed,randomize=args.randomize,collision_mode=args.collision_mode)
        from gene_agent.visualization.pick_bowl.scene import PickBowlScene
        import viser
        server=viser.ViserServer(host='127.0.0.1',port=args.port)
        server.scene.set_up_direction('+z')
        server.gui.configure_theme(dark_mode=False,control_width='medium',show_logo=False)
        server.gui.add_markdown('## 本次运行 · 场景预览\n使用正式场景初始化；**没有执行机器人运动或验证任务成功**。可拖动视角检查摆位。')
        PickBowlScene(server,model,data,{},{}).set_collision_visible(False)
        camera_target=(.35,.20,.25) if args.task=='microwave' else (.30,0.,.20)
        server.initial_camera.position=tuple(np.asarray(camera_target)+[-.75,.75,.65]);server.initial_camera.look_at=camera_target
        server.initial_camera.up=(0.,0.,1.);server.initial_camera.fov=.8
        manifest.update(physics_used=True,initialization=record,qpos=data.qpos.tolist(),sim_time_s=float(data.time))
    server.gui.add_markdown(f"**输入：** `{args.scene}`\n**摆位：** {'随机' if args.randomize else '固定'}；seed={args.seed}。\n修改场景或 seed 后请重新预览。")
    manifest['mujoco_imported']='mujoco' in sys.modules
    if args.execution_mode=='visual' and manifest['mujoco_imported']:raise RuntimeError('纯视觉初态预览意外导入 MuJoCo')
    atomic_json(args.output/'preview_manifest.json',manifest)
    print(f'本次运行场景：http://127.0.0.1:{server.get_port()}；未执行机器人动作',flush=True)
    try:
        while True:time.sleep(.1)
    except KeyboardInterrupt:pass
    finally:server.stop()

if __name__=='__main__':main()
