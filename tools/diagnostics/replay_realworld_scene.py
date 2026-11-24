"""导入冻结真机场景并离线重跑；不连接相机、Pose 服务或机器人 SDK。"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main(argv=None):
    import argparse,json,os
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',required=True,help='initial_scene 目录或 manifest.json')
    parser.add_argument('--action',choices=['inspect','pick_bowl','request'],default='inspect')
    parser.add_argument('--scene-id',default='initial')
    parser.add_argument('--selection',type=int,default=0,help='保存的候选筛选下标；每次筛选使用其自己的现场')
    parser.add_argument('--request',default='0',help='冻结请求下标或文件名，保留完整 batch')
    parser.add_argument('--output',help='新输出目录')
    parser.add_argument('--render',action='store_true',help='渲染冻结现场代理，非实际执行画面')
    parser.add_argument('--allow-code-change',action='store_true',help='请求重跑沿用既有严格代码变更审计')
    args=parser.parse_args(argv)
    if (args.action!='inspect' or args.render) and not args.output:parser.error('该操作需要 --output 新目录')
    out=Path(args.output).resolve() if args.output else None
    if out is not None:
        if out.exists() and any(out.iterdir()):raise ValueError('输出目录已有内容，不能覆盖')
        out.mkdir(parents=True,exist_ok=True)
    if args.render:os.environ.setdefault('MUJOCO_GL','egl')
    from gene_agent.realworld.scene_archive import verify_scene_archive,load_scene_archive,_write
    root,manifest=verify_scene_archive(args.bundle)
    if args.action=='request':
        from gene_agent.realworld.scene_archive import prepare_relocated_request
        from gene_agent.motion.curobo.diagnostics import replan_bundle
        path=prepare_relocated_request(root,args.request,out)
        report=replan_bundle(path,out/'replan',allow_code_change=args.allow_code_change)
        print(json.dumps({'mode':'offline_request_replan_no_execution','report':str(report)},ensure_ascii=False));return 0
    selection=None
    if args.action=='pick_bowl':
        if not 0<=args.selection<len(manifest['selections']):raise ValueError('没有保存该候选筛选输入')
        selection=json.loads((root/manifest['selections'][args.selection]).read_text())
        from gene_agent.realworld.scene_archive import _hash
        project=Path(__file__).resolve().parents[2]
        code_changes=[dict(path=name,recorded_sha256=expected,
            current_sha256=_hash(project/name) if (project/name).is_file() else None)
            for name,expected in manifest.get('source_code_hashes',{}).items()
            if not (project/name).is_file() or _hash(project/name)!=expected]
        if code_changes and not args.allow_code_change:
            raise ValueError('场景包采集后的项目代码已改变；验证修复请显式 --allow-code-change，结果保留变更列表')
    scene,model,data=load_scene_archive(root,scene_id=selection['scene_id'] if selection else args.scene_id)
    summary=dict(mode='frozen_observation_proxy',physics_stepped=False,source_manifest=str(root/'manifest.json'),
        scene_id=scene['offline_replay']['scene_id'],model=scene['scene_xml_path'],
        snapshot_sha256=scene['snapshot']['snapshot_sha256'],qpos=data.qpos.tolist(),
        provenance=scene['provenance'],limitations=manifest['claim'])
    if out is not None:_write(out/'restored_scene.json',summary)
    if args.render:
        import mujoco,numpy as np
        from PIL import Image
        cam=mujoco.MjvCamera();cam.lookat[:]=[.25,.20,.15];cam.distance=1.7;cam.azimuth=140;cam.elevation=-25
        with mujoco.Renderer(model,height=480,width=640) as renderer:
            renderer.update_scene(data,camera=cam);Image.fromarray(renderer.render()).save(out/'scene_preview.png')
        _write(out/'scene_preview.json',dict(mode='frozen_observation_proxy',physics_stepped=False,
            source_manifest=str(root/'manifest.json'),camera=dict(lookat=np.asarray(cam.lookat).tolist(),
            distance=cam.distance,azimuth=cam.azimuth,elevation=cam.elevation)))
    if args.action=='pick_bowl':
        from gene_agent.feedback.realworld_diagnostics import RealworldRunDiagnostics
        from gene_agent.realworld.microwave.pick_planning import plan_observed_bowl_grasp_stages
        from gene_agent.motion.configuration import motion_configuration
        with motion_configuration(manifest['metadata'].get('motion')):
            with RealworldRunDiagnostics(out,metadata=dict(mode='offline_pick_replan',execute=False)) as recorder:
                recorder.capture_initial(scene)
                result=plan_observed_bowl_grasp_stages(scene,output_dir=out/'selection',
                    seed=selection['seed'],frozen_candidates=selection['candidates'],
                    frozen_parameters=selection['parameters'],profile_path=root/selection['profile_ref'])
        summary['code_changes']=code_changes
        summary['grasp_prefix_planned']=result['grasp_prefix_planned']
        summary['physical_pick_success']=False
    if out is not None:_write(out/'restored_scene.json',summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    return 0


if __name__=='__main__':raise SystemExit(main())
