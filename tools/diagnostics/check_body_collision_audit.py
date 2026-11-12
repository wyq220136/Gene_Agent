"""复现本体迁移遗漏，验证局部安装例外与非安装碰撞保护。"""
from pathlib import Path
import argparse
import copy
import json
import hashlib
import sys
import traceback
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import mujoco
import numpy as np
from gene_agent.robots import build_robot
from gene_agent.simulation import build_scene
from gene_agent.motion import SynchronousPlanner,load_realman_motion_config
from gene_agent.motion.collision import MountingCollisionValidator
from gene_agent.motion.curobo.scene import export_collision_scene
from gene_agent.knowledge import bowl_grasp_candidates,knob_grasp_candidates
from tools.diagnostics.check_reachability import body_pose,screen,summarize
from tools.diagnostics.check_scene_collision import collision_probe



def evaluate_audit(report):
    """汇总已保存数值证据；轨迹成功和整段采样复核都是必要条件。

    此函数不初始化GPU，因此可以对历史本轮结果重验汇总，不重跑候选求解。
    旧版只有安装例外检查而没有motion证据的报告，不能冒充新版整体验收。
    """
    motion=report.get('small_motion_smoke',{})
    verification=motion.get('collision_verification',{})
    motion_pass=(bool(motion.get('success')) and bool(verification.get('valid'))
                 and verification.get('sample_count',0)>1)
    if report.get('source') == 'current_scene':
        motion_pass = motion_pass and bool(motion.get('gripper_geometry_synchronized')) and bool(motion.get('gripper_snapshot_id'))
    negative=report.get('protection_negative_cases',{})
    return bool(report['strict_start_collision']['colliding_count']>0
                and report['scoped_start_validation']['valid']
                and {'nonbase_vs_body','base_vs_external'}<=set(negative)
                and all(not result['valid'] for result in negative.values())
                and report['geometry_probe_costs'][0]>0
                and report['geometry_probe_costs'][1]==0 and motion_pass)


def run(out,use_current_scene):
    out.mkdir(parents=True,exist_ok=True)
    if use_current_scene:
        model,data,metadata=build_scene(out/'source_scene')
        robot=metadata['robot']
    else:
        # 固定前轮已验快照，避免新微波炉资产迁移影响本体语义A/B对照。
        source=Path('runs/scene-collision/nominal/source_scene/scene.xml').resolve()
        model=mujoco.MjModel.from_xml_path(str(source));data=mujoco.MjData(model)
        robot=build_robot(out/'robot_metadata_current',gripper_mode='fixed_urdf')
        for name,q in robot['home_joint_positions'].items():
            data.qpos[model.joint(name).qposadr[0]]=q
            data.ctrl[model.actuator(f'{name}_position').id]=q
    for _ in range(round(3/model.opt.timestep)):mujoco.mj_step(model,data)
    mujoco.mj_forward(model,data)
    snapshot=export_collision_scene(model,data,robot,'right',out/'collision')
    settings=load_realman_motion_config('right',robot,gripper_snapshot=snapshot.get('gripper_snapshot'))
    policy=snapshot['mounting_contact_policy']
    planner=SynchronousPlanner(**settings,scene_model=snapshot['scene_model'],scene_scope=snapshot['scope'],scene_snapshot_sha256=snapshot['snapshot_sha256'],mounting_contact_policy=policy)
    names=robot['joint_groups']['right'];q=[float(data.qpos[model.joint(name).qposadr[0]]) for name in names]
    spheres=planner.kinematics.compute_kinematics(planner._state(names,q)).robot_spheres.detach().cpu().numpy().reshape(-1,4)
    valid=spheres[:,3]>0
    rawcost=collision_probe(snapshot,spheres[valid,:3],spheres[valid,3])
    import torch
    code_paths=[Path(__file__),Path(__file__).resolve().parents[2]/'gene_agent/motion/collision.py',Path(__file__).resolve().parents[2]/'gene_agent/motion/curobo/core.py',Path(__file__).resolve().parents[2]/'gene_agent/motion/curobo/scene.py']
    report={'code_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in code_paths},
            'versions':{'torch':torch.__version__,'cuda':torch.version.cuda,'mujoco':mujoco.__version__,'gpu':torch.cuda.get_device_name()},
            'ik_seed_count':16,'trajopt_seed_count':4,'random_seed':123,'max_attempts':1,
            'GT_poses':{name:body_pose(model,data,name).tolist() for name in ['bowl','microwave']},
            'knob_concept_used':metadata['objects']['knob_concept'] if use_current_scene else 'frozen_legacy_default',
            'source':'current_scene' if use_current_scene else 'frozen_previous_scene', 'snapshot_sha256':snapshot['snapshot_sha256'],
            'policy':policy,'strict_start_collision':{'sphere_count':int(valid.sum()),'colliding_count':int(np.count_nonzero(rawcost)),'max_cost':max(rawcost)},
            'scoped_start_validation':planner._validator.check([q],planner.planner_joint_names),
            'scene_object_counts':{k:len(v) for k,v in snapshot['scene_model'].items()},'joint_names':names,'start_rad':q}
    # 已知净空/碰撞的几何探针不受安装policy影响：原始场景没有删任何障碍。
    record=next(x for x in snapshot['records'] if x['geom_name']=='scene_table')
    table=np.asarray(record['T_base_geom'])[:3,3]
    report['geometry_probe_costs']=collision_probe(snapshot,[table,[5.,5.,5.]],[.001,.001])
    kin=planner.kinematics.config.kinematics_config
    ids=kin.link_sphere_idx_map.detach().cpu().numpy();idx_to_name={v:k for k,v in kin.link_name_to_idx_map.items()}
    link7=next(i for i in range(len(spheres)) if idx_to_name[int(ids[i])]=='link_7' and spheres[i,3]>0)
    base=next(i for i in range(len(spheres)) if idx_to_name[int(ids[i])]=='base_link' and spheres[i,3]>0)
    negatives={}
    for label,index,as_body in [('nonbase_vs_body',link7,True),('base_vs_external',base,False)]:
        scene=copy.deepcopy(snapshot['scene_model']);test_name=f'audit_{label}'
        scene.setdefault('cuboid',{})[test_name]={'dims':[.02,.02,.02],'pose':[*spheres[index,:3].tolist(),1,0,0,0]}
        test_policy=copy.deepcopy(policy)
        if as_body:test_policy['obstacle_names'].append(test_name)
        validator=MountingCollisionValidator(planner.kinematics,scene,test_policy)
        negatives[label]=validator.check([q],planner.planner_joint_names)
        negatives[label]['probe_robot_link']=idx_to_name[int(ids[index])]
        negatives[label]['synthetic_obstacle']=scene['cuboid'][test_name]
    report['protection_negative_cases']=negatives
    for kind,candidates in [('bowl',bowl_grasp_candidates(body_pose(model,data,'bowl'))),('knob',knob_grasp_candidates(body_pose(model,data,'microwave'), **({'concept':metadata['objects']['knob_concept']} if use_current_scene else {})))]:
        report[kind]=screen(planner,names,q,candidates)
        for c in report[kind]:c.update(collision_checked=True,full_scene_collision_checked=True,collision_snapshot_sha256=snapshot['snapshot_sha256'])
        print(kind,summarize(report[kind]),flush=True)
    report['summary']={kind:summarize(report[kind]) for kind in ['bowl','knob']}
    target_q=np.asarray(q).copy();target_q[0]+=.04
    report['small_motion_witness_rad']=target_q.tolist()
    report['small_motion_target_world_tcp']=planner.fk(names,target_q).tolist()
    report['small_motion_smoke']=planner.plan(names,q,planner.fk(names,target_q))
    print('small_motion_smoke',report['small_motion_smoke']['success'],report['small_motion_smoke']['status'],flush=True)
    report['audit_pass']=evaluate_audit(report)
    (out/'audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('audit_pass',report['audit_pass'],'start',report['scoped_start_validation'],flush=True)
    return report['audit_pass']

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=Path('runs/body-collision-audit'));parser.add_argument('--current-scene',action='store_true');args=parser.parse_args()
    try:raise SystemExit(0 if run(args.output,args.current_scene) else 1)
    except Exception:
        args.output.mkdir(parents=True,exist_ok=True);(args.output/'raw_error.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
