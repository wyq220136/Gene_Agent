"""实际 cuRobo 失败证据及迁移后同 batch 重跑；只做离线规划。"""
import json
from pathlib import Path
import numpy as np
import pytest


def test_gpu_failed_batch_candidate_keeps_fk_constraints_and_portable_replay(tmp_path):
    torch=pytest.importorskip('torch');pytest.importorskip('curobo')
    if not torch.cuda.is_available():pytest.skip('需要 CUDA')
    import mujoco
    from gene_agent.robots import build_robot,set_gripper_width
    from gene_agent.motion.curobo.core import SynchronousPlanner,load_realman_motion_config
    from gene_agent.motion.curobo.gripper import export_gripper_snapshot
    from gene_agent.motion.curobo.adapter import CuroboPlannerAdapter
    from gene_agent.motion.curobo.diagnostics import replan_bundle
    from gene_agent.motion.model import JointState,MotionRequest,MotionOperation,PlanningSceneRef,PoseTarget
    from gene_agent.feedback.realworld_diagnostics import RealworldRunDiagnostics,inspect_realworld_run
    from gene_agent.realworld.scene_archive import load_scene_archive,prepare_relocated_request
    robot=build_robot(tmp_path/'robot',profile='realman_dual_standard')
    model=mujoco.MjModel.from_xml_path(robot['xml_path']);data=mujoco.MjData(model)
    root=Path(__file__).resolve().parents[2]
    q=json.loads((root/'config/robots/realman_right_default_pose.json').read_text())['joints_rad']
    for name,value in zip(robot['joint_groups']['right'],q):data.qpos[model.joint(name).qposadr[0]]=value
    set_gripper_width(model,data,robot,'right',.041);mujoco.mj_forward(model,data)
    gripper=export_gripper_snapshot(model,data,robot,'right')
    settings=load_realman_motion_config('right',robot,gripper_snapshot=gripper)
    world={'cuboid':{'far':{'dims':[.01]*3,'pose':[10.,10.,10.,1.,0.,0.,0.]}}}
    snapshot=dict(scene_model=world,scope='offline geometric control',snapshot_sha256='control',
        qpos_snapshot=data.qpos.tolist(),records=[],gripper_snapshot=gripper,active_group='right',
        mounting_contact_policy={'scene_collision_ignore_links':['base_link','link_1']})
    scene=dict(scene_xml_path=robot['xml_path'],scene_metadata={'robot':robot},robot_metadata=robot,
        snapshot=snapshot,provenance={'source_frame_number':7,'source_timestamp_ms':100.,
                                    'source_paths':{},'physics_stepped':False})
    planner=SynchronousPlanner(**settings,scene_model=world,scene_scope=snapshot['scope'],
        scene_snapshot_sha256='control',mounting_contact_policy=snapshot['mounting_contact_policy'])
    try:
        planner.diagnostic_snapshot=snapshot
        with RealworldRunDiagnostics(tmp_path/'run',options={'mode':'on_failure','budget_s':5.,'max_solver_calls':0}) as recorder:
            recorder.capture_initial(scene)
            adapter=CuroboPlannerAdapter(planner,group='right',scene_revision=0)
            near=planner.fk(planner.joint_names,q);far=near.copy();far[0,3]+=3.
            requests=tuple(MotionRequest(name,MotionOperation.IK,'right',JointState(tuple(planner.joint_names),tuple(q)),
                PoseTarget('world_tcp',pose),PlanningSceneRef(0,'control',snapshot['scope']),seed=123)
                for name,pose in [('near:final_ik',near),('far:final_ik',far)])
            results=adapter.evaluate_candidates(requests)
            assert [r.success for r in results]==[True,False]
            failure=results[1].failure.details
            assert len(failure['solver_diagnostic']['joint_positions_rad'])==7
            checked=failure['diagnostic']['candidates']['checked']
            assert len(checked)==1
            assert checked[0]['position_error_m']>1.
            assert checked[0]['strict_pose_converged'] is False
            assert isinstance(checked[0]['constraint_check']['joint_limit_violation'],bool)
            assert checked[0]['constraint_feasible'] is not None
            assert results[1].metrics['replay_request_ref']==results[0].metrics['replay_request_ref']
        events=inspect_realworld_run(tmp_path/'run')['events']
        assert len(events)==2 and events[1]['summary']['start_state_valid'] is True
        moved=tmp_path/'portable';(tmp_path/'run/initial_scene').rename(moved)
        restored,m,d=load_scene_archive(moved)
        np.testing.assert_allclose(d.qpos,data.qpos)
        prepared=prepare_relocated_request(moved,0,tmp_path/'relocated')
        frozen=json.loads(prepared.read_text())
        assert len(frozen['requests'])==2
        assert frozen['requests'][1]['target']['matrix']==requests[1].target.matrix.tolist()
        report=json.loads(replan_bundle(prepared,tmp_path/'replan',warm_repeats=0).read_text())
        assert [r['success'] for r in report['attempts'][0]['results']]==[True,False]
        assert report['mode']=='offline_motion_replan_no_execution'
    finally:
        planner.close()


def test_gpu_diagnostic_pairs_honor_realworld_base_scene_policy(tmp_path):
    torch=pytest.importorskip('torch');pytest.importorskip('curobo')
    if not torch.cuda.is_available():pytest.skip('需要 CUDA')
    from test_realworld_scene_collision_policy import _table_scene
    from gene_agent.motion.curobo import SynchronousPlanner
    from gene_agent.motion.curobo.diagnostics import sphere_collisions,DiagnosticBudget,DiagnosticOptions
    settings,q,mounting,world,_=_table_scene(tmp_path)
    planner=SynchronousPlanner(**settings,scene_model=world,scene_scope='offline table pair control',
        mounting_contact_policy={**mounting,'scene_collision_ignore_links':['base_link','link_1']})
    try:
        result=sphere_collisions(planner,q,{'active_group':'right','records':[]},
            DiagnosticBudget(DiagnosticOptions(mode='on_failure',budget_s=5.,max_solver_calls=0)))
        table=[p for p in result['pairs'] if p.get('second',{}).get('mesh_id')=='table']
        assert table and all(p['allowed'] for p in table),result
    finally:planner.close()
