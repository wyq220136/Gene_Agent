"""真机场景按 AffordToolAct link 策略；仿真默认严格，自碰撞始终保留。"""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2]


def _settings(tmp_path):
    import mujoco
    from gene_agent.robots import build_robot,set_gripper_width
    from gene_agent.motion.curobo.core import load_realman_motion_config
    from gene_agent.motion.curobo.gripper import export_gripper_snapshot
    robot=build_robot(tmp_path/'robot',profile='realman_dual_standard')
    model=mujoco.MjModel.from_xml_path(robot['xml_path']); data=mujoco.MjData(model)
    q=json.loads((ROOT/'config/robots/realman_right_default_pose.json').read_text())['joints_rad']
    for name,value in zip(robot['joint_groups']['right'],q):
        data.qpos[model.joint(name).qposadr[0]]=value
    set_gripper_width(model,data,robot,'right',.041); mujoco.mj_forward(model,data)
    gripper=export_gripper_snapshot(model,data,robot,'right')
    return load_realman_motion_config('right',robot,gripper_snapshot=gripper),q


def _table_scene(tmp_path):
    torch=pytest.importorskip('torch'); pytest.importorskip('curobo')
    if not torch.cuda.is_available(): pytest.skip('需要 CUDA 碰撞检查')
    from gene_agent.motion.curobo import SynchronousPlanner
    settings,q=_settings(tmp_path)
    mounting={'robot_links':['base_link','link_1'],'obstacle_names':['body']}
    far={'cuboid':{'body':{'dims':[.1]*3,'pose':[10.,10.,10.,1.,0.,0.,0.]}}}
    planner=SynchronousPlanner(**settings,scene_model=far,scene_scope='offline table policy control',
                               mounting_contact_policy=mounting)
    try:
        assert planner._start_configuration_check(q)['valid']
        planner._configuration_validity([q]); checker=planner._collision_checker
        checker.setup_batch_tensors(1,1)
        state=checker.get_kinematics(torch.tensor(q,device='cuda',dtype=torch.float32).reshape(1,1,7))
        spheres=state.robot_spheres.detach().cpu().numpy().reshape(-1,4)
        world=(np.asarray(settings['world_from_base'])[:3,:3]@spheres[:,:3].T).T+np.asarray(settings['world_from_base'])[:3,3]
        # 桌面保持0.6×0.6m、厚2cm。高度选择使基座球发生毫米级重叠。
        sphere=2; cx,cy,cz=world[sphere]; r=spheres[sphere,3]
        horizontal=np.maximum(np.abs(np.array([cx,cy])-[.1,.3])-[.3,.3],0.)
        top=float(cz-np.sqrt(r*r-np.dot(horizontal,horizontal))+.002)
    finally:
        planner.close()
    T=np.eye(4); T[:3,3]=[.1,.3,top-.01]
    base_table=np.linalg.inv(np.asarray(settings['world_from_base']))@T
    from scipy.spatial.transform import Rotation
    quat=Rotation.from_matrix(base_table[:3,:3]).as_quat()[[3,0,1,2]]
    scene=deepcopy(far); scene['cuboid']['table']={'dims':[.6,.6,.02],
        'pose':[*base_table[:3,3],*quat]}
    return settings,q,mounting,scene,spheres


def test_gpu_realworld_policy_matches_afford_but_keeps_other_links_and_self(tmp_path):
    from gene_agent.motion.curobo import SynchronousPlanner
    settings,q,mounting,scene,spheres=_table_scene(tmp_path)
    planner=SynchronousPlanner(**settings,scene_model=scene,scene_scope='offline table policy control',
                               mounting_contact_policy=mounting)
    try:
        # 同一实例切换策略必须使起点缓存失效，不沿用之前的通过/拒绝。
        for ignore,expected in (([],False),(['base_link','link_1'],True),([],False),(['base_link','link_1'],True)):
            policy={**mounting,'scene_collision_ignore_links':ignore}
            planner.update_contact_policy(policy)
            assert not planner._start_validation_cache
            result=planner._start_configuration_check(q)
            assert result['valid'] is expected,result
            assert result['optimizer_valid'] is True
            if expected:
                detail=planner._collision_diagnostic(q,context_label='start_state')
                assert not detail['scene_collision'],detail
                assert 'table' in detail['ignored_scene_contacts']
                assert result['strict_mounting_check']['max_collision_costs']['self']==0.
            else:
                assert result['strict_mounting_check']['max_collision_costs']['external']>0.
    finally:
        planner.close()
    # 在非忽略的 link_2 处增加实际障碍，同一真机策略仍须拦截。
    scene['cuboid']['arm_block']={'dims':[.02]*3,'pose':[*spheres[14,:3].tolist(),1.,0.,0.,0.]}
    planner=SynchronousPlanner(**settings,scene_model=scene,scene_scope='offline table policy control',
                               mounting_contact_policy={**mounting,'scene_collision_ignore_links':['base_link','link_1']})
    try:
        result=planner._start_configuration_check(q)
        assert not result['valid']
        assert 'arm_block' in result['collision_diagnostic']['colliding_obstacles']
        # 位置更新和几何更新都要保留策略并清除起点缓存。
        moved=deepcopy(scene)
        moved['cuboid']['arm_block']['pose']=[10.,10.,10.,1.,0.,0.,0.]
        planner.update_scene(scene_model=moved,scene_scope='offline moved blocker',scene_snapshot_sha256='moved',mounting_contact_policy=planner.mounting_contact_policy)
        assert planner._start_configuration_check(q)['valid']
        resized=deepcopy(moved);resized['cuboid']['arm_block']['dims']=[.03]*3
        planner.update_scene(scene_model=resized,scene_scope='offline resized blocker',scene_snapshot_sha256='resized',mounting_contact_policy=planner.mounting_contact_policy)
        assert planner._start_configuration_check(q)['valid']
        resized['cuboid']['arm_block']['pose']=scene['cuboid']['arm_block']['pose']
        planner.update_scene(scene_model=resized,scene_scope='offline blocker returned',scene_snapshot_sha256='returned',mounting_contact_policy=planner.mounting_contact_policy)
        assert not planner._start_configuration_check(q)['valid']
    finally:
        planner.close()


def test_gpu_realworld_scene_ignore_preserves_base_self_collision(tmp_path):
    torch=pytest.importorskip('torch');pytest.importorskip('curobo')
    if not torch.cuda.is_available():pytest.skip('需要 CUDA 碰撞检查')
    from gene_agent.motion.curobo import SynchronousPlanner
    settings,_=_settings(tmp_path)
    # 固定的限位内姿态，碰撞对涉及 base_link/link_1；仅优化期屏蔽会漏掉它。
    q=[1.9743019106203237,1.3684246032280196,-.34805140807256807,2.317011780010832,
       -3.056661759146179,-.28312768907673025,.9168326112583491]
    planner=SynchronousPlanner(**settings,scene_model={'cuboid':{}},
        scene_scope='offline complete self collision control',
        mounting_contact_policy={'scene_collision_ignore_links':['base_link','link_1']})
    try:
        result=planner._start_configuration_check(q)
        assert result['optimizer_valid'] is True,result
        assert not result['valid'],result
        assert result['strict_mounting_check']['max_collision_costs']['self']>.07
        diagnostic=result['collision_diagnostic']
        assert diagnostic['joint_limit_violation'] is False
        assert diagnostic['self_collision'] is True
        assert diagnostic['scene_collision'] is False
        assert diagnostic['self_collision_sphere_pairs']
        assert any('self_collision' in row['collision_types'] for row in diagnostic['robot_collision_spheres_world'])
    finally:
        planner.close()


def test_gpu_realworld_policy_reaches_ik_and_trajectory_on_supported_curobo(tmp_path):
    from gene_agent.motion.curobo import SynchronousPlanner
    settings,q,mounting,scene,_=_table_scene(tmp_path)
    planner=SynchronousPlanner(**settings,scene_model=scene,scene_scope='offline full solver table control',
        mounting_contact_policy={**mounting,'scene_collision_ignore_links':['base_link','link_1']})
    try:
        ik=planner.solve_ik(planner.joint_names,q,planner.fk(planner.joint_names,q))
        assert ik['success'],ik
        goal=list(q);goal[0]+=.01
        trajectory=planner.plan_joint_target(planner.joint_names,q,goal,planner.fk(planner.joint_names,goal))
        assert trajectory['success'],trajectory
        assert ik['collision_verification']['valid']
        assert trajectory['collision_verification']['valid']
        assert trajectory['collision_verification']['sample_count']>1
        assert planner._planner.trajopt_solver.config.interpolation_dt==.025
    finally:
        planner.close()
