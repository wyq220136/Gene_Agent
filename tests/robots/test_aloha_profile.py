"""用独立小模型检查源配置、TCP 和实际夹爪锁关节，不依赖本机大资产。"""
import hashlib
from pathlib import Path
import numpy as np
import pytest
import yaml


def source(tmp_path):
    """六轴双臂参考：TCP局部X平移12cm，旋转diag(1,-1,-1)。"""
    config = dict(arm_joints_name=[], move_group=['l6', 'r6'], gripper_bias=.12,
                  global_trans_matrix=[[1,0,0],[0,-1,0],[0,0,-1]], delta_matrix=np.eye(3).tolist(),
                  gripper_scale=[-.01,.045], gripper_name=[])
    joints=[]
    for side in ['l','r']:
        names=[f'{side}{i}' for i in range(1,7)];config['arm_joints_name'].append(names)
        config['gripper_name'].append({'base': f'{side}7', 'mimic': [[f'{side}8',1.,0.]]})
        for i in (7,8):
            joints.append(f'<joint name="{side}{i}" type="prismatic"><child link="{side}_finger{i}"/><limit lower="0" upper=".04785" velocity="1"/></joint>')
        for n in names:
            joints.append(f'<joint name="{n}" type="revolute"><limit lower="-2" upper="2" velocity="1"/></joint>')
    (tmp_path/'robot.urdf').write_text('<robot>'+''.join(joints)+'</robot>')
    config['urdf_path']='robot.urdf'
    (tmp_path/'config.yml').write_text(yaml.safe_dump(config))
    for group,side in [('left','l'),('right','r')]:
        cfg={'robot_cfg':{'kinematics':{'base_link':side+'base','ee_link':side+'6',
             'urdf_path':'${ASSETS_PATH}/robot.urdf','collision_spheres':{side+'6':[{'center':[0,0,0],'radius':.03}]},
             'lock_joints':{side+'7':.04,side+'8':.04},
             'cspace':{'joint_names':[side+str(i) for i in range(1,9)],'retract_config':[0]*8}}}}
        (tmp_path/f'curobo_{group}_tmp.yml').write_text(yaml.safe_dump(cfg))
    return config


def test_source_metadata_frame_and_asset_identity(tmp_path):
    from gene_agent.robots.aloha import load_aloha_metadata
    cfg=source(tmp_path)
    bases={'left':np.eye(4).tolist(),'right':np.eye(4).tolist()};bases['right'][0][3]=.4
    robot=load_aloha_metadata(embodiment_dir=tmp_path,resolved_config=cfg,world_from_bases=bases,link_from_ee_frames={g:np.eye(4).tolist() for g in ['left','right']})
    assert robot['joint_groups']['left']==['l1','l2','l3','l4','l5','l6']
    assert robot['tool_links']['right']=='r6'
    np.testing.assert_allclose(robot['transforms']['left']['link_from_tcp'],[[1,0,0,.12],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
    assert robot['source_hashes'][str(tmp_path/'robot.urdf')]==hashlib.sha256((tmp_path/'robot.urdf').read_bytes()).hexdigest()
    assert robot['grippers']['left']['width_range_m']==[0.,.09]


def test_planning_locks_actual_fingers_and_keeps_six_arm_dofs(tmp_path):
    from gene_agent.robots.aloha import load_aloha_metadata, make_aloha_gripper_snapshot
    from gene_agent.motion.curobo.profiles import load_motion_config
    cfg=source(tmp_path)
    robot=load_aloha_metadata(embodiment_dir=tmp_path,resolved_config=cfg,world_from_bases={g:np.eye(4).tolist() for g in ['left','right']},link_from_ee_frames={g:np.eye(4).tolist() for g in ['left','right']})
    first=make_aloha_gripper_snapshot(robot,'left',{'l7':.01,'l8':.012})
    config=load_motion_config('left',robot,gripper_snapshot=first)
    assert config['tool_frame']=='l6'
    assert config['planner_joint_names']==['l1','l2','l3','l4','l5','l6']
    assert config['robot_config']['kinematics']['lock_joints']=={'l7':.01,'l8':.012}
    second=make_aloha_gripper_snapshot(robot,'left',{'l7':.02,'l8':.02})
    assert second['snapshot_id']!=first['snapshot_id']
    assert config['robot_config']['kinematics']['lock_joints']['l7']==.01
    first['joint_positions_m']['l7']=.03
    with pytest.raises(ValueError): load_motion_config('left',robot,gripper_snapshot=first)


def test_aloha_snapshot_matches_existing_planner_sphere_contract(tmp_path):
    from gene_agent.robots.aloha import load_aloha_metadata, make_aloha_gripper_snapshot
    from gene_agent.motion.curobo.gripper import gripper_result_metadata, _digest
    cfg=source(tmp_path)
    robot=load_aloha_metadata(embodiment_dir=tmp_path,resolved_config=cfg,world_from_bases={g:np.eye(4).tolist() for g in ['left','right']},link_from_ee_frames={g:np.eye(4).tolist() for g in ['left','right']})
    snap=make_aloha_gripper_snapshot(robot,'left',{'l7':.01,'l8':.01})
    assert snap['merged_sphere_sha256']==_digest(robot['planning_assets']['left']['robot_config']['kinematics']['collision_spheres'])
    assert gripper_result_metadata(snap)['gripper_snapshot_id']==snap['snapshot_id']


def test_diagnostic_rejects_missing_source_before_creating_output(tmp_path):
    import subprocess,sys
    output=tmp_path/'result'
    result=subprocess.run([sys.executable,'tools/diagnostics/check_embodiment.py','--source','robotwin','--external-root',str(tmp_path/'absent'),'--output',str(output)],capture_output=True,text=True)
    assert result.returncode!=0
    assert not output.exists()
    assert 'hit_tools.yaml' in result.stderr


def test_native_null_extra_links_normalized_without_mutating_source(tmp_path):
    from gene_agent.robots.aloha import load_aloha_metadata,make_aloha_gripper_snapshot
    from gene_agent.motion.curobo.profiles import load_motion_config
    from gene_agent.motion.curobo.core import add_tcp_tool_frame
    cfg=source(tmp_path)
    path=tmp_path/'curobo_left_tmp.yml';raw=yaml.safe_load(path.read_text())
    raw['robot_cfg']['kinematics']['extra_links']=None;path.write_text(yaml.safe_dump(raw))
    robot=load_aloha_metadata(embodiment_dir=tmp_path,resolved_config=cfg,world_from_bases={g:np.eye(4).tolist() for g in ['left','right']},link_from_ee_frames={g:np.eye(4).tolist() for g in ['left','right']})
    snap=make_aloha_gripper_snapshot(robot,'left',{'l7':.01,'l8':.01})
    result=load_motion_config('left',robot,gripper_snapshot=snap)
    adapted=add_tcp_tool_frame(result['robot_config'],result['tool_frame'],np.asarray(result['link_from_tcp']))
    assert adapted['kinematics']['extra_links']['gene_tcp']['parent_link_name']=='l6'
    assert robot['planning_assets']['left']['robot_config']['kinematics']['extra_links'] is None


def test_legacy_native_schema_adapts_to_modern_tool_frames(tmp_path):
    from gene_agent.robots.aloha import load_aloha_metadata, make_aloha_gripper_snapshot
    from gene_agent.motion.curobo.profiles import load_motion_config
    cfg=source(tmp_path)
    path=tmp_path/'curobo_left_tmp.yml';raw=yaml.safe_load(path.read_text())
    raw['robot_cfg']['kinematics']['use_usd_kinematics']=False
    path.write_text(yaml.safe_dump(raw))
    robot=load_aloha_metadata(embodiment_dir=tmp_path,resolved_config=cfg,world_from_bases={g:np.eye(4).tolist() for g in ['left','right']},link_from_ee_frames={g:np.eye(4).tolist() for g in ['left','right']})
    snap=make_aloha_gripper_snapshot(robot,'left',{'l7':.01,'l8':.01})
    kin=load_motion_config('left',robot,gripper_snapshot=snap)['robot_config']['kinematics']
    assert kin['tool_frames']==['l6']
    assert kin['cspace']['default_joint_position']==[0,0,0,0,0,0,.01,.01]
    assert 'retract_config' not in kin['cspace']
    assert 'ee_link' not in kin and 'use_usd_kinematics' not in kin
    assert robot['planning_assets']['left']['robot_config']['kinematics']['ee_link']=='l6'
    robot['planning_assets']['left']['robot_config']['kinematics']['use_usd_kinematics']=True
    with pytest.raises(ValueError,match='USD'):
        load_motion_config('left',robot,gripper_snapshot=snap)


def test_known_collision_requires_expected_obstacle_evidence():
    from tools.diagnostics.check_embodiment import known_collision_check
    from types import SimpleNamespace
    def result(kind,obstacles):
        return SimpleNamespace(success=False,failure=SimpleNamespace(kind=SimpleNamespace(value=kind),details={'collision_verification':{'start_check':{'collision_diagnostic':{'colliding_obstacles':obstacles}}}}))
    assert not known_collision_check(result('path_not_found',[]))
    assert not known_collision_check(result('start_collision',['other']))
    assert known_collision_check(result('start_collision',['diagnostic_tcp_block']))


def test_native_joint_child_frame_is_applied_once(tmp_path):
    from gene_agent.robots.aloha import load_aloha_metadata
    cfg=source(tmp_path)
    child=np.diag([1.,-1.,-1.,1.])
    robot=load_aloha_metadata(embodiment_dir=tmp_path,resolved_config=cfg,
        world_from_bases={g:np.eye(4).tolist() for g in ['left','right']},
        link_from_ee_frames={g:child.tolist() for g in ['left','right']})
    tcp=np.eye(4);tcp[0,3]=.12
    np.testing.assert_allclose(robot['transforms']['left']['link_from_tcp'],tcp,atol=1e-12)
