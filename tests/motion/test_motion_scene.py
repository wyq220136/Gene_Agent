"""碰撞快照的坐标、启用语义和未知类型失败策略。"""
import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from gene_agent.motion.curobo.scene import export_collision_scene


def fixture_model(extra=''):
    model = mujoco.MjModel.from_xml_string(f'''<mujoco><worldbody>
      <body name="arm_mount"><geom name="own" type="sphere" size=".1"/></body>
      <body name="other_mount" pos="1 0 0"><geom name="other" type="box" size=".1 .2 .3"/></body>
      <geom name="transparent_contact" type="box" size=".2 .3 .4" pos="1 2 3" rgba="1 0 0 0"/>
      <geom name="visual" type="box" size=".1 .1 .1" contype="0" conaffinity="0"/>
      {extra}</worldbody></mujoco>''')
    data = mujoco.MjData(model); mujoco.mj_forward(model, data)
    transform = np.eye(4); transform[:3, :3] = Rotation.from_euler('z', .5).as_matrix(); transform[:3, 3] = [.3, .4, .5]
    metadata = dict(group_root_bodies={'arm': 'arm_mount', 'other': 'other_mount'}, joint_groups={'arm': [], 'other': []}, world_frame='test_world', transforms={'arm': {'world_from_urdf_base': transform.tolist()}})
    return model, data, metadata, transform


def test_masks_frames_and_other_arm(tmp_path):
    model, data, metadata, transform = fixture_model()
    result = export_collision_scene(model, data, metadata, 'arm', tmp_path)
    assert {r['geom_name'] for r in result['records']} == {'other', 'transparent_contact'}
    assert {r['geom_name'] for r in result['excluded']} == {'own', 'visual'}
    assert result['other_arm_frozen'] and result['frozen_groups'] == ['other']
    record = next(r for r in result['records'] if r['geom_name'] == 'transparent_contact')
    expected = np.eye(4); expected[:3, 3] = [1, 2, 3]
    np.testing.assert_allclose(np.array(record['T_base_geom']), np.linalg.inv(transform)@expected)
    np.testing.assert_allclose(result['scene_model']['cuboid'][record['obstacle_name']]['dims'], [.4, .6, .8])
    repeat = export_collision_scene(model, data, metadata, 'arm', tmp_path/'second')
    assert repeat['snapshot_sha256'] == result['snapshot_sha256']
    model.geom_pos[model.geom('transparent_contact').id, 0] += .1; mujoco.mj_forward(model, data)
    changed = export_collision_scene(model, data, metadata, 'arm', tmp_path/'changed')
    assert changed['snapshot_sha256'] != result['snapshot_sha256']


def test_explicit_attached_object_transfer_changes_scene_hash(tmp_path):
    model, data, metadata, _ = fixture_model()
    before = export_collision_scene(model, data, metadata, 'arm', tmp_path / 'before')
    transfer = {'other_mount': [[1.0, 0.0, 0.1, 0.05]]}
    after = export_collision_scene(model, data, metadata, 'arm', tmp_path / 'after',
                                   transferred_objects=transfer)
    assert not any(row['geom_name'] == 'other' for row in after['records'])
    excluded = next(row for row in after['excluded'] if row['geom_name'] == 'other')
    assert excluded['reason'] == 'transferred_to_attached_robot_proxy'
    assert excluded['object_instance_id'] == 'other_mount'
    assert after['transferred_objects'] == transfer
    assert after['snapshot_sha256'] != before['snapshot_sha256']
    with pytest.raises(ValueError, match='必须存在'):
        export_collision_scene(model, data, metadata, 'arm', tmp_path / 'bad',
                               transferred_objects={'unknown': transfer['other_mount']})


def test_dynamic_articulation_is_excluded_only_when_requested(tmp_path):
    model, data, metadata, _ = fixture_model()
    strict = export_collision_scene(model, data, metadata, 'arm', tmp_path / 'strict')
    moving = export_collision_scene(model, data, metadata, 'arm', tmp_path / 'moving',
                                    moving_body_names=('other_mount',))
    assert any(row['geom_name'] == 'other' for row in strict['records'])
    assert not any(row['geom_name'] == 'other' for row in moving['records'])
    assert any(row['geom_name'] == 'transparent_contact' for row in moving['records'])
    excluded = next(row for row in moving['excluded'] if row['geom_name'] == 'other')
    assert excluded['reason'] == 'moving_articulation_requires_per_sample_validation'
    assert moving['moving_body_names'] == ['other_mount']
    assert moving['snapshot_sha256'] != strict['snapshot_sha256']
    with pytest.raises(ValueError, match='活动组'):
        export_collision_scene(model, data, metadata, 'arm', tmp_path / 'bad-own',
                               moving_body_names=('arm_mount',))


def test_explicit_stage_collision_policy_disables_body_without_mislabeling_motion(tmp_path):
    model, data, metadata, _ = fixture_model()
    strict = export_collision_scene(model, data, metadata, 'arm', tmp_path / 'strict')
    relaxed = export_collision_scene(model, data, metadata, 'arm', tmp_path / 'relaxed',
                                     disabled_body_names=('other_mount',))
    assert not any(row['geom_name'] == 'other' for row in relaxed['records'])
    assert any(row['geom_name'] == 'transparent_contact' for row in relaxed['records'])
    excluded = next(row for row in relaxed['excluded'] if row['geom_name'] == 'other')
    assert excluded['reason'] == 'stage_collision_policy_disabled'
    assert excluded['disabled_body_name'] == 'other_mount'
    assert relaxed['disabled_body_names'] == ['other_mount']
    assert relaxed['moving_body_names'] == []
    assert relaxed['snapshot_sha256'] != strict['snapshot_sha256']
    with pytest.raises(ValueError, match='活动组'):
        export_collision_scene(model, data, metadata, 'arm', tmp_path / 'bad-own-disabled',
                               disabled_body_names=('arm_mount',))
    with pytest.raises(ValueError, match='不存在'):
        export_collision_scene(model, data, metadata, 'arm', tmp_path / 'bad-unknown',
                               disabled_body_names=('unknown',))


def test_unknown_enabled_shape_fails(tmp_path):
    # plane 已由有限支撑代理支持；使用仍不支持的 ellipsoid 检查拒绝分支。
    model, data, metadata, _ = fixture_model('<geom name="ellipsoid" type="ellipsoid" size=".1 .2 .3"/>')
    with pytest.raises(ValueError, match='不支持'):
        export_collision_scene(model, data, metadata, 'arm', tmp_path)


def test_cylinder_has_explicit_proxy(tmp_path):
    model, data, metadata, _ = fixture_model('<geom name="cylinder" type="cylinder" size=".05 .02"/>')
    result = export_collision_scene(model, data, metadata, 'arm', tmp_path)
    record = next(r for r in result['records'] if r['geom_name']=='cylinder')
    assert record['approximation'] == 'cylinder_circumscribed_64_segments'
    assert record['obstacle_name'] in result['scene_model']['mesh']


def test_compiled_mesh_recentering_is_applied_once(tmp_path):
    """源mesh非零原点，MuJoCo编译会重心/主轴变换；导出世界顶点须仍匹配源几何。"""
    import trimesh
    xml = '''<mujoco><asset><mesh name="offset" vertex="2 0 0 3 0 0 2 1 0 2 0 1"/></asset><worldbody>
    <body name="arm_mount"/><geom name="shifted_mesh" type="mesh" mesh="offset" pos="1 2 3"/>
    </worldbody></mujoco>'''
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model); mujoco.mj_forward(model, data)
    metadata = {'group_root_bodies': {'arm': 'arm_mount'}, 'joint_groups': {'arm': []}, 'world_frame': 'world', 'transforms': {'arm': {'world_from_urdf_base': np.eye(4).tolist()}}}
    result = export_collision_scene(model,data,metadata,'arm',tmp_path)
    record=result['records'][0]
    mesh=trimesh.load(record['mesh_path'],force='mesh',process=False)
    transform=np.asarray(record['T_world_geom'])
    actual=mesh.vertices@transform[:3,:3].T+transform[:3,3]
    expected=np.array([[3,2,3],[4,2,3],[3,3,3],[3,2,4]])
    # 凸包可重排顶点，使用最近邻集合距离而非数组顺序比较。
    assert np.max(np.min(np.linalg.norm(actual[:,None]-expected[None],axis=2),axis=1))<1e-6


def test_post_step_gt_uses_current_qpos_without_mutation(tmp_path):
    """导出入口自己刷新副本缓存，不能要求调用方额外forward或更改其warmstart。"""
    model=mujoco.MjModel.from_xml_string('''<mujoco><option gravity="0 0 0" timestep=".01"/>
    <worldbody><body name="custom_active_root"/><body name="moving"><freejoint/>
    <geom name="moving_geom" type="sphere" size=".1" mass="1"/></body></worldbody></mujoco>''')
    data=mujoco.MjData(model); data.qvel[0]=1.
    mujoco.mj_step(model,data)
    before_xpos=data.geom_xpos.copy(); before_warmstart=data.qacc_warmstart.copy()
    assert not np.allclose(data.geom_xpos[0,0],data.qpos[0])
    metadata={'group_root_bodies':{'arm':'custom_active_root'},'joint_groups':{'arm':[]},'world_frame':'test',
              'transforms':{'arm':{'world_from_urdf_base':np.eye(4).tolist()}}}
    result=export_collision_scene(model,data,metadata,'arm',tmp_path)
    np.testing.assert_allclose(np.asarray(result['records'][0]['T_world_geom'])[:3,3],data.qpos[:3])
    np.testing.assert_array_equal(data.geom_xpos,before_xpos)
    np.testing.assert_array_equal(data.qacc_warmstart,before_warmstart)
    assert result['active_group_root_body']=='custom_active_root'


def test_explicit_contact_pair_fails_instead_of_silent_omission(tmp_path):
    model=mujoco.MjModel.from_xml_string('''<mujoco><worldbody><body name="root"/>
    <geom name="a" type="sphere" size=".1" contype="0" conaffinity="0"/>
    <geom name="b" type="sphere" size=".1" pos="0 0 .1" contype="0" conaffinity="0"/>
    </worldbody><contact><pair geom1="a" geom2="b"/></contact></mujoco>''')
    data=mujoco.MjData(model)
    with pytest.raises(NotImplementedError,match='contact pair'):
        export_collision_scene(model,data,{'group_root_bodies':{'arm':'root'},'joint_groups':{'arm':[]}},'arm',tmp_path)


@pytest.mark.parametrize('field',['geom_gap'])
def test_nonzero_contact_padding_fails_explicitly(tmp_path,field):
    model,data,metadata,_=fixture_model()
    getattr(model,field)[0]=.001
    with pytest.raises(NotImplementedError,match='margin/gap'):
        export_collision_scene(model,data,metadata,'arm',tmp_path)


def test_static_internal_pair_is_recorded_and_margin_expands_boxes(tmp_path):
    model=mujoco.MjModel.from_xml_string('''<mujoco><worldbody><body name="active"/>
    <geom name="cabinet" type="box" size=".1 .2 .3" margin=".0001"/>
    <geom name="door" type="box" size=".1 .2 .3" pos="1 0 0"/>
    </worldbody><contact><pair geom1="cabinet" geom2="door" margin=".0002"/></contact></mujoco>''')
    data=mujoco.MjData(model)
    metadata={'group_root_bodies':{'arm':'active'},'joint_groups':{'arm':[]},'world_frame':'world',
              'transforms':{'arm':{'world_from_urdf_base':np.eye(4).tolist()}}}
    result=export_collision_scene(model,data,metadata,'arm',tmp_path)
    assert len(result['static_internal_contact_pairs'])==1
    assert len(result['records'])==2
    for record in result['records']:
        assert record['contact_margin_m']==pytest.approx(.0002)
        np.testing.assert_allclose(result['scene_model']['cuboid'][record['obstacle_name']]['dims'],[.2004,.4004,.6004])
