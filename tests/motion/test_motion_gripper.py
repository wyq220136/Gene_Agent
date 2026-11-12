"""动态夹爪规划几何契约；CPU覆盖证据，不声称物理抓取或执行安全。"""
import numpy as np
import pytest
from gene_agent.motion.curobo.gripper import cover_bounds, validate_gripper_snapshot


def test_conservative_spheres_cover_box_interior_and_corners():
    bounds=np.array([[-.004,-.013,-.031],[.004,.013,.031]])
    spheres=cover_bounds(bounds,max_cell_m=.012)
    rng=np.random.default_rng(7)
    points=np.concatenate([rng.uniform(*bounds,size=(1000,3)),
                           np.array([[x,y,z] for x in bounds[:,0] for y in bounds[:,1] for z in bounds[:,2]])])
    centers=np.array([s['center'] for s in spheres]);radii=np.array([s['radius'] for s in spheres])
    distance=np.linalg.norm(points[:,None]-centers[None],axis=-1)-radii
    assert np.max(np.min(distance,axis=1)) <= 1e-12
    assert max(radii) <= np.sqrt(3)*.012/2+1e-12


def test_snapshot_tampering_is_rejected():
    with pytest.raises(ValueError,match='snapshot'):
        validate_gripper_snapshot({'snapshot_id':'wrong','width_m':.04})

@pytest.fixture(scope='module')
def dynamic_robot(tmp_path_factory):
    import mujoco
    from gene_agent.robots import build_robot
    metadata=build_robot(tmp_path_factory.mktemp('gripper'))
    model=mujoco.MjModel.from_xml_path(metadata['xml_path'])
    return model,metadata


def test_dynamic_width_snapshot_and_legacy_coverage(dynamic_robot):
    import mujoco
    from gene_agent.motion.curobo.gripper import export_gripper_snapshot
    from gene_agent.motion import load_realman_motion_config
    model,metadata=dynamic_robot
    previous=None
    uncovered=0
    for width in [0.,.007,.041,.08]:
        data=mujoco.MjData(model)
        for name in metadata['grippers']['right']['joint_names']:
            data.qpos[model.joint(name).qposadr[0]]=width/2
        snap=export_gripper_snapshot(model,data,metadata,'right')
        assert abs(snap['width_m']-width)<1e-10
        assert len(snap['geometries'])==len(metadata['grippers']['right']['collision_geom_names'])
        assert snap['snapshot_id']!=previous
        previous=snap['snapshot_id']
        uncovered+=snap['legacy_coverage']['uncovered_probe_count']
        settings=load_realman_motion_config('right',metadata,gripper_snapshot=snap)
        assert len(settings['joint_names'])==7
        assert settings['gripper_snapshot']['snapshot_id']==snap['snapshot_id']
        for geom in snap['geometries']:
            bounds=np.array(geom['bounds_planning_link_m'])
            points=np.array([[x,y,z] for x in bounds[:,0] for y in bounds[:,1] for z in bounds[:,2]])
            balls=np.array([[*s['center'],s['radius']] for s in geom['added_spheres']])
            assert np.max(np.min(np.linalg.norm(points[:,None]-balls[None,:,:3],axis=-1)-balls[None,:,3],axis=1))<=1e-8
    assert uncovered>0, '旧球在完整开口范围内不能被误标为已覆盖'


def test_no_snapshot_remains_explicit_diagnostic(dynamic_robot):
    from gene_agent.motion import load_realman_motion_config
    from gene_agent.motion.curobo.gripper import gripper_result_metadata
    model,metadata=dynamic_robot
    config=load_realman_motion_config('right',metadata)
    assert config['gripper_snapshot'] is None
    result=gripper_result_metadata(None)
    assert result['gripper_geometry_synchronized'] is False
    assert result['gripper_snapshot_id'] is None
    assert result['gripper_limitations']


def test_snapshot_does_not_change_arm_spheres_or_ignore(dynamic_robot):
    import copy,mujoco
    from gene_agent.motion import load_realman_motion_config
    from gene_agent.motion.curobo.gripper import export_gripper_snapshot
    model,metadata=dynamic_robot
    base=load_realman_motion_config('right',metadata)
    snapshot=export_gripper_snapshot(model,mujoco.MjData(model),metadata,'right')
    derived=load_realman_motion_config('right',metadata,gripper_snapshot=snapshot)
    original=base['robot_config']['kinematics'];actual=derived['robot_config']['kinematics']
    for name,spheres in original['collision_spheres'].items():
        assert actual['collision_spheres'][name][:len(spheres)]==spheres
    assert actual['self_collision_ignore']==original['self_collision_ignore']
    assert actual['collision_link_names']==original['collision_link_names']
    changed=copy.deepcopy(metadata);changed['grippers']['right']['source_hashes']['config']='invalid'
    with pytest.raises(ValueError,match='source'):
        load_realman_motion_config('right',changed,gripper_snapshot=snapshot)


def test_same_gripper_opening_has_stable_snapshot_across_arm_poses(dynamic_robot):
    """活动臂 FK 数值抖动不能改变 link-local 夹爪碰撞几何身份。"""
    import mujoco
    from gene_agent.motion.curobo.gripper import export_gripper_snapshot

    model, metadata = dynamic_robot
    data = mujoco.MjData(model)
    joints = metadata['joint_groups']['right']
    for name, value in zip(joints, [.2, -.3, .4, -.2, .1, .25, -.15], strict=True):
        data.qpos[model.joint(name).qposadr[0]] = value
    mujoco.mj_forward(model, data)
    first = export_gripper_snapshot(model, data, metadata, 'right')
    for name, value in zip(joints, [-.4, .2, -.1, .3, -.2, -.15, .25], strict=True):
        data.qpos[model.joint(name).qposadr[0]] = value
    mujoco.mj_forward(model, data)
    second = export_gripper_snapshot(model, data, metadata, 'right')
    assert first['snapshot_id'] == second['snapshot_id']
    assert first['merged_sphere_sha256'] == second['merged_sphere_sha256']

def test_current_scene_audit_requires_synchronized_gripper():
    from tools.diagnostics.check_body_collision_audit import evaluate_audit
    report={'source':'current_scene','strict_start_collision':{'colliding_count':8},
            'scoped_start_validation':{'valid':True},
            'protection_negative_cases':{'nonbase_vs_body':{'valid':False},'base_vs_external':{'valid':False}},
            'geometry_probe_costs':[.01,0.],
            'small_motion_smoke':{'success':True,'collision_verification':{'valid':True,'sample_count':21}}}
    assert not evaluate_audit(report)
    report['small_motion_smoke'].update(gripper_geometry_synchronized=True,gripper_snapshot_id='synchronized')
    assert evaluate_audit(report)
