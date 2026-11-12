"""原生场景快照必须保留目标与另一臂，且拒绝过期状态和未知几何。"""
from types import SimpleNamespace
import numpy as np
import pytest
from gene_agent.skills.contracts import OperationState
from gene_agent.motion import CollisionPolicy


def state():
    return OperationState('current','fixture',dict(episode_id='episode',world_frame='world',sim_time_s=0.,robots={'left':{'gripper':{'joint_positions_m':{'g7':.01,'g8':.01}}}},objects={}))


class Box:
    def __init__(self):self.half_size=np.array([.1,.2,.3]);self.local_pose=Pose()


class Pose:
    def __init__(self):self.p=np.zeros(3)
    def to_transformation_matrix(self):return np.eye(4)


def fixture(tmp_path):
    def entity(name,shape):
        c=SimpleNamespace(get_collision_shapes=lambda:[shape],get_entity_pose=lambda:Pose())
        return SimpleNamespace(name=name,components=[c])
    env=SimpleNamespace(scene=SimpleNamespace(entities=[entity('active_link',Box()),entity('other_arm',Box()),entity('tool',Box())]))
    robot=dict(transforms={'left':{'world_from_urdf_base':np.eye(4).tolist()}},
        grippers={'left':{'joint_names':['g7','g8']}},source_hashes={},
        planning_assets={'left':{'robot_config':{'kinematics':{'collision_link_names':['active_link'],'collision_spheres':{}}}}})
    runtime=SimpleNamespace(observe=lambda **kw:state())
    from gene_agent.simulation.robotwin.scene import RobotwinSceneProvider
    return RobotwinSceneProvider(env,runtime,robot_metadata=robot,output_dir=tmp_path)


def test_world_snapshot_keeps_frozen_robot_and_target(tmp_path):
    provider=fixture(tmp_path)
    snap=provider.snapshot(state=state(),group='left',collision_policy=CollisionPolicy())
    names={r['body_name'] for r in snap['records']}
    assert names=={'other_arm','tool'}
    assert snap['state_id']=='current' and snap['episode_id']=='episode'
    assert snap['scene_model']['cuboid']['tool__0_0']['dims']==[.2,.4,.6]
    assert snap['gripper_snapshot']['joint_positions_m']=={'g7':.01,'g8':.01}


def test_stale_snapshot_rejected(tmp_path):
    provider=fixture(tmp_path)
    stale=OperationState('old','fixture',state().values)
    with pytest.raises(ValueError,match='state'):
        provider.snapshot(state=stale,group='left',collision_policy=CollisionPolicy())


def test_native_cylinder_mesh_conservatively_covers_source_solid():
    from gene_agent.simulation.robotwin.scene import RobotwinSceneProvider
    class PhysxCollisionShapeCylinder:
        radius=.01
        half_length=.04
    mesh=RobotwinSceneProvider._mesh(PhysxCollisionShapeCylinder())
    assert mesh.bounds[:,0]==pytest.approx([-.04,.04])
    # 每个侧面的支撑平面都在原圆柱半径之外，避免内接polygon漏碰撞。
    radial=np.abs(mesh.face_normals[:,0])<1e-6
    distances=np.abs(np.sum(mesh.face_normals[radial]*mesh.triangles_center[radial],axis=1))
    assert len(distances)>0 and np.min(distances)>=.01-1e-9
    assert np.max(np.linalg.norm(mesh.vertices[:,1:],axis=1))<=.0101


def test_held_geometry_transfers_to_attachment_with_local_contact_pair(tmp_path):
    from gene_agent.motion import CollisionAllowance
    provider=fixture(tmp_path)
    cp=CollisionPolicy('lift', (CollisionAllowance('gene_held_tool','other_arm','fixture support'),))
    snap=provider.snapshot(state=state(),group='left',collision_policy=cp,
        held_object={'object_id':'tool','T_link_object':np.eye(4).tolist()})
    assert 'tool__0_0' not in snap['scene_model']['cuboid']
    assert 'other_arm__0_0' in snap['scene_model']['cuboid']
    assert snap['attachment']['spheres_link']
    assert snap['mounting_contact_policy']['robot_links']==['gene_held_tool']
    assert snap['mounting_contact_policy']['obstacle_names']==['other_arm__0_0']
    assert snap['transferred_objects']['tool']['source_body']=='tool'


def test_contact_allowance_unknown_body_rejected(tmp_path):
    from gene_agent.motion import CollisionAllowance
    cp=CollisionPolicy('lift',(CollisionAllowance('gene_held_tool','missing','no'),))
    with pytest.raises(ValueError,match='missing'):
        fixture(tmp_path).snapshot(state=state(),group='left',collision_policy=cp,
            held_object={'object_id':'tool','T_link_object':np.eye(4).tolist()})


def test_snapshot_keeps_visual_capture_separate_and_nonfatal(tmp_path):
    snap=fixture(tmp_path).snapshot(state=state(),group='left',collision_policy=CollisionPolicy())
    assert snap['visual_snapshot']['state_id']=='current'
    assert snap['visual_snapshot']['warnings']  # fixture deliberately lacks robot visual state
    assert 'tool__0_0' in snap['scene_model']['cuboid']


def test_grasp_contact_keeps_tool_and_other_world_geometry(tmp_path):
    from gene_agent.motion import CollisionAllowance
    provider=fixture(tmp_path)
    # 原生metadata明确哪些link属于活动夹爪，不能由策略任意豁免机械臂。
    provider._RobotwinSceneProvider__robot['grippers']['left']['contact_links']=['finger_a','finger_b']
    cp=CollisionPolicy('grasp_finger_contact',tuple(CollisionAllowance(link,'tool','预期指部抓取接触') for link in ['finger_a','finger_b']))
    snap=provider.snapshot(state=state(),group='left',collision_policy=cp)
    assert set(snap['scene_model']['cuboid'])=={'tool__0_0','other_arm__0_0'}
    assert snap['mounting_contact_policy']['robot_links']==['finger_a','finger_b']
    assert snap['mounting_contact_policy']['obstacle_names']==['tool__0_0']
    assert snap['transferred_objects']=={}
    bad=CollisionPolicy('grasp_finger_contact',(CollisionAllowance('active_link','tool','not a finger'),))
    with pytest.raises(ValueError):provider.snapshot(state=state(),group='left',collision_policy=bad)
