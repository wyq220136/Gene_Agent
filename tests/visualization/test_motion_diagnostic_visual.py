import hashlib
import numpy as np
import trimesh
from gene_agent.visualization.motion_diagnostic_visual import load_visual_scene


def test_visual_asset_color_pose_and_hash_are_preserved(tmp_path):
    mesh=trimesh.creation.box(extents=[.1,.2,.3])
    mesh.visual.face_colors=[200,40,20,255]
    path=tmp_path/'visual.glb';path.write_bytes(trimesh.Scene(mesh).export(file_type='glb'))
    pose=np.eye(4);pose[:3,3]=[1,2,3]
    row=dict(name='tool',body_name='tool',file_path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),world_from_body=pose.tolist(),body_from_visual=np.eye(4).tolist())
    bundle=dict(configuration_ref='snapshot',snapshot={'visual_snapshot':{'objects':[row]}})
    result=load_visual_scene(bundle)
    assert result['status']=='available'
    np.testing.assert_allclose(result['meshes'][0]['mesh'].centroid,[1,2,3])
    assert result['meshes'][0]['mesh'].visual.face_colors[0,0]==200
    path.write_bytes(b'changed')
    assert not load_visual_scene(bundle)['meshes']
    assert 'hash changed' in load_visual_scene(bundle)['warnings'][0]


def test_unbound_start_visual_is_not_shown_as_ik_candidate():
    result=load_visual_scene(dict(configuration_ref='ik_candidate',snapshot={'visual_snapshot':{'objects':[]}}))
    assert result['status']=='missing'
    assert 'not bound' in result['warnings'][0]


def test_candidate_fk_moves_visual_and_attachment_together(tmp_path):
    urdf=tmp_path/'arm.urdf'
    urdf.write_text('''<robot name="test"><link name="base"/><link name="tip"/><joint name="slide" type="prismatic"><parent link="base"/><child link="tip"/><axis xyz="1 0 0"/><limit lower="0" upper="1" effort="1" velocity="1"/></joint></robot>''')
    path=tmp_path/'box.glb';path.write_bytes(trimesh.Scene(trimesh.creation.box(extents=[.1]*3)).export(file_type='glb'))
    def row(body):return dict(name=body,body_name=body,file_path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),world_from_body=np.eye(4).tolist(),body_from_visual=np.eye(4).tolist())
    offset=np.eye(4);offset[2,3]=.2
    robot=dict(urdf_path=str(urdf),urdf_sha256=hashlib.sha256(urdf.read_bytes()).hexdigest(),joint_positions={'slide':0},links=['base','tip'],world_from_base=np.eye(4).tolist())
    bundle=dict(configuration_ref='ik_candidate',configuration_joint_positions_rad=[.5],replay_config={'joint_names':['slide']},snapshot={'visual_snapshot':dict(objects=[row('tip'),row('hammer')],active_robot=robot,attachment=dict(body_name='hammer',link='tip',T_link_object=offset.tolist()))})
    result=load_visual_scene(bundle)
    assert result['status']=='available',result['warnings']
    np.testing.assert_allclose(result['meshes'][0]['mesh'].centroid,[.5,0,0],atol=1e-8)
    np.testing.assert_allclose(result['meshes'][1]['mesh'].centroid,[.5,0,.2],atol=1e-8)


def test_fk_uses_declared_planning_base_not_urdf_root(tmp_path):
    urdf=tmp_path/'arm.urdf'
    urdf.write_text('''<robot name="test"><link name="root"/><link name="arm_base"/><link name="tip"/><joint name="mount" type="fixed"><parent link="root"/><child link="arm_base"/><origin xyz="10 0 0"/></joint><joint name="slide" type="prismatic"><parent link="arm_base"/><child link="tip"/><axis xyz="1 0 0"/><limit lower="0" upper="1" effort="1" velocity="1"/></joint></robot>''')
    path=tmp_path/'box.glb';path.write_bytes(trimesh.Scene(trimesh.creation.box()).export(file_type='glb'))
    robot=dict(urdf_path=str(urdf),urdf_sha256=hashlib.sha256(urdf.read_bytes()).hexdigest(),joint_positions={'slide':0},links=['tip'],base_link='arm_base',required_joints=['slide'],world_from_base=np.eye(4).tolist())
    row=dict(name='tip',body_name='tip',file_path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),world_from_body=np.eye(4).tolist(),body_from_visual=np.eye(4).tolist())
    result=load_visual_scene(dict(configuration_ref='ik_candidate',configuration_joint_positions_rad=[.5],replay_config={'joint_names':['slide']},snapshot={'visual_snapshot':dict(objects=[row],active_robot=robot)}))
    np.testing.assert_allclose(result['meshes'][0]['mesh'].centroid,[.5,0,0],atol=1e-8)
