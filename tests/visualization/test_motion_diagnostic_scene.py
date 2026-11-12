import numpy as np
from gene_agent.visualization.motion_diagnostics import build_diagnostic_scene


def test_base_transform_applied_once_and_preview_label():
    base=np.eye(4);base[0,3]=2
    bundle={'snapshot':{'base_from_world':np.linalg.inv(base).tolist(),'scene_model':{'cuboid':{'board':{'dims':[1,1,1],'pose':[1,0,0,1,0,0,0]}}}},
        'collisions':{'spheres':[{'sphere_index':0,'object_id':'hammer','link':'gene_held_tool','label':'right/hammer/gene_held_tool/sphere#0','center_world_m':[3,0,0],'effective_radius_m':.02}],'pairs':[]},'candidates':{}}
    scene=build_diagnostic_scene(bundle)
    assert scene['mode']=='planned_diagnostic'
    np.testing.assert_allclose(scene['meshes'][0]['mesh'].centroid,[3,0,0],atol=1e-12)
    assert 'hammer' in scene['spheres'][0]['label']
    assert scene['spheres'][0]['center_world_m']==[3,0,0]


def test_missing_spheres_not_fabricated():
    scene=build_diagnostic_scene({'snapshot':{'scene_model':{}},'candidates':{}})
    assert not scene['spheres']
    assert scene['warnings']


def test_frozen_robot_mesh_has_robot_role(tmp_path):
    from gene_agent.visualization.motion_diagnostics import build_diagnostic_scene
    urdf=tmp_path/'robot.urdf';urdf.write_text('<robot name="r"><link name="other_arm"/></robot>')
    b=dict(effective_robot_config={'kinematics':{'urdf_path':str(urdf)}},snapshot={'scene_model':{'cuboid':{'other_arm__0_0':{'dims':[1,1,1],'pose':[0,0,0,1,0,0,0]}}},'records':[{'obstacle_name':'other_arm__0_0','body_name':'other_arm'}]})
    assert build_diagnostic_scene(b)['meshes'][0]['role']=='robot'


def test_changed_collision_mesh_is_not_shown_as_saved_evidence(tmp_path):
    import hashlib
    import trimesh
    from gene_agent.visualization.motion_diagnostics import build_diagnostic_scene
    path=tmp_path/'mesh.obj';path.write_text(trimesh.creation.box().export(file_type='obj'))
    sha=hashlib.sha256(path.read_bytes()).hexdigest()
    b={'snapshot':{'records':[dict(obstacle_name='board',mesh_sha256=sha)],'scene_model':{'mesh':{'board':dict(file_path=str(path),pose=[0,0,0,1,0,0,0])}}}}
    assert len(build_diagnostic_scene(b)['meshes'])==1
    path.write_text(trimesh.creation.box(extents=[2,2,2]).export(file_type='obj'))
    result=build_diagnostic_scene(b)
    assert not result['meshes']
    assert 'hash changed' in result['warnings'][0]
