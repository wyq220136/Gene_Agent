"""场景包内容、迁移和篡改检查；不读取设备或推进物理。"""
import json
from pathlib import Path
import mujoco
import numpy as np
import pytest
from gene_agent.realworld.scene_archive import SceneArchive, load_scene_archive, verify_scene_archive
from test_microwave_plan_only import observed_scene


def toy_scene(tmp_path):
    source=tmp_path/'source';(source/'meshes').mkdir(parents=True)
    (source/'meshes/tetra.obj').write_text('v 0 0 0\nv .1 0 0\nv 0 .1 0\nv 0 0 .1\nf 1 3 2\nf 1 2 4\nf 1 4 3\nf 2 3 4\n')
    xml=source/'model.xml';xml.write_text('''<mujoco><compiler meshdir="meshes"/><asset><mesh name="tetra" file="tetra.obj"/></asset><worldbody><body name="arm"><joint name="hinge" type="hinge"/><geom type="mesh" mesh="tetra"/></body></worldbody></mujoco>''')
    camera=source/'camera.json';camera.write_text(json.dumps({'frame_number':7,'depth_unit':'millimeter'}))
    (source/'rgb.png').write_bytes(b'original rgb input')
    (source/'depth.png').write_bytes(b'original depth input')
    return dict(scene_xml_path=str(xml),snapshot={'qpos_snapshot':[.3],'snapshot_sha256':'frozen','scope':'test','scene_model':{}},
        provenance={'source_frame_number':7,'source_timestamp_ms':100.,'source_paths':{'camera':str(camera)},'physics_stepped':False},
        robot_metadata={'xml_path':str(xml)},scene_metadata={'robot':{'xml_path':str(xml)}})


def test_moved_bundle_restores_geometry_and_state_without_original_inputs(tmp_path):
    scene=toy_scene(tmp_path)
    model=mujoco.MjModel.from_xml_path(scene['scene_xml_path']);data=mujoco.MjData(model)
    data.qpos[:]=.3;mujoco.mj_forward(model,data)
    archive=SceneArchive(tmp_path/'bundle');archive.add_scene(scene);archive.add_perception(scene,[])
    (tmp_path/'source').rename(tmp_path/'source_unavailable')
    moved=tmp_path/'moved';archive.root.rename(moved)
    restored,m,d=load_scene_archive(moved)
    np.testing.assert_allclose(d.qpos,data.qpos,atol=1e-12)
    np.testing.assert_allclose(d.geom_xpos,data.geom_xpos,atol=1e-6)
    assert restored['provenance']['source_timestamp_ms']==100.
    assert restored['snapshot']['snapshot_sha256']=='frozen'
    assert Path(restored['provenance']['source_paths']['camera']).is_relative_to(moved)
    assert restored['offline_replay']['physics_stepped'] is False
    _,manifest=verify_scene_archive(moved)
    assert any(row['source_path'].endswith('depth.png') for row in manifest['files'].values() if row['source_path'])


def test_modified_geometry_or_missing_frame_is_rejected(tmp_path):
    scene=toy_scene(tmp_path);archive=SceneArchive(tmp_path/'bundle');archive.add_scene(scene)
    relative=archive.manifest['scenes']['initial']['model_ref']
    (archive.root/relative).write_text('<mujoco/>')
    with pytest.raises(ValueError,match='hash'):
        load_scene_archive(archive.root)


def test_incomplete_perception_does_not_invent_robot_state(tmp_path):
    scene=toy_scene(tmp_path);archive=SceneArchive(tmp_path/'bundle')
    archive.add_perception(None,[Path(scene['scene_xml_path']).parent])
    _,manifest=verify_scene_archive(archive.root)
    assert manifest['status']=='initial_scene_unavailable'
    with pytest.raises(ValueError,match='伪造'):
        load_scene_archive(archive.root)


def test_real_robot_archive_preserves_scene_policy_and_both_arms(observed_scene,tmp_path):
    archive=SceneArchive(tmp_path/'bundle');archive.add_scene(observed_scene)
    restored,model,data=load_scene_archive(archive.root)
    assert restored['snapshot']['mounting_contact_policy']==observed_scene['snapshot']['mounting_contact_policy']
    np.testing.assert_allclose(data.qpos,observed_scene['snapshot']['qpos_snapshot'])
    for group in ('left','right'):
        assert restored['robot_metadata']['joint_groups'][group]==observed_scene['robot_metadata']['joint_groups'][group]
    assert restored['provenance']['aligned_object_poses_world']==observed_scene['provenance']['aligned_object_poses_world']


def test_wrong_camera_frame_is_marked_partial_instead_of_claiming_same_frame(tmp_path):
    scene=toy_scene(tmp_path);scene['provenance']['source_frame_number']=8
    archive=SceneArchive(tmp_path/'bundle');archive.add_scene(scene);archive.add_perception(scene,[])
    _,manifest=verify_scene_archive(archive.root)
    assert manifest['perception_status']=='frame_mismatch'
    assert manifest['status']=='partial'
