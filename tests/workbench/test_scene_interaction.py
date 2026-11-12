"""相机投影与物体拖动的几何契约，不依赖浏览器像素坐标。"""
from types import SimpleNamespace
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from gene_agent.workbench.scene_editor import screen_to_xy_plane


def test_table_save_updates_microwave_bounds_and_preserves_source(tmp_path):
    import json
    from gene_agent.workbench.service import Workbench
    source=tmp_path/'data/input';source.mkdir(parents=True)
    original={'schema_version':1,'microwave_pose_world':np.eye(4).tolist(),'bowl_pose_world':np.eye(4).tolist()}
    (source/'task.json').write_text(json.dumps(original))
    table={'center_xy_m':[.25,.3],'size_xy_m':[1.4,1.2],'top_z_m':-.12,'thickness_m':.04}
    service=Workbench(tmp_path)
    try:
        result=service.save_scene({'task':'microwave','scene':'data/input','table':table})
        saved=__import__('json').loads((tmp_path/result['path']/'task.json').read_text())
        assert saved['table']==table
        assert np.allclose(saved['placement_rules']['constraints']['bounds_xy_m'],[-.45,.95,-.3,.9])
        assert json.loads((source/'task.json').read_text())==original
        assert result['table']==table
    finally:service.jobs.close()


def test_sweep_table_save_updates_json_and_contact_geometry(tmp_path):
    import json,xml.etree.ElementTree as ET
    from gene_agent.workbench.service import Workbench
    source=tmp_path/'data/example';source.mkdir(parents=True)
    (source/'robot').mkdir();(source/'robot/robot_metadata.json').write_text('{}')
    original={'objects':{'brush':{'T_world_object':np.eye(4).tolist()}},
              'table':{'center_xy_m':[.4,0],'size_xy_m':[1,1],'top_z_m':0,'thickness_m':.04}}
    (source/'scene.json').write_text(json.dumps(original))
    (source/'scene.xml').write_text('<mujoco><worldbody><geom name="scene_table" type="box" pos=".4 0 -.02" size=".5 .5 .02"/></worldbody></mujoco>')
    service=Workbench(tmp_path)
    table={'center_xy_m':[.2,.1],'size_xy_m':[1.4,1.2],'top_z_m':.03,'thickness_m':.06}
    try:
        result=service.save_scene({'scene':'data/example','table':table})
        geom=ET.parse(tmp_path/result['path']/'scene.xml').find('./worldbody/geom')
        assert np.allclose(np.fromstring(geom.get('pos'),sep=' '),[.2,.1,0])
        assert np.allclose(np.fromstring(geom.get('size'),sep=' '),[.7,.6,.03])
        assert result['table']==table
        assert json.loads((source/'scene.json').read_text())==original
    finally:service.jobs.close()


def test_microwave_custom_table_reaches_compiled_geometry(tmp_path):
    import json
    from benchmarks.microwave_task.scene import environment_settings,build_initial_scene
    root=__import__('pathlib').Path(__file__).resolve().parents[2]
    settings=json.loads((root/'data/task_1/simulation/mujoco_task1_physical.json').read_text())
    settings['table']={'center_xy_m':[.25,.3],'size_xy_m':[1.4,1.2],'top_z_m':-.12,'thickness_m':.04}
    model,data,_,_=build_initial_scene(environment_settings(settings),tmp_path)
    table=model.geom('scene_table')
    assert np.allclose(table.size,[.7,.6,.02])
    assert np.allclose(data.geom_xpos[table.id],[.25,.3,-.14])


@pytest.mark.parametrize('change',[{'size_xy_m':[-1,1]},{'size_xy_m':[1,float('nan')]},
                                 {'thickness_m':0},{'center_xy_m':[0, float('inf')]}])
def test_invalid_table_rejected(change):
    from gene_agent.scene.table import validate_table
    table={'center_xy_m':[0,0],'size_xy_m':[1,1],'top_z_m':0,'thickness_m':.04}
    with pytest.raises(ValueError):validate_table({**table,**change})


def test_old_backend_cannot_silently_ignore_table(monkeypatch):
    import io
    from unittest.mock import Mock
    from gene_agent.workbench.scene_editor import SceneEditor
    editor=SceneEditor.__new__(SceneEditor)
    editor.save_button=SimpleNamespace(disabled=False);editor.url='http://localhost:1'
    editor.status=SimpleNamespace(content='');editor.dirty=True
    opener=Mock(return_value=io.BytesIO(b'{"token":"test"}'))
    monkeypatch.setattr('urllib.request.urlopen',opener)
    with pytest.raises(ValueError,match='旧版本'):editor._save(None)
    assert opener.call_count==1 and editor.dirty
    assert not editor.save_button.disabled


def test_editor_table_resize_and_translation_preserve_objects(tmp_path):
    import mujoco,viser
    from gene_agent.visualization.pick_bowl.scene import PickBowlScene
    from gene_agent.workbench.scene_editor import SceneEditor
    model=mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom name="scene_table" type="box" pos=".4 0 -.02" size=".5 .5 .02"/><body name="bowl" pos=".3 .2 .1"><geom type="box" size=".03 .03 .03"/></body></worldbody></mujoco>')
    data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    server=viser.ViserServer(host='127.0.0.1',port=0,verbose=False)
    try:
        case=SimpleNamespace(model=model,data=data,scene={'objects':{'bowl':{}}},task='microwave',rules={},scene_dir=tmp_path)
        editor=SceneEditor(server,case,PickBowlScene(server,model,data,{},{}),workbench_url='http://127.0.0.1:1',output=tmp_path)
        before=editor.poses['bowl'].copy()
        editor.resize_table([1.4,1.2,.08])
        assert editor.table_config()['top_z_m']==0
        assert editor.geoms['table'][0][0].dimensions==(1.4,1.2,.08)
        editor.select('table')
        pose=editor.poses['table'].copy();pose[0,3]=.2
        editor.set_pose('table',pose)
        assert np.allclose(editor.poses['bowl'],before)
        assert editor.fields[2].label=='桌面顶高 Z (m)'
        assert editor.control.disable_rotations
        payload=editor.payload()
        assert payload['table']['center_xy_m']==[.2,0]
        assert [r['name'] for r in payload['objects']]==['bowl']
        assert (tmp_path/'draft_state.json').is_file()
    finally:server.stop()


def camera():
    # OpenCV camera: +Z 朝下看桌面，+Y 屏幕向下。
    return SimpleNamespace(position=np.array([0,0,1.]),
        wxyz=np.array([0,1,0,0.]),fov=np.pi/2,aspect=1.)


def test_screen_drag_keeps_height_and_world_scale():
    c=camera()
    start=screen_to_xy_plane(c,(.5,.5),.2)
    end=screen_to_xy_plane(c,(.6,.5),.2)
    assert np.allclose(start,[0,0,.2])
    assert np.allclose(end,[.16,0,.2])
    assert np.allclose(screen_to_xy_plane(c,(.5,.6),.2),[0,-.16,.2])


def test_parallel_camera_cannot_make_unbounded_drag():
    c=camera();q=Rotation.from_euler('y',np.pi/2).as_quat();c.wxyz=q[[3,0,1,2]]
    with pytest.raises(ValueError):screen_to_xy_plane(c,(.5,.5),0)


def test_saved_sample_reloads_exact_state_without_sampling(tmp_path):
    import mujoco
    from gene_agent.workbench.sample_viewer import save_snapshot,load_snapshot
    model=mujoco.MjModel.from_xml_string('<mujoco><worldbody><body name="box"><freejoint/><geom type="box" size=".1 .1 .1" mass="1"/></body></worldbody></mujoco>')
    data=mujoco.MjData(model);data.qpos[:3]=[.4,.2,.1];data.time=.6
    mujoco.mj_forward(model,data)
    save_snapshot(SimpleNamespace(model=model,data=data,task='sweep'),tmp_path,
        {'seed':7,'valid':True,'source':'data/example'})
    _,again,meta=load_snapshot(tmp_path)
    assert meta['seed']==7 and meta['task']=='sweep'
    assert np.allclose(again.body('box').xpos,[.4,.2,.1])
    assert again.time==.6


def test_direct_drag_moves_whole_object_without_height_or_end_drift(tmp_path):
    import threading
    from unittest.mock import Mock
    from gene_agent.workbench.scene_editor import SceneEditor
    editor=SceneEditor.__new__(SceneEditor)
    editor.lock=threading.RLock();editor.mesh_drags={}
    root=np.eye(4);root[:3,3]=[.5,.3,.04]
    door=root.copy();door[:3,3]+=[.03,-.1,.15]
    support=root.copy();support[:3,3]+=[0,0,-.08]
    editor.base={'microwave':root.copy()};editor.poses={'microwave':root.copy()}
    editor.selection=SimpleNamespace(value='微波炉');editor.labels={'微波炉':'microwave'}
    editor.geoms={'microwave':[(SimpleNamespace(),p) for p in (root,door,support)]}
    editor._sync=Mock();editor.status=SimpleNamespace(content='');editor.saved=None;editor.dirty=False;editor.output=tmp_path
    event=SimpleNamespace(phase='start',client_id=1,target=SimpleNamespace(name='/cabinet'),
        client=SimpleNamespace(camera=camera()),start_position=(0,0,.2),
        start_screen_pos=(.5,.5),end_screen_pos=(.5,.5))
    editor._mesh_drag('microwave',event)
    event.phase='update';event.end_screen_pos=(.6,.5)
    editor._mesh_drag('microwave',event)
    moved=editor.poses['microwave'].copy()
    event.phase='end';editor._mesh_drag('microwave',event)
    assert not editor.mesh_drags
    assert np.allclose(editor.poses['microwave'],moved)
    assert np.allclose(moved[:3,3],[.66,.3,.04])
    from gene_agent.visualization.pick_bowl.scene import _pose
    for (handle,before) in editor.geoms['microwave']:
        assert np.allclose(handle.position,before[:3,3]+[.16,0,0])
