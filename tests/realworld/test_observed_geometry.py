"""实测点云/mesh 重建的坐标、可见范围和预览一致性；不使用设备或推理服务。"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from xml.etree import ElementTree as ET

import mujoco
import numpy as np
from PIL import Image
import pytest
import trimesh

from gene_agent.realworld.microwave.observed_geometry import (
    capture_cloud, fit_observed_table, rgbd_surface, reconstruct_plan_scene)
from gene_agent.realworld.microwave.perception_setup import asset_geometry, BOWL, MICROWAVE
from gene_agent.realworld.microwave.plan_only_world import PlanOnlyWorld
from gene_agent.visualization.pick_bowl.scene import (
    _compiled_mesh, _compiled_observed_cloud, _geom_pose)
from test_microwave_plan_only import observed_scene

ROOT=Path(__file__).resolve().parents[2]


@pytest.fixture
def capture(tmp_path):
    folder=tmp_path/"perception"
    dest=folder/"capture";dest.mkdir(parents=True)
    h,w=120,160
    rng=np.random.default_rng(4)
    depth=(800+rng.integers(-2,3,(h,w))).astype(np.uint16)
    rgb=np.zeros((h,w,3),np.uint8);rgb[...,0]=np.arange(w,dtype=np.uint8)
    rgb[...,1]=np.arange(h,dtype=np.uint8)[:,None];rgb[...,2]=90
    Image.fromarray(depth).save(dest/"depth.png")
    Image.fromarray(rgb).save(dest/"rgb.png")
    K=[[150.,0.,80.],[0.,150.,60.],[0.,0.,1.]]
    T=np.diag([1.,-1.,-1.,1.]);T[2,3]=.8
    camera={"serial":"test-camera","frame_number":42,"captured_at_unix_ms":1000.,
            "depth_unit":"millimeter","intrinsic_matrix":K}
    (dest/"camera.json").write_text(json.dumps(camera))
    setup={"camera":{"serial":"test-camera","width":w,"height":h,
                     "T_sdk_camera":T.tolist(),"intrinsic_matrix":K},
           "reference_frame":{"frame_number":42},
           "input_sha256":{str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in dest.iterdir()}}
    return folder,setup


def test_capture_reprojects_raw_depth_and_applies_each_transform_once(capture):
    folder,setup=capture
    world_sdk=np.eye(4);world_sdk[:3,3]=[.3,-.2,.1]
    points,rgb,valid,transform,source=capture_cloud(folder,setup,T_world_sdk=world_sdk)
    z=float(np.asarray(Image.open(folder/"capture/depth.png"))[60,80])*.001
    expected=(world_sdk@np.asarray(setup["camera"]["T_sdk_camera"])@np.array([0.,0.,z,1.]))[:3]
    np.testing.assert_allclose(points[60,80],expected,atol=1e-12)
    assert tuple(rgb[60,80])==(80,60,90)
    assert valid[60,80] and source["frame_number"]==42
    assert source["depth_scale_m"]==.001


@pytest.mark.parametrize("fault",["frame","image","unit"])
def test_mismatched_or_changed_frozen_capture_is_rejected(capture,fault):
    folder,setup=capture
    if fault=="image":
        Image.fromarray(np.zeros((120,160,3),np.uint8)).save(folder/"capture/rgb.png")
    else:
        p=folder/"capture/camera.json";camera=json.loads(p.read_text())
        camera["frame_number" if fault=="frame" else "depth_unit"]=99 if fault=="frame" else "unknown"
        p.write_text(json.dumps(camera))
    with pytest.raises(ValueError):
        capture_cloud(folder,setup,T_world_sdk=np.eye(4))


@pytest.mark.parametrize("camera_z,up",[(1.,True),(-1.,False)])
def test_table_uses_visible_plane_instead_of_legacy_xy(camera_z,up):
    rng=np.random.default_rng(7)
    x=rng.uniform(.4,1.4,3000);y=rng.uniform(-.7,.1,3000)
    points=np.column_stack((x,y,-.2+.02*x+.01*y+rng.normal(0,.001,3000)))
    result=fit_observed_table(points,[.7,-.3,camera_z],thickness_m=.02)
    assert result["table_size_xy_m"]==pytest.approx([1.,.8],abs=.03)
    assert result["top_center_world_m"][:2]==pytest.approx([.9,-.3],abs=.03)
    assert result["world_up_consistent"] is up
    assert result["rms_m"]<.002
    assert result["footprint_is_visible_extent"] is True
    T=np.asarray(result["table_pose_world"])
    np.testing.assert_allclose(T[:3,3]+T[:3,2]*.01,result["top_center_world_m"],atol=1e-12)


def test_depth_surface_never_bridges_holes_or_depth_discontinuities():
    v,u=np.indices((20,20));points=np.stack((u*.005,v*.005,np.zeros_like(u)),axis=-1).astype(float)
    points[:,10:,2]+=.2
    valid=np.ones((20,20),bool);valid[4:7,4:7]=False
    vertices,faces=rgbd_surface(points,valid)
    assert valid.reshape(-1)[faces].all()
    assert np.max(np.ptp(vertices[faces][:,:,2],axis=1))<.001


def test_rebuilt_model_uses_actual_mesh_and_table_without_changing_robot_state(
        observed_scene,capture,tmp_path):
    folder,setup=capture
    original=deepcopy(observed_scene)
    assets=asset_geometry()
    setup.update(assets=assets,meshes={
        BOWL:str(ROOT/"data/task_1/objs/Bowl/Bowl_0/meshes/obj_mesh.obj"),
        MICROWAVE:str(ROOT/"data/task_1/objs/Microwave/Microwave_0/meshes/handle_mesh.obj")},
        closed_reference_poses_sdk={
            BOWL:observed_scene["provenance"]["aligned_object_poses_world"][BOWL],
            MICROWAVE:(np.asarray(observed_scene["provenance"]["aligned_object_poses_world"][MICROWAVE])@
                       np.linalg.inv(assets["alignment"])).tolist()})
    scene=reconstruct_plan_scene(observed_scene,perception_dir=folder,setup=setup,
        output_dir=tmp_path/"geometry",T_world_sdk=np.eye(4))
    assert observed_scene==original
    np.testing.assert_allclose(scene["snapshot"]["qpos_snapshot"],original["snapshot"]["qpos_snapshot"])
    assert scene["provenance"]["source_frame_number"]==original["provenance"]["source_frame_number"]
    assert scene["provenance"]["table_pose_world"]!=original["provenance"]["table_pose_world"]
    assert scene["provenance"]["observed_geometry"]["table_world_up_consistent"]
    assert any(r["geom_name"]=="microwave_support_collision" for r in scene["snapshot"]["records"])
    table_record=next(r for r in scene["snapshot"]["records"] if r["geom_name"]=="scene_table")
    np.testing.assert_allclose(table_record["T_world_geom"],scene["provenance"]["table_pose_world"],atol=1e-8)
    assert not any(r["geom_name"]=="observed_rgbd_surface" for r in scene["snapshot"]["records"])
    model=mujoco.MjModel.from_xml_path(scene["scene_xml_path"])
    data=mujoco.MjData(model);data.qpos[:]=scene["snapshot"]["qpos_snapshot"];mujoco.mj_forward(model,data)
    assert model.camera("observed_rgbd_camera").id>=0
    gid=model.geom("observed_microwave_cabinet_visual").id
    actual_mesh=_compiled_mesh(model,gid)
    T=_geom_pose(data,gid)
    vertices=np.asarray(actual_mesh.vertices)@T[:3,:3].T+T[:3,3]
    raw=trimesh.load_mesh(ROOT/"data/task_1/objs/Microwave/Microwave_0/meshes/obj_mesh.obj",process=False)
    expected_T=np.asarray(scene["provenance"]["aligned_object_poses_world"][MICROWAVE])@np.linalg.inv(assets["alignment"])
    expected=np.asarray(raw.vertices)@expected_T[:3,:3].T+expected_T[:3,3]
    np.testing.assert_allclose([vertices.min(0),vertices.max(0)],[expected.min(0),expected.max(0)],atol=1e-6)
    gid=model.geom("observed_rgbd_surface").id
    cloud,colors=_compiled_observed_cloud(model,gid)
    assert len(cloud)>100 and len(cloud)==len(colors)
    assert colors.dtype==np.uint8 and np.all(colors[:,2]==90)
    assert np.ptp(colors[:,0])>20
    w=PlanOnlyWorld(scene,tmp_path/"preview",max_start_error_rad=.03)
    w.event("initial")
    w.save_replay()
    from gene_agent.simulation.skill_viewer import SkillMujocoViewer
    viewer=SkillMujocoViewer(tmp_path/"preview")
    camera,meta=viewer._camera("observed")
    assert camera.fixedcamid>=0 and meta["source"]=="PERCEPTION_RGBD"


def test_up_mismatch_saves_scene_and_stops_before_planner(observed_scene,tmp_path):
    from gene_agent.realworld.microwave.plan_only import plan_microwave_task1
    observed_scene["provenance"]["observed_geometry"]={
        "table_world_up_consistent":False,"table_normal_world":[0,0,-1]}
    result=plan_microwave_task1(observed_scene,planner_pool=object(),
        output_dir=tmp_path/"blocked",subtask_options={"door":{"max_start_error_rad":.02}})
    assert result["status"]=="planning_failed" and result["completed_steps"]==0
    assert result["first_failure"]["code"]=="observed_world_up_mismatch"
    assert result["task_success"] is None
    from gene_agent.visualization.replay import ReplayTimeline
    timeline=ReplayTimeline.load(tmp_path/"blocked")
    assert timeline.mode=="planned"
    assert timeline.seek(0).qpos is not None
    assert Path(timeline.view.stages[-1].collision_snapshot_ref).is_file()


def test_accepted_calibration_may_omit_resolution(capture):
    folder,setup=capture
    setup["camera"]["width"]=setup["camera"]["height"]=None
    points,rgb,valid,transform,source=capture_cloud(folder,setup,T_world_sdk=np.eye(4))
    assert points.shape[:2]==rgb.shape[:2]==valid.shape
