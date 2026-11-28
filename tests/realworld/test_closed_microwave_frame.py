"""闭门机身重建：只测几何与接线，不连接设备，不改变原采纳门限。"""
from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from test_automatic_microwave_setup import prepared, prepared_no_knob
from gene_agent.realworld.microwave import door_plane as module
from gene_agent.realworld.microwave.perception_setup import MICROWAVE
from gene_agent.realworld.microwave.operation_perception import OperationPerception


def test_plane_and_up_rebuild_root_while_preserving_actual_handle_center(prepared_no_knob,monkeypatch,tmp_path):
    _,record=prepared_no_knob
    assets=record["assets"];alignment=np.asarray(assets["alignment"])
    observer=module.DoorPlaneObserver(assets)
    old=np.eye(4);old[:3,:3]=Rotation.from_euler("z",15,degrees=True).as_matrix()
    old[:3,3]=[.5,.3,.05]
    pose=old@np.linalg.inv(alignment)
    table_rotation=Rotation.from_euler("xy",[3,-2],degrees=True).as_matrix()
    expected_rotation=table_rotation@Rotation.from_euler("z",8,degrees=True).as_matrix()
    asset_normal=np.linalg.inv(alignment)[:3,:3]@observer.door.pose[:3,2]
    normal=expected_rotation@asset_normal
    monkeypatch.setattr(observer,"correct",lambda *args:(pose.copy(),{
        "applied":True,"normal_sdk":normal.tolist(),"yaw_correction_deg":-7.}))
    actual_handle=(pose@alignment@np.asarray(assets["T_asset_handle"]))[:3,3]
    result,diagnostic=observer.reconstruct_closed(pose,tmp_path,np.eye(4),table_rotation[:,2])
    assert diagnostic["applied"]
    root=result@alignment
    np.testing.assert_allclose(root[:3,:3],expected_rotation,atol=1e-12)
    np.testing.assert_allclose((root@np.asarray(assets["T_asset_handle"]))[:3,3],actual_handle,atol=1e-12)
    assert np.linalg.norm(root[:3,3]-old[:3,3])>.001  # 只转mesh原点会破坏把手锚点。
    joint=root@np.asarray(assets["joints"]["door"]["T_asset_joint"])
    np.testing.assert_allclose(diagnostic["T_sdk_door_joint"],joint,atol=1e-12)
    np.testing.assert_allclose(diagnostic["door_axis_sdk"],-table_rotation[:,2],atol=1e-12)

    # 实际运行使用保存的新参考；不再次修改，不另建显示用的机身/门轴。
    updated=deepcopy(record)
    updated["closed_reference_poses_sdk"][MICROWAVE]=result.tolist()
    updated["microwave_frame_reconstruction"]=diagnostic
    monkeypatch.setattr(module.DoorPlaneObserver,"correct",lambda *args:pytest.fail("重复修正闭门参考"))
    scope=OperationPerception(updated)
    np.testing.assert_allclose(scope.reader.root,root,atol=1e-12)
    np.testing.assert_allclose(scope.anchor["T_world_scene_asset"],root,atol=1e-12)
    point,axis,_=scope.reader._joint("door")
    np.testing.assert_allclose(point,joint[:3,3],atol=1e-12)
    np.testing.assert_allclose(axis,-table_rotation[:,2],atol=1e-12)


def test_task17_angle_above_existing_fifteen_degree_limit_still_falls_back(prepared_no_knob,monkeypatch,tmp_path):
    _,record=prepared_no_knob
    alignment=np.asarray(record["assets"]["alignment"])
    observer=module.DoorPlaneObserver(record["assets"])
    root=np.eye(4);root[:3,:3]=Rotation.from_euler("z",15.926160484612913,degrees=True).as_matrix()
    pose=root@np.linalg.inv(alignment)
    points=np.array([[x,0.,z] for x in np.linspace(-.1,.1,20) for z in np.linspace(-.1,.1,20)])
    monkeypatch.setattr(module,"_load_rgbd_points",lambda *args,**kwargs:points)
    monkeypatch.setattr(module,"select_door_region_points",lambda *args:(np.ones(len(points),bool),np.array([-.1,.1,-.1,.1])))
    (tmp_path/"camera.json").write_text("{}")
    (tmp_path/"rgb.png").touch();(tmp_path/"depth.png").touch()
    result,diagnostic=observer.reconstruct_closed(pose,tmp_path,np.eye(4),[0.,0.,1.])
    assert not diagnostic["applied"]
    assert diagnostic["door_plane"]["max_yaw_correction_deg"]==15.
    assert abs(diagnostic["door_plane"]["yaw_correction_deg"])==pytest.approx(15.926160484612913,abs=.001)
    np.testing.assert_array_equal(result,pose)


def test_auxiliary_reconstruction_error_retains_pose_without_new_exit(prepared_no_knob,monkeypatch,tmp_path):
    _,record=prepared_no_knob
    observer=module.DoorPlaneObserver(record["assets"])
    pose=np.asarray(record["closed_reference_poses_sdk"][MICROWAVE])
    def broken(*args):raise RuntimeError("auxiliary fit unavailable")
    monkeypatch.setattr(observer,"correct",broken)
    result,diagnostic=observer.reconstruct_closed(pose,tmp_path,np.eye(4),[0.,0.,1.])
    np.testing.assert_array_equal(result,pose)
    assert not diagnostic["applied"]
    assert diagnostic["error_type"]=="RuntimeError"


@pytest.mark.parametrize("applied",[False,True])
def test_saved_prepare_result_does_not_repeat_reference_rgbd_processing(prepared_no_knob,monkeypatch,applied):
    _,record=prepared_no_knob
    record=deepcopy(record)
    record["microwave_frame_reconstruction"]={
        "applied":applied,"door_plane":{"applied":applied,"normal_sdk":[0.,-1.,0.]}}
    monkeypatch.setattr(module.DoorPlaneObserver,"reference_capture",lambda *args:pytest.fail("重复读取参考RGB-D"))
    monkeypatch.setattr(module.DoorPlaneObserver,"correct",lambda *args:pytest.fail("重复参考拟合"))
    scope=OperationPerception(record)
    assert scope.reference_plane["applied"] is applied
