"""CPU substitutes only: keyframe scheduling, occlusion, provenance and plane geometry."""
from dataclasses import replace
from types import SimpleNamespace
import json
import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from test_automatic_microwave_setup import prepared, prepared_no_knob, _partial_observation
from gene_agent.realworld.microwave.operation_perception import OperationPerception
from gene_agent.realworld.microwave.perception_setup import BOWL, MICROWAVE
from gene_agent.realworld.checks import has_accepted_realworld_completion, has_realworld_success_evidence


def consume(scope, record, frame, *, tcp=None, bowl=None):
    keys=set(scope.selected_sessions(record['sessions']))
    value=_partial_observation(record, keys, frame=frame)
    if bowl is not None and BOWL in keys:
        value=replace(value, object_poses_sdk={**value.object_poses_sdk, BOWL:bowl})
    return scope.consume(value, tcp_pose_sdk=np.eye(4) if tcp is None else tcp)[0]


def test_pick_cache_and_sdk_attachment_lift_keyframe(prepared_no_knob):
    _,record=prepared_no_knob
    scope=OperationPerception(record); scope.begin_operation('pick_bowl')
    first=consume(scope,record,1)
    original=first.object_poses_sdk[BOWL].copy()
    for frame,name in enumerate(['pregrasp','approach','close'],2):
        scope.before_interaction(name,'grasp' if name=='close' else 'motion')
        # Scheduling before dispatch is insufficient to create an attachment.
        assert scope.selected_sessions(record['sessions'])=={}
        scope.after_interaction(name,'grasp' if name=='close' else 'motion')
        value=consume(scope,record,frame)
        np.testing.assert_allclose(value.object_poses_sdk[BOWL],original)
    scope.after_interaction('lift','motion')
    assert set(scope.selected_sessions(record['sessions']))=={BOWL}
    tcp=np.eye(4);tcp[2,3]=.12
    actual=tcp@original
    # FoundationPose's arbitrary yaw around circular bowl opening is harmless.
    actual[:3,:3]=actual[:3,:3]@Rotation.from_euler('y',130,degrees=True).as_matrix()
    lifted=consume(scope,record,5,tcp=tcp,bowl=actual)
    np.testing.assert_allclose(lifted.object_poses_sdk[BOWL],tcp@original,atol=1e-12)
    assert scope.last_measurement_check['passed']
    assert scope.keyframe_counts[BOWL]==2


def test_bad_visible_lift_is_not_replaced_by_assumed_success(prepared_no_knob):
    _,record=prepared_no_knob
    scope=OperationPerception(record);scope.begin_operation('pick_bowl')
    consume(scope,record,1);scope.after_interaction('close','grasp');consume(scope,record,2)
    scope.after_interaction('lift','motion')
    tcp=np.eye(4);tcp[2,3]=.12
    with pytest.raises(ValueError,match='关键帧'):
        consume(scope,record,3,tcp=tcp,bowl=np.eye(4))


def test_insert_occlusion_no_pose_requests_and_release_freezes_pose(prepared_no_knob):
    _,record=prepared_no_knob
    scope=OperationPerception(record);scope.begin_operation('pick_bowl')
    consume(scope,record,1);scope.after_interaction('close','grasp');consume(scope,record,2)
    scope.begin_operation('insert_bowl')
    tcp=np.eye(4)
    for frame,name in enumerate(['held_turn','preplace','insert'],3):
        tcp=tcp.copy();tcp[0,3]+=.1
        scope.after_interaction(name,'motion')
        assert scope.selected_sessions(record['sessions'])=={}
        value=consume(scope,record,frame,tcp=tcp)
        assert value.perception_refresh['estimated_object_ids']==[BOWL]
    scope.after_interaction('release','release')
    released=consume(scope,record,6,tcp=tcp).object_poses_sdk[BOWL].copy()
    scope.before_interaction('settle','release_settled')
    tcp=tcp.copy();tcp[2,3]+=.2
    value=consume(scope,record,7,tcp=tcp)
    np.testing.assert_allclose(value.object_poses_sdk[BOWL],released)
    assert scope.keyframe_counts[BOWL]==1
    scene={'snapshot':{'snapshot_sha256':'hash'},'provenance':{
        'source_frame_number':7,'live_state':{'measured':True},
        'perception_refresh':value.perception_refresh}}
    check=scope.check_state('place',scene).to_dict()
    assert has_accepted_realworld_completion(check)
    assert not has_realworld_success_evidence(check)
    assert check['evidence']['physical_verification'] is False


def test_command_not_completed_cannot_create_assumed_grasp(prepared_no_knob):
    _,record=prepared_no_knob
    scope=OperationPerception(record);scope.begin_operation('pick_bowl')
    consume(scope,record,1);scope.before_interaction('close','grasp')
    consume(scope,record,2)
    assert scope.attachment_sdk is None
    assert not scope.grasp_assumed


def test_extract_contact_approach_uses_known_pose_without_claiming_grasp(prepared_no_knob):
    _,record=prepared_no_knob
    scope=OperationPerception(record);scope.begin_operation('extract_bowl')
    value=consume(scope,record,1)
    scene={'snapshot':{'snapshot_sha256':'hash'},'provenance':{
        'source_frame_number':1,'live_state':{'measured':True},
        'perception_refresh':value.perception_refresh}}
    assert not scope.grasp_assumed
    result=scope.check_state('path',scene,{'success':True})
    assert has_accepted_realworld_completion(result.to_dict())
    assert scope.check_state('path',scene,{'success':False}).status=='unknown'
    assert scope.check_state('grasp',scene).status=='unknown'


def test_door_intermediate_estimate_and_mandatory_final_keyframe(prepared_no_knob):
    _,record=prepared_no_knob
    scope=OperationPerception(record);scope.begin_operation('open_door')
    consume(scope,record,1);scope.set_door_final_phase('end')
    scope.after_door_phase(SimpleNamespace(name='pull_0',phase_id='mid',door_angle_rad=.2))
    assert scope.selected_sessions(record['sessions'])=={}
    value=consume(scope,record,2)
    assert scope.angles['door']['angle_rad']==.2
    assert value.perception_refresh['estimated_object_ids']==[MICROWAVE]
    scope.after_door_phase(SimpleNamespace(name='retreat',phase_id='end',door_angle_rad=.3))
    assert set(scope.selected_sessions(record['sessions']))=={MICROWAVE}
    assert scope.angles['door']['source']=='executed_door_phase_assumption'


def test_closed_door_noise_padding_but_not_gross_wrong_angle(prepared_no_knob):
    from gene_agent.realworld.microwave.live_perception import LiveArticulationReader
    _,record=prepared_no_knob
    reading=LiveArticulationReader(record)(_partial_observation(record,{BOWL,MICROWAVE},door=math.radians(-2.26)))
    assert reading['door']['angle_rad']==0.
    with pytest.raises(ValueError,match='超出'):
        LiveArticulationReader(record)(_partial_observation(record,{BOWL,MICROWAVE},door=math.radians(-12.)))


def test_cold_registration_expires_then_requires_new_frame(tmp_path,monkeypatch):
    from test_pose_tracking import _settings
    from gene_agent.realworld import pose_tracking as module
    from gene_agent.perception_modules.pose_est.pose_client import PoseSessionMissing
    class Client:
        def track_frame(self,*a,**kw):raise PoseSessionMissing('gone')
    settings=_settings(tmp_path,Client());settings.pop('now_ms')
    settings['sessions']={BOWL:'bowl-session'}
    stamps=iter([1000.5,1006.])
    monkeypatch.setattr(module.time,'time',lambda:next(stamps))
    def recover(obj,session,**kw):
        return {'ok':True,'session_id':session,'camera_id':'0','T_camera_object':np.eye(4).tolist()}
    with pytest.raises(module.PoseInitializationNeedsFreshFrame):
        module.track_observed_scene_from_capture(**settings,recover_session=recover)
    diagnostic=json.loads((settings['output_dir']/'capture_provenance.json').read_text())
    assert diagnostic['execution_frame_usable'] is False
    assert diagnostic['processing_age_s']==6.


def test_door_plane_yaw_uses_calibrated_roi_and_preserves_translation(prepared_no_knob,monkeypatch,tmp_path):
    from gene_agent.realworld.microwave import door_plane as module
    _,record=prepared_no_knob
    observer=module.DoorPlaneObserver(record['assets'])
    pose=np.asarray(record['closed_reference_poses_sdk'][MICROWAVE])
    door=pose@observer.door.pose
    handle_in_door=np.linalg.inv(observer.door.pose)@observer.handle.pose
    xs=np.linspace(handle_in_door[0,3]-.08,handle_in_door[0,3]-.025,22)
    ys=np.linspace(-.06,.06,20)
    local=np.array([[x,y,0.] for x in xs for y in ys])
    # Tilt actual plane yaw by 7 degrees around the panel center.
    turn=Rotation.from_euler('z',7,degrees=True).as_matrix()
    points=local@door[:3,:3].T@turn.T+door[:3,3]
    monkeypatch.setattr(module,'_load_rgbd_points',lambda *a,**k:points)
    (tmp_path/'camera.json').write_text('{}')
    (tmp_path/'rgb.png').touch();(tmp_path/'depth.png').touch()
    corrected,diag=observer.correct(pose,tmp_path,np.eye(4))
    assert diag['applied'],diag
    assert diag['yaw_correction_deg']==pytest.approx(7.,abs=1e-8)
    np.testing.assert_array_equal(corrected[:3,3],pose[:3,3])


def test_undeclared_estimates_do_not_pass_realworld_checks():
    assert not has_accepted_realworld_completion({'status':'pass','source':'prediction','evidence':{}})
    assert not has_accepted_realworld_completion({'status':'pass','source':'open_loop_object_state',
        'evidence':{'authorized_open_loop':True}})


def test_final_door_angle_uses_measured_value_not_commanded_ninety(prepared_no_knob):
    from gene_agent.realworld.microwave.live_perception import LiveArticulationReader
    _,record=prepared_no_knob
    reader=LiveArticulationReader(record)
    reader.door_plane_angle=math.radians(80.)
    value=_partial_observation(record,{BOWL,MICROWAVE},door=math.radians(80.))
    result=reader(value)['door']
    assert result['angle_rad']==pytest.approx(math.radians(80.))
    assert 'door_plane_and_handle_orbit' in result['source']
    reader=LiveArticulationReader(record);reader.door_plane_angle=math.radians(40.)
    with pytest.raises(ValueError,match='相差超过10度'):
        reader(value)


def test_plane_reference_corrects_actual_scene_root_not_only_viewer(prepared_no_knob,monkeypatch,tmp_path):
    from gene_agent.realworld.microwave import door_plane as module
    _,record=prepared_no_knob
    original=np.asarray(record['closed_reference_poses_sdk'][MICROWAVE])
    corrected=original.copy()
    corrected[:3,:3]=Rotation.from_euler('z',7,degrees=True).as_matrix()@original[:3,:3]
    monkeypatch.setattr(module.DoorPlaneObserver,'reference_capture',lambda self,setup:tmp_path)
    monkeypatch.setattr(module.DoorPlaneObserver,'correct',lambda *a:(corrected,{'applied':True}))
    scope=OperationPerception(record)
    np.testing.assert_allclose(scope.reader.root,corrected@np.asarray(record['assets']['alignment']))
    np.testing.assert_array_equal(scope.poses[MICROWAVE][:3,3],original[:3,3])
    # Persisted calibration/reference input remains untouched.
    np.testing.assert_array_equal(record['closed_reference_poses_sdk'][MICROWAVE],original)


def test_post_contact_reuses_already_settled_final_keyframe(prepared_no_knob):
    _,record=prepared_no_knob;waits=[]
    scope=OperationPerception(record,sleep=waits.append);scope.begin_operation('open_door')
    consume(scope,record,1);scope.set_door_final_phase('end')
    scope.after_door_phase(SimpleNamespace(name='retreat',phase_id='end',door_angle_rad=0.))
    scope.prepare_capture();consume(scope,record,2)
    assert waits==[2.]
    scope.after_door_phase('post_contact');scope.prepare_capture()
    assert scope.selected_sessions(record['sessions'])=={}
    assert waits==[2.]


def test_runtime_completion_hook_only_after_success():
    from test_runtime import FakeRobot, _runtime, _stage
    robot=FakeRobot();runtime=_runtime(robot);calls=[]
    runtime.after_interaction=lambda name,kind:calls.append((name,kind))
    runtime.run_motion_stage(_stage(),execute=False)
    assert calls==[]
    runtime.run_motion_stage(_stage(scene_hash='stale'),execute=True)
    assert calls==[]
    result=runtime.run_motion_stage(_stage(),execute=True)
    assert result['status']=='executed'
    assert calls==[(_stage().name,'motion')]


def test_full_task_continues_explicit_occluded_insert_and_reports_unverified(tmp_path,monkeypatch):
    from gene_agent.realworld.microwave import task as module
    visual={'status':'pass','source':'vision'}
    assumed={'status':'pass','source':'open_loop_object_state','evidence':{
        'policy':'affordtoolact_operation_scoped','authorized_open_loop':True,
        'physical_verification':False,'sdk_state_observed':True,
        'source_frame_number':7,'collision_snapshot_sha256':'hash'}}
    def door(**kw):
        skills=['handle_pull','push_open'] if kw['part']=='open' else ['push_close']
        return {'status':'executed','success':True,'skills':[
            {'skill_id':name,'postcondition':visual} for name in skills]}
    def completed(key,check=visual,**extra):
        return lambda **kw:{'status':'executed',key:True,'postcondition':check,**extra}
    monkeypatch.setattr(module,'run_observed_door_subtask',door)
    monkeypatch.setattr(module,'run_observed_door_return_default_tcp',completed('success'))
    monkeypatch.setattr(module,'run_observed_bowl_pick',completed('pick_success'))
    monkeypatch.setattr(module,'run_observed_bowl_insert',completed('insert_success',assumed))
    monkeypatch.setattr(module,'run_observed_bowl_extract',completed('extract_success',grasp={'grasp_success':True}))
    monkeypatch.setattr(module,'run_observed_table_place',completed('table_place_success'))
    result=module.run_observed_microwave_task1(observe_collision_scene=lambda:{},
        make_runtime=lambda *a:None,planner_pool=object(),output_dir=tmp_path/'task',
        execute=True,include_knob=False,subtask_options={key:{} for key in ['door','pick','insert','extract','table']})
    assert result['success'],result.get('first_failure')
    assert result['task_success'] is True
    assert result['verified_task_success'] is False
    assert result['physical_verification'] is False
    assert 'insert_bowl' in result['assumptions_used']


def test_observer_cold_bootstrap_retries_with_new_frame_and_new_sdk_sample(tmp_path,monkeypatch):
    import time
    from contextlib import contextmanager
    from gene_agent.realworld.microwave import scene as module
    from gene_agent.realworld.pose_tracking import PoseInitializationNeedsFreshFrame
    from gene_agent.realworld.perception import ObservedScene
    calls=[];samples=[];captures=[]
    def capture(*,output_dir,**kw):
        output_dir.mkdir(parents=True)
        captures.append(output_dir)
        (output_dir/'camera.json').write_text(json.dumps({'frame_number':len(captures),
            'captured_at_unix_ms':time.time()*1000}))
        return output_dir
    def track(**kw):
        calls.append(kw)
        if len(calls)==1:raise PoseInitializationNeedsFreshFrame('cold bootstrap')
        return ObservedScene('right','0',2,time.time()*1000,np.eye(4),{BOWL:np.eye(4)}, {},True)
    class Policy:
        def prepare_capture(self):pass
        def selected_sessions(self,sessions):return sessions
        def consume(self,observed,**kw):return observed,{'door':{}},None
        def begin_operation(self,*a):pass
        def after_door_phase(self,*a):pass
        def before_interaction(self,*a):pass
        def after_interaction(self,*a):pass
        def set_door_final_phase(self,*a):pass
        def accept_current_hold(self,*a):pass
    class Sampler:
        @contextmanager
        def sample(self,*a):
            row={'right':{'tcp_pose_sdk':np.eye(4)},'left':{'tcp_pose_sdk':np.eye(4)}}
            samples.append(row)
            yield SimpleNamespace(result=lambda:row)
    monkeypatch.setattr(module,'capture_frozen_scene_frame',capture)
    monkeypatch.setattr(module,'track_observed_scene_from_capture',track)
    monkeypatch.setattr(module,'build_live_microwave_collision_scene',lambda **kw:kw)
    observer=module.make_live_microwave_observer(output_dir=tmp_path/'observations',
        camera_endpoint='unused',expected_serial='unused',pose_client=object(),sessions={BOWL:'bowl'},
        side='right',camera_id='0',T_sdk_camera=np.eye(4),robots={},grippers={},angle_reader=None,
        scene_parameters={'active_group':'right'},max_age_s=5.,microwave_pose_mesh_path='handle.obj',
        operation_perception=Policy(),state_sampler=Sampler())
    result=observer()
    assert len(captures)==len(calls)==len(samples)==2
    assert result['observed'].frame_number==2
    assert result['robot_samples'] is samples[1]
    assert all(call['fresh_estimation'] is False for call in calls)

