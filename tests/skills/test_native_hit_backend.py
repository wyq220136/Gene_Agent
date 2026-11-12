from types import SimpleNamespace
from dataclasses import replace
import numpy as np
from gene_agent.skills.contracts import OperationState,StageOutcome
from gene_agent.tasks.policies.novel_tool_task.hit_backend import NativeHitBackend


def make_state(tcp,tool,identity):
    target=np.eye(4);target[2,3]=.8
    return OperationState(identity,'actual',{'episode_id':'one','objects':{'tool':{'T_world_object':tool.tolist()},'target':{'T_world_object':target.tolist()}},'robots':{'arm':{'T_world_tcp':tcp.tolist()}}})


def test_strike_rejects_slipped_tool_even_when_arm_reaches_goal(tmp_path):
    tool=np.diag([1.,-1.,-1.,1.]);tool[2,3]=.855
    start=make_state(tool,tool,'start')
    recorder=SimpleNamespace(root=tmp_path,append_state=lambda *a:None,append_event=lambda *a:None)
    def move(name,state,goal,**kwargs):
        return StageOutcome(name,True,state,make_state(goal,tool,'end'),details={'execution_status':'executed'})
    motion=SimpleNamespace(anchor=np.eye(4),request=SimpleNamespace(group='arm',object_id='tool'),recorder=recorder,move_target=move)
    backend=NativeHitBackend(motion,{'strike_point_object_m':[0,0,0],'strike_normal_object':[0,0,1],'target_point_object_m':[0,0,0],'clearance_m':.055,'stroke_m':.09},target_body='nail')
    result=backend.execute(SimpleNamespace(name='strike'),start)
    assert not result.success
    assert result.check.status=='fail'
    assert result.check.evidence['relative_translation_m']>.08


def test_native_termination_is_preserved_without_claiming_trajectory_completion(tmp_path):
    tool=np.diag([1.,-1.,-1.,1.]);tool[2,3]=.855
    start=make_state(tool,tool,'start')
    outcome=StageOutcome('strike',False,start,start,details={'execution_status':'partial'},failure={'code':'native_terminated'})
    motion=SimpleNamespace(anchor=np.eye(4),request=SimpleNamespace(group='arm',object_id='tool'),recorder=SimpleNamespace(append_state=lambda *a:None,append_event=lambda *a:None),move_target=lambda *a,**kw:outcome)
    backend=NativeHitBackend(motion,{'strike_point_object_m':[0,0,0],'strike_normal_object':[0,0,1],'target_point_object_m':[0,0,0],'clearance_m':.055,'stroke_m':.09},target_body='nail')
    assert backend.execute(SimpleNamespace(name='strike'),start) is outcome


def test_align_tries_next_yaw_only_after_unexecuted_planning_failure(tmp_path):
    tool=np.diag([1.,-1.,-1.,1.]);tool[2,3]=.855
    start=make_state(tool,tool,'start');calls=[]
    def move(name,state,goal,**kwargs):
        calls.append(goal)
        if len(calls)==1:return StageOutcome(name,False,state,state,details={'execution_status':'not_executed'},failure={'code':'goal_ik_no_solution','source':'motion'})
        return StageOutcome(name,True,state,make_state(goal,goal,'end'),details={'execution_status':'executed'})
    recorder=SimpleNamespace(root=tmp_path,append_state=lambda *a:None,append_event=lambda *a:None)
    motion=SimpleNamespace(anchor=np.eye(4),request=SimpleNamespace(group='arm',object_id='tool'),recorder=recorder,move_target=move)
    parameters=dict(strike_point_object_m=[0,0,0],strike_normal_object=[0,0,1],target_point_object_m=[0,0,0],clearance_m=.055,stroke_m=.09,alignment_yaw_offsets_rad=[0,-.3])
    result=NativeHitBackend(motion,parameters,target_body='nail').execute(SimpleNamespace(name='align'),start)
    assert result.success and len(calls)==2


def test_align_does_not_retry_after_partial_execution(tmp_path):
    tool=np.diag([1.,-1.,-1.,1.]);tool[2,3]=.855
    start=make_state(tool,tool,'start');calls=[]
    def move(name,state,goal,**kwargs):
        calls.append(goal)
        return StageOutcome(name,False,state,state,details={'execution_status':'partial'},failure={'code':'tracking_error','source':'robotwin'})
    recorder=SimpleNamespace(root=tmp_path,append_state=lambda *a:None,append_event=lambda *a:None)
    motion=SimpleNamespace(anchor=np.eye(4),request=SimpleNamespace(group='arm',object_id='tool'),recorder=recorder,move_target=move)
    parameters=dict(strike_point_object_m=[0,0,0],strike_normal_object=[0,0,1],target_point_object_m=[0,0,0],clearance_m=.055,stroke_m=.09,alignment_yaw_offsets_rad=[0,-.3])
    result=NativeHitBackend(motion,parameters,target_body='nail').execute(SimpleNamespace(name='align'),start)
    assert not result.success and len(calls)==1 and result.details['execution_status']=='partial'


def test_withdraw_is_linear_from_actual_pose_and_checks_actual_hold(tmp_path):
    tool=np.diag([1.,-1.,-1.,1.]);tool[:3,3]=[.01,.02,.815]
    start=make_state(tool,tool,'actual_strike');calls=[]
    def move(name,state,goal,**kwargs):
        calls.append((name,state,goal,kwargs))
        return StageOutcome(name,True,state,make_state(goal,goal,'withdraw_end'),
            details={'execution_status':'executed'})
    recorder=SimpleNamespace(root=tmp_path,append_state=lambda *a:None,append_event=lambda *a:None)
    motion=SimpleNamespace(anchor=np.eye(4),request=SimpleNamespace(group='arm',object_id='tool'),recorder=recorder,move_target=move)
    p=dict(strike_point_object_m=[0,0,0],strike_normal_object=[0,0,1],target_point_object_m=[0,0,0],clearance_m=.055,stroke_m=.09,withdraw_distance_m=.1)
    result=NativeHitBackend(motion,p,target_body='nail').execute(SimpleNamespace(name='withdraw'),start)
    assert result.success and result.check.passed
    name,state,goal,kwargs=calls[0]
    assert name=='withdraw' and state is start and kwargs['linear']
    np.testing.assert_allclose(goal[:3,3],tool[:3,3]+[0,0,.1])
    assert kwargs['collision_policy'].to_dict()['allowances'][0]['first']=='gene_held_tool'
    assert kwargs['collision_policy'].to_dict()['allowances'][0]['second']=='nail'


def test_rotation_failure_remains_json_serializable_and_false(tmp_path):
    import json
    from scipy.spatial.transform import Rotation
    tool=np.diag([1.,-1.,-1.,1.]);tool[2,3]=.855
    start=make_state(tool,tool,'start')
    def move(name,state,goal,**kwargs):
        slipped=goal.copy();slipped[:3,:3]=goal[:3,:3]@Rotation.from_euler('x',.3).as_matrix()
        return StageOutcome(name,True,state,make_state(goal,slipped,'end'),details={'execution_status':'executed'})
    recorder=SimpleNamespace(root=tmp_path,append_state=lambda *a:None,append_event=lambda row:json.dumps(row))
    motion=SimpleNamespace(anchor=np.eye(4),request=SimpleNamespace(group='arm',object_id='tool'),recorder=recorder,move_target=move)
    p=dict(strike_point_object_m=[0,0,0],strike_normal_object=[0,0,1],target_point_object_m=[0,0,0],clearance_m=.055,stroke_m=.09)
    result=NativeHitBackend(motion,p,target_body='nail').execute(SimpleNamespace(name='strike'),start)
    assert result.success is False
    assert result.failure['code']=='tool_postcondition_failed'


def test_strike_time_scale_is_forwarded_without_changing_goal_checks(tmp_path):
    tool=np.diag([1.,-1.,-1.,1.]);tool[2,3]=.855
    start=make_state(tool,tool,'start');scales=[]
    def move(name,state,goal,**kwargs):
        scales.append(kwargs['time_scale'])
        return StageOutcome(name,True,state,make_state(goal,goal,'end'),details={'execution_status':'executed'})
    recorder=SimpleNamespace(root=tmp_path,append_state=lambda *a:None,append_event=lambda *a:None)
    motion=SimpleNamespace(anchor=np.eye(4),request=SimpleNamespace(group='arm',object_id='tool'),recorder=recorder,move_target=move)
    p=dict(strike_point_object_m=[0,0,0],strike_normal_object=[0,0,1],target_point_object_m=[0,0,0],clearance_m=.055,stroke_m=.09,strike_time_scale=2.)
    result=NativeHitBackend(motion,p,target_body='nail').execute(SimpleNamespace(name='strike'),start)
    assert scales==[2.] and result.success and result.check.passed


def test_withdraw_accepts_declared_fk_substitute_and_preserves_original_goal(tmp_path):
    tool=np.diag([1.,-1.,-1.,1.]);tool[2,3]=.815
    start=make_state(tool,tool,'strike_end');calls=[]
    budget={'max_position_error_m':.01,'max_rotation_error_rad':.04}
    def move(name,state,goal,**kw):
        calls.append(kw.get('approximate_ik_budget'));actual=goal.copy();actual[0,3]+=.009
        approximation={'approximate':True,'budget':budget,'requested_tcp_pose_world':goal.tolist(),
            'actual_endpoint_tcp_pose_world':actual.tolist()}
        return StageOutcome(name,True,state,make_state(actual,actual,'end'),details={
            'execution_status':'executed','planning':{'metrics':{'approximate_ik':approximation}}})
    recorder=SimpleNamespace(root=tmp_path,append_state=lambda *a:None,append_event=lambda *a:None)
    motion=SimpleNamespace(anchor=np.eye(4),request=SimpleNamespace(group='arm',object_id='tool'),recorder=recorder,move_target=move)
    p=dict(strike_point_object_m=[0,0,0],strike_normal_object=[0,0,1],target_point_object_m=[0,0,0],
        clearance_m=.055,stroke_m=.09,withdraw_distance_m=.1,approximate_ik_by_stage={'withdraw':budget})
    outcome=NativeHitBackend(motion,p,target_body='nail').execute(SimpleNamespace(name='withdraw'),start)
    assert calls==[budget] and outcome.success
    assert outcome.check.evidence['approximate_ik']['requested_tcp_pose_world'][0][3]==0.
    assert outcome.check.evidence['strike_point_error_m']<1e-9


def test_hit_refuses_approximate_budget_for_precise_strike():
    import pytest
    p=dict(strike_point_object_m=[0,0,0],strike_normal_object=[0,0,1],target_point_object_m=[0,0,0],
        clearance_m=.055,stroke_m=.09,approximate_ik_by_stage={'strike':{'max_position_error_m':.01,'max_rotation_error_rad':.05}})
    with pytest.raises(ValueError,match='align|withdraw'):
        NativeHitBackend(SimpleNamespace(anchor=np.eye(4)),p,target_body='nail')


def test_withdraw_refuses_budget_that_covers_the_entire_retreat():
    import pytest
    p=dict(strike_point_object_m=[0,0,0],strike_normal_object=[0,0,1],target_point_object_m=[0,0,0],
        clearance_m=.055,stroke_m=.09,withdraw_distance_m=.1,
        approximate_ik_by_stage={'withdraw':{'max_position_error_m':.11,'max_rotation_error_rad':.04}})
    with pytest.raises(ValueError,match='撤离|withdraw'):
        NativeHitBackend(SimpleNamespace(anchor=np.eye(4)),p,target_body='nail')


def test_approximate_withdraw_requires_actual_outward_motion(tmp_path):
    tool=np.diag([1.,-1.,-1.,1.]);tool[2,3]=.815
    start=make_state(tool,tool,'strike_end')
    budget={'max_position_error_m':.095,'max_rotation_error_rad':.04}
    def move(name,state,goal,**kw):
        # 替代FK离起点仅5mm，实际执行又落后5mm；均不能代替真正撤离。
        substitute=tool.copy();substitute[2,3]+=.005
        approximation={'approximate':True,'budget':budget,'requested_tcp_pose_world':goal.tolist(),
            'actual_endpoint_tcp_pose_world':substitute.tolist()}
        return StageOutcome(name,True,state,make_state(tool,tool,'end'),details={
            'execution_status':'executed','planning':{'metrics':{'approximate_ik':approximation}}})
    recorder=SimpleNamespace(root=tmp_path,append_state=lambda *a:None,append_event=lambda *a:None)
    motion=SimpleNamespace(anchor=np.eye(4),request=SimpleNamespace(group='arm',object_id='tool'),recorder=recorder,move_target=move)
    p=dict(strike_point_object_m=[0,0,0],strike_normal_object=[0,0,1],target_point_object_m=[0,0,0],
        clearance_m=.055,stroke_m=.09,withdraw_distance_m=.1,approximate_ik_by_stage={'withdraw':budget})
    result=NativeHitBackend(motion,p,target_body='nail').execute(SimpleNamespace(name='withdraw'),start)
    assert not result.success
    assert result.check.evidence['actual_withdrawal_m']==0.
