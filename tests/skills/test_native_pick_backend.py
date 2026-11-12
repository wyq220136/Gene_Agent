"""原生后端复用Pick阶段，接近失败不得换路，证据不足不能过抓取检查。"""
from contextlib import contextmanager
from types import SimpleNamespace
import numpy as np
from gene_agent.skills.contracts import OperationState,StageOutcome
from gene_agent.skills.library.pick import run
from gene_agent.skills.library.pick.model import PickRequest
from gene_agent.motion import JointState,JointTrajectory,MotionResult,MotionFailure,MotionFailureKind,MotionOperation,PlannerMode


class Runtime:
    def __init__(self):self.steps=0
    def observe(self,**kwargs):
        return OperationState(str(self.steps),'fixture',dict(episode_id='one',scene_revision=0,world_frame='world',sim_time_s=float(self.steps),
            objects={'tool':{'T_world_object':np.eye(4).tolist()}},robots={'arm':{'joint_names':['joint'],'joint_positions_rad':[float(self.steps)],'T_world_tcp':np.eye(4).tolist(),'gripper':{'width_m':.08}}}))
    def _execute(self,stage_id,expected_state_id,**kwargs):
        start=self.observe();assert expected_state_id==start.state_id
        self.steps+=1;return StageOutcome(stage_id,True,start,self.observe(),details={'execution_status':'executed'})
    def execute_gripper(self,**kwargs):return self._execute(**kwargs)
    def execute_trajectory(self,trajectory,**kwargs):return self._execute(**kwargs)
    def hold(self,**kwargs):return self._execute(**kwargs)


class Planner:
    def __init__(self,fail_linear=False):self.fail_linear=fail_linear
    @contextmanager
    def stage(self,**kwargs):yield self
    def plan_to(self,target,*,request_id):return self._plan(request_id,MotionOperation.PLAN,False)
    def plan_linear(self,target,*,request_id):return self._plan(request_id,MotionOperation.LINEAR_TCP,self.fail_linear)
    def _plan(self,id,op,fail):
        return MotionResult(id,op,not fail,'fixture',PlannerMode.CUDA_GRAPH_BENCHMARK,0.,
            trajectory=None if fail else JointTrajectory(('joint',),(0.,.1),((0.,),(1.,))),
            failure=MotionFailure(MotionFailureKind.PATH_NOT_FOUND,'fixture','blocked',{}) if fail else None)


def backend(tmp_path,fail_linear=False):
    from gene_agent.tasks.policies.novel_tool_task.pick_backend import make_pick_backend
    from gene_agent.feedback.native_run import NativeRunRecorder
    runtime=Runtime();request=PickRequest('tool','arm')
    obj=make_pick_backend(runtime=runtime,scene_provider=SimpleNamespace(snapshot=lambda **kw:{}),
        planning_session=Planner(fail_linear),request=request,candidate={'T_object_tcp':np.eye(4).tolist()},
        recorder=NativeRunRecorder(tmp_path/'run',manifest={'episode_id':'one'}),robot_metadata={'grippers':{'arm':{'approach_axis_tcp':[0,0,1]}}})
    return obj,request,runtime


def test_shared_pick_pregrasp_prefix_uses_actual_state(tmp_path):
    obj,request,runtime=backend(tmp_path)
    result=run(obj,request,stop_after='pregrasp')
    assert result.success and result.executed and result.end_state.state_id=='2'
    assert [s.stage for s in result.stages]==['open','pregrasp']


def test_approach_failure_stops_without_other_path_or_execution(tmp_path):
    obj,request,runtime=backend(tmp_path,True)
    result=run(obj,request,stop_after='approach')
    assert not result.success and result.failure_stage=='approach'
    assert result.end_state.state_id=='2' and runtime.steps==2
    assert result.failure_code=='path_not_found'


def test_missing_actual_aperture_is_unknown_not_grasp_success(tmp_path):
    obj,request,runtime=backend(tmp_path)
    result=run(obj,request,stop_after='grasp')
    assert not result.success and result.failure_stage=='grasp_check'
    assert result.stages[-1].check.status=='unknown'
    assert runtime.steps==4


def test_hold_check_waits_and_checks_executed_end_state(tmp_path):
    from dataclasses import replace
    obj,request,runtime=backend(tmp_path)
    obj.anchor=np.eye(4);obj.lift_start=runtime.observe()
    obj.request=replace(request,settings=replace(request.settings,hold_duration_s=.7))
    observed=[]
    def hold(**kw):
        observed.append(kw['duration_s'])
        outcome=runtime._execute(**kw)
        end=outcome.end
        # 模拟保持期间工具滑落；必须检查新state而非lift瞬时状态。
        values={**end.values,'objects':{'tool':{'T_world_object':np.diag([1,1,1,1]).tolist()}}}
        values['objects']['tool']['T_world_object'][0][3]=.1
        return replace(outcome,end=replace(end,values=values))
    runtime.hold=hold
    stage=SimpleNamespace(name='hold_check')
    result=obj.execute(stage,runtime.observe())
    assert observed==[.7] and result.end.state_id=='1'
    assert not result.success and result.check.status=='fail'
    assert result.execution_status=='executed'


def test_grasp_width_interval_is_explicit_and_still_rejects_empty_closure(tmp_path):
    from dataclasses import replace
    obj,_,runtime=backend(tmp_path)
    obj.candidate['contact_width_range_m']=[.016,.026]
    state=runtime.observe()
    def at_width(width):
        values={**state.values,'robots':{'arm':{**state.values['robots']['arm'],'gripper':{'width_m':width}}}}
        return replace(state,values=values)
    assert obj._check('grasp_check',at_width(.019)).check.passed
    assert not obj._check('grasp_check',at_width(0.)).check.passed
    assert not obj._check('grasp_check',at_width(.08)).check.passed


def test_arm_selection_uses_observed_target_and_robot_base():
    from gene_agent.tasks.policies.novel_tool_task import execution
    assert hasattr(execution,'select_group')
    left=np.eye(4);left[0,3]=-.3;right=np.eye(4);right[0,3]=.3
    robot={'joint_groups':{'left':[],'right':[]},'transforms':{'left':{'world_from_urdf_base':left},'right':{'world_from_urdf_base':right}}}
    target=np.eye(4);target[0,3]=.2
    state=OperationState('now','gt',{'objects':{'target':{'T_world_object':target}}})
    assert execution.select_group({'group_selection':'nearest_target'},robot,state)=='right'
    target[0,3]=-.2
    assert execution.select_group({'group_selection':'nearest_target'},robot,state)=='left'


def test_transport_slowdown_preserves_planned_path(tmp_path):
    from gene_agent.motion import CollisionPolicy
    obj,_,runtime=backend(tmp_path)
    seen=[]
    def execute(trajectory,**kwargs):
        seen.append(trajectory)
        return runtime._execute(**kwargs)
    runtime.execute_trajectory=execute
    obj.move_target('align',runtime.observe(),np.eye(4),linear=False,collision_policy=CollisionPolicy(),time_scale=2.)
    assert seen[0].times_s==(0.,.2)
    assert seen[0].positions_rad==((0.,),(1.,))


def test_only_approach_enables_explicit_finger_tool_contact(tmp_path):
    obj,request,runtime=backend(tmp_path)
    obj.candidate['allow_finger_tool_contact']=True
    obj.robot['grippers']['arm']['contact_links']=['finger_a','finger_b']
    calls=[]
    def move(name,start,target,**kwargs):
        calls.append((name,kwargs['collision_policy']))
        return StageOutcome(name,True,start,start,details={'execution_status':'not_executed'})
    obj.move_target=move
    obj._move('pregrasp',runtime.observe());obj._move('approach',runtime.observe())
    assert calls[0][1].allowances==()
    assert {(p.first,p.second) for p in calls[1][1].allowances}=={('finger_a','tool'),('finger_b','tool')}


def test_pregrasp_uses_configured_budget_but_approach_remains_strict(tmp_path):
    from dataclasses import replace
    obj,request,runtime=backend(tmp_path)
    budget={'max_position_error_m':.01,'max_rotation_error_rad':.05}
    obj.request=replace(request,settings=replace(request.settings,pregrasp_approximate_ik_budget=budget))
    class BudgetPlanner(Planner):
        def plan_to(self,target,*,request_id,approximate_ik_budget=None):
            return self._plan(request_id,MotionOperation.PLAN,approximate_ik_budget!=budget)
        def plan_linear(self,target,*,request_id,approximate_ik_budget=None):
            assert approximate_ik_budget is None
            return self._plan(request_id,MotionOperation.LINEAR_TCP,True)
    obj.session=BudgetPlanner()
    pre=obj._move('pregrasp',runtime.observe())
    assert pre.success and pre.execution_status=='executed'
    approach=obj._move('approach',pre.end)
    assert not approach.success and approach.execution_status=='not_executed'


def test_approximate_execution_checks_actual_tcp_against_declared_budget(tmp_path):
    from dataclasses import replace
    from gene_agent.motion import CollisionPolicy
    obj,_,runtime=backend(tmp_path)
    budget={'max_position_error_m':.01,'max_rotation_error_rad':.05}
    class ApproxPlanner(Planner):
        def plan_to(self,target,*,request_id,**kw):
            plan=super().plan_to(target,request_id=request_id)
            return replace(plan,metrics={'approximate_ik':{'approximate':True,'budget':budget,
                'requested_tcp_pose_world':target.matrix.tolist(),'actual_endpoint_tcp_pose_world':target.matrix.tolist()}})
    obj.session=ApproxPlanner()
    original=runtime.execute_trajectory
    def drifted(*args,**kwargs):
        outcome=original(*args,**kwargs);values=dict(outcome.end.values)
        values['robots']['arm']['T_world_tcp'][0][3]=.02
        return replace(outcome,end=replace(outcome.end,values=values))
    runtime.execute_trajectory=drifted
    outcome=obj.move_target('pregrasp',runtime.observe(),np.eye(4),linear=False,
        collision_policy=CollisionPolicy(),approximate_ik_budget=budget)
    assert not outcome.success and outcome.failure['code']=='approximate_ik_postcondition_failed'
    assert outcome.check.evidence['position_error_m']==.02
