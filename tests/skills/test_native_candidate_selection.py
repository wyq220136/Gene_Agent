from types import SimpleNamespace
from gene_agent.tasks.policies.novel_tool_task.candidate_selection import first_feasible_candidate


def test_grasp_must_have_feasible_grasp_and_future_alignment():
    flags=[False,True,True, True,False,False, True,False,True]
    results=[SimpleNamespace(success=x) for x in flags]
    assert first_feasible_candidate(results,3)==2
    assert first_feasible_candidate(results[:6],3) is None


def test_bounded_alignment_is_retained_for_real_planning_without_claiming_ik_success():
    budget={'max_position_error_m':.005,'max_rotation_error_rad':.03}
    seed=dict(position_error_m=.004,rotation_error_rad=.02,joint_positions_rad=[.1])
    failed=SimpleNamespace(success=False,failure=SimpleNamespace(details={'ik_diagnostic':{'seeds':[seed]}}))
    row=[SimpleNamespace(success=True),failed]
    assert first_feasible_candidate(row,2) is None
    assert first_feasible_candidate(row,2,alignment_budget=budget)==0
    assert failed.success is False
    seed['position_error_m']=.02
    assert first_feasible_candidate(row,2,alignment_budget=budget) is None


def test_pregrasp_prescreen_forwards_budget_and_uses_substitute_joint_endpoint():
    from contextlib import contextmanager
    import numpy as np
    from gene_agent.motion import JointState
    from gene_agent.tasks.policies.novel_tool_task.candidate_selection import screen_approach_path
    budget={'max_position_error_m':.01,'max_rotation_error_rad':.05}
    endpoint=JointState(('j',),(.7,));starts=[]
    class Session:
        @contextmanager
        def stage(self,**kwargs):starts.append(kwargs['start']);yield self
        def plan_to(self,target,*,approximate_ik_budget=None,**kw):
            return SimpleNamespace(success=approximate_ik_budget==budget,goal_state=endpoint,to_dict=lambda:{})
        def plan_linear(self,target,**kw):
            assert 'approximate_ik_budget' not in kw
            return SimpleNamespace(success=True,to_dict=lambda:{'success':True})
    backend=SimpleNamespace(session=Session(),request=SimpleNamespace(group='arm',settings=SimpleNamespace(
        approach_distance_m=.08,pregrasp_approximate_ik_budget=budget)),robot={'grippers':{'arm':{'approach_axis_tcp':[1,0,0]}}})
    okay,_=screen_approach_path(backend,{},JointState(('j',),(0.,)),0,np.eye(4),index=0)
    assert okay and starts[-1]==endpoint


def test_path_screen_uses_planned_pregrasp_end_and_rejects_bad_line():
    from contextlib import contextmanager
    import numpy as np
    from gene_agent.motion import JointState
    from gene_agent.tasks.policies.novel_tool_task.candidate_selection import screen_approach_path
    initial=JointState(('j',),(0.,));pre_end=JointState(('j',),(1.,))
    starts=[];calls=[]
    class Session:
        @contextmanager
        def stage(self,**kw):
            starts.append(kw['start']);yield self
        def plan_to(self,target,**kw):
            calls.append(('pre',target.matrix.copy()))
            return SimpleNamespace(success=True,goal_state=pre_end,to_dict=lambda:{'success':True})
        def plan_linear(self,target,**kw):
            calls.append(('line',target.matrix.copy()))
            return SimpleNamespace(success=False,to_dict=lambda:{'success':False})
    backend=SimpleNamespace(session=Session(),request=SimpleNamespace(group='arm',settings=SimpleNamespace(approach_distance_m=.15)),robot={'grippers':{'arm':{'approach_axis_tcp':[1,0,0]}}})
    ok,evidence=screen_approach_path(backend,{},initial,0,np.eye(4),index=2)
    assert not ok
    assert starts==[initial,pre_end]
    assert calls[0][1][0,3]==-.15
    assert calls[1][1][0,3]==0
    assert evidence['approach']['success'] is False


def test_selection_skips_candidate_with_failed_path_without_executing(tmp_path,monkeypatch):
    import numpy as np
    from contextlib import contextmanager
    from gene_agent.tasks.policies.novel_tool_task import candidate_selection as module
    from gene_agent.skills.contracts import OperationState
    state=OperationState('same','fixture',{'episode_id':'e','scene_revision':0,'objects':{'tool':{'T_world_object':np.eye(4).tolist()}},'robots':{'arm':{'joint_names':['j'],'joint_positions_rad':[0.]}}})
    @contextmanager
    def stage(**kw):
        yield SimpleNamespace(evaluate_ik=lambda *a,**k:[SimpleNamespace(success=True,to_dict=lambda:{'success':True})]*2)
    pool=[{'T_object_tcp':np.eye(4).tolist(),'id':str(i)} for i in range(2)]
    pool[1]['T_object_tcp'][1][3]=.01
    calls=[]
    def screen(*a,**k):
        calls.append(k['index']);return k['index']==1,{'fixture':True}
    monkeypatch.setattr(module,'screen_approach_path',screen)
    backend=SimpleNamespace(candidate={'grasp_candidates':pool,'screen_approach_path':True},request=SimpleNamespace(group='arm',object_id='tool'),runtime=SimpleNamespace(observe=lambda **k:state),scene=SimpleNamespace(snapshot=lambda **k:{}),session=SimpleNamespace(stage=stage),recorder=SimpleNamespace(root=tmp_path,append_event=lambda e:None))
    assert module.select_grasp(backend,state)
    assert calls==[0,1]
    assert backend.candidate['selected_grasp']['id']=='1'
    assert backend.local[1,3]==.01


def test_grasp_ik_contact_policy_does_not_leak_to_alignment():
    from contextlib import contextmanager
    from gene_agent.tasks.policies.novel_tool_task.candidate_selection import evaluate_candidate_ik
    calls=[]
    @contextmanager
    def stage(**kwargs):
        def evaluate(targets,request_ids):
            calls.append((kwargs,targets,request_ids))
            return list(targets)
        yield SimpleNamespace(evaluate_ik=evaluate)
    results=evaluate_candidate_ik(SimpleNamespace(stage=stage),list(range(6)),3,
        dict(snapshot='strict',group='arm'),contact_snapshot='contact',contact_policy='finger')
    assert results==list(range(6))
    assert calls[0][1]==[0,3]
    assert calls[0][0]['snapshot']=='contact'
    assert calls[0][0]['collision_policy']=='finger'
    assert calls[1][1]==[1,2,4,5]
    assert calls[1][0]['snapshot']=='strict'
    assert 'collision_policy' not in calls[1][0]
    assert calls[1][2]==['candidate/arm/1','candidate/arm/2','candidate/arm/4','candidate/arm/5']


def test_predicted_lift_uses_attachment_and_rejects_collision():
    import numpy as np
    from contextlib import contextmanager
    from gene_agent.tasks.policies.novel_tool_task.candidate_selection import screen_held_lift
    from gene_agent.motion import JointState
    captured={}
    def snapshot(**kw):
        captured.update(kw);return {'held':True}
    @contextmanager
    def stage(**kw):
        captured['stage']=kw
        def linear(target,**kw):
            captured['target']=target.matrix
            return SimpleNamespace(success=False,to_dict=lambda:{'success':False})
        yield SimpleNamespace(plan_linear=linear)
    backend=SimpleNamespace(request=SimpleNamespace(group='arm',object_id='tool',settings=SimpleNamespace(lift_distance_m=.08)),robot={'transforms':{'arm':{'link_from_tcp':np.eye(4)}}},scene=SimpleNamespace(snapshot=snapshot),session=SimpleNamespace(stage=stage))
    local=np.eye(4);local[1,3]=.03
    state=SimpleNamespace(values={'scene_revision':2})
    q=JointState(('j',),(1.,))
    ok,record=screen_held_lift(backend,state,np.eye(4),local,q,index=0)
    assert not ok and not record['success']
    assert captured['held_object']['T_link_object'][1][3]==-.03
    assert captured['stage']['start'] is q
    assert captured['target'][2,3]==.08
    assert captured['collision_policy'].policy_id=='lift_support'
