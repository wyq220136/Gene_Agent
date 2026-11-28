"""CPU回归：多tool聚合指标不能冒充TCP距离，日志更正不改变准入。"""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from gene_agent.motion.curobo.core import summarize_ik_failure, solver_failure_diagnostic
from gene_agent.motion.curobo.pose_error_logging import annotate_tcp_pose_errors
from gene_agent.feedback.motion_diagnostics import summarize_motion_failure


def _raw():
    return SimpleNamespace(success=np.array([[False, False]]), feasible=np.array([[False, False]]),
        # Reproduce the upstream multi-tool concatenation/reshape aggregate.
        position_error=np.array([[.03055627, .03115835]]),
        rotation_error=np.array([[.03055621, .03115829]]),
        solution=np.array([[[1., 0.], [2., 0.]]]))


def _solver():
    return SimpleNamespace(config=SimpleNamespace(position_tolerance=.005, orientation_tolerance=.05))


def _poses():
    values=np.broadcast_to(np.eye(4),(2,4,4)).copy()
    values[0,:3,3]=[.00667681667,0,0]
    values[1,:3,3]=[.00653109432,0,0]
    values[0,:3,:3]=Rotation.from_euler('z',.03055629367).as_matrix()
    values[1,:3,:3]=Rotation.from_euler('z',.03115834130).as_matrix()
    return values


def test_multitool_metrics_are_preserved_without_false_metre_label():
    raw=_raw();d=summarize_ik_failure(raw,_solver())
    assert d['selected_seed_index']==0
    seed=d['seeds'][0]
    assert seed['position_error_m'] is None
    assert seed['rotation_error_rad'] is None
    assert seed['backend_error_metrics']['position_error']==pytest.approx(.03055627)
    assert seed['pose_error_source']=='world_tcp_fk_unavailable'


def test_recomputes_all_seeds_once_and_summary_uses_physical_error():
    raw=_raw();before=deepcopy(vars(raw));d=summarize_ik_failure(raw,_solver());calls=[]
    def fk(names,values):
        calls.append((names,values.copy()))
        return _poses()
    annotate_tcp_pose_errors(d,np.eye(4),('j1','j2'),fk_batch=fk)
    assert len(calls)==1
    np.testing.assert_array_equal(calls[0][1],raw.solution[0])
    assert d['seeds'][0]['position_error_m']==pytest.approx(.00667681667)
    assert d['seeds'][0]['rotation_error_rad']==pytest.approx(.03055629367)
    assert d['selected_seed_index']==0  # second seed's smaller TCP error must not change policy
    assert d['successful_seed_count']==d['pose_converged_seed_count']==0
    assert d['tcp_pose_converged_seed_count']==0
    summary=summarize_motion_failure({'success':False,'failure_category':'goal_ik_no_solution',
                                      'ik_diagnostic':d})
    assert summary.best_position_error_m==pytest.approx(.00667681667)
    assert summary.best_rotation_error_rad==pytest.approx(.03055629367)
    assert summary.pose_error_source=='world_tcp_fk'
    for name,value in before.items():np.testing.assert_array_equal(getattr(raw,name),value)


def test_world_tcp_offset_and_geodesic_angle():
    world_base=np.eye(4);world_base[:3,3]=[.6,-.2,1.]
    world_base[:3,:3]=Rotation.from_euler('xyz',[30,10,90],degrees=True).as_matrix()
    link_tcp=np.eye(4);link_tcp[2,3]=.15
    target=world_base@link_tcp
    candidate=target.copy();candidate[:3,3]+=[.003,.004,0]
    candidate[:3,:3]=target[:3,:3]@Rotation.from_euler('x',7,degrees=True).as_matrix()
    d={'joint_positions_rad':[.1], 'position_tolerance_m':.005,'rotation_tolerance_rad':.05}
    annotate_tcp_pose_errors(d,target,('joint',),fk_poses=candidate[None])
    assert d['position_error_m']==pytest.approx(.005)
    assert d['rotation_error_rad']==pytest.approx(np.deg2rad(7))
    assert d['tcp_pose_error_reference']['frame']=='world_tcp'


@pytest.mark.parametrize('mode',['exception','shape','nan','nonrigid'])
def test_invalid_fk_marks_unknown_instead_of_reusing_aggregate(mode):
    d=summarize_ik_failure(_raw(),_solver())
    def invalid(*args):
        if mode=='exception':raise RuntimeError('FK unavailable')
        if mode=='shape':return np.eye(4)
        values=_poses()
        if mode=='nan':values[0,0,3]=np.nan
        if mode=='nonrigid':values[0,0,0]=2.
        return values
    annotate_tcp_pose_errors(d,np.eye(4),('j1','j2'),fk_batch=invalid)
    assert 'tcp_pose_error_recalculation_error' in d
    assert all(row['position_error_m'] is None for row in d['seeds'])
    assert all(row['rotation_error_rad'] is None for row in d['seeds'])
    assert d['seeds'][0]['backend_error_metrics']['position_error']==pytest.approx(.03055627)
    assert d['selected_seed_index']==0 and d['successful_seed_count']==0


@pytest.mark.parametrize('actual_error,collision,expected_success',[
    (.012,False,True),(.035,False,False),(.012,True,False)])
def test_real_pose_budget_prevents_false_rejection_and_keeps_real_failures(actual_error,collision,expected_success):
    from test_approximate_ik_plan import Planner
    from gene_agent.motion.curobo.approximate import ApproximateIKBudget,plan_with_approximate_ik
    class CorrectedLogsPlanner(Planner):
        def plan(self,*args,**kwargs):
            report=super().plan(*args,**kwargs)
            if not report.get('success'):
                seed=report['ik_diagnostic']['seeds'][0]
                seed['joint_positions_rad']=[1.-actual_error]
                report['ik_diagnostic']['selected_joint_positions_rad']=[1.-actual_error]
                # Backend aggregate is 30mm-labelled even when the physical error
                # is 12mm; for a bad pose it can conversely appear deceptively small.
                seed['position_error_m']=.03 if actual_error < .02 else .001
                annotate_tcp_pose_errors(report['ik_diagnostic'],args[2],args[0],
                    fk_batch=lambda names,q:np.array([self.fk(names,row) for row in q]))
            return report
        def fk(self,names,q):
            pose=np.eye(4);pose[0,3]=1.-actual_error
            return pose
    planner=CorrectedLogsPlanner(position_error=.03,collision=collision)
    target=np.eye(4);target[0,3]=1.
    result=plan_with_approximate_ik(planner,('j1',),(.2,),target,budget=ApproximateIKBudget(.02,.1))
    assert result['success'] is expected_success
    assert len(planner.targets)==(2 if expected_success else 1)
    if expected_success:
        assert result['approximate_ik']['position_error_m']==pytest.approx(actual_error)


def test_already_successful_plan_does_not_enter_error_diagnostics():
    from gene_agent.motion.curobo.approximate import ApproximateIKBudget,plan_with_approximate_ik
    success={'success':True,'positions_rad':[[0.],[1.]],'interpolation_dt_s':.025}
    class Planner:
        calls=0
        def plan(self,*args,**kwargs):self.calls+=1;return success
        def fk(self,*args):raise AssertionError('normal success must not be reclassified by diagnostic FK')
    planner=Planner()
    result=plan_with_approximate_ik(planner,('j1',),(.2,),np.eye(4),budget=ApproximateIKBudget(.02,.1))
    assert result['success'] is True
    assert planner.calls==1


def test_backend_convergence_is_distinct_from_tcp_pose_convergence():
    d=solver_failure_diagnostic(SimpleNamespace(position_error=np.array([.03]),
        rotation_error=np.array([.03]),feasible=np.array([False])),'ik',_solver())
    d['joint_positions_rad']=[0.]
    annotate_tcp_pose_errors(d,np.eye(4),('j1',),fk_poses=np.eye(4)[None])
    assert d['pose_converged'] is False
    assert d['pose_converged_source']=='backend_aggregate_metrics'
    assert d['tcp_pose_converged'] is True
    assert d['constraint_feasible'] is False
