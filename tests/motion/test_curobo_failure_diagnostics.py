import numpy as np
from gene_agent.motion.curobo.diagnostics import DiagnosticOptions, DiagnosticBudget, build_sphere_identity, candidate_diagnostics


def test_sphere_identity_contains_source_and_snapshot():
    rows = build_sphere_identity(['wrist','gene_held_tool'], [[0,0,0,.02],[0,0,0,.01]],
        model_hash='model', snapshot_hash='scene', group='right',
        held_object={'object_id':'hammer','sphere_sources':[{'source_body':'hammer_body','source_shape_id':'convex_2'}]})
    assert rows[1]['object_id'] == 'hammer'
    assert rows[1]['source_shape_id'] == 'convex_2'
    assert rows[1]['snapshot_hash'] == 'scene'
    assert 'hammer' in rows[1]['label'] and 'gene_held_tool' in rows[1]['label']
    assert rows[0]['object_id'] == 'robot'


def test_zero_budget_never_starts_candidate_fk():
    class Planner:
        def fk(self,*args):raise AssertionError('must not compute')
    result=candidate_diagnostics(Planner(), ['q'], np.eye(4), [{'index':0,'joint_positions_rad':[0], 'position_error_m':0,'rotation_error_rad':0}], DiagnosticBudget(DiagnosticOptions(budget_s=0)))
    assert result['checked'] == []
    assert result['nearest_checked_feasible'] is None


def test_close_but_colliding_candidate_not_usable():
    class Planner:
        def fk(self,*args):return np.eye(4)
        def _collision_diagnostic(self,*args):return dict(joint_limit_violation=False,self_collision=True,scene_collision=False)
    result=candidate_diagnostics(Planner(), ['q'], np.eye(4), [{'index':0,'joint_positions_rad':[0], 'position_error_m':0,'rotation_error_rad':0}], DiagnosticBudget(DiagnosticOptions()))
    assert result['checked'][0]['position_error_m']==0
    assert result['nearest_checked_feasible'] is None
    assert result['checked'][0]['constraint_feasible'] is False


def test_budget_caps_additional_solver_calls():
    b=DiagnosticBudget(DiagnosticOptions())
    assert b.allow('ik',solver=True)
    assert not b.allow('another',solver=True)
    assert b.to_dict()['solver_calls']==1


def test_float32_sphere_identity_is_json_serializable():
    import json
    rows=build_sphere_identity(['arm'],np.array([[0,0,0,.02]],dtype=np.float32),
        model_hash='m',snapshot_hash='s',group='right')
    json.dumps(rows,allow_nan=False)


def test_unsupported_target_frame_returns_explicit_status():
    from types import SimpleNamespace
    from gene_agent.motion.curobo.diagnostics import collect_failure_diagnostics
    request=SimpleNamespace(target=SimpleNamespace(frame='tool_link',matrix=np.eye(4)))
    result=collect_failure_diagnostics(object(),request,{}, {},DiagnosticOptions())
    assert result['status']=='unsupported_target_frame'


def test_candidate_uses_recorded_tolerances_and_unknown_when_missing():
    class Planner:
        def fk(self,*args):
            pose=np.eye(4);pose[0,3]=.003;return pose
        def _collision_diagnostic(self,*args):return dict(joint_limit_violation=False,self_collision=False,scene_collision=False)
    args=(Planner(),['q'],np.eye(4),[dict(index=0,joint_positions_rad=[0],position_error_m=.003,rotation_error_rad=0)])
    result=candidate_diagnostics(*args,DiagnosticBudget(DiagnosticOptions()),tolerances={'position_tolerance_m':.001,'rotation_tolerance_rad':.01})
    assert result['checked'][0]['strict_pose_converged'] is False
    assert result['checked'][0]['position_tolerance_m']==.001
    unknown=candidate_diagnostics(*args,DiagnosticBudget(DiagnosticOptions()))
    assert unknown['checked'][0]['strict_pose_converged'] is None


def test_failed_trajectories_keep_batch_boundaries():
    from gene_agent.motion.curobo.diagnostics import retain_failed_trajectories
    paths=np.arange(3*4*2).reshape(3,4,2)
    rows=retain_failed_trajectories(paths,2)
    assert len(rows)==2
    assert rows[0]['positions_rad']==paths[0].tolist()
    assert rows[1]['positions_rad']==paths[1].tolist()
    assert rows[0]['returned_trajectory_count']==3


def test_trajectory_diagnostics_bound_sampling_and_select_invalid_path():
    from gene_agent.motion.curobo.diagnostics import check_failed_trajectories,select_failure_configuration
    class Planner:
        def _configuration_validity(self, q):
            assert len(q)<=64
            return [float(v[0])==0 for v in q]
    paths=retain_paths=np.array([np.zeros((200,2)),np.ones((200,2)),np.ones((200,2))*2])
    from gene_agent.motion.curobo.diagnostics import retain_failed_trajectories
    rows=retain_failed_trajectories(paths,2)
    result=check_failed_trajectories(Planner(),rows,DiagnosticBudget(DiagnosticOptions()))
    assert result['sampled_waypoints']==128
    assert rows[1]['first_invalid_waypoint']==0
    q,ref=select_failure_configuration({},rows,None,[9,9])
    assert q==[1,1] and ref['trajectory_index']==1 and ref['waypoint_index']==0


def test_effective_spheres_reuses_public_to_planner_joint_mapping():
    from types import SimpleNamespace
    from gene_agent.motion.curobo.diagnostics import effective_spheres
    class Tensor:
        def __init__(self,value):self.value=np.asarray(value)
        def detach(self):return self
        def cpu(self):return self
        def numpy(self):return self.value
    params=SimpleNamespace(link_sphere_idx_map=Tensor([0]),link_name_to_idx_map={'arm':0},link_spheres=Tensor([[0,0,0,.02]]))
    calls=[]
    def mapped(names,q):
        calls.append((names,list(q)));return 'mapped_private_names'
    def compute(state):
        assert state=='mapped_private_names'
        return SimpleNamespace(robot_spheres=Tensor([[.1,0,0,.02]]))
    planner=SimpleNamespace(kinematics=SimpleNamespace(config=SimpleNamespace(kinematics_config=params),compute_kinematics=compute),
        _state=mapped,joint_names=['public_q'],robot_config={},scene_snapshot_sha256='s',held_object=None,world_from_base=np.eye(4))
    rows,_=effective_spheres(planner,[.2],{'active_group':'right'})
    assert calls==[(['public_q'],[.2])]
    assert rows[0]['center_world_m']==[.1,0,0]
