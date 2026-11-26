"""纯几何候选复用必须保留精确输入、顺序和消费者之间的隔离。"""
import numpy as np
import pytest
from gene_agent.knowledge.operation import door_push as module
from gene_agent.knowledge.operation.rotation import AxisSpec


def inputs():
    pose = np.eye(4)
    pose[:3, 3] = [.413217, .3, .2]
    return pose, dict(panel_width_m=.36, panel_height_m=.22,
        panel_thickness_m=.016, axis=AxisSpec([.4,.48,.2], [0,0,-1], 'world', 'reuse-test'),
        mode='right_open', start_angle_rad=.2, target_angle_rad=.3, step_rad=.1,
        reference_origin_world=np.array([0.,0.,0.]), overrides={})


def test_repeated_candidate_generation_reuses_work_and_isolates_mutation(monkeypatch):
    pose, kw = inputs()
    first = module.generate_mode_door_push_candidate_sets(pose, **kw)
    expected = first[0].candidates[0].tcp_pose_world.copy()
    first[0].candidates[0].parameters['surface'] = 'corrupted'
    def unexpected(*a, **kw):
        raise AssertionError('identical geometry rebuilt')
    monkeypatch.setattr(module, 'door_push_pose', unexpected)
    second = module.generate_mode_door_push_candidate_sets(pose.copy(), **kw)
    np.testing.assert_array_equal(second[0].candidates[0].tcp_pose_world, expected)
    assert second[0].candidates[0].parameters['surface'] != 'corrupted'
    assert not second[0].candidates[0].tcp_pose_world.flags.writeable
    assert not second[0].axis_point_world.flags.writeable
    assert second[0].candidates[0].candidate_id == first[0].candidates[0].candidate_id


@pytest.mark.parametrize('changed', ['pose','angle','axis','origin','override','size','source'])
def test_candidate_reuse_never_hides_changed_inputs(monkeypatch, changed):
    pose, kw = inputs()
    module.generate_mode_door_push_candidate_sets(pose, **kw)
    if changed == 'pose': pose[0,3] += 1e-10
    elif changed == 'angle': kw['start_angle_rad'] += 1e-10
    elif changed == 'axis': kw['axis'] = AxisSpec([.4+1e-10,.48,.2],[0,0,-1],'world','reuse-test')
    elif changed == 'origin': kw['reference_origin_world'][0] += 1e-10
    elif changed == 'override': kw['overrides'] = {'tip_x_offset_deg': 1e-10}
    elif changed == 'size': kw['panel_width_m'] += 1e-10
    elif changed == 'source': kw['axis'] = AxisSpec([.4,.48,.2],[0,0,-1],'world','different-source')
    def changed_call(*a, **kw):
        raise RuntimeError('new geometry evaluated')
    monkeypatch.setattr(module, 'door_push_pose', changed_call)
    with pytest.raises(RuntimeError, match='new geometry evaluated'):
        module.generate_mode_door_push_candidate_sets(pose, **kw)


@pytest.mark.parametrize('mode', ['right_open','right_close','left_open','left_close'])
def test_cached_candidates_exactly_match_uncached_geometry_and_order(mode):
    pose, kw = inputs()
    kw['mode'] = mode
    if mode.endswith('_close'):
        kw['start_angle_rad'],kw['target_angle_rad']=.3,.2
    expected = module._generate_mode_door_push_candidate_sets(pose, **kw)
    actual = module.generate_mode_door_push_candidate_sets(pose, **kw)
    assert [x.to_dict() for x in actual] == [x.to_dict() for x in expected]
