"""任务无关的摆位规则及固定/随机语义。"""
import numpy as np
import pytest

from gene_agent.scene.placement import sample_placement,validate_rules


def nominal():
    a=np.eye(4);a[:3,3]=[.4,.3,0]
    b=np.eye(4);b[:3,3]=[.6,.3,.02]
    return {'a':a,'b':b}


def rules():
    return {'schema_version':1,'source':'codex','objects':{
        'a':{'mode':'random','frame':'world','translation_half_range_m':[.02,.03,0],'yaw_half_range_deg':10},
        'b':{'mode':'fixed','frame':'world','translation_half_range_m':[0,0,0],'yaw_half_range_deg':0}}}


def test_random_deterministic_fixed_stays_fixed():
    one=sample_placement(nominal(),rules(),seed=7,randomize=True)
    two=sample_placement(nominal(),rules(),seed=7,randomize=True)
    assert one==two
    assert np.allclose(one['sampled_poses_world']['b'],nominal()['b'])
    assert not np.allclose(one['sampled_poses_world']['a'],nominal()['a'])
    fixed=sample_placement(nominal(),rules(),seed=99,randomize=False)
    for n,p in nominal().items():assert np.allclose(fixed['sampled_poses_world'][n],p)


def test_relative_fixed_follows_reference_object():
    r=rules();r['objects']['b']['frame']='a'
    result=sample_placement(nominal(),r,seed=8,randomize=True)['sampled_poses_world']
    assert np.allclose(np.linalg.inv(result['a'])@result['b'],
                       np.linalg.inv(nominal()['a'])@nominal()['b'])


@pytest.mark.parametrize('change',[
    lambda r:r['objects']['a'].update(frame='b'),
    lambda r:r['objects']['a'].update(translation_half_range_m=[float('nan'),0,0]),
    lambda r:r['objects']['a'].update(mode='whatever'),
])
def test_invalid_rules_fail(change):
    r=rules();r['objects']['b']['frame']='a';change(r)
    with pytest.raises(ValueError):validate_rules(r,tuple(nominal()))


def test_constraints_fail_with_diagnostics_no_ik_filtering():
    r=rules();r['constraints']={'bounds_xy_m':[0,.1,0,.1]};r['max_attempts']=3
    row=sample_placement(nominal(),r,seed=7,randomize=True)
    assert row['valid'] is False
    assert len(row['rejected_attempts'])==3
    assert row['ik_filter'] is False


def test_microwave_resolve_nonbaseline_seed_and_per_object_modes():
    from benchmarks.microwave_task.scene import resolve_settings
    from gene_agent.workbench.scene_inputs import suggested_rules
    a=np.eye(4);a[:3,3]=[.52,.29,.04]
    b=np.eye(4);b[:3,3]=[.255,.12,-.078]
    settings={'schema_version':1,'seed':123,'microwave_pose_world':a.tolist(),'bowl_pose_world':b.tolist()}
    settings['placement_rules']=suggested_rules('microwave',settings)
    sampled=resolve_settings(settings,seed=17,randomize=True)
    assert np.allclose(sampled['microwave_pose_world'],a)
    assert not np.allclose(sampled['bowl_pose_world'],b)
    assert sampled['seed']==17 and sampled['placement_record']['valid']
    fixed=resolve_settings(settings,seed=17,randomize=False)
    assert np.allclose(fixed['bowl_pose_world'],b)
    assert np.allclose(settings['bowl_pose_world'],b)


@pytest.mark.parametrize('object_name,z', [('bowl',-1.),('microwave',.14)])
def test_microwave_rejects_unsupported_or_below_table(tmp_path,object_name,z):
    import json
    from pathlib import Path
    from benchmarks.microwave_task.scene import resolve_settings,build_initial_scene
    from gene_agent.workbench.scene_inputs import suggested_rules
    settings=json.loads(Path('data/task_1/simulation/mujoco_task1_physical.json').read_text())
    settings['placement_rules']=suggested_rules('microwave',settings)
    settings[object_name+'_pose_world'][2][3]=z
    _,_,_,record=build_initial_scene(resolve_settings(settings,randomize=False),tmp_path/'scene')
    assert not record['valid']
    assert record['initial_check']['invalid_reasons']
def test_microwave_invalid_initial_state_reports_object_and_reason(tmp_path,monkeypatch):
    import numpy as np
    import pytest
    from gene_agent.tasks.policies.microwave_task import simulation
    placement={'valid':False,'invalid_reasons':[{'object':'bowl','reason':'table_boundary'}],
               'initial_check':{'invalid_reasons':[{'object':'bowl','reason':'outside_table_support'}]}}
    monkeypatch.setattr(simulation,'environment_settings',lambda s:s)
    monkeypatch.setattr(simulation,'build_initial_scene',lambda *a:(None,None,None,placement))
    with pytest.raises(ValueError,match='bowl.*outside_table_support'):
        simulation._prepare_scene(tmp_path,{'microwave_pose_world':np.eye(4),'bowl_pose_world':np.eye(4)})

