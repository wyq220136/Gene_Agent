"""运行状态必须隔离episode/诊断，并保持原仿真只读查询语义。"""
import pytest
from gene_agent.skills.contracts import OperationState


def test_observation_ids_are_episode_scoped_and_privileged_fields_filtered():
    from gene_agent.simulation.contracts import operational_state
    raw=dict(state_id='same',sim_time_s=0.,world_frame='world',scene_revision=0,
             objects={'tool':{'T_world_object':[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]}},
             robots={},nail_state={'depth':1},seed=123,simulation_attachments=[{'weld':True}])
    first=operational_state(raw,episode_id='one',source='fixture',allowed_fields=('objects','robots'))
    second=operational_state(raw,episode_id='two',source='fixture',allowed_fields=('objects','robots'))
    assert first.state_id != second.state_id
    assert first.values['objects']['tool']['T_world_object'][0][0]==1
    assert not {'seed','nail_state','simulation_attachments'} & first.values.keys()
    raw['objects']['tool']['T_world_object'][0][0]=9
    assert first.values['objects']['tool']['T_world_object'][0][0]==1
    with pytest.raises(ValueError): operational_state(raw,episode_id='one',source='fixture',allowed_fields=('seed',))


def test_mujoco_adapter_observe_does_not_advance_and_rejects_old_episode():
    from gene_agent.simulation.mujoco_adapter import MujocoRuntimeAdapter
    import mujoco
    from gene_agent.simulation.runtime import MujocoRuntime
    model=mujoco.MjModel.from_xml_string('<mujoco><worldbody><body name="bowl"><joint name="joint_a"/><geom size=".1"/><site name="tcp"/></body></worldbody></mujoco>')
    data=mujoco.MjData(model)
    runtime=MujocoRuntime(model,data,{'robot':{'joint_groups':{'right':['joint_a']},'tcp_sites':{'right':'tcp'},'grippers':{}},'world_frame':'world'})
    first=MujocoRuntimeAdapter(runtime,episode_id='one',allowed_observation_fields=('objects','robots'))
    obs=first.observe(object_names=('bowl',),robot_groups=('right',))
    assert data.time==0.
    assert obs.values['robots']['right']['joint_names']==['joint_a']
    assert 'simulation_attachments' not in obs.values
    second=MujocoRuntimeAdapter(runtime,episode_id='two',allowed_observation_fields=('objects','robots'))
    with pytest.raises(ValueError,match='state'):
        second.hold(duration_s=.01,stage_id='hold',expected_state_id=obs.state_id)
    assert data.time==0.
