"""原生执行必须沿原控制路径，维持状态、时间和失败终态。"""
from types import SimpleNamespace
import numpy as np
import pytest
from gene_agent.motion import JointTrajectory


class Pose:
    def __init__(self, p=(0,0,0)):
        self.p=np.array(p,dtype=float);self.q=np.array([1.,0,0,0])
    def to_transformation_matrix(self):
        result=np.eye(4);result[:3,3]=self.p;return result


class Entity:
    def __init__(self):
        self.q=np.array([0.,0.,.01,.01]);self.v=np.zeros(4)
    def get_qpos(self): return self.q.copy()
    def get_qvel(self): return self.v.copy()
    def get_active_joints(self): return [SimpleNamespace(name=n) for n in ['a','b','g7','g8']]


class Native:
    def __init__(self):
        entity=Entity()
        self.robot=SimpleNamespace(left_entity=entity,get_left_tcp_pose=lambda:[.1,0,0,1,0,0,0])
        self.scene=SimpleNamespace(get_timestep=lambda:.01)
        self.tool=SimpleNamespace(get_pose=lambda:Pose([.2,0,.8]))
        self.seed=999;self.nail_state={'depth':99};self.steps=0;self.stop_at=None
    def set_gripper(self, *, set_tag, left_pos):
        current=(self.robot.left_entity.q[2]+.01)/.055
        return dict(num_step=2,result=np.array([current,left_pos]),per_step=(left_pos-current)/2)
    def take_dense_action(self, seq, save_freq=None):
        n=max(len(seq['left_arm']['position']) if seq.get('left_arm') is not None else 0,
              seq['left_gripper']['num_step'] if seq.get('left_gripper') is not None else 0)
        for i in range(n):
            if seq.get('left_arm') is not None:self.robot.left_entity.q[:2]=seq['left_arm']['position'][i]
            if seq.get('left_gripper') is not None:self.robot.left_entity.q[2:]=.055*seq['left_gripper']['result'][i]-.01
            self.steps+=1
            callback=getattr(self,'_after_physics_step',None)
            if callback:callback()
            if self.stop_at==self.steps:raise RuntimeError('native control failure')
        return True


def metadata():
    return dict(joint_groups={'left':['a','b']},grippers={'left':dict(joint_names=['g7','g8'],base_joint='g7',mimic=[['g8',1.,0.]],native_range=[-.01,.045],width_range_m=[0.,.09])},world_frame='world')


def runtime(env, episode='episode'):
    from gene_agent.simulation.robotwin.runtime import RobotwinRuntime
    return RobotwinRuntime(env,episode_id=episode,robot_metadata=metadata(),allowed_observation_fields=('objects','robots'),object_bindings={'tool':'tool'})


def test_readonly_actual_observation_and_episode_staleness():
    env=Native(); r=runtime(env)
    state=r.observe(object_names=('tool',),robot_groups=('left',))
    assert env.steps==0 and state.values['objects']['tool']['T_world_object'][0][3]==.2
    assert state.values['robots']['left']['joint_positions_rad']==[0.,0.]
    assert not {'seed','nail_state','native_env','diagnostics'} & state.values.keys()
    with pytest.raises(ValueError,match='state'):
        runtime(env,'other').hold(duration_s=.1,stage_id='hold',expected_state_id=state.state_id)
    assert env.steps==0


def test_timed_trajectory_tracks_actual_state_and_preserves_hook():
    env=Native();seen=[];env._after_physics_step=lambda:seen.append(env.steps)
    r=runtime(env);start=r.observe(object_names=('tool',),robot_groups=('left',))
    result=r.execute_trajectory(JointTrajectory(('a','b'),(0.,.03),((0.,0.),(.03,.06))),group='left',stage_id='move',expected_state_id=start.state_id)
    assert result.success and env.steps==3 and len(seen)==3
    assert result.end.values['robots']['left']['joint_positions_rad']==pytest.approx([.03,.06])
    assert result.details['control_dt_s']==.01
    assert result.end.values['sim_time_s']==pytest.approx(.03)
    env._after_physics_step();assert len(seen)==4
    with pytest.raises(ValueError):r.execute_trajectory(JointTrajectory(('b','a'),(0.,.1),((0.,0.),(.1,.1))),group='left',stage_id='wrong',expected_state_id=result.end.state_id)


def test_partial_exception_keeps_actual_end_and_does_not_claim_arrival():
    env=Native();env.stop_at=2;r=runtime(env)
    start=r.observe(object_names=('tool',),robot_groups=('left',))
    result=r.execute_trajectory(JointTrajectory(('a','b'),(0.,.04),((0.,0.),(.04,.08))),group='left',stage_id='move',expected_state_id=start.state_id)
    assert not result.success and result.details['execution_status']=='partial'
    assert result.end.values['robots']['left']['joint_positions_rad']==pytest.approx([.02,.04])
    assert result.failure['code']=='execution_exception'


def test_zero_width_uses_original_native_close_preload():
    env=Native();r=runtime(env);start=r.observe(object_names=(),robot_groups=('left',))
    result=r.execute_gripper(group='left',width_m=0.,duration_s=.02,stage_id='close',expected_state_id=start.state_id)
    assert result.success
    # 此替身无硬限位，仅验证原控制命令的闭合预载仍为 -.01。
    assert env.robot.left_entity.q[2]==pytest.approx(-.01)


def test_recording_failure_does_not_change_control_or_native_hook():
    from gene_agent.simulation.robotwin.runtime import RobotwinRuntime
    env=Native();seen=[];env._after_physics_step=lambda:seen.append(env.steps)
    def broken(*args):raise OSError('disk full')
    r=RobotwinRuntime(env,episode_id='one',robot_metadata=metadata(),allowed_observation_fields=('objects','robots'),on_sample=broken)
    start=r.observe(object_names=(),robot_groups=('left',))
    result=r.execute_trajectory(JointTrajectory(('a','b'),(0.,.03),((0.,0.),(.03,.06))),group='left',stage_id='move',expected_state_id=start.state_id)
    assert result.success and env.steps==3 and len(seen)==3
    assert 'disk full' in result.details['recording_errors'][0]


def test_hold_uses_physics_dt_and_preserves_native_drive_target():
    env=Native()
    env.robot.left_arm_joints=[SimpleNamespace(get_drive_target=lambda:[.2]),SimpleNamespace(get_drive_target=lambda:[.3])]
    def wrong_delay(*args,**kwargs):raise AssertionError('native delay is not seconds')
    env.delay=wrong_delay
    r=runtime(env);start=r.observe(object_names=(),robot_groups=('left',))
    outcome=r.hold(duration_s=.03,stage_id='hold',expected_state_id=start.state_id)
    assert outcome.success and env.steps==3
    assert env.robot.left_entity.q[:2]==pytest.approx([.2,.3])
    assert outcome.end.values['sim_time_s']==pytest.approx(.03)


def test_open_gripper_requires_actual_aperture():
    env=Native();original=env.take_dense_action
    def stuck(seq,**kwargs):
        count=seq['left_gripper']['num_step']
        for i in range(count):
            env.steps+=1
            if getattr(env,'_after_physics_step',None):env._after_physics_step()
    env.take_dense_action=stuck
    r=runtime(env);start=r.observe(object_names=(),robot_groups=('left',))
    result=r.execute_gripper(group='left',width_m=.08,duration_s=.03,stage_id='open',expected_state_id=start.state_id)
    assert not result.success and result.failure['code']=='gripper_tracking_error'
    assert result.end.values['robots']['left']['gripper']['width_m']==pytest.approx(.02)


def test_nonzero_close_allows_contact_to_stop_before_command():
    env=Native()
    def blocked(seq,**kw):
        env.steps+=1
        if getattr(env,'_after_physics_step',None):env._after_physics_step()
    env.take_dense_action=blocked
    r=runtime(env);start=r.observe(object_names=(),robot_groups=('left',))
    result=r.execute_gripper(group='left',width_m=.01,duration_s=.03,stage_id='close',expected_state_id=start.state_id)
    assert result.success and result.end.values['robots']['left']['gripper']['width_m']==pytest.approx(.02)


def test_gripper_keeps_original_native_sequence_and_only_extends_hold():
    env=Native();calls=[];seen=[]
    def source_plan(**kwargs):
        calls.append(kwargs)
        return dict(num_step=3,result=np.array([.3,.6,.6]),per_step=.15)
    env.set_gripper=source_plan
    original=env.take_dense_action
    def execute(seq,**kwargs):
        seen.append(seq['left_gripper']);return original(seq,**kwargs)
    env.take_dense_action=execute
    r=runtime(env);start=r.observe(object_names=(),robot_groups=('left',))
    r.execute_gripper(group='left',width_m=.046,duration_s=.05,stage_id='open',expected_state_id=start.state_id)
    assert calls==[dict(set_tag='left',left_pos=pytest.approx(.6))]
    assert seen[0]['result']==pytest.approx([.3,.6,.6,.6,.6])
    assert seen[0]['per_step']==.15 and env.steps==5


def test_failure_image_capture_uses_render_only(tmp_path):
    from types import SimpleNamespace
    from gene_agent.simulation.robotwin.diagnostics import RobotwinDiagnostics
    import numpy as np
    calls=[]
    camera=SimpleNamespace(name='head_camera',width=2,height=2,get_intrinsic_matrix=lambda:np.eye(3),get_extrinsic_matrix=lambda:np.eye(4))
    env=SimpleNamespace(_update_render=lambda:calls.append('render'),cameras=SimpleNamespace(update_picture=lambda:calls.append('picture'),get_rgb=lambda:{'head_camera':{'rgb':np.zeros((2,2,3),dtype=np.uint8)}},static_camera_list=[camera]))
    row=RobotwinDiagnostics(env).capture_failure_image(tmp_path/'actual.png',stage_id='align',state_id='s')
    assert row['mode']=='actual' and row['state_id']=='s'
    assert calls==['render','picture']
    assert (tmp_path/'actual.png').is_file()
