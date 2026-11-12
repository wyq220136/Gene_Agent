"""用原生 dense controller 执行轨迹；评分/预算仍由原 evaluator 管理。"""
from __future__ import annotations
import hashlib
import json
import math
import traceback
from dataclasses import replace
import numpy as np
from scipy.spatial.transform import Rotation

from gene_agent.skills.contracts import OperationState, StageOutcome
from ..contracts import RuntimeCapabilities, operational_state


def pose_matrix(pose):
    """SAPIEN pose/七维wxyz边界显式转成 T_world_object。"""
    if hasattr(pose, 'to_transformation_matrix'):
        return np.asarray(pose.to_transformation_matrix(), dtype=float)
    values = np.asarray(pose, dtype=float)
    if values.shape != (7,) or not np.isfinite(values).all():
        raise ValueError('native pose 必须为有限 xyz+wxyz')
    result = np.eye(4)
    result[:3,:3] = Rotation.from_quat(values[[4,5,6,3]]).as_matrix()
    result[:3,3] = values[:3]
    return result


class RobotwinRuntime:
    """只暴露允许观测/执行；不公开native_env或诊断端口。"""
    capabilities = RuntimeCapabilities(diagnostics=frozenset({'contacts','native_video'}))

    def __init__(self, native_env, *, episode_id, robot_metadata, allowed_observation_fields, object_bindings=None, on_sample=None):
        self.__env = native_env
        self.__episode = str(episode_id)
        self.__robot = robot_metadata
        self.__fields = tuple(allowed_observation_fields)
        self.__objects = dict(object_bindings or {})
        self.__steps = 0
        self.__query = dict(object_names=tuple(self.__objects),robot_groups=tuple(robot_metadata['joint_groups']))
        self.__on_sample = on_sample

    def _raw(self):
        """全部登记对象和机器人来自当前状态，不读取setting或评分状态。"""
        env = self.__env
        robots = {}
        for group,names in self.__robot['joint_groups'].items():
            entity = getattr(env.robot, f'{group}_entity')
            active = {j.name:i for i,j in enumerate(entity.get_active_joints())}
            q, v = np.asarray(entity.get_qpos()), np.asarray(entity.get_qvel())
            descriptor = self.__robot['grippers'][group]
            fingers = {n:float(q[active[n]]) for n in descriptor['joint_names']}
            robots[group] = dict(joint_names=list(names),joint_positions_rad=[float(q[active[n]]) for n in names],
                joint_velocities_rad_s=[float(v[active[n]]) for n in names],
                T_world_tcp=pose_matrix(getattr(env.robot,f'get_{group}_tcp_pose')()).tolist(),
                gripper=dict(width_m=max(0.,sum(fingers.values())),joint_positions_m=fingers))
        objects = {public:dict(T_world_object=pose_matrix(getattr(env,native).get_pose()).tolist()) for public,native in self.__objects.items()}
        raw = dict(sim_time_s=self.__steps*float(env.scene.get_timestep()),scene_revision=0,world_frame=self.__robot['world_frame'],objects=objects,robots=robots)
        raw['state_id'] = hashlib.sha256(json.dumps(raw,sort_keys=True,allow_nan=False).encode()).hexdigest()
        return raw

    def observe(self, *, object_names=(), robot_groups=()):
        """读取不推进物理，任何未登记名字拒绝；返回独立公共状态。"""
        if set(object_names)-self.__objects.keys() or set(robot_groups)-self.__robot['joint_groups'].keys():
            raise ValueError('请求未知物体或机器人组')
        if len(set(object_names))!=len(object_names) or len(set(robot_groups))!=len(robot_groups):
            raise ValueError('观测名称不能重复')
        self.__query = dict(object_names=tuple(object_names),robot_groups=tuple(robot_groups))
        return self._state()

    def _state(self):
        """hash基于整个可观察世界，字段选择不改变state身份。"""
        raw = self._raw()
        raw['objects'] = {k:raw['objects'][k] for k in self.__query['object_names']}
        raw['robots'] = {k:raw['robots'][k] for k in self.__query['robot_groups']}
        return operational_state(raw,episode_id=self.__episode,source='robotwin_gt',allowed_fields=self.__fields)

    def _perform(self, stage_id, expected, command, *, target=None, group=None):
        """串接原physics hook；部分执行/终止/异常均保留实际末态。"""
        start = self._state()
        if start.state_id != expected:
            raise ValueError('expected_state_id 已过期或属于另一episode')
        env = self.__env
        local = '_after_physics_step' in vars(env)
        previous = getattr(env,'_after_physics_step',None)
        before = self.__steps
        samples = []
        recording_errors = []
        def after_step():
            self.__steps += 1
            if (self.__steps-before)==1 or (self.__steps-before)%max(1,round(.05/env.scene.get_timestep()))==0:
                try:
                    state = self._state()
                    samples.append(dict(state_id=state.state_id,**state.values))
                    if self.__on_sample:self.__on_sample(stage_id,state)
                except Exception as exc:
                    # 记录器故障不改变控制/原生评分与预算；终态仍独立读取验证。
                    if len(recording_errors)<3:recording_errors.append(f'{type(exc).__name__}: {exc}')
            if previous:previous()
        env._after_physics_step = after_step
        failure = None
        terminated = truncated = False
        error = None
        try:
            command()
        except Exception as exc:
            error = exc
            if type(exc).__name__ == 'EpisodeTimeLimit':
                truncated = True; code = 'time_limit'
            elif type(exc).__name__ == 'EpisodeFinished':
                terminated = bool(getattr(env,'eval_success',False))
                truncated = not terminated
                code = 'native_terminated' if terminated else 'action_budget'
            else:code = 'execution_exception'
            failure = dict(code=code,source='robotwin',message=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
        finally:
            if local:env._after_physics_step=previous
            else:delattr(env,'_after_physics_step')
        try:
            end = self._state()
        except Exception as exc:
            end = OperationState(f'unavailable:{start.state_id}','unavailable',{'state_unavailable':True,'error':str(exc)})
            failure = failure or dict(code='observation_unavailable',source='robotwin',message=str(exc))
        executed = self.__steps > before
        status = 'partial' if executed and failure else 'executed' if executed else 'not_executed'
        details = dict(execution_status=status,control_dt_s=float(env.scene.get_timestep()),
                       physics_steps=self.__steps-before,terminated=terminated,truncated=truncated,
                       samples=samples,recording_errors=recording_errors)
        if target is not None and not end.values.get('state_unavailable'):
            actual = self._raw()['robots'][group]['joint_positions_rad']
            err = float(np.max(np.abs(np.asarray(actual)-target)))
            details['endpoint_error_rad'] = err
            if err > .02 and failure is None:
                failure = dict(code='tracking_error',source='robotwin',message=f'关节终点误差 {err:.6f} rad > 0.02 rad')
                details['execution_status']='partial' if executed else 'not_executed'
        return StageOutcome(stage_id,failure is None,start,end,details=details,failure=failure)

    def execute_trajectory(self, trajectory, *, group, stage_id, expected_state_id):
        """按原物理dt重采样原时间轴，不经take_action重新TOPP或设置qpos。"""
        if group not in self.__robot['joint_groups'] or tuple(self.__robot['joint_groups'][group]) != trajectory.joint_names:
            raise ValueError('轨迹关节名称/顺序与活动组不一致')
        times = np.asarray(trajectory.times_s)
        if len(times)<2 or times[0] != 0. or times[-1]<=0:
            raise ValueError('原生轨迹至少两点，必须从0秒开始')
        dt = float(self.__env.scene.get_timestep())
        ticks = np.minimum(np.arange(1,math.ceil(times[-1]/dt)+1)*dt,times[-1])
        q = np.asarray(trajectory.positions_rad)
        positions = np.column_stack([np.interp(ticks,times,q[:,i]) for i in range(q.shape[1])])
        velocity = np.gradient(positions,dt,axis=0) if len(ticks)>1 else np.zeros_like(positions)
        velocity[-1] = 0.  # 段末保持位置，不遗留非零速度目标。
        seq = {f'{side}_{kind}':None for side in ('left','right') for kind in ('arm','gripper')}
        seq[f'{group}_arm'] = dict(position=positions,velocity=velocity)
        return self._perform(stage_id,expected_state_id,lambda:self.__env.take_dense_action(seq,save_freq=None),target=q[-1],group=group)

    def execute_gripper(self, *, group, width_m, duration_s, stage_id, expected_state_id):
        """米制总开口映射到native命令；闭合保留原-.01m控制预载。"""
        descriptor = self.__robot['grippers'][group]
        width = float(width_m); duration = float(duration_s)
        if not np.isfinite(width) or not descriptor['width_range_m'][0]<=width<=descriptor['width_range_m'][1] or not np.isfinite(duration) or duration<=0:
            raise ValueError('夹爪目标/时长无效')
        lo,hi = descriptor['native_range']
        target = 0. if width==0. else (width/2-lo)/(hi-lo)
        # 原set_gripper包含原生插值、限速与末端保持段；不缩短它来满足请求时长。
        native=dict(self.__env.set_gripper(set_tag=group,**{f'{group}_pos':target}))
        n=max(int(native['num_step']),math.ceil(duration/self.__env.scene.get_timestep()))
        values=np.asarray(native['result'],dtype=float)
        native['result']=np.pad(values,(0,n-len(values)),mode='edge')
        native['num_step']=n
        seq = {f'{side}_{kind}':None for side in ('left','right') for kind in ('arm','gripper')}
        seq[f'{group}_gripper'] = native
        outcome=self._perform(stage_id,expected_state_id,lambda:self.__env.take_dense_action(seq,save_freq=None))
        if outcome.success and width>outcome.start.values['robots'][group]['gripper']['width_m']:
            # close允许物体阻挡；open必须达到要求的实际净开口，才能进入pregrasp。
            actual=outcome.end.values['robots'][group]['gripper']['width_m']
            if abs(actual-width)>.004:
                outcome=replace(outcome,success=False,failure=dict(code='gripper_tracking_error',source='robotwin',
                    message=f'实际开口 {actual:.6f}m，目标 {width:.6f}m，超过0.004m容差'))
        return outcome

    def hold(self, *, duration_s, stage_id, expected_state_id):
        """按原物理dt保持现有drive target，不能把native delay的次数当秒。"""
        if not np.isfinite(duration_s) or duration_s<=0:raise ValueError('保持时间必须正且有限')
        if self._state().state_id!=expected_state_id:raise ValueError('expected_state_id已过期或属于另一episode')
        group=next(iter(self.__robot['joint_groups']))
        joints=getattr(self.__env.robot,f'{group}_arm_joints')
        target=np.asarray([float(j.get_drive_target()[0]) for j in joints])
        n=math.ceil(duration_s/self.__env.scene.get_timestep())
        seq={f'{side}_{kind}':None for side in ('left','right') for kind in ('arm','gripper')}
        seq[f'{group}_arm']=dict(position=np.tile(target,(n,1)),velocity=np.zeros((n,len(target))))
        return self._perform(stage_id,expected_state_id,lambda:self.__env.take_dense_action(seq,save_freq=None))
