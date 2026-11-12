"""将已有 MuJoCo 执行器映射到公共运行契约；物理实现不复制。"""
from __future__ import annotations
from gene_agent.skills.contracts import OperationState, StageOutcome
from .contracts import RuntimeCapabilities, operational_state


class MujocoRuntimeAdapter:
    """保留原实际状态与执行检查，policy不持有原生runtime。"""
    capabilities = RuntimeCapabilities()

    def __init__(self, runtime, *, episode_id, allowed_observation_fields):
        self.__runtime = runtime
        self.__episode = episode_id
        self.__fields = tuple(allowed_observation_fields)
        self.__query = dict(object_names=(), robot_groups=tuple(runtime.robot_metadata['joint_groups']))

    def observe(self, *, object_names=(), robot_groups=()):
        """冻结实际观测；不推进模拟，也不泄露weld/诊断。"""
        self.__query = dict(object_names=tuple(object_names), robot_groups=tuple(robot_groups))
        return self._state(self.__runtime.query(**self.__query))

    def _state(self, raw):
        """为旧native状态加入独立episode身份。"""
        return operational_state(raw,episode_id=self.__episode,source='mujoco_gt',allowed_fields=self.__fields)

    def _perform(self, stage_id, expected, command, **kwargs):
        """发令前核对state；异常后读取实际终态，观测失败不能填预测。"""
        raw = self.__runtime.query(**self.__query)
        start = self._state(raw)
        if start.state_id != expected:
            raise ValueError('expected_state_id 已过期或属于另一episode')
        try:
            row = command(planned_state_id=raw['state_id'], **kwargs)
        except Exception as error:
            row = dict(success=False,execution_status='unknown',failure_type='execution_exception',error=f'{type(error).__name__}: {error}')
        try:
            end = self._state(self.__runtime.query(**self.__query))
        except Exception as error:
            end = OperationState(f'unavailable:{start.state_id}', 'unavailable', {'state_unavailable': True,'error':str(error)})
            row = {**row,'success':False,'observation_error':str(error)}
        success = bool(row.get('success'))
        return StageOutcome(stage_id,success,start,end,details=row,
            failure=None if success else {'code':row.get('failure_type','execution_failed'),'source':'mujoco','message':row.get('error','')})

    def execute_trajectory(self, trajectory, *, group, stage_id, expected_state_id):
        """保留原具名弧度轨迹及时间轴。"""
        return self._perform(stage_id,expected_state_id,self.__runtime.execute_joint_trajectory,
            group=group,joint_names=trajectory.joint_names,times_s=trajectory.times_s,
            positions_rad=trajectory.positions_rad,endpoint_tolerance_rad=.02,settle_timeout_s=.5)

    def execute_gripper(self, *, group, width_m, duration_s, stage_id, expected_state_id):
        """米制命令继续交给旧夹爪映射校验。"""
        return self._perform(stage_id,expected_state_id,self.__runtime.execute_gripper_command,
                             group=group,width_m=width_m,duration_s=duration_s)

    def hold(self, *, duration_s, stage_id, expected_state_id):
        """保持真实控制目标，不修改qpos。"""
        return self._perform(stage_id,expected_state_id,self.__runtime.execute_hold,duration_s=duration_s)
