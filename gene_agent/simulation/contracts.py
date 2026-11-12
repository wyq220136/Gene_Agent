"""原生运行/规划场景/开发诊断的最小同步边界，不加载物理引擎。"""
from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass
import hashlib
from typing import Protocol

from gene_agent.skills.contracts import OperationState, StageOutcome
from gene_agent.motion.model import JointTrajectory, CollisionPolicy


@dataclass(frozen=True)
class RuntimeCapabilities:
    """显式声明可用能力；缺失接触/恢复不能解释成已验证无异常。"""
    observations: frozenset[str] = frozenset({'objects', 'robots'})
    controls: frozenset[str] = frozenset({'joint_trajectory', 'gripper_width', 'hold'})
    diagnostics: frozenset[str] = frozenset()
    checkpoint: bool = False


class RuntimePort(Protocol):
    """policy只能消费操作观测与动作，不拥有reset/评分/开发诊断。"""
    capabilities: RuntimeCapabilities
    def observe(self, *, object_names: tuple[str, ...], robot_groups: tuple[str, ...]) -> OperationState: ...
    def execute_trajectory(self, trajectory: JointTrajectory, *, group: str, stage_id: str, expected_state_id: str) -> StageOutcome: ...
    def execute_gripper(self, *, group: str, width_m: float, duration_s: float, stage_id: str, expected_state_id: str) -> StageOutcome: ...
    def hold(self, *, duration_s: float, stage_id: str, expected_state_id: str) -> StageOutcome: ...


class PlanningSceneProvider(Protocol):
    """独立提供同一实际state的碰撞快照，不假定model/data。"""
    def snapshot(self, *, state: OperationState, group: str, collision_policy: CollisionPolicy) -> dict: ...


class DiagnosticsPort(Protocol):
    """开发记录器持有；不挂到运行时policy输入上。"""
    def capture(self, *, stage_id: str, state_id: str) -> dict: ...


def operational_state(raw: dict, *, episode_id: str, source: str, allowed_fields: tuple[str, ...]) -> OperationState:
    """建立episode作用域的只读观测副本，仅复制明确允许的公共字段。"""
    if not episode_id or not source or not raw.get('state_id'):
        raise ValueError('状态需要episode、source及native state_id')
    if set(allowed_fields) - {'objects', 'robots'}:
        raise ValueError('当前公开观测仅支持 objects/robots')
    key = hashlib.sha256(f'{episode_id}:{raw["state_id"]}'.encode()).hexdigest()
    values = {k:deepcopy(raw[k]) for k in ('scene_revision','sim_time_s','world_frame') if k in raw}
    values.update({k:deepcopy(raw.get(k, {})) for k in allowed_fields})
    values.update(episode_id=episode_id, observation_provenance=source)
    return OperationState(key, source, values)
