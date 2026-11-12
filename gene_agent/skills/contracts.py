"""共享检查与执行事实；不依赖 Task、设备或规划后端。"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal, Mapping

@dataclass(frozen=True)
class CheckResult:
    """`unknown` 表示证据不足，不能当作通过。"""

    status: Literal["pass", "fail", "unknown"]
    source: str
    evidence: Mapping[str, object]
    reason: str = ""

    def __post_init__(self) -> None:
        if self.status not in {"pass", "fail", "unknown"}:
            raise ValueError("检查状态只能是 pass、fail 或 unknown")
        if not self.source:
            raise ValueError("检查结果必须注明证据来源")

    @property
    def passed(self) -> bool:
        return self.status == "pass"

    def to_dict(self) -> dict:
        return {
            "status": self.status, "source": self.source,
            "evidence": dict(self.evidence), "reason": self.reason,
        }


@dataclass(frozen=True)
class OperationState:
    state_id: str
    source: str
    values: Mapping

    def __post_init__(self):
        if not self.state_id or not self.source:
            raise ValueError('状态必须有版本和来源')


@dataclass(frozen=True)
class OperationStage:
    name: str
    requires_check: bool = False
    # 纯检查与“执行动作后检查”不同；不能仅凭 requires_check 推断未运动。
    observation_only: bool = False


@dataclass(frozen=True)
class StageOutcome:
    stage: str
    success: bool
    start: OperationState
    end: OperationState
    check: CheckResult | None = None
    details: Mapping = field(default_factory=dict)
    failure: Mapping | None = None

    @property
    def execution_status(self):
        """读取生产者的执行证据；失败且缺证据时不能推断没有运动。"""
        row = self.details
        status = row.get('execution_status')
        if status in {'executed', 'partial', 'not_executed', 'unknown', 'prediction'}:
            return 'executed' if status == 'partial' else status
        if self.end.source.startswith(('prediction', 'predicted', 'preview', 'kinematic_')) or self.end.values.get('predicted'):
            return 'prediction'
        if row.get('evidence_mode') == 'planning_only':
            return 'not_executed'
        if 'executed' in row:
            return 'executed' if row['executed'] else 'not_executed'
        if 'execution' in row:
            execution = row['execution']
            if execution is None:
                return 'not_executed'
            if isinstance(execution, Mapping) and execution.get('status') in {'not_sent', 'rejected', 'not_executed'}:
                return 'not_executed'
            return 'executed'
        if row.get('observation_only'):
            return 'not_executed'
        # 兼容原 perform 契约：成功表示动作完成；失败不作此推断。
        return 'executed' if row.get('success', self.success) else 'unknown'


@dataclass(frozen=True)
class OperationResult:
    success: bool
    executed: bool
    stages: tuple[StageOutcome, ...]
    end_state: OperationState
    failure_stage: str | None = None
    failure_code: str | None = None
    failure: Mapping | None = None
    skill: str | None = None
    requested_scope: str | None = None
    execution_status: str | None = None

    def to_dict(self):
        def ref(state):
            return {'state_id': state.state_id, 'source': state.source}
        return {'success': self.success, 'executed': self.executed,
            'execution_status': self.execution_status,
            'skill': self.skill, 'requested_scope': self.requested_scope,
            'failure_stage': self.failure_stage, 'failure_code': self.failure_code,
            'failure': None if self.failure is None else dict(self.failure),
            'end_state': {**ref(self.end_state), 'values': dict(self.end_state.values)},
            'stages': [s.stage for s in self.stages],
            'stage_outcomes': [{'stage': s.stage, 'success': s.success,
                'start': ref(s.start), 'end': ref(s.end),
                'execution_status': s.execution_status,
                'check': None if s.check is None else s.check.to_dict(),
                'failure': None if s.failure is None else dict(s.failure)} for s in self.stages]}

