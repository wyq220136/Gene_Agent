"""执行后端协议与回调适配；阶段和结果定义见 contracts。"""
from typing import Protocol
from gene_agent.skills.contracts import OperationState, OperationStage, StageOutcome

class OperationBackend(Protocol):
    evidence_mode: str
    def observe(self) -> OperationState: ...
    def execute(self, stage: OperationStage, start: OperationState) -> StageOutcome: ...


class CallbackOperationBackend:
    """薄适配已有执行器；回调必须完成动作并提供终态，不负责阶段排序。"""
    def __init__(self, *, observe, perform, evidence_mode, execution_enabled=True,
                 failure_details=None):
        self._observe = observe
        self._perform = perform
        self.evidence_mode = evidence_mode
        self.execution_enabled = execution_enabled
        self._failure_details = failure_details

    def observe(self):
        return self._observe()

    def execute(self, stage, start):
        row = self._perform(stage.name)
        failure = row.get('failure')
        success = bool(row['success'])
        if not success and failure is None and self._failure_details is not None:
            # 只读取原适配报告，不重跑动作或验证；错误归因仍由生产者负责。
            failure = self._failure_details(stage.name)
        try:
            end = self.observe()
        except Exception as error:
            # 命令可能已部分执行，丢失观测时不能把发令前状态当作终态。
            end = OperationState(f'unavailable:{stage.name}:{start.state_id}', 'unavailable', {
                'state_unavailable': True, 'last_known_state_id': start.state_id,
                'observation_error': f'{type(error).__name__}: {error}',
            })
            failure = failure or {'code': 'observation_unavailable', 'source': self.evidence_mode,
                                  'message': str(error)}
            success = False
        return StageOutcome(stage.name, success, start, end,
                            check=row.get('check'), details=row, failure=failure)
