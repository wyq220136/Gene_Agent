"""同步公共阶段循环：失败保留实际终态，未知检查不能伪装成通过。"""
from gene_agent.skills.contracts import OperationResult
from dataclasses import replace


def _predicted(source, values):
    return (source.startswith(('prediction', 'predicted', 'preview', 'kinematic_'))
            or bool(values.get('predicted')))


def run_operation(stages, *, backend, initial_state=None):
    current = initial_state if initial_state is not None else backend.observe()
    predicted = backend.evidence_mode == 'prediction'
    enabled = not predicted and getattr(backend, 'execution_enabled', True)
    executed = False
    unknown = False
    rows = []
    def result(success, stage=None, code=None, failure=None):
        status = ('prediction' if predicted else
                  'partial' if executed and not success else
                  'executed' if executed else 'unknown' if unknown else 'not_executed')
        return OperationResult(success, executed, tuple(rows), current, stage, code,
                               failure, execution_status=status)
    if not predicted and _predicted(current.source, current.values):
        return result(False, 'initial_state', 'predicted_start_in_execution')
    for stage in stages:
        outcome = backend.execute(stage, current)
        if stage.observation_only:
            outcome = replace(outcome, details={**outcome.details, 'observation_only': True})
        if not predicted and not enabled and not any(
                key in outcome.details for key in ('executed', 'execution', 'execution_status')):
            # 兼容声明只做预检的旧后端；明确返回的执行证据仍优先于开关。
            outcome = replace(outcome, details={**outcome.details, 'executed': False})
        rows.append(outcome)
        if not predicted:
            executed |= outcome.execution_status == 'executed'
            unknown |= outcome.execution_status == 'unknown'
        code = None
        if outcome.stage != stage.name or outcome.start.state_id != current.state_id:
            code = 'state_discontinuity'
        elif not predicted and _predicted(outcome.end.source, outcome.end.values):
            code = 'predicted_end_in_execution'
        elif not outcome.success:
            code = (outcome.failure or {}).get('code') or 'stage_failed'
        elif stage.requires_check:
            check = outcome.check
            if check is not None and not predicted and _predicted(check.source, check.evidence):
                code = 'predicted_check_in_execution'
            elif check is None or check.status == 'fail' or (check.status == 'unknown' and not predicted):
                code = 'postcondition_unknown' if check is None or check.status == 'unknown' else 'postcondition_failed'
        if code is None and not predicted and not enabled:
            code = 'execution_disabled'
        current = outcome.end
        if code:
            return result(False, stage.name, code, outcome.failure if not outcome.success else None)
    return result(True)
