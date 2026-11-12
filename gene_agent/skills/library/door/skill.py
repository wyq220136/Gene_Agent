"""门阶段的同步推进；实际角度与接触检查由后端提供。"""
from dataclasses import replace
from gene_agent.skills.runtime.runner import run_operation


def run(backend, *, stages, stop_after=None):
    """阶段可以延迟生成，以消费拉门后重新观测的门角。"""
    if stop_after is not None:
        # 独立调试要求明确的阶段序列，非法名称必须在发令前拒绝。
        if not isinstance(stages, (list,tuple)):
            raise ValueError('阶段停止需提供明确的门阶段序列')
        names = [stage.name for stage in stages]
        if stop_after not in names:
            raise ValueError('未知门停止阶段')
        stages = stages[:names.index(stop_after)+1]
    result = run_operation(stages, backend=backend)
    return replace(result, skill='door', requested_scope=stop_after or 'complete')
