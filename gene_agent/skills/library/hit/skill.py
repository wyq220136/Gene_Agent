"""对齐、直线敲击与可选撤离；失败保留实际终态并停止。"""
from dataclasses import replace
from gene_agent.skills.contracts import OperationStage
from gene_agent.skills.runtime.runner import run_operation


def run(backend, *, withdraw=False):
    """这里只报告动作执行；完整任务成功仍由原benchmark评分。"""
    stages=(OperationStage('align'),OperationStage('strike'))
    if withdraw:stages+=(OperationStage('withdraw'),)
    result=run_operation(stages,backend=backend)
    result=replace(result,skill='hit',requested_scope='hit')
    finish=getattr(backend,'finish',None)
    if finish is not None:finish(result)
    return result
