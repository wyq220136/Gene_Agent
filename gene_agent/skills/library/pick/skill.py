"""通用同步 Pick 阶段；物体几何、物理证据和设备命令由后端提供。"""
from gene_agent.skills.contracts import OperationStage
from gene_agent.skills.runtime.runner import run_operation
from dataclasses import asdict, replace
from .model import PickRequest


def _bound(backend, request):
    # 在观测/动作之前校验，不能让调用参数与实际设备绑定各用一份配置。
    bound = getattr(backend, 'request', None)
    if not isinstance(request, PickRequest) or not isinstance(bound, PickRequest):
        raise ValueError('Pick backend 必须显式绑定 PickRequest')
    if asdict(bound) != asdict(request):
        raise ValueError('Pick 请求与 backend 绑定的对象、机器人或参数不一致')


def run(backend, request: PickRequest, *, stop_after='pick'):
    """执行所选前缀；成功只针对 requested_scope，不代替 Task 评分。"""
    _bound(backend, request)
    result = run_pick_flow(backend, stop_after=stop_after, lift=request.lift)
    result = replace(result, skill='pick', requested_scope=stop_after)
    finish = getattr(backend, 'finish', None)
    if finish is not None:
        finish(result)
    return result


def resume(backend, request: PickRequest):
    """从有效已抓取状态继续；先复核夹持证据，绝不重新 open/close。"""
    _bound(backend, request)
    result = run_pick_flow(backend, lift=request.lift, resume_after_grasp=True)
    result = replace(result, skill='pick', requested_scope='resume_after_grasp')
    finish = getattr(backend, 'finish', None)
    if finish is not None:
        finish(result)
    return result


def run_pick_flow(backend, *, stop_after='pick', lift=True, verify_hold=True,
                  resume_after_grasp=False):
    """独立 grasp 与完整 Pick 使用同一阶段定义。

    接续已完成 grasp 时仍先检查夹持证据，不能绕过确认直接建立 attachment。
    后端不得把仅预检的成功当执行成功；未执行时应停在当前验证前缀。
    """
    if stop_after not in {'pregrasp', 'approach', 'grasp', 'pick'}:
        raise ValueError('未知抓取停止范围')
    if resume_after_grasp and stop_after != 'pick':
        raise ValueError('已抓取后的接续只能请求 Pick 后半段')
    names = ['open', 'pregrasp', 'approach', 'close', 'grasp_check', 'establish_hold']
    if lift:
        names.append('lift')
        if verify_hold:
            names.append('hold_check')
    if stop_after in {'pregrasp', 'approach', 'grasp'}:
        last = 'grasp_check' if stop_after == 'grasp' else stop_after
        names = names[:names.index(last) + 1]
    if resume_after_grasp:
        names = names[names.index('grasp_check'):]
    return run_operation([OperationStage(n, n in {'grasp_check', 'hold_check'},
                         observation_only=n in {'grasp_check', 'hold_check'}) for n in names],
                         backend=backend)
