"""插入和桌面放置的同步共享编排；证据与动作实现由环境适配器提供。"""
from dataclasses import replace
from gene_agent.skills.contracts import OperationStage
from gene_agent.skills.runtime.runner import run_operation


def run_transfer_flow(backend, *, operation='insert', verify_hold=True,
                      approve_path=True, verify_release=True, verify_final=True, stop_after=None):
    """先移动携物目标，再松爪、detach、落稳、检查和撤离。

    真机的路径人工审批与 MuJoCo 规划器的碰撞检查来源不同，后端显式
    选择审批阶段。低层执行入口可以只报告动作前缀，不宣称任务后置成功；
    完整 Skill 必须启用 final_check。detach 始终在 settle 之前，确保仿真
    weld 在落稳时已解除。所有启用的检查都交由公共 runner 拒绝 unknown。
    """
    if operation not in {'insert', 'place'}:
        raise ValueError('搬运操作必须是 insert 或 place')
    allowed = {None, 'complete', 'preplace', operation, 'release'}
    if stop_after not in allowed:
        raise ValueError('未知搬运停止范围')
    if stop_after == 'release' and not verify_release:
        raise ValueError('释放调试必须启用释放检查')
    names = ['preplace']
    if verify_hold:
        names.append('hold_check')
    if approve_path:
        names.append('transfer_path_check')
    names.extend([operation, 'release', 'detach', 'settle'])
    if verify_release:
        names.append('release_check')
    names.extend(['retreat_1', 'retreat_2'] if operation == 'insert' else ['retreat'])
    if verify_final:
        names.append('final_check')
    if stop_after not in (None, 'complete'):
        last = 'release_check' if stop_after == 'release' else stop_after
        names = names[:names.index(last) + 1]
    result = run_operation([OperationStage(name, name.endswith('_check'),
                          observation_only=name.endswith('_check')) for name in names],
                         backend=backend)

    return replace(result, skill=operation, requested_scope=(stop_after or ('complete' if verify_final else 'motion_prefix')))


def insert(backend, *, stop_after=None):
    """持物放入、释放与撤离；完整模式要求全部后置检查。"""
    return run_transfer_flow(backend, operation='insert', stop_after=stop_after)


def place(backend, *, stop_after=None):
    """持物放置、释放与撤离；低层仿真特例只由适配器配置。"""
    return run_transfer_flow(backend, operation='place', stop_after=stop_after)


def run_extract_flow(backend, *, stop_after='extract', resume_after_grasp=False):
    """独立抓取、完整取出和从已抓取状态继续使用相同的检查决策。"""
    if stop_after not in {'grasp', 'extract'}:
        raise ValueError('未知取出停止范围')
    if resume_after_grasp and stop_after != 'extract':
        raise ValueError('抓取后只能继续携物取出')
    names = ['open', 'pre_extract_1', 'pre_extract_2', 'grasp', 'close', 'grasp_check']
    if resume_after_grasp:
        names = ['grasp_check']
    if stop_after == 'extract':
        names += ['attach', 'post_extract_1_hold_check', 'post_extract_1',
                  'post_extract_2_hold_check', 'post_extract_2', 'extract_check']
    result = run_operation([OperationStage(n, n.endswith('_check'),
                          observation_only=n.endswith('_check')) for n in names],
                         backend=backend)

    return replace(result, skill='extract', requested_scope=stop_after)


def extract(backend, *, stop_after='extract'):
    return run_extract_flow(backend, stop_after=stop_after)


def resume_extract(backend):
    return run_extract_flow(backend, resume_after_grasp=True)
