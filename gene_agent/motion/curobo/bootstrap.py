"""独立命令启动时减少长期导入对象的重复GC扫描；不改变图捕获的回收/同步。"""
from contextlib import contextmanager
import gc
from time import perf_counter


def _load_runtime():
    # 在任何solver/图创建之前载入常用库，避免每次捕图扫描这些长期对象。
    from curobo.motion_planner import MotionPlanner, MotionPlannerCfg  # noqa: F401
    from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg  # noqa: F401
    import trimesh  # noqa: F401
    import torch
    return torch


@contextmanager
def curobo_import_gc_scope(*, enabled=True):
    """仅用于独立同步命令的启动边界，不在solver或共享服务内部调用。

    已有CUDA会话或冻结集合时跳过；新建的任务对象仍参与正常GC，cuRobo
    与PyTorch的显式collect/synchronize保持原样。退出时（含异常）恢复。
    """
    started = perf_counter()
    report = {'requested': bool(enabled), 'applied': False, 'frozen_objects': 0,
              'reason': 'disabled', 'setup_s': 0., 'cleanup_s': 0.}
    if not enabled:
        yield report
        return
    if gc.get_freeze_count():
        report['reason'] = 'existing_frozen_objects'
        yield report
        return
    torch = _load_runtime()
    if torch.cuda.is_initialized():
        report.update(reason='cuda_already_initialized', setup_s=perf_counter()-started)
        yield report
        return
    gc.collect()
    gc.freeze()
    report.update(applied=True, reason='frozen_imports', frozen_objects=gc.get_freeze_count(),
                  setup_s=perf_counter()-started)
    try:
        yield report
    finally:
        cleanup_started = perf_counter()
        gc.unfreeze()
        gc.collect()
        report['cleanup_s'] = perf_counter()-cleanup_started
