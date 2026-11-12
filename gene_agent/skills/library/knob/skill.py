"""旋钮共享执行流程；后端提供接触模式、具体动作和实际检查证据。"""
from dataclasses import replace
from gene_agent.skills.contracts import OperationStage
from gene_agent.skills.runtime.runner import run_operation


def run_knob_flow(backend, *, target_names, grasp_mode="physical", stop_after=None):
    """开爪后读取所选目标，夹持检查通过才能旋转，撤离后检查终态。

    target_names 可延迟选择候选，以便实机先获得开爪后的新观测。
    旋转的关节命令、TCP 轨迹或显式仿真运动学更新均由后端承担。
    """
    if stop_after not in {None, 'complete', 'open', 'pregrasp', 'approach', 'grasp', 'rotate'}:
        raise ValueError('未知旋钮停止范围')
    if grasp_mode not in {"physical", "declared_kinematic"}:
        raise ValueError("未知旋钮接触模式")
    if grasp_mode == "declared_kinematic" and backend.evidence_mode != "mujoco_mixed_knob":
        raise ValueError("运动学旋钮许可仅适用于显式 MuJoCo 混合仿真后端")

    def stages():
        yield OperationStage('open')
        if stop_after == 'open':
            return
        names = list(target_names())
        if names[:4] != ['open', 'pregrasp', 'approach', 'close']:
            raise ValueError('旋钮前缀必须为 open→pregrasp→approach→close')
        remainder = names[4:]
        if remainder[:1] == ['grasp_settle']:
            remainder = remainder[1:]
        rotation_count = 0
        while remainder and remainder[0].startswith('rotate_'):
            rotation_count += 1
            remainder = remainder[1:]
        if (not rotation_count or remainder not in (
                ['release', 'retreat'], ['release', 'release_settle', 'retreat'])
                or len(set(names)) != len(names)):
            raise ValueError('旋钮必须先旋转，再释放和撤离；阶段名称不得重复')
        for name in names[1:]:
            if name == 'release' and stop_after == 'rotate':
                return
            yield OperationStage(name)
            if name in {'pregrasp', 'approach'} and name == stop_after:
                return
            if name == 'close':
                yield OperationStage('grasp_check', grasp_mode == 'physical', observation_only=True)
                if grasp_mode == 'declared_kinematic':
                    yield OperationStage('kinematic_eligibility_check', True, observation_only=True)
                if stop_after == 'grasp':
                    return
        yield OperationStage('rotation_check', True, observation_only=True)
    result = run_operation(stages(), backend=backend)
    return replace(result, skill='knob', requested_scope=stop_after or 'complete')


def run(backend, *, target_names, stop_after=None):
    """实际旋转；完整模式必须通过夹持与最终旋转检查。"""
    return run_knob_flow(backend, target_names=target_names, stop_after=stop_after)
