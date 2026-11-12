"""微波炉操作的语义分组；只选择既有步骤，不管理后端或改变动作。"""
from __future__ import annotations
from dataclasses import dataclass

# 同一操作在完整 Task 中可出现多次；这里选择第一次调用，卸载的前缀
# 自然包含第二次开门。原始 step_id 保留，避免两次开门的记录互相覆盖。
SUBTASK_STEPS = {
    'open_door': ('open_door_first',),
    'close_door': ('before_close_first_transition', 'close_door_first'),
    'rotate_knob': ('rotate_knob',),
    'pick_bowl': ('pick_bowl',),
    'insert_bowl': ('insert_bowl',),
    'extract_bowl': ('extract_bowl',),
    'place_bowl_on_table': ('place_bowl_on_table',),
    'load_bowl': ('pick_bowl', 'insert_bowl'),
    'unload_bowl': ('extract_bowl', 'place_bowl_on_table'),
}
SUBTASK_NAMES = tuple(SUBTASK_STEPS)


@dataclass(frozen=True)
class SubtaskSelection:
    name: str
    prefix_steps: tuple[str, ...]
    target_steps: tuple[str, ...]

    def steps(self, preparation: str) -> tuple[str, ...]:
        if preparation not in {'prefix', 'current'}:
            raise ValueError('准备模式必须是 prefix 或 current')
        return ((self.prefix_steps if preparation == 'prefix' else ()) + self.target_steps)


def resolve_subtask(name: str = 'all', *, policy=None) -> SubtaskSelection:
    from .policy import MicrowaveTaskPolicy, microwave_task_steps
    order = tuple(s.name for s in microwave_task_steps(policy or MicrowaveTaskPolicy()))
    if name == 'all':
        return SubtaskSelection(name, (), order)
    if name not in SUBTASK_STEPS:
        raise ValueError(f'未知微波炉子任务：{name}')
    target = SUBTASK_STEPS[name]
    if any(step not in order for step in target):
        raise ValueError('当前任务配置禁用了所选旋钮操作')
    first = order.index(target[0])
    if order[first:first + len(target)] != target:
        raise ValueError('子任务分组与当前 Task 次序不一致')
    return SubtaskSelection(name, order[:first], target)

# 实际回放复用既有 loader 的筛选；视频默认只导出目标操作，不把准备前缀混入。
REPLAY_SKILLS = {
    'open_door': 'open_door', 'close_door': 'close_door',
    'rotate_knob': 'rotate_knob_0', 'pick_bowl': 'pick', 'insert_bowl': 'insert',
    'extract_bowl': 'extract', 'place_bowl_on_table': 'place',
    'load_bowl': 'object_insert', 'unload_bowl': 'object_extract',
}


# 调试停止仅用于单个操作；组合 Task 不可在中间暂停后继续消费不完整结果。
STAGE_STOPS = {
    'insert_bowl': ('preplace', 'insert', 'release'),
    'place_bowl_on_table': ('preplace', 'place', 'release'),
    'extract_bowl': ('grasp',),
    'rotate_knob': ('open', 'pregrasp', 'approach', 'grasp', 'rotate'),
}


def validate_stage_stop(selection, stop_after):
    if stop_after is not None and stop_after not in STAGE_STOPS.get(selection, ()):
        raise ValueError(f'停止范围不适用：{selection}/{stop_after}；允许值：{STAGE_STOPS.get(selection, ())}')
