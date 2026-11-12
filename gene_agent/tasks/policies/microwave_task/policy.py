"""微波炉 Task 决策；默认 TCP 过渡不属于门几何 Knowledge。"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from gene_agent.knowledge.geometry import validate_pose
from gene_agent.skills.library.door.sequence import DoorPhase
from gene_agent.skills.runtime.operation import OperationBackend
from gene_agent.skills.contracts import OperationResult, OperationStage
from gene_agent.skills.runtime.runner import run_operation


def operation_parameters(settings):
    """兼容旧输入的策略参数；不接收物体摆位、物理模型或评分阈值。"""
    from copy import deepcopy
    keys = ('pick_depth_m', 'insert_depth_offset_m',
            'insert_tcp_table_angle_max_deg', 'insert_max_bowl_tilt_deg',
            'insert_retreat_rotation_local_deg',
            'insert_retreat_clearance_margin_m',
            'insert_retreat_minimum_lift_m',
            'knob_azimuth_deg', 'knob_tip_x_deg')
    return {key: deepcopy(settings[key]) for key in keys if key in settings}


@dataclass(frozen=True)
class MicrowaveTaskPolicy:
    """显式环境差异；不把缺少的仿真轨迹伪称为已执行。

    旋钮后第二次开门前由仿真适配器插入安全退离和默认 TCP 过渡；
    其余次序与检查由同一执行器负责。
    """
    include_knob: bool = True
    before_reopen_transition: bool = True


@dataclass(frozen=True)
class MicrowaveTaskStep:
    name: str
    kind: str
    door_part: str | None
    success_key: str
    result_name: str


def open_door(name):
    return MicrowaveTaskStep(name, "door", "open", "success", "door_subtask_result.json")


def close_door(name):
    return MicrowaveTaskStep(name, "door", "close", "success", "door_subtask_result.json")


def transition(name):
    return MicrowaveTaskStep(name, "transition", None, "success", "transition_result.json")


def load_bowl():
    """请求抓取、放入（含释放撤离）；适配核对持物证据，失败由执行循环停止。"""
    yield MicrowaveTaskStep("pick_bowl", "pick", None, "pick_success", "pick_result.json")
    yield MicrowaveTaskStep("insert_bowl", "insert", None, "insert_success", "insert_result.json")


def unload_bowl():
    yield MicrowaveTaskStep("extract_bowl", "extract", None, "extract_success", "extract_result.json")
    yield MicrowaveTaskStep("place_bowl_on_table", "table", None, "table_place_success", "table_place_result.json")


def iter_task_steps(policy: MicrowaveTaskPolicy = MicrowaveTaskPolicy()):
    """任务次序的唯一来源；这里只描述操作，不提前规划轨迹。

    每项执行时重新读取实际状态。任意执行/后置检查失败，run_operation
    立即返回实际终态，后续操作不会执行；本策略尚无自动恢复分支。
    """
    yield open_door("open_door_first")
    yield transition("after_open_transition")
    yield from load_bowl()
    yield transition("before_close_first_transition")
    yield close_door("close_door_first")
    if policy.include_knob:
        yield MicrowaveTaskStep("rotate_knob", "knob", None, "rotation_success", "skill_result.json")
    if policy.before_reopen_transition:
        yield transition("before_reopen_transition")
    yield open_door("open_door_second")
    yield transition("after_reopen_transition")
    yield from unload_bowl()
    yield transition("before_close_second_transition")
    yield close_door("close_door_second")


def microwave_task_steps(policy: MicrowaveTaskPolicy = MicrowaveTaskPolicy()):
    """原选择器与适配的兼容查询；不独立维护另一份步骤表。"""
    return tuple(iter_task_steps(policy))


def run_microwave_task1_flow(backend: OperationBackend, *, policy=MicrowaveTaskPolicy(),
                             selection="all", preparation="prefix") -> OperationResult:
    """同步执行所选流程，返回 OperationResult（不含 benchmark 最终评分）。

    backend 遵循 skills.runtime.operation.OperationBackend：observe 提供实际状态，
    execute 返回 StageOutcome，检查 pass/fail/unknown；未知不能算执行成功。
    policy 只含流程开关，目标参数由已准备的 backend 提供。selection 使用
    subtasks.SUBTASK_NAMES 或 all；prefix 执行必要前缀，current 仅选中操作，
    均不隐式 reset。场景/设备生命周期、输出目录和真机授权由调用者管理。
    失败返回已执行前缀、failure_stage/code 和实际 end_state，后续动作停止。
    完整评分与报告引用见外层仿真 Session/现场 Task 返回的运行摘要。
    """
    from .subtasks import resolve_subtask
    steps = resolve_subtask(selection, policy=policy).steps(preparation)
    return run_operation((OperationStage(name, requires_check=True)
                          for name in steps), backend=backend)


def default_tcp_transition(
    default_tcp_pose_world: np.ndarray, *, door_angle_rad: float,
    gripper_width_m: float,
    phase_id: str = "task/open_to_close/return_default_tcp",
) -> DoorPhase:
    """接收当前机器人 home FK，不从门 frame 或任意历史起点推断目标。

    开门后的撤离保持旧 Task1 接触白名单；第二轮开门前回 home 保留
    完整场景碰撞。这里只表达请求，不启用任何路径 fallback。
    """
    policies = {
        "task/open_to_close/return_default_tcp": ("linear_tcp", "door_contact"),
        "task/before_knob/normalize_right_arm": ("plan", "avoid_door"),
        "task/before_reopen/return_default_tcp": ("plan", "avoid_door"),
    }
    if phase_id not in policies:
        raise ValueError(f"未知微波炉 Task 过渡：{phase_id}")
    pose = validate_pose(default_tcp_pose_world, field_name="机器人默认 TCP")
    if (not np.isfinite(door_angle_rad) or not np.isfinite(gripper_width_m)
            or gripper_width_m < 0):
        raise ValueError("Task 过渡门角、夹爪开口必须有限且开口非负")
    motion, contact = policies[phase_id]
    return DoorPhase(phase_id, "task_transition", "return_default_tcp", pose.copy(),
                     float(door_angle_rad), float(gripper_width_m), motion, contact)


def compose_task1_door_cycle(
    skill_phases: tuple[DoorPhase, ...],
    *,
    default_tcp_pose_world: np.ndarray,
) -> tuple[DoorPhase, ...]:
    """按旧版 Task1 的边界组合三个门 Skill。

    返回默认 TCP 是 ``push_open`` 与 ``push_close`` 之间的 Task 过渡，
    不属于任一门 Skill。目标由当前机器人默认关节状态做 FK 得到，不能
    写死为 Realman 的固定世界 pose。
    """
    open_indices = [i for i, phase in enumerate(skill_phases)
                    if phase.skill_id == "push_open"]
    close_indices = [i for i, phase in enumerate(skill_phases)
                     if phase.skill_id == "push_close"]
    if not open_indices or not close_indices or max(open_indices) >= min(close_indices):
        raise ValueError("Task1 组合要求 push_open 位于 push_close 之前")
    last_open = skill_phases[max(open_indices)]
    transition = default_tcp_transition(
        default_tcp_pose_world,
        door_angle_rad=last_open.door_angle_rad,
        gripper_width_m=last_open.gripper_width_m,
    )
    insert_at = max(open_indices) + 1
    return skill_phases[:insert_at] + (transition,) + skill_phases[insert_at:]
