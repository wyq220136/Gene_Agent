"""同步执行米制平行夹爪开口命令。"""
from __future__ import annotations

import math

import mujoco
import numpy as np

from gene_agent.feedback.state import integration_state, state_id
from gene_agent.robots.gripper import measure_gripper_width, set_gripper_width
from .execution import _capture_replay_sample, _is_position_actuator


# 执行器目标始终严格限制在配置范围内；实测的动力学关节在端点附近
# 可短暂越过硬限位。长时间夹持后实测下限越界约 70 微米；
# 100 微米仅用于接受下一条命令的起点，
# 参考开口仍由下方 np.clip 截回合法范围；更大的越界继续报错。
START_WIDTH_LIMIT_TOLERANCE_M = 1e-4


def execute_gripper_command(
    runtime,
    *,
    group: str,
    width_m: float,
    duration_s: float,
    planned_state_id: str,
    timeout_s: float | None = None,
    step_observer=None,
    replay_object_names=None,
    replay_attachment=None,
    replay_interval_s: float = .05,
) -> dict:
    """线性插值总开口；只写 gripper position actuator ctrl。"""
    model, data = runtime.model, runtime.data
    descriptor = runtime.robot_metadata.get("grippers", {}).get(group)
    if not descriptor or descriptor.get("mode") not in {"parallel_slide_proxy", "pika_urdf"}:
        raise ValueError(f"机器人组 {group} 没有可执行的米制双指夹爪")
    target = float(width_m)
    duration = float(duration_s)
    low, high = (float(value) for value in descriptor["width_range_m"])
    if not np.isfinite(target) or not low <= target <= high:
        raise ValueError(f"开口必须在[{low},{high}]米且有限")
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError("duration_s 必须为正有限值")
    if timeout_s is not None and (not np.isfinite(timeout_s) or timeout_s <= 0):
        raise ValueError("timeout_s 必须为正有限值")
    if step_observer is not None and not callable(step_observer):
        raise ValueError("step_observer 必须可调用")
    if planned_state_id != state_id(model, data):
        raise ValueError("planned_state_id 已过期，必须基于当前状态重新规划或控制")
    if not np.isfinite(replay_interval_s) or replay_interval_s <= 0:
        raise ValueError("replay_interval_s 必须为正有限值")
    actuator_ids = []
    for name in descriptor.get("actuator_names", []):
        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if actuator_id < 0 or not _is_position_actuator(model, actuator_id):
            raise ValueError(f"夹爪执行器必须是有效 position actuator: {name}")
        actuator_ids.append(actuator_id)
    if len(actuator_ids) != 2 or len(set(actuator_ids)) != 2:
        raise ValueError("parallel slide gripper 必须恰好包含两个不同执行器")

    start_width = float(
        measure_gripper_width(model, data, runtime.robot_metadata, group)["width_m"]
    )
    if not low - START_WIDTH_LIMIT_TOLERANCE_M <= start_width \
            <= high + START_WIDTH_LIMIT_TOLERANCE_M:
        raise ValueError(
            f"当前实测夹爪开口{start_width:.9f}米超出[{low},{high}]米"
            f"及{START_WIDTH_LIMIT_TOLERANCE_M:g}米动力学容差")
    start_identifier = state_id(model, data)
    start_time = float(data.time)
    timestep = float(model.opt.timestep)
    steps = int(math.ceil(duration / timestep))
    samples = []
    replay_samples = []
    max_error = 0.0
    status = "completed"
    failure_type = None
    if not np.all(np.isfinite(integration_state(model, data))):
        status, failure_type = "failed", "non_finite_state"
    unstable_warnings = (
        mujoco.mjtWarning.mjWARN_BADQPOS,
        mujoco.mjtWarning.mjWARN_BADQVEL,
        mujoco.mjtWarning.mjWARN_BADQACC,
    )
    warning_counts = {
        warning: int(data.warning[warning].number) for warning in unstable_warnings
    }
    if status == "completed":
        _capture_replay_sample(runtime, replay_samples, object_names=replay_object_names,
                               group=group, attachment=replay_attachment, force=True)

    for step in range(1, steps + 1) if status == "completed" else ():
        if timeout_s is not None and data.time - start_time + timestep > timeout_s:
            status, failure_type = "failed", "execution_timeout"
            break
        elapsed = min(step * timestep, duration)
        # 测量结果在限位上可能多几个浮点尾数；只把发送给执行器的
        # 参考值截到合法区间，实测开口和反馈仍保留原始数值。
        reference = float(np.clip(
            np.interp(elapsed, [0.0, duration], [start_width, target]), low, high,
        ))
        set_gripper_width(model, data, runtime.robot_metadata, group, reference)
        mujoco.mj_step(model, data)
        measured = measure_gripper_width(
            model, data, runtime.robot_metadata, group
        )
        actual = float(measured["width_m"])
        max_error = max(max_error, abs(reference - actual))
        samples.append({
            "time_s": float(data.time - start_time),
            "reference_width_m": reference,
            "actual_width_m": actual,
            "symmetry_error_m": float(measured["symmetry_error_m"]),
            "joint_positions_m": measured["joint_positions_m"],
        })
        _capture_replay_sample(runtime, replay_samples, object_names=replay_object_names,
                               group=group, attachment=replay_attachment,
                               interval_s=replay_interval_s)
        if not np.all(np.isfinite(integration_state(model, data))):
            status, failure_type = "failed", "non_finite_state"
            break
        if any(
            int(data.warning[warning].number) > warning_counts[warning]
            for warning in unstable_warnings
        ):
            status, failure_type = "failed", "mujoco_instability"
            break
        if step_observer is not None:
            frozen = mujoco.MjData(model)
            mujoco.mj_copyData(frozen, model, data)
            mujoco.mj_forward(model, frozen)
            step_observer(model, frozen, dict(samples[-1]))
    if replay_object_names is not None:
        _capture_replay_sample(runtime, replay_samples, object_names=replay_object_names,
                               group=group, attachment=replay_attachment, force=True)
    return {
        "success": status == "completed",
        "status": status,
        "failure_type": failure_type,
        "group": group,
        "target_width_m": target,
        "start_width_m": start_width,
        "start_state_id": start_identifier,
        "end_state_id": state_id(model, data),
        "sim_duration_s": float(data.time - start_time),
        "max_width_error_m": max_error,
        "samples": samples,
        "replay_samples": replay_samples,
    }


__all__ = ["execute_gripper_command"]
