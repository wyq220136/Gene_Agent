"""用具名 position actuator 同步执行短关节轨迹。"""
from __future__ import annotations

import math
from contextlib import contextmanager

import mujoco
import numpy as np

from gene_agent.feedback.state import integration_state, state_id


def _is_position_actuator(model, actuator_id: int) -> bool:
    kp = float(model.actuator_gainprm[actuator_id, 0])
    return bool(
        model.actuator_gaintype[actuator_id] == mujoco.mjtGain.mjGAIN_FIXED
        and model.actuator_biastype[actuator_id] == mujoco.mjtBias.mjBIAS_AFFINE
        and model.actuator_dyntype[actuator_id] == mujoco.mjtDyn.mjDYN_NONE
        and kp > 0
        and model.actuator_biasprm[actuator_id, 0] == 0
        and model.actuator_biasprm[actuator_id, 1] == -kp
        and model.actuator_trntype[actuator_id] == mujoco.mjtTrn.mjTRN_JOINT
    )


def _position_actuator_for_joint(model, joint_id: int) -> int:
    matches = [
        actuator_id
        for actuator_id in range(model.nu)
        if _is_position_actuator(model, actuator_id)
        and int(model.actuator_trnid[actuator_id, 0]) == joint_id
    ]
    if len(matches) != 1:
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        raise ValueError(f"关节 {name} 必须恰好有一个 position actuator")
    return matches[0]


def scale_position_actuator_kp(model, joint_names, scale: float) -> dict[str, float]:
    """按具名关节缩放仿真 position actuator 的刚度，保留原力矩上限。

    任务装配层传入当前机械臂组的关节名，因此这里不假设机器人型号、
    手臂数量或关节命名。先找齐全部 actuator 再修改模型，避免中途
    遇到缺失关节时只改了一半。夹爪不在关节组中，增益不会受影响。
    """
    value = float(scale)
    if not np.isfinite(value) or value <= 0:
        raise ValueError("position actuator kp 比例必须为正有限数")
    names = tuple(joint_names)
    if not names or len(set(names)) != len(names):
        raise ValueError("position actuator 关节名不能为空或重复")
    selected = []
    for name in names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"未知关节：{name}")
        actuator_id = _position_actuator_for_joint(model, joint_id)
        selected.append((name, actuator_id, float(model.actuator_gainprm[actuator_id, 0])))
    result = {}
    for name, actuator_id, original_kp in selected:
        kp = original_kp * value
        model.actuator_gainprm[actuator_id, 0] = kp
        model.actuator_biasprm[actuator_id, 1] = -kp
        result[name] = kp
    return result


@contextmanager
def temporary_position_actuator_kp_scale(model, joint_names, scale: float):
    """仅在一段物理轨迹执行期间调整具名关节刚度，结束后精确恢复模型。

    检查点的模型指纹包含 actuator 参数；如果在阶段结束后还保留临时
    刚度，下一次恢复同一场景会被正确拒绝。这里保存完整参数行，并在
    正常结束和异常路径都写回，保证阶段间以及检查点看到原始模型。
    """
    names = tuple(joint_names)
    ids = []
    for name in names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"未知关节：{name}")
        ids.append(_position_actuator_for_joint(model, joint_id))
    gain_before = model.actuator_gainprm[ids].copy()
    bias_before = model.actuator_biasprm[ids].copy()
    try:
        gains = scale_position_actuator_kp(model, names, scale)
        yield gains
    finally:
        model.actuator_gainprm[ids] = gain_before
        model.actuator_biasprm[ids] = bias_before


def _capture_replay_sample(runtime, samples: list[dict], *, object_names, group: str,
                           attachment=None, interval_s: float = .05,
                           force: bool = False) -> None:
    """按仿真时间抽样完整实际状态；最后一步强制保留以核对阶段边界。"""
    if object_names is None:
        return
    now = float(runtime.data.time)
    if samples and abs(now - samples[-1]["sim_time_s"]) <= 1e-9:
        return
    if not force and samples and now - samples[-1]["sim_time_s"] < interval_s - 1e-9:
        return
    samples.append(runtime.replay_sample(
        object_names=object_names, robot_groups=(group,), attachment=attachment,
    ))


def execute_joint_trajectory(
    runtime,
    *,
    group: str,
    joint_names,
    times_s,
    positions_rad,
    planned_state_id: str,
    timeout_s: float | None = None,
    step_observer=None,
    stop_condition=None,
    endpoint_tolerance_rad: float | None = None,
    settle_timeout_s: float = 0.0,
    replay_object_names=None,
    replay_attachment=None,
    replay_interval_s: float = .05,
) -> dict:
    """按 model timestep 执行；可在发完指令后等待实际关节进入终点容差。"""
    model = runtime.model
    data = runtime.data
    groups = runtime.robot_metadata.get("joint_groups", {})
    if group not in groups:
        raise ValueError(f"未知机器人组: {group}")

    canonical_names = list(groups[group])
    supplied_names = list(joint_names)
    if len(set(supplied_names)) != len(supplied_names):
        raise ValueError("joint_names 不能重复")
    if set(supplied_names) != set(canonical_names) or len(supplied_names) != len(canonical_names):
        raise ValueError(f"轨迹关节必须与机器人组 {group} 完全一致")

    times = np.asarray(times_s, dtype=float)
    positions = np.asarray(positions_rad, dtype=float)
    if times.ndim != 1 or times.size < 2:
        raise ValueError("times_s 至少包含两个一维时间点")
    if positions.shape != (times.size, len(supplied_names)):
        raise ValueError("positions_rad shape 必须为 [时间点, 关节]")
    if not np.all(np.isfinite(times)) or not np.all(np.isfinite(positions)):
        raise ValueError("轨迹时间和位置必须有限")
    if times[0] != 0 or np.any(np.diff(times) <= 0):
        raise ValueError("times_s 必须从 0 开始并严格递增")
    if timeout_s is not None and (not np.isfinite(timeout_s) or timeout_s <= 0):
        raise ValueError("timeout_s 必须为正有限值")
    if step_observer is not None and not callable(step_observer):
        raise ValueError("step_observer 必须可调用")
    if stop_condition is not None and not callable(stop_condition):
        raise ValueError("stop_condition 必须可调用")
    if endpoint_tolerance_rad is not None and (
        not np.isfinite(endpoint_tolerance_rad) or endpoint_tolerance_rad <= 0
    ):
        raise ValueError("endpoint_tolerance_rad 必须为正有限值")
    if not np.isfinite(settle_timeout_s) or settle_timeout_s < 0:
        raise ValueError("settle_timeout_s 必须为非负有限值")
    if endpoint_tolerance_rad is None and settle_timeout_s > 0:
        raise ValueError("指定稳定等待时必须同时指定终点容差")
    if planned_state_id != state_id(model, data):
        raise ValueError("planned_state_id 已过期，必须基于当前状态重新规划")
    if not np.isfinite(replay_interval_s) or replay_interval_s <= 0:
        raise ValueError("replay_interval_s 必须为正有限值")

    source_columns = [supplied_names.index(name) for name in canonical_names]
    canonical_positions = positions[:, source_columns]
    joint_ids = []
    for name in canonical_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0 or int(model.jnt_type[joint_id]) != int(
            mujoco.mjtJoint.mjJNT_HINGE
        ):
            raise ValueError(f"机器人组 {group} 仅支持具名 hinge 关节: {name}")
        joint_ids.append(joint_id)
    actuator_ids = [_position_actuator_for_joint(model, joint_id) for joint_id in joint_ids]
    qpos_addresses = [int(model.jnt_qposadr[joint_id]) for joint_id in joint_ids]
    gears = np.asarray([model.actuator_gear[actuator_id, 0] for actuator_id in actuator_ids])
    if not np.all(np.isfinite(gears)) or np.any(gears == 0):
        raise ValueError("position actuator gear 必须为非零有限值")

    start_id = state_id(model, data)
    start_time = float(data.time)
    duration = float(times[-1])
    timestep = float(model.opt.timestep)
    steps = int(math.ceil(duration / timestep))
    samples = []
    replay_samples = []
    max_error = 0.0
    status = "completed"
    failure_type = None
    failure_evidence = None
    failure_step_index = None

    if not np.all(np.isfinite(integration_state(model, data))):
        status = "failed"
        failure_type = "non_finite_state"

    unstable_warnings = (
        mujoco.mjtWarning.mjWARN_BADQPOS,
        mujoco.mjtWarning.mjWARN_BADQVEL,
        mujoco.mjtWarning.mjWARN_BADQACC,
    )
    warning_counts = {
        warning: int(data.warning[warning].number) for warning in unstable_warnings
    }
    if status == "completed":
        # 以阶段真实起点为采样时钟原点，使 0.1 秒视频帧能引用同一
        # state_id 的完整状态；末帧仍由下方强制采样核对边界。
        _capture_replay_sample(runtime, replay_samples, object_names=replay_object_names,
                               group=group, attachment=replay_attachment, force=True)

    for step in range(1, steps + 1) if status == "completed" else ():
        # MuJoCo 只能按完整 timestep 前进；若下一步会越过预算，保持当前失败状态。
        if timeout_s is not None and data.time - start_time + timestep > timeout_s:
            status = "failed"
            failure_type = "execution_timeout"
            break
        elapsed = min(step * timestep, duration)
        reference = np.asarray(
            [np.interp(elapsed, times, canonical_positions[:, column]) for column in range(len(canonical_names))]
        )
        data.ctrl[actuator_ids] = gears * reference
        mujoco.mj_step(model, data)
        actual = np.asarray(data.qpos[qpos_addresses], dtype=float)
        error = np.abs(reference - actual)
        max_error = max(max_error, float(np.max(error, initial=0.0)))
        samples.append(
            {
                "time_s": float(data.time - start_time),
                "reference_positions_rad": reference.tolist(),
                "actual_positions_rad": actual.tolist(),
            }
        )
        _capture_replay_sample(runtime, replay_samples, object_names=replay_object_names,
                               group=group, attachment=replay_attachment,
                               interval_s=replay_interval_s)
        if not np.all(np.isfinite(integration_state(model, data))):
            status = "failed"
            failure_type = "non_finite_state"
            break
        if any(
            int(data.warning[warning].number) > warning_counts[warning]
            for warning in unstable_warnings
        ):
            status = "failed"
            failure_type = "mujoco_instability"
            break
        if step_observer is not None or stop_condition is not None:
            frozen = mujoco.MjData(model)
            mujoco.mj_copyData(frozen, model, data)
            mujoco.mj_forward(model, frozen)
            if step_observer is not None:
                step_observer(model, frozen, dict(samples[-1]))
            if stop_condition is not None:
                evidence = stop_condition(model, frozen, dict(samples[-1]))
                if evidence is not None:
                    if not isinstance(evidence, dict) or not evidence.get("failure_type"):
                        raise ValueError("stop_condition 必须返回 None 或含 failure_type 的字典")
                    status = "failed"
                    failure_type = str(evidence["failure_type"])
                    failure_evidence = dict(evidence)
                    failure_step_index = len(samples) - 1
                    break

    # 控制轨迹已经发完时，末端可能仍明显落后。阶段边界必须用真实 qpos
    # 判定，不能把“控制命令完成”写成“到达目标”。等待期间继续记录真实采样。
    settle_start = float(data.time)
    if status == "completed" and endpoint_tolerance_rad is not None:
        final_reference = canonical_positions[-1]
        settle_steps = int(math.ceil(settle_timeout_s / timestep))
        for _ in range(settle_steps):
            actual = np.asarray(data.qpos[qpos_addresses], dtype=float)
            if float(np.max(np.abs(final_reference - actual))) <= endpoint_tolerance_rad:
                break
            if timeout_s is not None and data.time - start_time + timestep > timeout_s:
                status, failure_type = "failed", "execution_timeout"
                break
            data.ctrl[actuator_ids] = gears * final_reference
            mujoco.mj_step(model, data)
            actual = np.asarray(data.qpos[qpos_addresses], dtype=float)
            error = np.abs(final_reference - actual)
            max_error = max(max_error, float(np.max(error)))
            samples.append({
                "time_s": float(data.time - start_time),
                "reference_positions_rad": final_reference.tolist(),
                "actual_positions_rad": actual.tolist(),
            })
            _capture_replay_sample(runtime, replay_samples, object_names=replay_object_names,
                                   group=group, attachment=replay_attachment,
                                   interval_s=replay_interval_s)
            if not np.all(np.isfinite(integration_state(model, data))):
                status, failure_type = "failed", "non_finite_state"
                break
            if any(int(data.warning[warning].number) > warning_counts[warning]
                   for warning in unstable_warnings):
                status, failure_type = "failed", "mujoco_instability"
                break
            if step_observer is not None or stop_condition is not None:
                frozen = mujoco.MjData(model)
                mujoco.mj_copyData(frozen, model, data)
                mujoco.mj_forward(model, frozen)
                if step_observer is not None:
                    step_observer(model, frozen, dict(samples[-1]))
                if stop_condition is not None:
                    evidence = stop_condition(model, frozen, dict(samples[-1]))
                    if evidence is not None:
                        if not isinstance(evidence, dict) or not evidence.get("failure_type"):
                            raise ValueError("stop_condition 必须返回 None 或含 failure_type 的字典")
                        status = "failed"
                        failure_type = str(evidence["failure_type"])
                        failure_evidence = dict(evidence)
                        failure_step_index = len(samples) - 1
                        break
    endpoint_error = float(np.max(np.abs(
        canonical_positions[-1] - np.asarray(data.qpos[qpos_addresses], dtype=float)
    )))
    if status == "completed" and endpoint_tolerance_rad is not None \
            and endpoint_error > endpoint_tolerance_rad:
        status, failure_type = "failed", "endpoint_tracking_error"
    if replay_object_names is not None:
        _capture_replay_sample(runtime, replay_samples, object_names=replay_object_names,
                               group=group, attachment=replay_attachment, force=True)
    return {
        "success": status == "completed",
        "status": status,
        "failure_type": failure_type,
        "failure_evidence": failure_evidence,
        "failure_step_index": failure_step_index,
        "group": group,
        "joint_names": canonical_names,
        "start_state_id": start_id,
        "end_state_id": state_id(model, data),
        "sim_duration_s": float(data.time - start_time),
        "max_tracking_error_rad": max_error,
        "endpoint_error_rad": endpoint_error,
        "endpoint_tolerance_rad": endpoint_tolerance_rad,
        "settle_duration_s": float(data.time - settle_start),
        "samples": samples,
        "replay_samples": replay_samples,
    }


def execute_hold(runtime, *, duration_s: float, planned_state_id: str,
                 step_observer=None, replay_object_names=None, replay_group: str = "right",
                 replay_attachment=None, replay_interval_s: float = .05) -> dict:
    """不改动任何ctrl或qpos，按仿真步推进并留下实际状态链。"""
    model, data = runtime.model, runtime.data
    duration = float(duration_s)
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError("保持时长必须为正有限值")
    if step_observer is not None and not callable(step_observer):
        raise ValueError("step_observer必须可调用")
    if state_id(model, data) != planned_state_id:
        raise ValueError("planned_state_id已过期，不能从旧状态开始保持")
    if not np.isfinite(replay_interval_s) or replay_interval_s <= 0:
        raise ValueError("replay_interval_s 必须为正有限值")
    start_time = float(data.time)
    warning_kinds = (mujoco.mjtWarning.mjWARN_BADQPOS,
                     mujoco.mjtWarning.mjWARN_BADQVEL,
                     mujoco.mjtWarning.mjWARN_BADQACC)
    warning_counts = {kind: int(data.warning[kind].number) for kind in warning_kinds}
    samples = []
    replay_samples = []
    failure_type = None
    _capture_replay_sample(runtime, replay_samples, object_names=replay_object_names,
                           group=replay_group, attachment=replay_attachment, force=True)
    for _ in range(int(math.ceil(duration / model.opt.timestep))):
        mujoco.mj_step(model, data)
        samples.append({"time_s": float(data.time - start_time)})
        _capture_replay_sample(runtime, replay_samples, object_names=replay_object_names,
                               group=replay_group, attachment=replay_attachment,
                               interval_s=replay_interval_s)
        if not np.all(np.isfinite(integration_state(model, data))):
            failure_type = "non_finite_state"
            break
        if any(int(data.warning[kind].number) > warning_counts[kind]
               for kind in warning_kinds):
            failure_type = "mujoco_instability"
            break
        if step_observer is not None:
            frozen = mujoco.MjData(model)
            mujoco.mj_copyData(frozen, model, data)
            mujoco.mj_forward(model, frozen)
            step_observer(model, frozen, dict(samples[-1]))
    if replay_object_names is not None:
        _capture_replay_sample(runtime, replay_samples, object_names=replay_object_names,
                               group=replay_group, attachment=replay_attachment, force=True)
    return {
        "success": failure_type is None,
        "status": "completed" if failure_type is None else "failed",
        "failure_type": failure_type,
        "start_state_id": planned_state_id,
        "end_state_id": state_id(model, data),
        "sim_duration_s": float(data.time - start_time),
        "samples": samples,
        "replay_samples": replay_samples,
    }
