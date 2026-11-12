"""单个运动阶段的真机校验与显式执行边界。

此模块不生成 Knowledge、Skill 或 Task，也不初始化 cuRobo/MuJoCo。
调用方必须针对最新观测重新生成本阶段规划；本类只核对规划来源、
实测起点和整条轨迹，然后可选择显式执行一个阶段。
"""

from __future__ import annotations

import math
import time
from typing import Callable, Sequence

import numpy as np

from gene_agent.motion.model import JointState
from gene_agent.skills.runtime.planning import SkillStage
from gene_agent.tasks.checks import CheckResult


class RealWorldRuntime:
    """同步阶段执行器；默认只验证，绝不会在构造时连接真机。"""

    def __init__(
        self, robot, *, scene_snapshot_sha256: str,
        max_start_error_rad: float,
        tcp_height_fn: Callable[[Sequence[float]], float],
        min_tcp_z_m: float,
        gripper=None,
        pre_execute_scene_check: Callable[[], CheckResult] | None = None,
        motion_speed_scale: float = 0.1,
        max_tcp_speed_mps: float = 0.1,
        canfd_frequency_hz: float = 100.0,
        tcp_speed_mps: float | None = None,
        execution_guard: Callable[[], None] | None = None,
    ) -> None:
        if not scene_snapshot_sha256:
            raise ValueError("现场碰撞快照 hash 不能为空")
        if not math.isfinite(max_start_error_rad) or max_start_error_rad <= 0:
            raise ValueError("起点误差阈值必须为正有限数")
        if not math.isfinite(min_tcp_z_m):
            raise ValueError("最低 TCP 高度必须为有限数")
        self.robot = robot
        self.gripper = gripper
        self.scene_snapshot_sha256 = scene_snapshot_sha256
        self.max_start_error_rad = float(max_start_error_rad)
        self.tcp_height_fn = tcp_height_fn
        self.min_tcp_z_m = float(min_tcp_z_m)
        self.pre_execute_scene_check = pre_execute_scene_check
        self.execution_guard = execution_guard
        scale = float(motion_speed_scale)
        maximum = float(max_tcp_speed_mps)
        frequency = float(canfd_frequency_hz)
        requested = None if tcp_speed_mps is None else float(tcp_speed_mps)
        if not math.isfinite(scale) or not 0.0 < scale <= 1.0:
            raise ValueError("motion_speed_scale 必须位于 (0, 1] 区间")
        if not math.isfinite(maximum) or maximum <= 0.0:
            raise ValueError("max_tcp_speed_mps 必须为正有限数")
        if not math.isfinite(frequency) or frequency <= 0.0:
            raise ValueError("canfd_frequency_hz 必须为正有限数")
        if requested is not None and (
            not math.isfinite(requested) or not 0.0 < requested <= maximum
        ):
            raise ValueError("tcp_speed_mps 必须位于 (0, max_tcp_speed_mps] 区间")
        self.execution_speed_profile = {
            "speed_scale": scale,
            "max_tcp_speed_mps": maximum,
            "canfd_frequency_hz": frequency,
            "tcp_speed_mps": requested,
        }

    @staticmethod
    def _failed(stage: str, code: str, message: str, *, details=None) -> dict:
        return {
            "source_format": "realworld_motion_stage", "stage": stage,
            "status": "failed", "success": False,
            "failure": {"code": code, "message": message,
                        "details": details or {}},
        }

    def _check_live_scene(self, stage: str) -> tuple[dict | None, dict | None]:
        """执行前重新观察现场；失败、未知或检查异常均禁止发送 SDK 命令。

        调用方负责将本阶段规划所用快照与新观测比较。预览不调用此检查，
        以免一次纯规划预览触发相机采集或现场副作用。
        """
        if self.pre_execute_scene_check is None:
            return None, None
        try:
            result = self.pre_execute_scene_check()
            if not isinstance(result, CheckResult):
                raise TypeError("现场复核必须返回 CheckResult")
        except Exception as error:
            return self._failed(stage, "scene_recheck_error",
                                f"执行前现场复核异常：{type(error).__name__}: {error}"), None
        evidence = result.to_dict()
        if not result.passed:
            return self._failed(stage, "scene_recheck_failed",
                                "现场已变化或证据不足，必须重新观测并规划",
                                details=evidence), evidence
        return None, evidence

    def run_motion_stage(self, stage: SkillStage, *, execute: bool = False) -> dict:
        """核对一个已规划运动阶段；`execute=True` 才允许发 SDK 命令。"""
        request, result = stage.request, stage.result
        if stage.status != "planned" or request is None or result is None or not result.success:
            failure = None if result is None or result.failure is None else result.failure.to_dict()
            return self._failed(stage.name, "planner_failed", "本阶段没有可执行的成功规划",
                                details={"planner_failure": failure})
        if request.request_id != result.request_id or request.operation != result.operation:
            return self._failed(stage.name, "plan_request_mismatch", "规划结果与请求不一致")
        if request.group != self.robot.group:
            return self._failed(stage.name, "robot_group_mismatch", "规划组与真机活动臂不一致")
        if request.scene.snapshot_sha256 != self.scene_snapshot_sha256:
            return self._failed(stage.name, "scene_mismatch", "现场碰撞快照已变化，必须重新规划",
                                details={"planned": request.scene.snapshot_sha256,
                                         "current": self.scene_snapshot_sha256})
        trajectory = result.trajectory
        if trajectory is None:
            return self._failed(stage.name, "trajectory_missing", "规划结果没有关节轨迹")
        try:
            measured_start = self.robot.read_joint_state()
        except Exception as error:
            return self._failed(stage.name, "robot_state_error", f"读取真机状态失败：{error}")
        if request.start.joint_names != measured_start.joint_names:
            return self._failed(stage.name, "joint_order_mismatch", "规划起点关节顺序与真机不一致")
        start_error = float(np.max(np.abs(
            np.asarray(request.start.positions_rad) - measured_start.positions_rad
        )))
        if start_error > self.max_start_error_rad:
            return self._failed(
                stage.name, "start_state_mismatch", "实测起点已偏离规划起点，必须重新规划",
                details={"max_error_rad": start_error,
                         "allowed_rad": self.max_start_error_rad},
            )
        options = {
            "measured_start": measured_start,
            "max_start_error_rad": self.max_start_error_rad,
            "tcp_height_fn": self.tcp_height_fn,
            "min_tcp_z_m": self.min_tcp_z_m,
        }
        try:
            self.robot.preflight_trajectory(trajectory, **options)
        except Exception as error:
            return self._failed(stage.name, "preflight_failed", f"整段轨迹预检失败：{error}")
        record = {
            "source_format": "realworld_motion_stage", "stage": stage.name,
            "status": "validated", "success": True,
            "request_id": request.request_id,
            "scene_snapshot_sha256": self.scene_snapshot_sha256,
            "trajectory_points": len(trajectory.times_s),
            "path_type": request.operation.value,
            "planning_duration_s": result.duration_s,
            "execution_speed_profile": dict(self.execution_speed_profile),
            "measured_start": measured_start.to_dict(),
            "measured_end": None,
            "claim": "仅检查规划、现场快照及真机起点；未执行运动",
        }
        if not execute:
            return record
        failure, recheck = self._check_live_scene(stage.name)
        if failure is not None:
            return failure
        if recheck is not None:
            record["scene_recheck"] = recheck
        started = time.monotonic()
        try:
            measured_end = self.robot.execute_trajectory(
                trajectory, **options, **self.execution_speed_profile,
                **({"execution_guard":self.execution_guard} if self.execution_guard is not None else {}),
            )
        except Exception as error:
            failure = self._failed(stage.name, "execution_failed",
                                   f"真机执行失败：{type(error).__name__}: {error}")
            if recheck is not None:
                failure["scene_recheck"] = recheck
            failure["measured_start"] = measured_start.to_dict()
            failure["execution_duration_s"] = time.monotonic() - started
            try:
                # 已下发的部分轨迹不能回滚；读出失败后的现场关节，供下一步
                # 人工恢复和重新规划。读取本身若失败，也要保留原 SDK 异常。
                failure["measured_end"] = self.robot.read_joint_state().to_dict()
            except Exception as state_error:
                failure["measured_end"] = None
                failure["state_read_error"] = (
                    f"{type(state_error).__name__}: {state_error}"
                )
            return failure
        record.update({
            "status": "executed", "measured_end": measured_end.to_dict(),
            "execution_duration_s": time.monotonic() - started,
            "claim": "仅证明关节轨迹已下发且实测终点到达；物体操作需另查后置条件",
        })
        return record

    def run_last_joint_stage(
        self, stage: str, *, group: str, joint_delta_rad: float,
        planned_start: JointState, planned_scene_snapshot_sha256: str,
        speed_scale: float = 0.1, execute: bool = False,
    ) -> dict:
        """按旧 RealMan 单次 movej 语义旋转第七轴，不冒充 cuRobo 轨迹。

        旧版该段不做场景碰撞规划；这里显式记录这一事实，并在下发前
        核对同帧快照、实测关节起点、目标 TCP 最低高度。调用方仍需
        单独确认旋钮附近的接触安全和真实旋转后置条件。
        """
        if planned_scene_snapshot_sha256 != self.scene_snapshot_sha256:
            return self._failed(stage, "scene_mismatch", "现场碰撞快照已变化，必须重新观测")
        if group != self.robot.group:
            return self._failed(stage, "robot_group_mismatch", "旋钮活动臂与 Runtime 不一致")
        delta = float(joint_delta_rad)
        speed = float(speed_scale)
        if not math.isfinite(delta) or not math.isfinite(speed) or speed <= 0:
            return self._failed(stage, "invalid_command", "末关节增量或速度无效")
        try:
            measured = self.robot.read_joint_state()
        except Exception as error:
            return self._failed(stage, "robot_state_error", f"读取真机状态失败：{error}")
        if measured.joint_names != planned_start.joint_names:
            return self._failed(stage, "joint_order_mismatch", "末关节命令的关节顺序不一致")
        if len(measured.positions_rad) != 7:
            return self._failed(stage, "joint_count_mismatch", "末关节命令需要七轴状态")
        start_error = float(np.max(np.abs(
            np.asarray(measured.positions_rad) - planned_start.positions_rad
        )))
        if start_error > self.max_start_error_rad:
            return self._failed(stage, "start_state_mismatch", "真机关节起点已变化，必须重新观测",
                                details={"max_error_rad": start_error})
        target = (*measured.positions_rad[:-1], measured.positions_rad[-1] + delta)
        try:
            height = float(self.tcp_height_fn(target))
            if not math.isfinite(height) or height < self.min_tcp_z_m:
                raise ValueError(f"末关节目标 TCP 高度 {height:.4f} m 低于 {self.min_tcp_z_m:.4f} m")
        except Exception as error:
            return self._failed(stage, "preflight_failed", f"末关节目标预检失败：{error}")
        record = {
            "source_format": "realworld_motion_stage", "stage": stage,
            "status": "validated", "success": True,
            "scene_snapshot_sha256": self.scene_snapshot_sha256,
            "collision_scope": "none_legacy_last_joint",
            "path_type": "direct_last_joint_movej",
            "joint_delta_rad": delta, "speed_scale": speed,
            "planned_start": planned_start.to_dict(),
            "measured_start": measured.to_dict(),
            "target_joint_state": JointState(measured.joint_names, target).to_dict(),
            "measured_end": None,
            "claim": "仅核对末关节目标与 TCP 高度；旧版此段不做场景碰撞规划",
        }
        if not execute:
            return record
        failure, recheck = self._check_live_scene(stage)
        if failure is not None:
            return failure
        if recheck is not None:
            record["scene_recheck"] = recheck
        started = time.monotonic()
        try:
            actual = self.robot.rotate_last_joint(
                delta, measured_start=measured, speed_scale=speed,
                max_start_error_rad=self.max_start_error_rad,
            )
            if actual.joint_names != measured.joint_names:
                raise RuntimeError("末关节动作后关节顺序变化")
            error = float(np.max(np.abs(np.asarray(actual.positions_rad) - target)))
            if error > self.max_start_error_rad:
                raise RuntimeError(f"末关节目标实测误差 {error:.5f} rad 超限")
        except Exception as error:
            failure = self._failed(stage, "execution_failed",
                                   f"末关节运动失败：{type(error).__name__}: {error}")
            if recheck is not None:
                failure["scene_recheck"] = recheck
            failure["measured_start"] = measured.to_dict()
            failure["execution_duration_s"] = time.monotonic() - started
            try:
                failure["measured_end"] = self.robot.read_joint_state().to_dict()
            except Exception as state_error:
                failure["state_read_error"] = f"{type(state_error).__name__}: {state_error}"
            return failure
        record.update(status="executed", measured_end=actual.to_dict(),
                      execution_duration_s=time.monotonic() - started,
                      claim="只证实末关节到达；旋钮是否被物理带动仍需现场检查")
        return record

    def run_gripper_stage(
        self, stage: str, *, group: str, target_width_m: float,
        planned_start, planned_start_width_m: float,
        max_width_error_m: float, planned_scene_snapshot_sha256: str,
        execute: bool = False, command_mode: str = "position",
        grasp_force: float = 0.5,
    ) -> dict:
        """使用真机实测关节和夹爪开口核对米制命令。

        ``target_width_m`` 仍表示 Skill 规划所用的几何宽度。
        ``position`` 才把它发为位置命令；``grasp`` 和 ``release``
        分别沿用旧版 RealMan 的力控抓取与松爪 SDK 语义。这里只记录
        指令和读数，真实夹持/放置仍由 Task 后置检查判定；预览不发令。
        """
        if not math.isfinite(max_width_error_m) or max_width_error_m <= 0:
            raise ValueError("夹爪起点误差阈值必须为正有限米数")
        if command_mode not in {"position", "grasp", "release"}:
            return self._failed(stage, "gripper_command_invalid", "未知夹爪硬件命令类型")
        force = float(grasp_force)
        if command_mode == "grasp" and (not math.isfinite(force) or not 0. <= force <= 1.):
            return self._failed(stage, "gripper_force_invalid", "抓取力必须在 0～1 范围内")
        if self.gripper is None:
            return self._failed(stage, "gripper_unavailable", "未配置真机夹爪")
        if group != self.robot.group:
            return self._failed(stage, "robot_group_mismatch", "夹爪阶段与活动臂不一致")
        if planned_scene_snapshot_sha256 != self.scene_snapshot_sha256:
            return self._failed(stage, "scene_mismatch", "现场碰撞快照已变化，必须重新规划")
        target = float(target_width_m)
        planned_width = float(planned_start_width_m)
        maximum = float(self.gripper.max_width_m)
        if (not all(math.isfinite(v) for v in (target, planned_width))
                or not 0 <= target <= maximum or not 0 <= planned_width <= maximum):
            return self._failed(stage, "gripper_width_invalid", "夹爪开口超出配置范围")
        try:
            measured_joints = self.robot.read_joint_state()
            measured_width = float(self.gripper.read_width_m())
        except Exception as error:
            return self._failed(stage, "robot_state_error", f"读取真机起点失败：{error}")
        if measured_joints.joint_names != planned_start.joint_names:
            return self._failed(stage, "joint_order_mismatch", "夹爪阶段关节顺序不一致")
        joint_error = float(np.max(np.abs(
            np.asarray(measured_joints.positions_rad)
            - np.asarray(planned_start.positions_rad)
        )))
        if joint_error > self.max_start_error_rad:
            return self._failed(stage, "start_state_mismatch", "真机关节已偏离规划起点",
                                details={"max_error_rad": joint_error})
        width_error = abs(measured_width - planned_width)
        if width_error > max_width_error_m:
            return self._failed(stage, "gripper_start_mismatch", "夹爪已偏离规划起点",
                                details={"error_m": width_error})
        record = {
            "source_format": "realworld_gripper_stage", "stage": stage,
            "status": "validated", "success": True, "group": group,
            "scene_snapshot_sha256": self.scene_snapshot_sha256,
            "target_width_m": target,
            # target_width_m 是用于规划的几何闭合宽度；旧真机抓取使用
            # pick_on 的力控命令，不应误译为位置闭合到该宽度。
            "hardware_command": ({"mode": command_mode, "force": force,
                                  "speed": 1.0, "continuous": True}
                                 if command_mode == "grasp" else
                                 {"mode": command_mode}),
            "measured_start_width_m": measured_width,
            "measured_end_width_m": None,
            "measured_joint_start": measured_joints.to_dict(),
            "claim": "只核对夹爪指令与真机起点；未执行运动",
        }
        if not execute:
            return record
        failure, recheck = self._check_live_scene(stage)
        if failure is not None:
            return failure
        if recheck is not None:
            record["scene_recheck"] = recheck
        try:
            # 阶段开始后若机械臂状态改变，必须撤销旧夹爪指令。
            current = self.robot.read_joint_state()
            drift = float(np.max(np.abs(
                np.asarray(current.positions_rad)
                - np.asarray(measured_joints.positions_rad)
            )))
            if current.joint_names != measured_joints.joint_names or drift > self.max_start_error_rad:
                return self._failed(stage, "start_state_mismatch", "发令前机械臂起点变化")
            if command_mode == "grasp":
                self.gripper.grasp(force=force)
            elif command_mode == "release":
                self.gripper.release()
            else:
                self.gripper.command_width_m(target)
            measured_end_width = float(self.gripper.read_width_m())
        except Exception as error:
            failure = self._failed(stage, "gripper_execution_failed",
                                   f"夹爪执行或终点读取失败：{type(error).__name__}: {error}")
            if recheck is not None:
                failure["scene_recheck"] = recheck
            failure["measured_start_width_m"] = measured_width
            return failure
        record.update({
            "status": "executed", "measured_end_width_m": measured_end_width,
            "claim": "夹爪命令已下发并读取开口；夹持物体需另查后置条件",
        })
        return record
