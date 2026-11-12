"""在线 MuJoCo 状态与同步 cuRobo 规划之间的可复用单阶段边界。"""
from __future__ import annotations

import json
from pathlib import Path
from ..configuration import CollisionMode, resolve_collision_mode, collision_mode_metadata

import numpy as np

from .core import SynchronousPlanner, load_realman_motion_config
from .session import CuroboPlannerPool
from ..model import JointTrajectory, PlannerMode
from ..path_metrics import measure_linear_tcp_path
from .approximate import ApproximateIKBudget, _pose_error, plan_with_approximate_ik
from .scene import (
    export_collision_scene, export_self_collision_only_snapshot,
    filter_collision_snapshot,
)


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


class CuroboRuntimeBridge:
    """每段查询当前状态、冻结当前场景、规划并执行；调用方决定任务阶段。"""

    def __init__(self, runtime, output_dir: Path, *, group: str = "right", on_failure=None,
                 visual_recorder=None, contact_policy_for_stage=None,
                 linear_tcp_stages=(), endpoint_tolerance_rad=None,
                 settle_timeout_s=0.0, attachment_for_stage=None,
                 replay_object_names=None, max_attempts_by_stage=None,
                 approximate_ik_by_stage=None, disabled_body_names_for_stage=None,
                 plan_postcheck_for_stage=None, snapshot_filter_for_stage=None,
                 execution_stop_condition_for_stage=None,
                 execution_time_scale_by_stage=None,
                 execution_position_kp_scale_by_stage=None,
                 max_joint_travel_rad_by_stage=None,
                 optimizer_collision_activation_distance_by_stage=None,
                 preferred_trajectory_by_stage=None,
                 execute_rejected_plan_prefixes: bool = False,
                 pivot_tcp_stages=None,
                 self_collision_only_stages=(),
                 robot_body_only_stages=(),
                 planner_mode: PlannerMode | str = PlannerMode.CUDA_GRAPH_BENCHMARK,
                 reuse_gripper_geometry: bool = True):
        self.runtime = runtime
        self.output = Path(output_dir)
        self.group = group
        self.on_failure = on_failure
        self.visual_recorder = visual_recorder
        self.contact_policy_for_stage = contact_policy_for_stage
        self.linear_tcp_stages = frozenset(linear_tcp_stages)
        self.pivot_tcp_stages = {
            str(stage): dict(limits) for stage, limits in
            dict(pivot_tcp_stages or {}).items()}
        if self.linear_tcp_stages & self.pivot_tcp_stages.keys():
            raise ValueError("同一阶段不能同时选择直线平移和原位旋转轨迹")
        self.endpoint_tolerance_rad = endpoint_tolerance_rad
        self.settle_timeout_s = settle_timeout_s
        self.attachment_for_stage = attachment_for_stage
        self.disabled_body_names_for_stage = disabled_body_names_for_stage
        self.plan_postcheck_for_stage = plan_postcheck_for_stage
        self.execution_stop_condition_for_stage = execution_stop_condition_for_stage
        self.snapshot_filter_for_stage = snapshot_filter_for_stage
        self.execution_time_scale_by_stage = dict(execution_time_scale_by_stage or {})
        self.execution_position_kp_scale_by_stage = dict(
            execution_position_kp_scale_by_stage or {})
        self.max_joint_travel_rad_by_stage = {
            str(stage): float(limit) for stage, limit in
            dict(max_joint_travel_rad_by_stage or {}).items()
        }
        self.optimizer_collision_activation_distance_by_stage = dict(
            optimizer_collision_activation_distance_by_stage or {})
        self.preferred_trajectory_by_stage = dict(preferred_trajectory_by_stage or {})
        self.execute_rejected_plan_prefixes = bool(execute_rejected_plan_prefixes)
        if any(not isinstance(value, (JointTrajectory, dict))
               for value in self.preferred_trajectory_by_stage.values()):
            raise TypeError("阶段首选轨迹必须是 JointTrajectory 或轨迹字典")
        if any(not np.isfinite(distance) or distance < 0 for distance in
               self.optimizer_collision_activation_distance_by_stage.values()):
            raise ValueError('阶段优化器软避障距离必须是非负有限米数')
        if any(not np.isfinite(scale) or scale <= 0
               for scale in self.execution_position_kp_scale_by_stage.values()):
            raise ValueError('阶段位置控制增益倍率必须为正有限值')
        if any(not np.isfinite(limit) or limit <= 0
               for limit in self.max_joint_travel_rad_by_stage.values()):
            raise ValueError("阶段最大单关节行程必须是正有限弧度")
        if any(not np.isfinite(scale) or scale < 1.0
               for scale in self.execution_time_scale_by_stage.values()):
            raise ValueError("阶段执行时间倍率必须是大于等于 1 的有限数")
        self.self_collision_only_stages = frozenset(self_collision_only_stages)
        self.robot_body_only_stages = frozenset(robot_body_only_stages)
        if self.self_collision_only_stages & self.robot_body_only_stages:
            raise ValueError("阶段不能同时选择空 world 和仅保留 robot_body")
        self.replay_object_names = replay_object_names
        self.max_attempts_by_stage = dict(max_attempts_by_stage or {})
        self.approximate_ik_by_stage = {
            str(stage): ApproximateIKBudget(**value) if isinstance(value, dict) else value
            for stage, value in dict(approximate_ik_by_stage or {}).items()
        }
        if any(not isinstance(value, ApproximateIKBudget) for value in self.approximate_ik_by_stage.values()):
            raise TypeError("approximate_ik_by_stage 必须映射到 ApproximateIKBudget 或误差预算字典")
        if self.pivot_tcp_stages.keys() & self.approximate_ik_by_stage.keys():
            raise ValueError("原位旋转阶段暂不支持近似 IK 目标")
        self.planner_mode = PlannerMode(planner_mode)
        # 仅在本bridge生命周期内跨阶段复用；容量1避免同时驻留多套大图。
        # 仍逐阶段导出实际世界、夹爪和attachment，不缓存执行起点或目标。
        self.planner_pool = CuroboPlannerPool(
            planner_factory=lambda **kwargs: SynchronousPlanner(**kwargs),
            max_entries=1, reuse_gripper_geometry=reuse_gripper_geometry,
        )

    def _preferred_plan(self, stage: str, planner, robot: dict,
                        target: np.ndarray) -> dict:
        """复核并采用候选筛选阶段选定的同一条关节分支。"""
        raw = self.preferred_trajectory_by_stage[stage]
        trajectory = raw.to_dict() if isinstance(raw, JointTrajectory) else dict(raw)
        names = tuple(trajectory.get("joint_names") or ())
        positions = np.asarray(trajectory.get("positions_rad") or (), dtype=float)
        times = np.asarray(trajectory.get("times_s") or (), dtype=float)
        if (not names or len(names) != len(set(names)) or positions.ndim != 2
                or positions.shape != (len(times), len(names)) or len(times) < 2
                or not np.isfinite(positions).all() or not np.isfinite(times).all()
                or times[0] < 0 or np.any(np.diff(times) <= 0)):
            raise ValueError(f"{stage} 首选轨迹结构无效")
        steps = np.diff(times)
        if not np.allclose(steps, steps[0], atol=1e-9, rtol=1e-6):
            raise ValueError(f"{stage} 首选轨迹必须使用均匀时间采样")

        current_by_name = dict(zip(
            robot["joint_names"], robot["joint_positions_rad"], strict=True,
        ))
        if any(name not in current_by_name for name in names):
            raise ValueError(f"{stage} 首选轨迹关节不属于当前机器人状态")
        actual_start = np.asarray([current_by_name[name] for name in names], dtype=float)
        start_error = np.abs(positions[0] - actual_start)
        start_limit = float(self.endpoint_tolerance_rad or .03)
        maximum_start_error = float(np.max(start_error))

        tcp_path = planner.fk_batch(names, positions)
        endpoint_position_error, endpoint_rotation_error = _pose_error(
            np.asarray(target, dtype=float), tcp_path[-1],
        )
        validation = planner.validate_joint_trajectory(names, positions)
        checks = {
            "source": "candidate_selection_trajectory_revalidated_at_actual_stage_start",
            "start_maximum_joint_error_rad": maximum_start_error,
            "start_error_limit_rad": start_limit,
            "endpoint_position_error_m": endpoint_position_error,
            "endpoint_rotation_error_rad": endpoint_rotation_error,
            "collision_verification": validation,
        }
        success = bool(
            maximum_start_error <= start_limit + 1e-12
            and endpoint_position_error <= .005
            and endpoint_rotation_error <= .05
            and validation.get("valid") is True
        )
        if stage in self.linear_tcp_stages:
            start_tcp = robot.get("T_world_tcp")
            if start_tcp is None:
                start_tcp = planner.fk(names, actual_start)
            metrics = measure_linear_tcp_path(
                tcp_path[:, :3, 3], np.asarray(start_tcp, dtype=float)[:3, 3],
                np.asarray(target, dtype=float)[:3, 3],
            )
            checks["linear_path_metrics"] = metrics.to_dict()
            success = bool(success and metrics.satisfies(
                lateral_m=.005, backward_step_m=.001,
                cumulative_backward_m=.002, overshoot_m=.002, endpoint_m=.005,
            ))
        return {
            "success": success,
            "status": "success" if success else "preferred_trajectory_rejected",
            "failure_category": None if success else "constraint_violation",
            "message": (None if success else
                        "候选阶段轨迹与实际起点、目标、直线性或当前碰撞世界不一致"),
            "joint_names": list(names),
            "positions_rad": positions.tolist(),
            "interpolation_dt_s": float(steps[0]),
            "trajectory_source": "selected_candidate_stage_branch",
            "preferred_trajectory_validation": checks,
            "collision_verification": validation,
            **({"linear_path_metrics": checks["linear_path_metrics"]}
               if "linear_path_metrics" in checks else {}),
        }

    def close(self) -> None:
        """调用方结束使用后可显式释放所有solver及CUDA Graph。"""
        self.planner_pool.clear()

    def _report_failure(self, stage: str, kind: str, details: dict) -> str | None:
        if self.on_failure is None:
            return None
        try:
            self.on_failure(stage, kind, details)
        except Exception as error:
            return f"{type(error).__name__}: {error}"
        return None

    def __call__(self, stage: str, target: np.ndarray, planned_state_id: str) -> dict:
        return self._run(stage, target, planned_state_id)

    def plan_only(self, stage: str, target: np.ndarray, planned_state_id: str) -> dict:
        """完成同样的规划和几何复核，但不向 runtime 发出执行命令。"""
        return self._run(stage, target, planned_state_id, execute=False)

    def plan_and_execute_candidates(
        self, stage: str, targets, planned_state_id: str,
    ) -> dict:
        """先按顺序筛完整路径，再对首个可行 pose 规划并执行阶段路径。

        这对应旧 Pick 的 post-pick 语义。筛选与最终规划共用同一实际
        起点、附着代理和碰撞快照；最终直线失败时不换下一个候选。
        """
        if stage not in self.linear_tcp_stages:
            raise ValueError("候选完整路径筛选当前只用于最终 TCP 直线阶段")
        candidates = tuple(np.asarray(target, dtype=float) for target in targets)
        if not candidates or any(target.shape != (4, 4) or not np.isfinite(target).all()
                                 for target in candidates):
            raise ValueError("候选目标必须是非空的有限 4x4 pose 列表")
        return self._run(stage, candidates[0], planned_state_id, candidates=candidates)

    def plan_only_candidates(self, stage: str, targets, planned_state_id: str) -> dict:
        """不执行运动，按顺序筛选第一个具有完整路径的目标。"""
        candidates = tuple(np.asarray(target, dtype=float) for target in targets)
        if not candidates or any(target.shape != (4, 4) or not np.isfinite(target).all()
                                 for target in candidates):
            raise ValueError("候选目标必须是非空的有限 4x4 pose 列表")
        return self._run(
            stage, candidates[0], planned_state_id,
            candidates=candidates, execute=False,
        )

    def plan_only_candidate_sequences(
        self, stage: str, sequences, planned_state_id: str,
    ) -> dict:
        """从同一实际起点逐段筛选候选，并保留选中候选的关节分支。"""
        normalized = []
        for sequence in sequences:
            rows = tuple(
                (str(segment), np.asarray(target, dtype=float))
                for segment, target in sequence
            )
            if (not rows or any(target.shape != (4, 4)
                                or not np.isfinite(target).all()
                                for _, target in rows)):
                raise ValueError("候选序列必须包含具名的有限 4x4 pose")
            normalized.append(rows)
        if not normalized:
            raise ValueError("候选序列不能为空")
        return self._run(
            stage, normalized[0][-1][1], planned_state_id,
            candidate_sequences=tuple(normalized), execute=False,
        )

    def _execute_rejected_prefix(
        self, stage: str, plan: dict, planned_state_id: str, attachment,
        *, end_index: int | None, execute: bool,
    ) -> dict | None:
        """在仿真中执行已验证轨迹的诊断前缀，仍保持原阶段判负。"""
        if not self.execute_rejected_plan_prefixes or not execute or end_index is None:
            return None
        positions = plan.get("positions_rad") or []
        names = plan.get("joint_names") or []
        dt = float(plan.get("interpolation_dt_s") or 0.)
        if (len(positions) < 2 or not names or not np.isfinite(dt) or dt <= 0):
            return None
        end = min(max(int(end_index), 1), len(positions) - 1)
        prefix = positions[:end + 1]
        time_scale = float(self.execution_time_scale_by_stage.get(stage, 1.0))
        times = (np.arange(len(prefix), dtype=float) * dt * time_scale).tolist()
        args = dict(
            group=self.group, joint_names=names, times_s=times,
            positions_rad=prefix, planned_state_id=planned_state_id,
            timeout_s=max(times[-1] + self.settle_timeout_s + 1.0, 1.0),
        )
        if stage in self.execution_position_kp_scale_by_stage:
            args["position_kp_scale"] = float(
                self.execution_position_kp_scale_by_stage[stage])
        if self.replay_object_names is not None:
            args["replay_object_names"] = self.replay_object_names
            args["replay_attachment"] = attachment
        if self.endpoint_tolerance_rad is not None:
            args.update(
                endpoint_tolerance_rad=self.endpoint_tolerance_rad,
                settle_timeout_s=self.settle_timeout_s,
            )
        if self.visual_recorder is not None:
            self.visual_recorder.begin_stage(stage, self.runtime.model, self.runtime.data)
            args["step_observer"] = self.visual_recorder.observer(stage)
        if self.execution_stop_condition_for_stage is not None:
            args["stop_condition"] = (
                lambda model, data, sample: self.execution_stop_condition_for_stage(
                    stage, model, data, sample,
                )
            )
        try:
            result = self.runtime.execute_joint_trajectory(**args)
        finally:
            if self.visual_recorder is not None:
                self.visual_recorder.end_stage(stage, self.runtime.model, self.runtime.data)
        result = dict(result)
        result.update({
            "diagnostic_only": True,
            "original_plan_status": plan.get("status"),
            "executed_prefix_end_index": end,
            "executed_waypoint_count": len(prefix),
            "requested_waypoint_count": len(positions),
        })
        _write_json(self.output / "executions" / f"{stage}.json", result)
        return result

    def _run(self, stage: str, target: np.ndarray, planned_state_id: str,
             *, candidates: tuple[np.ndarray, ...] | None = None,
             candidate_sequences=None,
             execute: bool = True) -> dict:
        observation = self.runtime.query(robot_groups=(self.group,))
        if observation["state_id"] != planned_state_id:
            raise ValueError(f"{stage} 收到过期 planned_state_id")
        robot_only = resolve_collision_mode() is CollisionMode.ROBOT_ONLY
        attachment = (self.attachment_for_stage(stage, observation)
                      if self.attachment_for_stage is not None and not robot_only else None)
        if attachment is not None:
            if not attachment.get("physical_hold_evidence"):
                raise ValueError(f"{stage} 附着代理缺少实际夹持证据")
            if not attachment.get("object_instance_id"):
                raise ValueError(f"{stage} 附着代理缺少物体实例ID")
            _write_json(self.output / "attachments" / f"{stage}.json", attachment)
        # 接触阶段可以暂时从规划场景中移除目标物体，但必须在执行前
        # 用完整的 MuJoCo 几何复核轨迹，避免掌部或非指垫穿入物体。
        disabled_bodies = (tuple(self.disabled_body_names_for_stage(stage, observation))
                           if self.disabled_body_names_for_stage is not None and not robot_only else ())
        export_options = ({"transferred_objects": {
            attachment["object_instance_id"]: attachment["spheres_world_m"]}}
            if attachment is not None else {})
        if disabled_bodies:
            export_options["disabled_body_names"] = disabled_bodies
        snapshot = export_collision_scene(
            self.runtime.model, self.runtime.data, self.runtime.robot_metadata,
            self.group, self.output / "collision" / stage,
            **export_options,
        )
        if self.snapshot_filter_for_stage is not None and not robot_only:
            snapshot = self.snapshot_filter_for_stage(stage, snapshot)
            if not isinstance(snapshot, dict) or not snapshot.get("snapshot_sha256"):
                raise ValueError("阶段碰撞筛选必须返回带 snapshot_sha256 的完整快照")
        if stage in self.robot_body_only_stages:
            # 旧 Extract 的 ignore_world 仍保留机器人基座代理，不等于
            # Insert 的完全空 world；先保存完整来源再派生可审计快照。
            snapshot = filter_collision_snapshot(
                snapshot, self.output / "collision" / f"{stage}_robot_body_only",
                keep_body_names={"robot_body"}, rejected_body_names=None,
                attached_object_id=(attachment["object_instance_id"]
                                    if attachment is not None else None),
                scope="旧取出 Skill 忽略普通场景，仅保留 robot_body；保留自碰撞",
            )
        if stage in self.self_collision_only_stages:
            # 与旧版 Insert 的 ignore_scene_collision 一致：保留刚导出的
            # 完整世界快照作为审计依据，再派生独立的仅自身碰撞快照供规划。
            # MuJoCo 中的实际物理接触不受此规划范围切换影响。
            snapshot = export_self_collision_only_snapshot(
                snapshot, self.output / "collision" / f"{stage}_self_only",
                attached_object_id=(attachment["object_instance_id"]
                                    if attachment is not None else None),
            )
        settings = load_realman_motion_config(
            self.group, self.runtime.robot_metadata,
            gripper_snapshot=snapshot.get("gripper_snapshot"),
        )
        if attachment is not None:
            spheres = np.asarray(attachment["spheres_world_m"], dtype=float)
            world_from_tcp = np.asarray(observation["robots"][self.group]["T_world_tcp"], dtype=float)
            world_from_tool = world_from_tcp @ np.linalg.inv(np.asarray(settings["link_from_tcp"]))
            tool_from_world = np.linalg.inv(world_from_tool)
            centers = (tool_from_world[:3, :3] @ spheres[:, :3].T).T + tool_from_world[:3, 3]
            settings["attachment_spheres_tool"] = np.column_stack((centers, spheres[:, 3])).tolist()
        policy = snapshot["mounting_contact_policy"]
        if self.contact_policy_for_stage is not None and not robot_only:
            policy = self.contact_policy_for_stage(stage, snapshot)
            if not isinstance(policy, dict):
                raise ValueError("阶段接触策略必须返回显式 dict")
        _write_json(self.output / "contact_policies" / f"{stage}.json", policy)
        lease = self.planner_pool.acquire(
            settings, {**snapshot, "mounting_contact_policy": policy},
            planner_mode=self.planner_mode,
            **({'optimizer_collision_activation_distance': float(
                self.optimizer_collision_activation_distance_by_stage.get(stage, .01))}
               if self.optimizer_collision_activation_distance_by_stage else {}),
        )
        planner = lease.planner
        robot = observation["robots"][self.group]
        plan_method = (planner.plan_pivot_tcp if stage in self.pivot_tcp_stages else
                       planner.plan_linear_tcp if stage in self.linear_tcp_stages else
                       planner.plan)
        # 未指定重试时沿用规划器自己的默认值，避免向只实现基础接口的
        # planner 传入额外关键字参数；显式配置的阶段才启用多次尝试。
        attempt_options = ({"max_attempts": int(self.max_attempts_by_stage[stage])}
                           if stage in self.max_attempts_by_stage else {})
        if stage in self.pivot_tcp_stages:
            attempt_options.update(self.pivot_tcp_stages[stage])
        candidate_screen = None
        selected_index = None
        if candidate_sequences is not None:
            candidate_screen = []
            selected_trajectories = None
            for index, sequence in enumerate(candidate_sequences):
                current_names = list(robot["joint_names"])
                current_positions = list(robot["joint_positions_rad"])
                segment_rows = []
                trajectories = {}
                usable = True
                for segment, pose in sequence:
                    method = (planner.plan_linear_tcp
                              if segment in self.linear_tcp_stages else planner.plan)
                    segment_attempts = (
                        {"max_attempts": int(self.max_attempts_by_stage[segment])}
                        if segment in self.max_attempts_by_stage else {})
                    trial = method(
                        current_names, current_positions, pose,
                        target_frame="world_tcp",
                        diagnostic_label=f"{stage}_candidate_{index:03d}_{segment}",
                        **segment_attempts,
                    )
                    postcheck = None
                    if (trial.get("success") and self.plan_postcheck_for_stage is not None
                            and not robot_only):
                        postcheck = self.plan_postcheck_for_stage(
                            segment, trial, snapshot)
                        if postcheck is not None and not postcheck.get("success"):
                            trial = dict(trial)
                            trial.update(
                                success=False,
                                status=postcheck.get(
                                    "status", "pre_execution_task_constraint_rejected"),
                                failure_category=postcheck.get(
                                    "failure_category", "task_constraint_postcheck_rejected"),
                                pre_execution_check=postcheck,
                            )
                    segment_rows.append({
                        "stage": segment,
                        "target_pose_world": pose.tolist(),
                        "result": trial,
                    })
                    if not trial.get("success") or len(trial.get("positions_rad") or []) < 2:
                        usable = False
                        break
                    dt = float(trial.get("interpolation_dt_s") or 0.)
                    positions = trial["positions_rad"]
                    names = trial.get("joint_names") or current_names
                    if not np.isfinite(dt) or dt <= 0:
                        usable = False
                        break
                    trajectories[segment] = {
                        "joint_names": list(names),
                        "times_s": (np.arange(len(positions), dtype=float) * dt).tolist(),
                        "positions_rad": positions,
                    }
                    current_names = list(names)
                    current_positions = list(positions[-1])
                candidate_screen.append({
                    "index": index,
                    "usable_stage_sequence": usable,
                    "segments": segment_rows,
                })
                if usable:
                    selected_index = index
                    selected_trajectories = trajectories
                    break
            plan = {
                "success": selected_index is not None,
                "status": ("success" if selected_index is not None
                           else "candidate_sequence_no_full_path"),
                "failure_category": (None if selected_index is not None
                                     else "candidate_screen_failed"),
                "candidate_screen": candidate_screen,
                "selected_candidate_index": selected_index,
                "selected_stage_trajectories": selected_trajectories,
                "candidate_screen_scene_sha256": snapshot["snapshot_sha256"],
            }
            _write_json(self.output / "plans" / f"{stage}.json", plan)
            result = {"success": bool(plan["success"]), "plan": plan,
                      "execution": None, "evidence_mode": "planning_only"}
            if not plan["success"]:
                error = self._report_failure(stage, "planner_failed", plan)
                if error:
                    result["failure_capture_error"] = error
            return result
        if candidates is not None:
            candidate_screen = []
            for index, pose in enumerate(candidates):
                # 旧 Pick 先用不要求直线的完整路径确定首个可行目标。
                # 这里只筛目标，机械臂尚未运动；失败结果保留供诊断。
                trial = planner.plan(
                    robot["joint_names"], robot["joint_positions_rad"], pose,
                    target_frame="world_tcp",
                    diagnostic_label=f"{stage}_candidate_{index:03d}",
                    **attempt_options,
                )
                usable = bool(trial.get("success") and len(trial.get("positions_rad") or []) >= 2)
                candidate_screen.append({
                    "index": index, "target_pose_world": pose.tolist(),
                    "usable_full_path": usable, "result": trial,
                })
                if usable:
                    target = pose
                    selected_index = index
                    break
            if selected_index is None:
                plan = {
                    "success": False, "status": "postpick_candidates_no_full_path",
                    "failure_category": "candidate_screen_failed",
                    "candidate_screen": candidate_screen,
                    "candidate_screen_scene_sha256": snapshot["snapshot_sha256"],
                }
                _write_json(self.output / "plans" / f"{stage}.json", plan)
                result = {"success": False, "plan": plan, "execution": None}
                error = self._report_failure(stage, "planner_failed", plan)
                if error:
                    result["failure_capture_error"] = error
                return result
        if stage in self.preferred_trajectory_by_stage:
            plan = self._preferred_plan(stage, planner, robot, target)
        elif stage in self.approximate_ik_by_stage:
            plan = plan_with_approximate_ik(
                planner, robot["joint_names"], robot["joint_positions_rad"], target,
                budget=self.approximate_ik_by_stage[stage], target_frame="world_tcp",
                linear_tcp=stage in self.linear_tcp_stages,
                max_attempts=attempt_options.get("max_attempts"),
            )
        else:
            plan = plan_method(
                robot["joint_names"], robot["joint_positions_rad"], target,
                target_frame="world_tcp",
                diagnostic_label=stage,
                **attempt_options,
            )
        plan["planner_cache"] = {
            "hit": lease.cache_hit, "setup_s": lease.setup_duration_s,
            "rebuild_reason": lease.rebuild_reason,
        }
        plan.update(collision_mode_metadata())
        if robot_only:
            plan["pre_execution_check"] = {"status": "skipped_by_collision_mode",
                                            "environment_geometry_checked": False}
        if candidate_screen is not None:
            plan["candidate_screen"] = candidate_screen
            plan["selected_candidate_index"] = selected_index
            plan["candidate_screen_scene_sha256"] = snapshot["snapshot_sha256"]
        if attachment is not None:
            plan["attachment_proxy"] = {
                "object_instance_id": attachment["object_instance_id"],
                "spheres_world_m": attachment["spheres_world_m"],
                "spheres_tool_m": settings["attachment_spheres_tool"],
                "physical_hold_evidence": attachment["physical_hold_evidence"],
                "coverage": attachment.get("coverage"),
            }
        _write_json(self.output / "plans" / f"{stage}.json", plan)
        if not plan.get("success"):
            diagnostic_end = None
            if (plan.get("failure_category") == "constraint_violation"
                    and (plan.get("collision_verification") or {}).get("valid") is True):
                diagnostic_end = len(plan.get("positions_rad") or []) - 1
            elif plan.get("failure_category") == "collision_postcheck_rejected":
                invalid = (plan.get("collision_verification") or {}).get(
                    "first_invalid_index")
                if invalid is not None:
                    diagnostic_end = max(1, int(invalid) - 1)
            diagnostic = self._execute_rejected_prefix(
                stage, plan, planned_state_id, attachment,
                end_index=diagnostic_end, execute=execute,
            )
            result = {"success": False, "plan": plan, "execution": diagnostic}
            error = self._report_failure(stage, "planner_failed", plan)
            if error:
                result["failure_capture_error"] = error
            return result
        positions = plan.get("positions_rad") or []
        dt = float(plan.get("interpolation_dt_s", 0.0))
        if len(positions) < 2 or not np.isfinite(dt) or dt <= 0:
            raise ValueError(f"{stage} planner 成功但没有有效轨迹")
        if stage in self.max_joint_travel_rad_by_stage:
            plan_names = tuple(plan.get("joint_names") or ())
            robot_names = tuple(robot["joint_names"])
            if not plan_names or len(set(plan_names)) != len(plan_names):
                raise ValueError(f"{stage} 轨迹缺少唯一的具名关节")
            start_by_name = dict(zip(
                robot_names, robot["joint_positions_rad"], strict=True,
            ))
            if any(name not in start_by_name for name in plan_names):
                raise ValueError(f"{stage} 轨迹关节不属于当前实际机器人状态")
            path = np.asarray(positions, dtype=float)
            if (path.ndim != 2 or path.shape[1] != len(plan_names)
                    or not np.isfinite(path).all()):
                raise ValueError(f"{stage} 轨迹必须是有限 waypoint × joint 数组")
            start = np.asarray([start_by_name[name] for name in plan_names], dtype=float)
            per_joint = np.max(np.abs(path - start[None, :]), axis=0)
            maximum_index = int(np.argmax(per_joint))
            maximum = float(per_joint[maximum_index])
            limit = self.max_joint_travel_rad_by_stage[stage]
            travel_check = {
                "success": bool(maximum <= limit + 1e-12),
                "reason_code": (None if maximum <= limit + 1e-12
                                else "stage_joint_travel_exceeded"),
                "maximum_joint_travel_rad": maximum,
                "maximum_joint_travel_deg": float(np.rad2deg(maximum)),
                "maximum_joint_name": plan_names[maximum_index],
                "per_joint_travel_rad": {
                    name: float(value) for name, value in zip(plan_names, per_joint)
                },
                "limit_rad": limit,
                "limit_deg": float(np.rad2deg(limit)),
                "source": "planned_joint_path_relative_to_actual_stage_start",
            }
            plan["joint_travel_check"] = travel_check
            if not travel_check["success"]:
                first_exceeded = int(np.argmax(
                    np.max(np.abs(path - start[None, :]), axis=1) > limit + 1e-12))
                travel_check["first_exceeded_waypoint_index"] = first_exceeded
                plan.update(
                    success=False,
                    status="joint_travel_rejected",
                    failure_category="joint_travel_exceeded",
                    message="轨迹需要过大的单关节反向运动，未下发执行",
                )
                _write_json(self.output / "plans" / f"{stage}.json", plan)
                diagnostic = self._execute_rejected_prefix(
                    stage, plan, planned_state_id, attachment,
                    end_index=first_exceeded, execute=execute,
                )
                result = {"success": False, "plan": plan, "execution": diagnostic}
                error = self._report_failure(
                    stage, "joint_travel_postcheck_rejected", plan,
                )
                if error:
                    result["failure_capture_error"] = error
                return result
        if self.plan_postcheck_for_stage is not None and not robot_only:
            postcheck = self.plan_postcheck_for_stage(stage, plan, snapshot)
            if postcheck is not None:
                if not isinstance(postcheck, dict) or not isinstance(postcheck.get("success"), bool):
                    raise ValueError("规划执行前复核必须返回带布尔 success 的字典或 None")
                plan["pre_execution_check"] = postcheck
                if not postcheck["success"]:
                    plan["success"] = False
                    plan["status"] = postcheck.get(
                        "status", "pre_execution_contact_rejected")
                    plan["failure_category"] = postcheck.get(
                        "failure_category", "collision_postcheck_rejected")
                    contacts = postcheck.get("forbidden_contacts") or []
                    invalid = min(
                        (int(row["waypoint_index"]) for row in contacts
                         if row.get("waypoint_index") is not None),
                        default=int(postcheck.get("waypoint_index") or 0),
                    )
                    try:
                        plan["failure_visualization"] = planner.export_pre_execution_failure(
                            positions, target,
                            first_invalid_waypoint=invalid,
                            diagnostic_label=f"{stage}_pre_execution_check",
                            precheck=postcheck,
                        )
                    except Exception as visualization_error:
                        plan["failure_visualization_error"] = (
                            f"{type(visualization_error).__name__}: {visualization_error}"
                        )
                    _write_json(self.output / "plans" / f"{stage}.json", plan)
                    prefix_end = (max(1, invalid - 1) if contacts else invalid)
                    diagnostic = self._execute_rejected_prefix(
                        stage, plan, planned_state_id, attachment,
                        end_index=prefix_end, execute=execute,
                    )
                    result = {"success": False, "plan": plan,
                              "execution": diagnostic}
                    error = self._report_failure(
                        stage, plan["failure_category"], postcheck)
                    if error:
                        result["failure_capture_error"] = error
                    return result
                _write_json(self.output / "plans" / f"{stage}.json", plan)
        if not execute:
            return {"success": True, "plan": plan, "execution": None,
                    "evidence_mode": "planning_only"}
        time_scale = float(self.execution_time_scale_by_stage.get(stage, 1.0))
        plan["execution_time_scale"] = time_scale
        _write_json(self.output / "plans" / f"{stage}.json", plan)
        times = (np.arange(len(positions), dtype=float) * dt * time_scale).tolist()
        execution_args = dict(
            group=self.group, joint_names=plan["joint_names"], times_s=times,
            positions_rad=positions, planned_state_id=planned_state_id,
            timeout_s=max(times[-1] + self.settle_timeout_s + 1.0, 1.0),
        )
        if stage in self.execution_position_kp_scale_by_stage:
            gain_scale = float(self.execution_position_kp_scale_by_stage[stage])
            execution_args['position_kp_scale'] = gain_scale
            plan['execution_position_kp_scale'] = gain_scale
            _write_json(self.output / 'plans' / f'{stage}.json', plan)
        if self.replay_object_names is not None:
            execution_args["replay_object_names"] = self.replay_object_names
            execution_args["replay_attachment"] = attachment
        if self.endpoint_tolerance_rad is not None:
            execution_args.update(
                endpoint_tolerance_rad=self.endpoint_tolerance_rad,
                settle_timeout_s=self.settle_timeout_s,
            )
        if self.visual_recorder is not None:
            self.visual_recorder.begin_stage(stage, self.runtime.model, self.runtime.data)
            execution_args["step_observer"] = self.visual_recorder.observer(stage)
        if self.execution_stop_condition_for_stage is not None:
            execution_args["stop_condition"] = (
                lambda model, data, sample: self.execution_stop_condition_for_stage(
                    stage, model, data, sample,
                )
            )
        execution = self.runtime.execute_joint_trajectory(**execution_args)
        if self.visual_recorder is not None:
            self.visual_recorder.end_stage(stage, self.runtime.model, self.runtime.data)
        failure_type = str(execution.get("failure_type") or "")
        if (not execution.get("success")
                and ("collision" in failure_type or "contact" in failure_type)):
            try:
                stopped = self.runtime.query(robot_groups=(self.group,))
                stopped_robot = stopped["robots"][self.group]
                by_name = dict(zip(
                    stopped_robot["joint_names"],
                    stopped_robot["joint_positions_rad"],
                    strict=True,
                ))
                stopped_positions = [by_name[name] for name in plan["joint_names"]]
                execution["failure_visualization"] = planner.export_execution_collision(
                    stopped_positions, target,
                    diagnostic_label=f"{stage}_execution_collision",
                    execution_failure={
                        "failure_type": execution.get("failure_type"),
                        "failure_step_index": execution.get("failure_step_index"),
                        "failure_evidence": execution.get("failure_evidence"),
                        "end_state_id": execution.get("end_state_id"),
                    },
                )
            except Exception as visualization_error:
                execution["failure_visualization_error"] = (
                    f"{type(visualization_error).__name__}: {visualization_error}"
                )
        _write_json(self.output / "executions" / f"{stage}.json", execution)
        result = {"success": bool(execution.get("success")), "plan": plan,
                  "execution": execution}
        approximation = plan.get("approximate_ik")
        if result["success"] and approximation and approximation.get("approximate"):
            # 规划通过只证明替代目标存在轨迹；阶段是否达成仍取实际执行后 TCP。
            after = self.runtime.query(robot_groups=(self.group,))
            actual = after["robots"][self.group].get("T_world_tcp")
            requested = np.asarray(approximation["requested_tcp_pose_world"], dtype=float)
            budget = approximation["budget"]
            if actual is None:
                postcheck = {"success": False, "reason": "actual_tcp_unavailable"}
            else:
                measured = np.asarray(actual, dtype=float)
                if measured.shape != (4, 4) or not np.isfinite(measured).all():
                    postcheck = {"success": False, "reason": "actual_tcp_invalid"}
                else:
                    position_error, rotation_error = _pose_error(requested, measured)
                    postcheck = {
                        "success": position_error <= budget["max_position_error_m"]
                                   and rotation_error <= budget["max_rotation_error_rad"],
                        "actual_state_id": after["state_id"],
                        "actual_tcp_pose_world": measured.tolist(),
                        "position_error_m": position_error,
                        "rotation_error_rad": rotation_error,
                        "budget": budget,
                    }
            result["approximate_ik_postcheck"] = postcheck
            result["success"] = bool(postcheck["success"])
            _write_json(self.output / "executions" / f"{stage}.json", result)
            if not result["success"]:
                error = self._report_failure(stage, "approximate_ik_postcheck_failed", postcheck)
                if error:
                    result["failure_capture_error"] = error
        if not execution.get("success"):
            error = self._report_failure(stage, "execution_failed", execution)
            if error:
                result["failure_capture_error"] = error
        return result


__all__ = ["CuroboRuntimeBridge"]
