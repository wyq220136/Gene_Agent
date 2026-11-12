"""后端无关 MotionRequest/Result 与同步 cuRobo 字典接口之间的适配。"""
from __future__ import annotations
from typing import Sequence, TYPE_CHECKING
from dataclasses import replace
import time
import numpy as np
from .approximate import ApproximateIKBudget
from ..model import (JointState as MotionJointState, JointTrajectory, MotionFailure,
                     MotionFailureKind, MotionOperation, MotionRequest, MotionResult)
if TYPE_CHECKING:
    from .core import SynchronousPlanner


class CuroboPlannerAdapter:
    """把现有同步 cuRobo 基线映射到后端无关的 Motion 契约。

    适配器只做输入一致性检查和结果归一化，不实现 Skill fallback。当前基线
    接触允许项必须与已构建的局部碰撞复核策略完全一致。
    """

    def __init__(self, planner: SynchronousPlanner, *, group: str, scene_revision: int, request_output_dir=None):
        self.planner = planner
        self.group = str(group)
        self.scene_revision = int(scene_revision)
        self.request_output_dir = request_output_dir

    def _record(self, requests, results, *, invocation=None):
        if self.request_output_dir is None:
            return tuple(results)
        started = time.perf_counter()
        try:
            from .diagnostics import save_planning_request
            path = save_planning_request(self.planner, requests, results,
                getattr(self.planner, 'diagnostic_snapshot', {}), self.request_output_dir, invocation=invocation)
            extra = dict(replay_request_ref=str(path.resolve()),request_recording_s=time.perf_counter()-started)
        except Exception as exc:
            # 记录出错不能把原成功/失败改写；缺少现场在摘要里明确保留。
            extra = dict(request_recording_error=f'{type(exc).__name__}: {exc}')
        return tuple(replace(result,metrics={**result.metrics,**extra}) for result in results)

    def fk(self, state: MotionJointState) -> np.ndarray:
        return self.planner.fk(state.joint_names, state.positions_rad)

    def _invalid(self, request: MotionRequest, summary: str) -> MotionResult:
        return MotionResult(
            request_id=request.request_id,
            operation=request.operation,
            success=False,
            backend="curobo",
            planner_mode=request.planner_mode,
            duration_s=0.0,
            failure=MotionFailure(MotionFailureKind.INVALID_REQUEST, summary),
        )

    def _backend_error(self, request: MotionRequest, summary: str) -> MotionResult:
        """后端违反返回协议时保留正确归因，不把责任推给请求调用者。"""
        return MotionResult(
            request_id=request.request_id,
            operation=request.operation,
            success=False,
            backend="curobo",
            planner_mode=request.planner_mode,
            duration_s=0.0,
            failure=MotionFailure(MotionFailureKind.BACKEND_ERROR, summary),
        )

    def _validate(self, request: MotionRequest, expected: MotionOperation) -> MotionResult | None:
        if request.operation is not expected:
            return self._invalid(request, f"请求 operation 必须是 {expected.value}")
        if request.group != self.group:
            return self._invalid(request, "请求 robot group 与 cuRobo adapter 不一致")
        if tuple(request.start.joint_names) != tuple(self.planner.joint_names):
            return self._invalid(request, "起始关节名称或顺序与 cuRobo 配置不一致")
        if request.scene.revision != self.scene_revision:
            return self._invalid(request, "scene revision 与当前 cuRobo scene snapshot 不一致")
        if request.scene.snapshot_sha256 != self.planner.scene_snapshot_sha256:
            return self._invalid(request, "scene snapshot hash 与当前 cuRobo scene snapshot 不一致")
        if request.scene.scope != self.planner.scene_scope:
            return self._invalid(request, "scene scope 与当前 cuRobo scene snapshot 不一致")
        if request.planner_mode is not self.planner.planner_mode:
            return self._invalid(request, "planner mode 与 cuRobo 实例不一致")
        if request.seed != self.planner.random_seed:
            return self._invalid(request, "random seed 与 cuRobo 实例不一致")
        policy = request.collision_policy
        conservative_grasp = (
            request.operation is MotionOperation.LINEAR_TCP
            and policy.policy_id == "finger_object_contact"
            and bool(policy.allowances)
            and all(item.first == "gripper_fingers" for item in policy.allowances)
        )
        effective=getattr(self.planner,'mounting_contact_policy',{}) or {}
        declared=effective.get('reason')
        local_contact=False
        if isinstance(declared,dict) and 'allowances' in declared:
            local_contact=(declared==policy.to_dict() and bool(policy.allowances)
                and set(effective.get('robot_links',()))=={p.first for p in policy.allowances}
                and bool(effective.get('obstacle_names')) and getattr(self.planner,'_validator',None) is not None)
            if not local_contact:return self._invalid(request,'请求与场景的局部接触复核策略不一致')
        if not conservative_grasp and not local_contact and (policy.policy_id != "strict" or policy.allowances):
            return self._invalid(request, "当前 cuRobo 基线只支持 strict 空碰撞允许表")
        return None

    @staticmethod
    def _failure_kind(operation: MotionOperation, category: str | None) -> MotionFailureKind:
        if category == "start_collision":
            return MotionFailureKind.START_COLLISION
        if category == "input_validation_error":
            return MotionFailureKind.INVALID_REQUEST
        if category == "collision_postcheck_rejected":
            return (MotionFailureKind.GOAL_COLLISION if operation is MotionOperation.IK
                    else MotionFailureKind.PATH_COLLISION)
        explicit = {
            "goal_ik_no_solution": MotionFailureKind.GOAL_IK_NO_SOLUTION,
            "ik_pose_not_converged": MotionFailureKind.GOAL_IK_NO_SOLUTION,
            "ik_constraint_rejected": MotionFailureKind.CONSTRAINT_VIOLATION,
            "goal_collision": MotionFailureKind.GOAL_COLLISION,
            "path_not_found": MotionFailureKind.PATH_NOT_FOUND,
            "path_collision": MotionFailureKind.PATH_COLLISION,
            "constraint_violation": MotionFailureKind.CONSTRAINT_VIOLATION,
        }
        if category in explicit:
            return explicit[category]
        if category == "solver_rejected":
            # 当前 cuRobo 原始状态不区分“纯几何无解”和“碰撞/约束导致无解”。
            # 不做隐藏的无碰撞二次求解，也不把未知原因伪装成某个精确类别。
            return MotionFailureKind.CONSTRAINT_VIOLATION
        return MotionFailureKind.BACKEND_ERROR

    def _convert(self, request: MotionRequest, raw: dict) -> MotionResult:
        try:
            duration = float(raw.get("duration_s", 0.0))
        except (TypeError, ValueError):
            return self._backend_error(request, "cuRobo backend 返回了非法 duration_s")
        if not np.isfinite(duration) or duration < 0.0:
            return self._backend_error(request, "cuRobo backend 返回了非法 duration_s")
        metrics = {
            "backend_status": str(raw.get("status", "unknown")),
            "scene_scope": self.planner.scene_scope,
            "scene_snapshot_sha256": self.planner.scene_snapshot_sha256,
        }
        for key in ("collision_mode", "environment_collision_enabled", "planning_attachment_enabled",
                    "self_collision_check", "collision_verification", "start_state_check",
                    "planner_cache", "timing", "pivot_path_metrics"):
            if key in raw:
                metrics[key] = raw[key]
        if raw.get("linear_path_metrics") is not None:
            metrics["linear_path_metrics"] = raw["linear_path_metrics"]
            metrics["linear_path_limits"] = {
                "lateral_m": .005, "backward_step_m": .001,
                "cumulative_backward_m": .002, "overshoot_m": .002,
                "endpoint_m": .005,
                "goal_orientation_rad": .05,
            }
        if request.collision_policy.policy_id == "finger_object_contact":
            metrics["effective_collision_policy"] = "strict_conservative"
            metrics["collision_policy_note"] = (
                "请求允许指尖接触，但本次仍检查全部规划碰撞；通过仅证明当前代理下严格无碰撞"
            )
        if raw.get("batch_total_duration_s") is not None:
            metrics["batch_total_duration_s"] = float(raw["batch_total_duration_s"])
            metrics["batch_size"] = int(raw["batch_size"])
        if raw.get("approximate_ik") is not None:
            metrics["approximate_ik"] = raw["approximate_ik"]
        if raw.get("success"):
            try:
                positions = np.asarray(raw.get("positions_rad"), dtype=float)
            except (TypeError, ValueError):
                return self._backend_error(request, "cuRobo backend 成功结果缺少合法关节解")
            if (positions.ndim != 2 or positions.shape[1] != len(self.planner.joint_names)
                    or len(positions) == 0 or not np.all(np.isfinite(positions))):
                return self._backend_error(request, "cuRobo backend 成功结果缺少合法关节解")
            goal = MotionJointState(tuple(self.planner.joint_names), tuple(positions[-1]))
            trajectory = None
            if request.operation in (MotionOperation.PLAN, MotionOperation.LINEAR_TCP, MotionOperation.PIVOT_TCP):
                try:
                    dt = float(raw.get("interpolation_dt_s", 0.0))
                except (TypeError, ValueError):
                    return self._backend_error(
                        request, "cuRobo backend 轨迹结果缺少合法 interpolation_dt_s"
                    )
                if not np.isfinite(dt) or dt <= 0.0:
                    return self._backend_error(
                        request, "cuRobo backend 轨迹结果缺少合法 interpolation_dt_s"
                    )
                trajectory = JointTrajectory(
                    tuple(self.planner.joint_names),
                    tuple(index * dt for index in range(len(positions))),
                    tuple(tuple(row) for row in positions),
                )
            return MotionResult(
                request.request_id, request.operation, True, "curobo", request.planner_mode,
                duration, goal_state=goal, trajectory=trajectory, metrics=metrics,
            )

        details = {
            "failure_category": raw.get("failure_category"),
            "raw_error": raw.get("raw_error"),
        }
        if raw.get("failure_category") == "solver_rejected":
            details["attribution"] = (
                "ambiguous: backend did not distinguish kinematic, collision, or path constraint"
            )
        if raw.get("solver_diagnostic") is not None:
            details["solver_diagnostic"] = raw["solver_diagnostic"]
        if raw.get("ik_diagnostic") is not None:
            details["ik_diagnostic"] = raw["ik_diagnostic"]
        if raw.get("ik_diagnostic_error") is not None:
            details["ik_diagnostic_error"] = raw["ik_diagnostic_error"]
        if raw.get("failure_detail") is not None:
            details["failure_detail"] = raw["failure_detail"]
        if raw.get("collision_verification") is not None:
            details["collision_verification"] = raw["collision_verification"]
        for key in ("approximate_ik_attempts", "approximate_ik_candidate_screen"):
            if key in raw:
                details[key] = raw[key]
        options = getattr(self.planner, 'diagnostic_options', None)
        if options is not None and options.mode == 'on_failure':
            try:
                from .diagnostics import collect_failure_diagnostics
                import json
                diagnostic = collect_failure_diagnostics(
                    self.planner, request, raw, getattr(self.planner, 'diagnostic_snapshot', {}), options)
                details['diagnostic'] = json.loads(json.dumps(diagnostic,default=lambda v:v.tolist(),allow_nan=False))
            except Exception as exc:
                details['diagnostic_error'] = f'{type(exc).__name__}: {exc}'
        summary = str(raw.get("message") or raw.get("status") or "cuRobo 规划失败")
        failure = MotionFailure(
            self._failure_kind(request.operation, raw.get("failure_category")),
            summary,
            backend_status=str(raw.get("status", "unknown")),
            details=details,
        )
        return MotionResult(
            request.request_id, request.operation, False, "curobo", request.planner_mode,
            duration, failure=failure, metrics=metrics,
        )

    def solve_ik(self, request: MotionRequest) -> MotionResult:
        invalid = self._validate(request, MotionOperation.IK)
        if invalid is not None:
            return invalid
        raw = self.planner.solve_ik(
            request.start.joint_names, request.start.positions_rad, request.target.matrix,
            target_frame=request.target.frame,
        )
        return self._record((request,), (self._convert(request, raw),))[0]

    def evaluate_candidates(self, requests: Sequence[MotionRequest]) -> tuple[MotionResult, ...]:
        """同一初态和场景的一批 IK 请求；输入顺序就是输出顺序。"""
        requests = tuple(requests)
        if not requests:
            return ()
        first = requests[0]
        ids = [item.request_id for item in requests]
        if len(ids) != len(set(ids)):
            raise ValueError("batch 候选 request_id 不能重复")
        for item in requests:
            if (item.operation is not MotionOperation.IK or item.group != first.group
                    or item.start != first.start or item.scene != first.scene
                    or item.target.frame != first.target.frame
                    or item.collision_policy != first.collision_policy
                    or item.seed != first.seed or item.planner_mode != first.planner_mode):
                raise ValueError("batch 候选必须共享 IK、group、起点、场景、目标 frame、碰撞策略和 seed/mode")
        errors = [self._validate(item, MotionOperation.IK) for item in requests]
        if any(error is not None for error in errors):
            raise ValueError("batch 请求与当前 planner 的 scene/group/seed/mode 不一致")
        raw = self.planner.solve_ik_batch(
            first.start.joint_names, first.start.positions_rad,
            [item.target.matrix for item in requests], target_frame=first.target.frame,
            diagnose_failures="when_no_success",
        )
        if len(raw) != len(requests):
            return tuple(self._backend_error(item, "cuRobo batch 结果数与候选数不一致") for item in requests)
        return self._record(requests, tuple(self._convert(item, result) for item, result in zip(requests, raw)), invocation="evaluate_candidates")

    def plan(self, request: MotionRequest) -> MotionResult:
        if request.operation not in (MotionOperation.PLAN, MotionOperation.LINEAR_TCP, MotionOperation.PIVOT_TCP):
            return self._invalid(request, "请求 operation 必须是 plan、linear_tcp 或 pivot_tcp")
        invalid = self._validate(request, request.operation)
        if invalid is not None:
            return invalid
        if request.operation is MotionOperation.PIVOT_TCP:
            method = getattr(self.planner, "plan_pivot_tcp", None)
            if method is None:
                return self._invalid(request, "当前backend不支持pivot_tcp")
            raw = method(request.start.joint_names, request.start.positions_rad,
                         request.target.matrix, target_frame=request.target.frame,
                         **request.pivot_limits.to_dict())
            return self._record((request,), (self._convert(request, raw),))[0]
        method = (self.planner.plan_linear_tcp if request.operation is MotionOperation.LINEAR_TCP
                  else self.planner.plan)
        if request.approximate_ik_budget is None:
            raw = method(
                request.start.joint_names, request.start.positions_rad, request.target.matrix,
                target_frame=request.target.frame,
            )
        else:
            from .core import plan_with_approximate_ik
            raw = plan_with_approximate_ik(
                self.planner, request.start.joint_names, request.start.positions_rad,
                request.target.matrix,
                budget=ApproximateIKBudget(**request.approximate_ik_budget),
                target_frame=request.target.frame,
                linear_tcp=request.operation is MotionOperation.LINEAR_TCP,
            )
        return self._record((request,), (self._convert(request, raw),))[0]
