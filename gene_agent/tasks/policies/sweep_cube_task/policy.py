"""按当前观测排序、归拢与有限重试的清扫策略；保留原决策行为。"""
import numpy as np

TARGET_BLOCKS = ("cube", "cube_red", "cube_blue")

class SweepPolicy:
    """清扫决策；operations 提供实际观测、动作、判据与记录，不持有 MuJoCo。

    参数仅使用 alignment_lateral_trigger_m（米）及 target_order_policy。
    run 返回操作是否完成；后端外层负责实际终态、完整评分与报告落盘。
    """
    def __init__(self, operations, *, alignment_lateral_trigger_m=.015,
                 target_order_policy="closest_first"):
        self.operations = operations
        self.alignment_lateral_trigger_m = float(alignment_lateral_trigger_m)
        self.target_order_policy = target_order_policy

    def _retry_target(self, target_name, stage_label, target_goal):
        # 盘外方块先走与首次清扫相同的侧向检查；盘内边缘只沿开口轴推进。
        if target_goal["tray_contact"]:
            return self.operations.sweep(target_name, stage_label, recovery_inside_pan=True)
        return self._sweep_target_with_alignment(target_name, stage_label)


    def _sweep_target_with_alignment(self, target_name, stage_label):
        actual = self.operations.observe()["objects"]
        pan = np.asarray(actual["dustpan"]["T_world_object"])
        cube = np.asarray(actual[target_name]["T_world_object"])
        local = pan[:3, :3].T @ (cube[:3, 3] - pan[:3, 3])
        lateral_trigger = self.alignment_lateral_trigger_m
        align_first = abs(local[1]) > lateral_trigger and local[0] > .082
        direct_preflight_failed = False
        if not align_first:
            before = self.operations.observe()["state_id"]
            if self.operations.sweep(target_name, stage_label, recovery_inside_pan=True):
                return True
            after = self.operations.observe()["state_id"]
            failure_info = self.operations.failure()
            failure = failure_info["stage"] or ""
            screen = failure_info["screen"]
            direct_preflight_failed = bool(
                abs(local[1]) > .015 and local[0] > .082
                and failure.endswith("__sequence_screen") and before == after
                and screen.get("evidence_mode") == "kinematic_preview_only"
                and screen.get("start_state_id") == after)
            if not direct_preflight_failed:
                return False
            # 只在无执行的纯预检失败后换归拢路线；原失败候选保留在 run 中。
            self.operations.clear_failure()
        if align_first or direct_preflight_failed:
            # 在铲口外先归拢；每段后使用真实位置，再生成下一次推扫。
            staging_local = local.copy()
            staging_local[0] = max(.14, staging_local[0])
            staging_local[1] = 0.
            staging = pan[:3, :3] @ staging_local + pan[:3, 3]
            if not self.operations.sweep(target_name, f"{stage_label}_align",
                                      cube_goal_world_xyz=staging):
                return False
            actual_cube = np.asarray(self.operations.observe()["objects"][target_name]["T_world_object"])
            aligned_local = pan[:3, :3].T @ (actual_cube[:3, 3] - pan[:3, 3])
            aligned = bool(abs(aligned_local[1]) <= .01)
            self.operations.record_alignment(f"{stage_label}_align", aligned_local, aligned)
            if not aligned:
                return False
        final_label = f"{stage_label}_after_align" if direct_preflight_failed else stage_label
        if not self.operations.sweep(target_name, final_label, recovery_inside_pan=True):
            return False
        return True


    def _sweep_pending_targets(self, order):
        """纯预检受阻时尝试其他方块；真实执行失败立即停，不跳过难样本。"""
        pending = list(order)
        attempts = {name: 0 for name in order}
        deferred_targets, executed_order, observed_completed = [], [], []
        self.operations.record_progress(deferred_targets=deferred_targets,
            executed_target_order=executed_order, observed_completed_targets=observed_completed)
        while pending:
            deferred = []
            progressed = False
            last_failure = None
            for name in pending:
                # 其他扫动可能已将本块顺带入盘；记录真实判据，不伪造执行。
                goal = self.operations.goals()["targets"][name]
                if goal["success"]:
                    observed_completed.append({
                        "target_name": name, "state_id": self.operations.observe()["state_id"], "goal": goal})
                    progressed = True
                    continue
                attempts[name] += 1
                label = name if attempts[name] == 1 else f"{name}_attempt{attempts[name]}"
                before = self.operations.observe()["state_id"]
                if self._sweep_target_with_alignment(name, label):
                    progressed = True
                    executed_order.append(name)
                    continue
                failure_info = self.operations.failure()
                failure = failure_info["stage"] or ""
                after = self.operations.observe()["state_id"]
                screen = failure_info["screen"]
                pure_preflight = (screen.get("evidence_mode") == "kinematic_preview_only"
                                  and screen.get("start_state_id") == after)
                if not failure.endswith("__sequence_screen") or (before != after and not pure_preflight):
                    return False
                # 只改尝试次序；所有失败候选和独立stage_label保留，后续重新观测。
                deferred_targets.append({"target_name": name,
                    "stage_label": label, "failure_stage": failure,
                    "state_id": after, "target_attempt_start_state_id": before,
                    "reason": "sequence_preflight_failed_without_execution"})
                deferred.append(name)
                last_failure = failure
                self.operations.clear_failure()
            if deferred and not progressed:
                self.operations.fail(last_failure)
                return False
            pending = deferred
        return True


    def run(self) -> bool:
        """抓刷→按观测排序→推扫/归拢→最多一次目标恢复→释放并检查。

        预检失败只在原证据允许时换顺序；真实执行失败立即停止，不擅自松爪。
        """
        if not self.operations.pick():
            return False
        current = self.operations.observe()["objects"]
        rotation = np.asarray(current["brush"]["T_world_object"])[:3, :3].copy()
        pan_xy = np.asarray(current["dustpan"]["T_world_object"])[:2, 3]
        order = sorted(TARGET_BLOCKS, key=lambda name: float(np.linalg.norm(
            np.asarray(current[name]["T_world_object"])[:2, 3] - pan_xy)))
        if self.target_order_policy == "farthest_first":
            order.reverse()
        self.operations.begin_sweeping(rotation, order)
        if not self._sweep_pending_targets(order):
            return False
        recovery_targets = []
        self.operations.record_progress(goal_after_first_pass=self.operations.goals(),
                                        recovery_targets=recovery_targets)
        for target_name in order:
            target_goal = self.operations.goals()["targets"][target_name]
            if target_goal["success"]:
                continue
            recovery_targets.append(target_name)
            if not self._retry_target(target_name, f"{target_name}_retry1", target_goal):
                return False
        if not self.operations.finish_sweeping():
            return False
        goal = self.operations.goals()
        self.operations.record_progress(goal=goal)
        if not goal["success"]:
            self.operations.fail("goal_postcondition")
            return False
        return True


def run(operations, *, alignment_lateral_trigger_m=.015,
        target_order_policy="closest_first") -> bool:
    """对外同步 policy 入口；已准备后端，不 reset、不管理设备授权。

    operations 能力见 SweepCubeRun 的公开方法；失败也保留后端实际状态与
    原始操作证据。完整 Task 报告由后端 run 的 finally 保存。
    """
    return SweepPolicy(operations,
        alignment_lateral_trigger_m=alignment_lateral_trigger_m,
        target_order_policy=target_order_policy).run()
