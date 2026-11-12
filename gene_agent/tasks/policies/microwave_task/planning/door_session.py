"""门 Skill 在线纯视觉会话：同源 Knowledge、场景编辑与同步规划。"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from datetime import datetime
from dataclasses import replace
import time
import numpy as np
import mujoco

from gene_agent.tasks.policies.microwave_task.planning.door_scene import door_stage_collision_options
from gene_agent.tasks.policies.microwave_task.operations.door import approximate_ik_budget_for, local_rotation_search_for
from gene_agent.tasks.policies.microwave_task.operations.door import build_door_knowledge, load_push_open_profile
from gene_agent.tasks.policies.microwave_task.planning.bowl import load_pick_bowl_case, prepare_pick_bowl_scene
from gene_agent.knowledge.geometry import validate_pose
from gene_agent.knowledge.operation.door_push import door_push_pose
from gene_agent.skills.library.door.sequence import (
    DoorPhase,
    door_phases,
)
from gene_agent.tasks.policies.microwave_task.policy import compose_task1_door_cycle
from gene_agent.motion.curobo.core import SynchronousPlanner, load_realman_motion_config
from gene_agent.motion.curobo.session import CuroboPlannerPool
from gene_agent.motion.curobo.scene import export_collision_scene
from gene_agent.motion.curobo.approximate import ApproximateIKBudget, plan_with_approximate_ik
from gene_agent.motion import PlannerMode
from gene_agent.feedback.motion_diagnostics import summarize_motion_failure
from gene_agent.skills.runtime.target_selection import plan_with_local_rotation_search
from gene_agent.visualization.skill_inspector.planned_timeline import (
    PlannedSegment,
    PlannedTimeline,
)
from gene_agent.visualization.grasp_tuner.microwave_scene import (
    set_preview_microwave_pose, _body_pose,
)

ROOT = Path(__file__).resolve().parents[5]
DEFAULT_CONFIG = ROOT / "data/task_1/skills/door_skill_debug.json"
TRANSITION_NAMES = {
    "pregrasp", "pre_contact", "retreat", "retract_up", "retract_local_z",
    "return_default_tcp",
}


def configure_curobo_run_log(output_dir: Path) -> Path:
    """把 cuRobo 的大量 warning 留在 run 内，终端只保留任务摘要。"""
    destination = Path(output_dir) / "curobo.log"
    destination.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("curobo")
    for handler in tuple(logger.handlers):
        if getattr(handler, "_gene_agent_door_log", False):
            logger.removeHandler(handler)
            handler.close()
    handler = logging.FileHandler(destination, encoding="utf-8")
    handler._gene_agent_door_log = True
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)
    logger.propagate = False
    return destination


def _read_state(path: str | None) -> dict:
    if not path:
        return {}
    payload = json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))
    if payload.get("input_valid") is False:
        raise ValueError(f"Knowledge 状态含有未通过校验的输入：{path}")
    return dict(payload.get("tuner_values") or {})


class DoorDebugSession:
    """GUI 和 CLI 调用同一会话；浏览阶段不会推进实际规划起点。"""

    def __init__(self, *, config_path: Path = DEFAULT_CONFIG, run_dir: Path | None = None,
                 planner_mode: PlannerMode | str | None = None):
        self.config_path = Path(config_path).resolve()
        self.config = json.loads(self.config_path.read_text(encoding="utf-8"))
        if self.config.get("schema_version") != 1:
            raise ValueError("门 Skill 配置版本必须为 1")
        # 在线 Skill/Task 默认追求热态交互速度，因此显式启用 CUDA Graph。
        # 调试 CUDA Graph 本身时仍可用构造参数切回 development，运行记录会
        # 保存实际模式，避免性能结果和诊断结果混淆。
        configured_mode = self.config.get(
            "planner_mode", PlannerMode.CUDA_GRAPH_BENCHMARK.value
        )
        self.planner_mode = PlannerMode(
            configured_mode if planner_mode is None else planner_mode
        )
        self.root = ROOT
        self.run_dir = Path(run_dir or self.config.get("scene_run", "runs/pick-bowl-planning-final"))
        if not self.run_dir.is_absolute():
            self.run_dir = self.root / self.run_dir
        # 干净检出没有配置指向的历史 runs：用仓库内现行资产构建同一
        # GT 场景，让门视觉界面可以直接启动。显式传入 run_dir 的调用方
        # 仍要求该目录存在；已有 run 也始终走原加载入口，保留它自己的
        # 场景采样、碰撞快照和诊断记录，不在这里覆盖或重建。
        self.case = (load_pick_bowl_case(self.run_dir)
                     if run_dir is not None or self.run_dir.exists()
                     else prepare_pick_bowl_scene(self.run_dir, seed=0))
        self.group = self.case.snapshot["active_group"]
        if self.group != "right":
            raise ValueError("本次 task_1 门 Knowledge 仅验证右臂；不能默认为另一臂可用")
        self.output_dir = self.root / "runs" / f"door-live-{datetime.now():%Y%m%d-%H%M%S}"
        self.revision = 0
        self.phase_index = 0
        self.plan_result: dict | None = None
        self.plan_revision: int | None = None
        self.planned_q: np.ndarray | None = None
        self._execution_diagnostic: dict | None = None
        self.timeline: PlannedTimeline | None = None
        self.timeline_selection: str | None = None
        self.animation_index = 0
        self.results: list[dict] = []
        # 完整 Skill/Task 规划拆成显式的小步状态机。Viser 每个 tick 只执行
        # 一个阶段，阶段开始前就能告诉用户正在算什么，阶段结束后立刻展示
        # 耗时和诊断；CLI 仍可通过 plan_sequence() 同步跑完整流程。
        self._sequence_state: dict | None = None
        self.knowledge_state = {key: _read_state(path) for key, path in
                                self.config.get("knowledge_states", {}).items()}
        # lambda 保留测试和其他 backend 注入能力；pool 只负责生命周期，
        # 不把 Door Skill 逻辑塞进 cuRobo wrapper。
        self.planner_pool = CuroboPlannerPool(
            planner_factory=lambda **kwargs: SynchronousPlanner(**kwargs),
            # 一轮门 Task 实测有六个不同的碰撞/夹爪结构。容量四会在每轮
            # 开始连续淘汰，导致 CUDA Graph 每次都在首次求解时重新建立。
            # 保留六个结构只改变 solver 生命周期，不改任何候选或碰撞规则。
            max_entries=6,
            reuse_gripper_geometry=bool(self.config.get("reuse_gripper_geometry", True)),
        )
        self._apply_initial_scene()
        self.rebuild()

    def planner_options(self) -> dict:
        """返回传给 planner pool 且参与缓存签名的求解器运行模式。"""
        return {"planner_mode": self.planner_mode}

    def _apply_initial_scene(self) -> None:
        pose = self.config.get("microwave_pose_world")
        if pose is None:
            settings = json.loads((self.root / "data/task_1/knowledge/microwave_knowledge_view.json").read_text(encoding="utf-8"))
            pose = _body_pose(self.case.data, "microwave").copy()
            pose[:2, 3] = settings["preview_microwave_xy_m"]
        set_preview_microwave_pose(self.case, validate_pose(pose, field_name="microwave_pose_world"))

    def _load_push_open_profile(self) -> dict | None:
        """与现场装配共用同一逐角推门 profile 校验。"""
        return load_push_open_profile(self.root, self.config, self.group)

    def push_open_recontact_poses(self, angle_rad: float) -> tuple[np.ndarray, np.ndarray]:
        """按当前门角与旧 profile 的下一推点生成换点前/后两个 TCP。"""
        profile = self._load_push_open_profile()
        if profile is None or "reposition_after_40" not in profile:
            raise ValueError("当前推门 Skill 没有 40° 后换点 profile")
        # 旧验证入口在 40° 后抬离，再取下一段 50° 的接触参数，
        # 但目标门角使用重新观测的实际角度。不能用 50° 几何直接
        # 横穿门板去换点。
        params = dict(profile["push_waypoints"][3])
        params.pop("angle_deg")
        frozen = mujoco.MjData(self.case.model)
        mujoco.mj_copyData(frozen, self.case.model, self.case.data)
        frozen.qpos[self.case.model.joint("microwave_door_joint").qposadr[0]] = angle_rad
        mujoco.mj_forward(self.case.model, frozen)
        geom = frozen.geom("microwave_door_panel_collision")
        panel = np.eye(4)
        panel[:3, :3] = geom.xmat.reshape(3, 3)
        panel[:3, 3] = geom.xpos
        size = self.case.model.geom("microwave_door_panel_collision").size
        common = dict(
            panel_width_m=float(2 * size[0]),
            panel_height_m=float(2 * size[2]),
            panel_thickness_m=float(2 * size[1]),
            surface=str(profile["surface"]),
        )
        pre = door_push_pose(
            panel, **common, **{
                **params,
                "standoff_m": float(profile["reposition_after_40"]["precontact_standoff_m"]),
            },
        )
        contact = door_push_pose(panel, **common, **{**params, "standoff_m": 0.})
        return pre, contact

    def _insert_push_open_reposition(self, phases: tuple[DoorPhase, ...]) -> tuple[DoorPhase, ...]:
        """仅显式 profile 带换点段时复刻旧的抬离→预接触→接触顺序。"""
        profile = self._load_push_open_profile()
        if profile is None or "reposition_after_40" not in profile:
            return phases
        waypoints = profile["push_waypoints"]
        if len(waypoints) < 4 or abs(float(waypoints[2]["angle_deg"]) - 40.) > 1e-9:
            raise ValueError("当前旧推门 profile 的换点要求第三个接触目标为 40°")
        index = next(
            (i for i, phase in enumerate(phases)
             if phase.skill_id == "push_open" and phase.name == "push_02"),
            None,
        )
        if index is None:
            raise ValueError("推门 Skill 缺少 40° 接触阶段")
        previous = phases[index]
        up = np.asarray(previous.tcp_pose_world, dtype=float).copy()
        up[2, 3] += float(profile["reposition_after_40"]["retreat_world_z_m"])
        pre, contact = self.push_open_recontact_poses(previous.door_angle_rad)
        inserted = (
            DoorPhase("", "push_open", "retract_up", up,
                      previous.door_angle_rad, .06, "linear_tcp", "door_contact"),
            DoorPhase("", "push_open", "new_precontact", pre,
                      previous.door_angle_rad, .06, "plan", "avoid_door"),
            DoorPhase("", "push_open", "new_contact", contact,
                      previous.door_angle_rad, .06, "linear_tcp", "door_contact"),
        )
        expanded = phases[:index + 1] + inserted + phases[index + 1:]
        next_open_index = 0
        result = []
        for phase in expanded:
            if phase.skill_id == "push_open":
                phase = replace(
                    phase, phase_id=f"push_open/{next_open_index:02d}_{phase.name}",
                )
                next_open_index += 1
            result.append(phase)
        return tuple(result)

    def _make_knowledge(self, kind: str, start_deg: float, target_deg: float):
        return build_door_knowledge(
            self.root, self.case, self.config, self.knowledge_state,
            kind=kind, start_deg=start_deg, target_deg=target_deg,
        )

    def rebuild(self) -> tuple[DoorPhase, ...]:
        # 用户在播放中编辑时，目标仍从当前编辑场景的关闭初态生成。
        self.reset_visual_state()
        pull_angle = float(self.config["pull_angle_deg"])
        push_start_angle = float(self.config.get("push_start_angle_deg", pull_angle))
        push_angle = float(self.config["push_angle_deg"])
        close_start_angle = float(self.config.get("close_start_angle_deg", push_angle))
        if not (0 < pull_angle < push_angle <= 90
                and 0 < push_start_angle < push_angle
                and 0 < close_start_angle < 180):
            raise ValueError("门角必须满足拉门角和推门起始角位于 (0, 推开角)，推开角 ≤ 90°，关门实测起始角位于 (0,180°)")
        # 默认 TCP 来自当前机器人 profile 的 home joint FK。它属于 Task
        # 编排，移动微波炉时不应跟着微波炉平移。
        self.default_tcp_pose_world = self._current_tcp_pose()
        # 三段都用同一场景姿态；门板只从 Knowledge 的名义零角旋转一次。
        handle = self._make_knowledge("handle_pull", 0., pull_angle)
        # 旧 Task1 的推门起始角是把手阶段后的重新估计值，不一定等于
        # 把手路线的名义终点。纯视觉调试器由任务配置显式提供该输入。
        opening = self._make_knowledge("push_open", push_start_angle, push_angle)
        closing = self._make_knowledge("push_close", close_start_angle, 0.)
        phases = compose_task1_door_cycle(
            door_phases(handle, opening, closing),
            default_tcp_pose_world=self.default_tcp_pose_world,
        )
        phases = self._insert_push_open_reposition(phases)
        microwave = _body_pose(self.case.data, "microwave")
        for index, phase in enumerate(phases):
            if phase.skill_id == "task_transition":
                override = self.config.get("task_transition_overrides_world", {}).get(
                    phase.phase_id)
                pose = (None if override is None else
                        validate_pose(override, field_name=phase.phase_id))
            else:
                override = self.config.get("stage_overrides_microwave", {}).get(phase.phase_id)
                pose = (None if override is None else microwave @ validate_pose(
                    override, field_name=phase.phase_id))
            if override is None:
                continue
            if phase.name not in TRANSITION_NAMES:
                raise ValueError(f"只允许覆盖过渡目标，接触目标由 Knowledge 决定：{phase.phase_id}")
            phases = phases[:index] + (replace(phase, tcp_pose_world=pose),) + phases[index+1:]
        self.phases = phases
        self.knowledge = {"handle_pull": handle, "push_open": opening, "push_close": closing}
        self.revision += 1
        self.phase_index = 0
        self.plan_result = None
        self.plan_revision = None
        self.planned_q = None
        # Explicit C-space targets are reserved for safe task-transition
        # waypoints whose configured home branch is known.  They do not alter
        # contact stages or bypass any post-plan collision validation.
        self.joint_targets_by_phase: dict[str, list[float]] = {}
        self.timeline = None
        self.timeline_selection = None
        self.results = []
        self._sequence_state = None
        self.reset_visual_state()
        self.stop_index = len(self.phases)
        return self.phases

    def reset_visual_state(self) -> None:
        robot = self.case.metadata["robot"]
        for name, value in robot["home_joint_positions"].items():
            joint_id = mujoco.mj_name2id(self.case.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id >= 0:
                joint = self.case.model.joint(joint_id)
                self.case.data.qpos[joint.qposadr[0]] = value
        self.case.data.qpos[self.case.model.joint("microwave_door_joint").qposadr[0]] = 0.
        mujoco.mj_forward(self.case.model, self.case.data)
        self.animation_index = 0

    def _current_tcp_pose(self) -> np.ndarray:
        """读取当前组 TCP 的世界 pose；调用方须先设置关节并 forward。"""
        site_name = self.case.metadata["robot"]["tcp_sites"][self.group]
        site = self.case.data.site(site_name)
        pose = np.eye(4)
        pose[:3, 3] = np.asarray(site.xpos, dtype=float)
        pose[:3, :3] = np.asarray(site.xmat, dtype=float).reshape(3, 3)
        return pose

    def set_microwave_pose(self, pose_world: np.ndarray) -> None:
        original = _body_pose(self.case.data, "microwave").copy()
        previous = self.config.get("microwave_pose_world")
        try:
            set_preview_microwave_pose(self.case, validate_pose(pose_world, field_name="microwave_pose_world"))
            self.config["microwave_pose_world"] = _body_pose(self.case.data, "microwave").tolist()
            self.rebuild()
        except Exception:
            set_preview_microwave_pose(self.case, original)
            if previous is None:
                self.config.pop("microwave_pose_world", None)
            else:
                self.config["microwave_pose_world"] = previous
            raise

    def set_stage_override(self, phase_id: str, pose_world: np.ndarray | None) -> None:
        phase = next((item for item in self.phases if item.phase_id == phase_id), None)
        if phase is None or phase.name not in TRANSITION_NAMES:
            raise ValueError("仅可编辑已存在的非接触过渡点")
        is_task_transition = phase.skill_id == "task_transition"
        key = ("task_transition_overrides_world" if is_task_transition
               else "stage_overrides_microwave")
        overrides = self.config.setdefault(key, {})
        previous = overrides.get(phase_id)
        if pose_world is None:
            overrides.pop(phase_id, None)
        else:
            pose = validate_pose(pose_world, field_name=phase_id)
            if is_task_transition:
                overrides[phase_id] = pose.tolist()
            else:
                microwave = _body_pose(self.case.data, "microwave")
                overrides[phase_id] = (np.linalg.inv(microwave) @ pose).tolist()
        try:
            self.rebuild()
        except Exception:
            if previous is None:
                overrides.pop(phase_id, None)
            else:
                overrides[phase_id] = previous
            raise

    def set_angles(self, pull_deg: float, push_deg: float) -> None:
        old = (self.config["pull_angle_deg"], self.config["push_angle_deg"])
        self.config.update(pull_angle_deg=float(pull_deg), push_angle_deg=float(push_deg))
        try:
            self.rebuild()
        except Exception:
            self.config.update(pull_angle_deg=old[0], push_angle_deg=old[1])
            raise

    def budget_for(self, phase_id: str) -> dict | None:
        return approximate_ik_budget_for(self.config, self.phase(phase_id))

    def local_rotation_search_for(self, phase_id: str) -> dict | None:
        """返回 Skill 允许的局部姿态候选；接触目标没有默认候选范围。"""
        return local_rotation_search_for(self.config, self.phase(phase_id))

    def set_stage_budget(self, phase_id: str, budget: dict | None) -> None:
        self.phase(phase_id)
        if budget is not None:
            ApproximateIKBudget(**budget)
        self.config.setdefault("approximate_ik_by_stage", {})[phase_id] = budget
        self.revision += 1
        self.plan_revision = None
        self.plan_result = None
        self.planned_q = None
        self.timeline = None
        self.timeline_selection = None
        self.results = []
        self.phase_index = 0
        self.reset_visual_state()

    def save_config(self, path: Path | None = None) -> Path:
        destination = Path(path or self.config_path).resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.config, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        return destination

    def start_skill(self, skill_id: str | None = None) -> None:
        """选择单 Skill 或门子任务；关门子任务先真实规划开门前序。"""
        self.reset_visual_state()
        if skill_id == "microwave_door_open":
            # 开门子任务只包含把手拉门及内侧推门，回默认 TCP 属于后续任务过渡。
            indices = [i for i, phase in enumerate(self.phases)
                       if phase.skill_id in {"handle_pull", "push_open"}]
        elif skill_id in {"microwave_door_close", "microwave_door_cycle"}:
            # 关门不能从 home 直接起步。先在本轮规划开门及任务过渡，
            # 再从真实终态执行 push_close；界面只展示最终关门分区。
            indices = list(range(len(self.phases)))
        else:
            indices = [i for i, phase in enumerate(self.phases)
                       if skill_id is None or phase.skill_id == skill_id]
        if not indices:
            raise ValueError(f"未知门 Skill：{skill_id}")
        self.phase_index = indices[0]
        self.stop_index = indices[-1] + 1
        self.planned_q = None
        self.plan_result = None
        self.plan_revision = None
        self.timeline = None
        self.timeline_selection = None
        self.case.data.qpos[self.case.model.joint("microwave_door_joint").qposadr[0]] = \
            self.phases[self.phase_index].door_angle_rad
        mujoco.mj_forward(self.case.model, self.case.data)

    def phase(self, phase_id: str) -> DoorPhase:
        return next(item for item in self.phases if item.phase_id == phase_id)

    def export_targets(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = [dict(phase_id=p.phase_id, skill_id=p.skill_id, name=p.name,
                     tcp_pose_world=p.tcp_pose_world.tolist(),
                     door_angle_deg=float(np.degrees(p.door_angle_rad)),
                     gripper_width_m=p.gripper_width_m, motion_kind=p.motion_kind,
                     contact_policy=p.contact_policy) for p in self.phases]
        path.write_text(json.dumps({"revision": self.revision, "kind": "knowledge_targets_only",
                                    "phases": rows}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return path

    def _prepare_phase_scene(self, phase: DoorPhase) -> None:
        """按旧 Task1 的阶段边界设置推门起始门角。

        把手拉门的名义终点与重新估计的推门起点可以不同。旧流程在
        ``prepare_push_stage`` 中先更新门碰撞位姿，再做 pre-contact
        规划；这里同步更新运动学门关节，确保导出的 collision 快照、
        时间轴起点和本阶段 Knowledge 使用同一个角度。
        """
        if phase.skill_id not in {"push_open", "push_close"} or phase.name != "pre_contact":
            return
        joint = self.case.model.joint("microwave_door_joint")
        address = joint.qposadr[0]
        if abs(float(self.case.data.qpos[address]) - phase.door_angle_rad) <= 1e-9:
            return
        self.case.data.qpos[address] = phase.door_angle_rad
        mujoco.mj_forward(self.case.model, self.case.data)

    def plan_next(self) -> dict:
        """从上一个已播放阶段的运动学终态规划；失败不推进阶段。"""
        stage_started = time.perf_counter()
        if self.phase_index >= self.stop_index:
            return {"success": True, "complete": True}
        phase = self.phases[self.phase_index]
        self._prepare_phase_scene(phase)
        data, model = self.case.data, self.case.model
        if phase.motion_kind == "gripper":
            self._set_gripper_width(phase.gripper_width_m)
            row = {"phase_id": phase.phase_id, "success": True, "operation": "gripper",
                   "gripper_width_m": phase.gripper_width_m, "revision": self.revision}
            self.results.append(row)
            self.phase_index += 1
            return row
        folder = self.output_dir / f"revision_{self.revision:03d}" / f"phase_{self.phase_index:02d}"
        # 旧 Task1 在整个把手阶段暂时关闭微波炉 collision；推门阶段
        # 恢复炉体，只把随关节转动的门板从静态世界移出并单独记录。
        collision_options = door_stage_collision_options(phase)
        export_started = time.perf_counter()
        snapshot = export_collision_scene(model, data, self.case.metadata["robot"],
                                          self.group, folder / "collision",
                                          **collision_options)
        export_duration = time.perf_counter() - export_started
        config_started = time.perf_counter()
        settings = load_realman_motion_config(self.group, self.case.metadata["robot"],
                                               gripper_snapshot=snapshot.get("gripper_snapshot"))
        config_duration = time.perf_counter() - config_started
        lease = self.planner_pool.acquire(
            settings, snapshot, **self.planner_options()
        )
        planner = lease.planner
        planner_duration = lease.setup_duration_s
        names = tuple(self.case.metadata["robot"]["joint_groups"][self.group])
        q = [float(data.qpos[model.joint(name).qposadr[0]]) for name in names]
        budget_data = self.budget_for(phase.phase_id)
        budget = ApproximateIKBudget(**budget_data) if budget_data is not None else None
        local_search = self.local_rotation_search_for(phase.phase_id)
        solve_started = time.perf_counter()
        explicit_joint_target = self.joint_targets_by_phase.get(phase.phase_id)
        if explicit_joint_target is not None:
            raw = planner.plan_joint_target(
                names, q, explicit_joint_target, phase.tcp_pose_world,
                target_frame="world_tcp", max_attempts=3,
                diagnostic_label=phase.phase_id.replace("/", "_"),
            )
            actual_operation = "plan_cspace"
        elif local_search:
            raw = plan_with_local_rotation_search(
                planner, names, q, phase.tcp_pose_world, budget=budget,
                x_degrees=local_search.get("x_degrees", (0.,)),
                y_degrees=local_search.get("y_degrees", (0.,)),
                target_frame="world_tcp", linear_tcp=phase.motion_kind == "linear_tcp",
            )
        else:
            raw = plan_with_approximate_ik(
                planner, names, q, phase.tcp_pose_world, budget=budget,
                target_frame="world_tcp", linear_tcp=phase.motion_kind == "linear_tcp")
        if explicit_joint_target is None:
            actual_operation = phase.motion_kind
        fallback = None
        fixed_close_pose = self.config.get("close_pre_contact_fallback_pose_world")
        if (not raw.get("success") and phase.skill_id == "push_close"
                and phase.name == "pre_contact" and fixed_close_pose is not None):
            # 旧 Task1 阶段三在原 pre-contact 规划失败时，允许用显式记录
            # 的固定 TCP pose 重试。只替换本阶段目标；接触点、逐角推门和
            # 撤离仍由原 Knowledge 生成。两次求解结果都留在阶段诊断中。
            original_failure = raw
            target = validate_pose(np.asarray(fixed_close_pose, dtype=float),
                                   field_name="close_pre_contact_fallback_pose_world")
            raw = plan_with_approximate_ik(
                planner, names, q, target, budget=budget,
                target_frame="world_tcp", linear_tcp=False)
            fallback = {
                "from_operation": phase.motion_kind,
                "to_operation": phase.motion_kind,
                "source_status": original_failure.get("status"),
                "source_failure_category": original_failure.get("failure_category"),
                "source_message": original_failure.get("message"),
                "source_failure_detail": original_failure.get("failure_detail"),
                "source_solver_diagnostic": original_failure.get("solver_diagnostic"),
                "source_ik_diagnostic": original_failure.get("ik_diagnostic"),
                "source_target_tcp_pose_world": phase.tcp_pose_world.tolist(),
                "retry_target_tcp_pose_world": target.tolist(),
                "success": bool(raw.get("success")),
            }
            if raw.get("success"):
                phase = replace(phase, tcp_pose_world=target)
                self.phases = (self.phases[:self.phase_index] + (phase,)
                               + self.phases[self.phase_index + 1:])
        if (not raw.get("success")
                and phase.phase_id == "task/open_to_close/return_default_tcp"
                and self.config.get("return_default_tcp_non_linear_fallback", False)):
            # 旧 Task1 只在推门结束回默认 TCP 的撤离段显式允许普通规划
            # 兜底。接触/推门段仍要求直线；这里记录两次尝试，避免界面将
            # 普通规划轨迹误称为直线轨迹。
            linear_failure = raw
            raw = planner.plan(names, q, phase.tcp_pose_world,
                               target_frame="world_tcp",
                               diagnostic_label=phase.phase_id.replace("/", "_"))
            actual_operation = "plan"
            fallback = {
                "from_operation": phase.motion_kind,
                "to_operation": actual_operation,
                "source_status": linear_failure.get("status"),
                "source_failure_category": linear_failure.get("failure_category"),
                "source_message": linear_failure.get("message"),
                "source_failure_detail": linear_failure.get("failure_detail"),
                "source_solver_diagnostic": linear_failure.get("solver_diagnostic"),
                "source_ik_diagnostic": linear_failure.get("ik_diagnostic"),
                "success": bool(raw.get("success")),
            }
        solve_duration = time.perf_counter() - solve_started
        row = {"phase_id": phase.phase_id, "revision": self.revision,
               "success": bool(raw.get("success")), "operation": actual_operation,
               "requested_operation": phase.motion_kind, "fallback": fallback,
               "requested_tcp_pose_world": phase.tcp_pose_world.tolist(),
               "approximate_ik": raw.get("approximate_ik"),
               "target_selection": raw.get("target_selection"),
               "target_selection_attempts": raw.get("target_selection_attempts"),
               "failure_category": raw.get("failure_category"),
               "status": raw.get("status"), "message": raw.get("message"),
               "planner_mode": raw.get("planner_mode", self.planner_mode.value),
               "use_cuda_graph": raw.get(
                   "use_cuda_graph",
                   self.planner_mode is PlannerMode.CUDA_GRAPH_BENCHMARK,
               ),
               "raw_error": raw.get("raw_error"),
               "start_state_check": raw.get("start_state_check"),
               "failure_detail": raw.get("failure_detail"),
               "ik_diagnostic": raw.get("ik_diagnostic"),
               "solver_diagnostic": raw.get("solver_diagnostic"),
               "collision_verification": raw.get("collision_verification"),
               "failure_visualization": raw.get("failure_visualization"),
               "failure_visualization_error": raw.get("failure_visualization_error"),
               "collision_snapshot_ref": str(folder / "collision" / "collision_snapshot.json"),
               "duration_s": raw.get("duration_s"),
               "interpolation_dt_s": raw.get("interpolation_dt_s"),
               "linear_path_metrics": raw.get("linear_path_metrics")}
        if not row["success"]:
            row["failure_summary"] = summarize_motion_failure(raw, snapshot).to_dict()
        if raw.get("success"):
            points = np.asarray(raw.get("positions_rad"), dtype=float)
            if points.ndim != 2 or points.shape[1] != len(names) or len(points) < 2:
                row.update(success=False, failure_category="invalid_trajectory")
            else:
                self.planned_q = points
                self.planned_joint_names = names
                self._execution_diagnostic = {
                    "phase_id": phase.phase_id,
                    "planner": planner,
                    "joint_names": tuple(names),
                    "target_pose_world": np.asarray(
                        phase.tcp_pose_world, dtype=float,
                    ).copy(),
                }
                self.animation_index = 0
                self.animation_door_start_rad = float(data.qpos[model.joint("microwave_door_joint").qposadr[0]])
        folder.mkdir(parents=True, exist_ok=True)
        row["timing"] = {
            "collision_export_s": export_duration,
            "motion_config_s": config_duration,
            "planner_setup_s": planner_duration,
            "planner_cache_hit": lease.cache_hit,
            "planner_rebuild_reason": lease.rebuild_reason,
            "candidate_and_plan_s": solve_duration,
            "backend_s": raw.get("duration_s"),
            # 在写出详细后端记录和阶段结果后再更新，避免只统计 solver
            # 而漏掉反馈生成成本。
            "stage_wall_s": None,
        }
        (folder / "plan.json").write_text(
            json.dumps(raw, ensure_ascii=False, indent=2, default=_json_value) + "\n", encoding="utf-8")
        result_path = folder / "result.json"
        # 第一次写入被计入最终墙钟；随后只重写 timing 数值。第二次写入
        # 的微小成本不影响 solver 秒级对比，但保存值已覆盖主要 I/O。
        row["timing"]["stage_wall_s"] = time.perf_counter() - stage_started
        result_path.write_text(
            json.dumps(row, ensure_ascii=False, indent=2, default=_json_value) + "\n", encoding="utf-8")
        row["timing"]["stage_wall_s"] = time.perf_counter() - stage_started
        result_path.write_text(
            json.dumps(row, ensure_ascii=False, indent=2, default=_json_value) + "\n", encoding="utf-8")
        self.results.append(row)
        self.plan_result = row
        self.plan_revision = self.revision
        return row

    def export_execution_collision(
        self, phase_id: str, joint_names, positions_rad,
        execution_failure: dict,
    ) -> dict:
        """Export the cuRobo collision model at an actual door-execution stop."""
        diagnostic = self._execution_diagnostic
        if diagnostic is None or diagnostic["phase_id"] != phase_id:
            raise RuntimeError("current door phase has no matching planner diagnostic")
        by_name = dict(zip(joint_names, positions_rad, strict=True))
        positions = [by_name[name] for name in diagnostic["joint_names"]]
        return diagnostic["planner"].export_execution_collision(
            positions,
            diagnostic["target_pose_world"],
            diagnostic_label=f"{phase_id.replace('/', '_')}_execution_collision",
            execution_failure=execution_failure,
        )

    def _current_joint_positions(self, joint_names: tuple[str, ...]) -> np.ndarray:
        return np.asarray([
            self.case.data.qpos[self.case.model.joint(name).qposadr[0]]
            for name in joint_names
        ], dtype=float)

    def _gripper_width(self) -> float:
        gripper = self.case.metadata["robot"]["grippers"][self.group]
        if gripper.get("mode") != "parallel_slide_proxy":
            return 0.0
        return float(sum(
            self.case.data.qpos[self.case.model.joint(name).qposadr[0]]
            for name in gripper["joint_names"]
        ))

    def begin_plan_sequence(self, skill_id: str | None = None) -> dict:
        """建立增量规划状态，但不在本次调用中运行 cuRobo。"""
        self.start_skill(skill_id)
        self._sequence_state = {
            "selection": skill_id or "door_cycle",
            "initial_qpos": self.case.data.qpos.copy(),
            "joint_names": tuple(
                self.case.metadata["robot"]["joint_groups"][self.group]),
            "segments": [],
            "started_s": time.perf_counter(),
            "total_stages": self.stop_index - self.phase_index,
        }
        return {
            "done": False,
            "selection": self._sequence_state["selection"],
            "completed_segments": 0,
            "total_stages": self._sequence_state["total_stages"],
            "next_phase_id": self.phases[self.phase_index].phase_id,
        }

    def _finish_plan_sequence(self, *, success: bool,
                              first_failure: dict | None = None) -> dict:
        """冻结已完成前缀为时间轴，并恢复规划开始时的可视状态。"""
        state = self._sequence_state
        if state is None:
            raise RuntimeError("当前没有进行中的 Skill/Task 规划")
        segments = list(state["segments"])
        self.case.data.qpos[:] = state["initial_qpos"]
        mujoco.mj_forward(self.case.model, self.case.data)
        self.planned_q = None
        self.timeline_selection = state["selection"] if segments else None
        self.timeline = PlannedTimeline(segments) if segments else None
        elapsed = time.perf_counter() - state["started_s"]
        report = {
            "done": True,
            "success": bool(success),
            "selection": state["selection"],
            "completed_segments": len(segments),
            "total_stages": state["total_stages"],
            "elapsed_wall_s": elapsed,
            "first_failure": first_failure,
        }
        if self.timeline is not None:
            report["duration_s"] = self.timeline.duration_s
            report["segment_count"] = len(segments)
            self.seek_timeline(0.0)
        self._sequence_state = None
        return report

    def plan_sequence_step(self) -> dict:
        """只规划当前一个阶段，供 Viser 在相邻 tick 间刷新进度。"""
        state = self._sequence_state
        if state is None:
            raise RuntimeError("请先调用 begin_plan_sequence()")
        if self.phase_index >= self.stop_index:
            return self._finish_plan_sequence(success=True)

        phase = self.phases[self.phase_index]
        self._prepare_phase_scene(phase)
        joint_names = state["joint_names"]
        q_start = self._current_joint_positions(joint_names)
        door_joint = self.case.model.joint("microwave_door_joint")
        door_start = float(self.case.data.qpos[door_joint.qposadr[0]])
        gripper_start = self._gripper_width()
        try:
            row = self.plan_next()
        except Exception:
            # 异常继续向上传递给界面/CLI，但必须先恢复运动学场景，避免
            # 下一次规划误用半途状态。
            self.case.data.qpos[:] = state["initial_qpos"]
            mujoco.mj_forward(self.case.model, self.case.data)
            self.planned_q = None
            self._sequence_state = None
            raise
        if not row.get("success"):
            return self._finish_plan_sequence(success=False, first_failure=row)

        if row.get("operation") == "gripper":
            positions = np.vstack((q_start, q_start))
            sample_dt = .25
            door_values = np.asarray([door_start, phase.door_angle_rad])
            gripper_values = np.asarray([gripper_start, phase.gripper_width_m])
        else:
            positions = np.asarray(self.planned_q, dtype=float).copy()
            # 某些后端不返回起点采样。补上实际起点，保证相邻阶段连续。
            if np.max(np.abs(positions[0] - q_start)) > 1e-4:
                positions = np.vstack((q_start, positions))
            sample_dt = float(row.get("interpolation_dt_s") or .025)
            door_values = np.linspace(door_start, phase.door_angle_rad, len(positions))
            # 这一阶段没有夹爪命令。时间轴必须沿用实际起始开口，
            # 否则回放会把拉门后 80 mm 的夹爪误画成 60 mm，和规划
            # 碰撞快照及物理执行看到的几何形状不一致。
            gripper_values = np.full(len(positions), gripper_start)
            while self.planned_q is not None:
                self.advance()
        state["segments"].append(PlannedSegment(
            stage_id=phase.phase_id,
            skill_id=phase.skill_id,
            joint_names=joint_names,
            positions_rad=positions,
            sample_dt_s=sample_dt,
            channels={
                "door_angle_rad": door_values,
                "gripper_width_m": gripper_values,
            },
            details=row,
        ))
        if self.phase_index >= self.stop_index:
            return self._finish_plan_sequence(success=True)
        return {
            "done": False,
            "success": True,
            "selection": state["selection"],
            "completed_segments": len(state["segments"]),
            "total_stages": state["total_stages"],
            "last_result": row,
            "next_phase_id": self.phases[self.phase_index].phase_id,
            "elapsed_wall_s": time.perf_counter() - state["started_s"],
        }

    def cancel_plan_sequence(self) -> None:
        """取消界面发起的增量规划，并恢复规划初态。"""
        if self._sequence_state is None:
            return
        state = self._sequence_state
        self.case.data.qpos[:] = state["initial_qpos"]
        mujoco.mj_forward(self.case.model, self.case.data)
        self.planned_q = None
        self._sequence_state = None

    def plan_sequence(self, skill_id: str | None = None) -> dict:
        """CLI 同步入口；内部复用 Viser 的逐阶段状态机。"""
        report = self.begin_plan_sequence(skill_id)
        while not report["done"]:
            report = self.plan_sequence_step()
        return report

    def seek_timeline(self, time_s: float):
        """把 MuJoCo 运动学场景设置到规划时间轴的指定时刻。"""
        if self.timeline is None:
            raise RuntimeError("尚未生成连续规划时间轴")
        sample = self.timeline.sample(time_s)
        for name, value in zip(sample.joint_names, sample.joint_positions_rad, strict=True):
            self.case.data.qpos[self.case.model.joint(name).qposadr[0]] = float(value)
        door_joint = self.case.model.joint("microwave_door_joint")
        self.case.data.qpos[door_joint.qposadr[0]] = sample.channels["door_angle_rad"]
        self._set_gripper_width(sample.channels["gripper_width_m"])
        mujoco.mj_forward(self.case.model, self.case.data)
        return sample

    def advance(self) -> dict:
        """播放 cuRobo 轨迹中的一个关节采样，不用失败 IK 种子制作动画。"""
        if self.planned_q is None or self.plan_revision != self.revision:
            raise RuntimeError("本阶段没有可播放的当前规划轨迹")
        phase = self.phases[self.phase_index]
        index = min(self.animation_index, len(self.planned_q) - 1)
        for name, q in zip(self.planned_joint_names, self.planned_q[index], strict=True):
            self.case.data.qpos[self.case.model.joint(name).qposadr[0]] = float(q)
        fraction = index / (len(self.planned_q) - 1)
        door_angle = (1. - fraction) * self.animation_door_start_rad + fraction * phase.door_angle_rad
        self.case.data.qpos[self.case.model.joint("microwave_door_joint").qposadr[0]] = door_angle
        mujoco.mj_forward(self.case.model, self.case.data)
        self.animation_index = index + 1
        done = self.animation_index >= len(self.planned_q)
        if done:
            self.planned_q = None
            self.phase_index += 1
        return {"phase_id": phase.phase_id, "sample": index, "samples": len(self.planned_q) if self.planned_q is not None else index + 1,
                "complete": done, "door_angle_deg": float(np.degrees(door_angle))}

    def _set_gripper_width(self, width: float) -> None:
        gripper = self.case.metadata["robot"]["grippers"][self.group]
        if gripper.get("mode") == "parallel_slide_proxy":
            opening = float(np.clip(width, *gripper["width_range_m"]))
            for name in gripper["joint_names"]:
                self.case.data.qpos[self.case.model.joint(name).qposadr[0]] = opening / 2.
            mujoco.mj_forward(self.case.model, self.case.data)


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"不能序列化规划返回值：{type(value).__name__}")
