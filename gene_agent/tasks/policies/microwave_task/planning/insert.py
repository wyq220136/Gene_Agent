"""旧微波炉 Bowl 放入动作的纯视觉场景适配器。"""

from __future__ import annotations
from gene_agent.motion.curobo.scene import planner_scene_options

import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from gene_agent.knowledge.geometry import validate_pose

from gene_agent.motion import (
    CuroboPlannerAdapter, JointState, PlannerMode, PlanningSceneRef,
    SynchronousPlanner, load_realman_motion_config,
)
from gene_agent.motion.curobo.gripper import export_gripper_snapshot
from gene_agent.motion.curobo.session import CuroboPlannerPool, CuroboPlanningSession
from gene_agent.motion.curobo.scene import export_self_collision_only_snapshot
from gene_agent.motion.model import PoseTarget
from gene_agent.skills.runtime.planning import SkillContext
from gene_agent.skills.library.transfer.planning.insertion import InsertObjectSkill, InsertionTargets

from gene_agent.scene.microwave_frame import estimate_legacy_microwave_frame
from gene_agent.tasks.policies.microwave_task.operations.bowl import fixed_place_targets, load_object_operation_profile
from gene_agent.tasks.policies.microwave_task.planning.bowl import PickBowlCase, tcp_path_for_positions


ROOT = Path(__file__).resolve().parents[5]


def _state(row: dict) -> JointState:
    return JointState(tuple(row["joint_names"]), tuple(row["positions_rad"]))


class _DeferredStageAdapter:
    """顺序使用的阶段首次调用时才获取实例，避免提前切换 world/采样状态。

    此内部适配器只用于下方同步放入/取出流水线；阶段结束后不可回头调用。
    pool 的生命周期由入口管理，不依赖此代理的垃圾回收。
    """

    def __init__(self, factory):
        self._factory = factory
        self._adapter = None
        self._cache_evidence = None

    def __getattr__(self, name):
        if self._adapter is None:
            self._adapter = self._factory()
            self._cache_evidence = dict(self._adapter.planner.planner_cache)
        return getattr(self._adapter, name)

    @property
    def cache_evidence(self):
        return self._cache_evidence


class _SessionStageAdapter:
    """旧顺序Skill的迁移适配：每条请求显式绑定起点，保留原采样边界。"""

    def __init__(self, session, snapshot, revision, spheres):
        self.session, self.snapshot = session, snapshot
        self.revision, self.spheres = revision, spheres
        self._first = True
        self.cache_evidence = None

    def _stage(self, start, collision_policy=None):
        return self.session.stage(
            snapshot=self.snapshot, start=start, group="right", scene_revision=self.revision,
            collision_policy=collision_policy, attachment_spheres_tool=self.spheres,
            restart_sampling=self._first,
        )

    def _record(self, motion):
        self._first = False
        if self.cache_evidence is None:
            self.cache_evidence = motion.cache_evidence

    def fk(self, state):
        with self._stage(state) as motion:
            self._record(motion)
            return motion.fk()

    def plan(self, request):
        with self._stage(request.start, request.collision_policy) as motion:
            self._record(motion)
            return motion.planner.plan(request)


def _planner(
    robot: dict, snapshot: dict, *, sphere_link_m: list[list[float]] | None,
    seed: int, revision: int, pool: CuroboPlannerPool | None = None,
) -> CuroboPlannerAdapter | _DeferredStageAdapter:
    if pool is not None:
        def acquire():
            config = load_realman_motion_config(
                "right", robot, gripper_snapshot=snapshot["gripper_snapshot"],
            )
            lease = pool.acquire(
                config, snapshot, attachment_spheres_tool=sphere_link_m,
                planner_mode=PlannerMode.CUDA_GRAPH_BENCHMARK, random_seed=seed,
            )
            lease.planner.reset_seed()
            lease.planner.planner_cache = lease.to_dict()
            return CuroboPlannerAdapter(lease.planner, group="right", scene_revision=revision)
        return _DeferredStageAdapter(acquire)
    config = load_realman_motion_config(
        "right", robot, gripper_snapshot=snapshot["gripper_snapshot"],
    )
    raw = SynchronousPlanner(
        **config, **planner_scene_options(snapshot),
        attachment_spheres_tool=sphere_link_m,
        planner_mode=PlannerMode.CUDA_GRAPH_BENCHMARK, random_seed=seed,
    )
    return CuroboPlannerAdapter(raw, group="right", scene_revision=revision)


def _scene_ref(snapshot: dict, revision: int) -> PlanningSceneRef:
    return PlanningSceneRef(
        revision, snapshot["snapshot_sha256"], snapshot["scope"],
    )


def _legacy_place_targets(case: PickBowlCase, T_tcp_bowl: np.ndarray):
    """以旧 profile frame 计算目标，同时公开资产对齐误差。

    当前 URDF 与旧 mesh 不同，因此这只是显式标记的实验性视觉映射；
    不会回写原始操作参数，也不把映射当成真机标定。
    """
    closed = mujoco.MjData(case.model)
    mujoco.mj_copyData(closed, case.model, case.data)
    joint = case.model.joint("microwave_door_joint")
    closed.qpos[joint.qposadr[0]] = 0.0
    mujoco.mj_forward(case.model, closed)
    alignment = estimate_legacy_microwave_frame(
        case.model, closed, ROOT / "data/task_1",
    )
    body = case.data.body("microwave")
    world_calibrated = np.eye(4)
    world_calibrated[:3, :3] = body.xmat.reshape(3, 3)
    world_calibrated[:3, 3] = body.xpos
    world_legacy = world_calibrated @ alignment.T_calibrated_legacy
    profile = load_object_operation_profile(ROOT / "data/task_1", "Bowl_0")
    targets = fixed_place_targets(world_legacy, profile, T_tcp_bowl)
    return targets, {
        "status": "experimental_visual_alignment_not_real_calibration",
        "T_calibrated_legacy": alignment.T_calibrated_legacy.tolist(),
        "handle_center_residual_m": alignment.handle_center_residual_m,
        "door_size_max_residual_m": alignment.door_size_max_residual_m,
        "source_profile": str(ROOT / "data/task_1/knowledge/microwave_object_manipulation.json"),
    }


def _sphere_link_from_world(snapshot: dict, world_tcp: np.ndarray, robot: dict) -> list[list[float]] | None:
    """从闭爪时保存的 world 球恢复固定的 link_7 球，不重新采样 mesh。"""
    if snapshot.get("collision_mode") == "robot_only":
        return None
    spheres = np.asarray(snapshot["transferred_objects"]["bowl"], dtype=float)
    if spheres.shape != (32, 4):
        raise ValueError("旧 Bowl 视觉附着必须恰有 32 个碰撞球")
    world_link = world_tcp @ np.linalg.inv(
        np.asarray(robot["transforms"]["right"]["link_from_tcp"], dtype=float),
    )
    local = (world_link[:3, :3].T @ (
        spheres[:, :3] - world_link[:3, 3]
    ).T).T
    return np.column_stack((local, spheres[:, 3])).tolist()


def _opened_gripper_snapshot(case: PickBowlCase, end: JointState, width_m: float) -> dict:
    frozen = mujoco.MjData(case.model)
    mujoco.mj_copyData(frozen, case.model, case.data)
    for name, value in zip(end.joint_names, end.positions_rad):
        frozen.qpos[case.model.joint(name).qposadr[0]] = value
    gripper = case.metadata["robot"]["grippers"]["right"]
    for name in gripper["joint_names"]:
        frozen.qpos[case.model.joint(name).qposadr[0]] = width_m / 2.0
    mujoco.mj_forward(case.model, frozen)
    return export_gripper_snapshot(case.model, frozen, case.metadata["robot"], "right")


def run_insert_bowl_skill_planning(
    case: PickBowlCase, *, seed: int = 123, pool: CuroboPlannerPool | None = None,
    motion_interface: str = "stage",
) -> dict:
    """从同一 run 的 Pick 实际终点规划旧固定 keypose 放入序列。"""
    if pool is None:
        with CuroboPlannerPool(planner_factory=SynchronousPlanner) as owned:
            return run_insert_bowl_skill_planning(case, seed=seed, pool=owned,
                                                 motion_interface=motion_interface)
    if motion_interface not in {"stage", "legacy"}:
        raise ValueError("motion_interface须为stage或legacy")
    if motion_interface == "legacy":
        return _run_insert_bowl_skill_planning(case, seed=seed, pool=pool)
    with CuroboPlanningSession(case.metadata["robot"], seed=seed, planner_pool=pool) as session:
        return _run_insert_bowl_skill_planning(case, seed=seed, pool=pool, session=session)


def _run_insert_bowl_skill_planning(case, *, seed, pool, session=None):
    def stage_planner(robot, snapshot, *, sphere_link_m, seed, revision, pool):
        if session is None:
            return _planner(robot, snapshot, sphere_link_m=sphere_link_m,
                            seed=seed, revision=revision, pool=pool)
        return _SessionStageAdapter(session, snapshot, revision, sphere_link_m)

    pick_path = case.output_dir / "skill_pick_bowl_result.json"
    pick = json.loads(pick_path.read_text(encoding="utf-8"))
    if not pick.get("success") or pick.get("instance_id") != "bowl":
        raise ValueError("放入 Skill 需要同一场景中成功的 Bowl Pick 规划")
    if pick.get("scene_revision") != case.revision:
        raise ValueError("Pick 报告与当前编辑场景 revision 不一致")
    lift = next(row for row in pick["stages"] if row["name"] == "lift")
    approach = next(row for row in pick["stages"] if row["name"] == "approach")
    lift_end = _state(lift["result"]["goal_state"])
    approach_end = _state(approach["result"]["goal_state"])
    robot = case.metadata["robot"]
    attached_path = case.output_dir / "visual_attachment/collision_snapshot.json"
    attached = json.loads(attached_path.read_text(encoding="utf-8"))
    if attached["snapshot_sha256"] != pick["visual_attachment"]["snapshot_sha256"]:
        raise ValueError("Pick 的附着碰撞快照已改变")
    # 球相对于 link_7 固定；使用 attach 时的实际 FK，而不是 lift 的 TCP。
    world_tcp_at_attach = pick["visual_attachment"].get("T_world_tcp_at_attach")
    if world_tcp_at_attach is None:  # 兼容尚未保存抓取实际 TCP 的旧报告。
        fk_source = stage_planner(robot, attached, sphere_link_m=None,
                             seed=seed, revision=case.revision + 1, pool=pool)
        world_tcp_at_attach = fk_source.fk(approach_end)
    else:
        world_tcp_at_attach = validate_pose(world_tcp_at_attach, field_name="T_world_tcp_at_attach")
    sphere_link = _sphere_link_from_world(attached, world_tcp_at_attach, robot)
    world_planner = stage_planner(robot, attached, sphere_link_m=sphere_link,
                             seed=seed, revision=case.revision + 1, pool=pool)
    T_tcp_bowl = np.asarray(pick["visual_attachment"]["T_tcp_bowl"], dtype=float)
    geometry, alignment = _legacy_place_targets(case, T_tcp_bowl)
    targets = InsertionTargets(*(
        PoseTarget("world_tcp", pose) for pose in (
            geometry.pre_place_tcp, geometry.place_tcp,
            geometry.post_place_1_tcp, geometry.post_place_2_tcp,
        )
    ))
    output = case.output_dir / "visual_insert"
    attached_self = export_self_collision_only_snapshot(
        attached, output / "attached_self_only", attached_object_id="bowl",
    )
    attached_ignore_planner = stage_planner(
        robot, attached_self, sphere_link_m=sphere_link,
        seed=seed, revision=case.revision + 2, pool=pool,
    )
    skill = InsertObjectSkill()
    result = skill.plan_attached(
        world_planner, attached_ignore_planner,
        SkillContext("right", lift_end, _scene_ref(attached, case.revision + 1), "bowl", seed),
        _scene_ref(attached_self, case.revision + 2), targets,
    )
    released_pose = None
    detached_self = None
    if result.stage("insert").status == "planned":
        end = result.stage("insert").result.goal_state
        actual_tcp = attached_ignore_planner.fk(end)
        released_pose = actual_tcp @ T_tcp_bowl
        open_width = float(robot["grippers"]["right"]["width_range_m"][1])
        opened = _opened_gripper_snapshot(case, end, open_width)
        detached_self = export_self_collision_only_snapshot(
            attached, output / "detached_self_only",
            gripper_snapshot=opened,
        )
        detached_planner = stage_planner(
            robot, detached_self, sphere_link_m=None,
            seed=seed, revision=case.revision + 3, pool=pool,
        )
        result = skill.plan_visual_release_and_retreat(
            detached_planner, result,
            SkillContext("right", end, _scene_ref(detached_self, case.revision + 3),
                         "bowl", seed),
            targets, release_width_m=open_width,
        )
    report = {
        "schema_version": 1,
        "task": "insert_bowl_skill_planning",
        "instance_id": "bowl",
        "model_ref": "source_scene/scene.xml",
        "scene_revision": case.revision,
        "source_pick_report": "skill_pick_bowl_result.json",
        "frame_alignment": alignment,
        "target_placed_object_world": geometry.placed_object.tolist(),
        "actual_released_object_world": None if released_pose is None else released_pose.tolist(),
        "T_tcp_bowl": T_tcp_bowl.tolist(),
        "visual_only": True,
        "motion_interface": "stage" if session is not None else "legacy",
        "planner_cache": {"preplace": world_planner.cache_evidence,
                          "insert": attached_ignore_planner.cache_evidence,
                          "retreat": None if detached_self is None else detached_planner.cache_evidence},
        **result.to_dict(),
    }
    for stage in report["stages"]:
        name = stage["name"]
        if name == "preplace":
            stage["collision_snapshot_ref"] = "visual_attachment/collision_snapshot.json"
            stage["attachment_ref"] = "visual_attachment/attachment_state.json"
        elif name == "insert":
            stage["collision_snapshot_ref"] = "visual_insert/attached_self_only/collision_snapshot.json"
            stage["attachment_ref"] = "visual_attachment/attachment_state.json"
        elif detached_self is not None and name in {"release", "detach", "retreat_1", "retreat_2"}:
            stage["collision_snapshot_ref"] = "visual_insert/detached_self_only/collision_snapshot.json"
            stage["object_pose_world"] = released_pose.tolist()
        trajectory = ((stage.get("result") or {}).get("trajectory") or {})
        if trajectory.get("positions_rad"):
            stage["tcp_path_world_m"] = tcp_path_for_positions(
                case, trajectory["joint_names"], trajectory["positions_rad"],
            )
    output.mkdir(parents=True, exist_ok=True)
    (case.output_dir / "skill_insert_bowl_result.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


__all__ = ["run_insert_bowl_skill_planning"]
