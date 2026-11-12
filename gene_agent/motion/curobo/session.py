"""复用固定机器人/碰撞拓扑的同步 cuRobo planner。"""

from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
import copy
from dataclasses import dataclass, field
from uuid import uuid4
import hashlib
import json
import time
from typing import Any, Callable

import numpy as np
from ..configuration import CollisionMode, resolve_collision_mode
from ..client import PlanningContext, StageMotion
from ..model import CollisionPolicy, JointState, PlannerMode, PlanningSceneRef
from .core import SynchronousPlanner, load_realman_motion_config
from .adapter import CuroboPlannerAdapter
from .profiles import load_motion_config


def _plain(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, float):
        # MuJoCo forward 后同一刚体几何会出现约 1e-15 m 的数值抖动；
        # 1e-9 m/rad 以下不会改变碰撞语义，不应让 planner 每阶段冷启动。
        rounded = round(value, 9)
        return 0.0 if rounded == 0.0 else rounded
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return repr(value)


def _digest(value: Any) -> str:
    payload = json.dumps(_plain(value), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _scene_structure(snapshot: dict) -> list[dict]:
    """忽略 pose 和临时导出路径，只比较碰撞拓扑与实际几何内容。"""
    scene = snapshot["scene_model"]
    records = {str(row["obstacle_name"]): row for row in snapshot.get("records", ())}
    result = []
    for kind, obstacles in sorted(scene.items()):
        for name, raw in sorted(obstacles.items()):
            row = records.get(str(name), {})
            geometry = {
                key: value for key, value in raw.items()
                if key not in {"pose", "file_path"}
            }
            if kind == "mesh":
                geometry["mesh_sha256"] = row.get("mesh_sha256")
                geometry["approximation"] = row.get("approximation")
            result.append({"kind": kind, "name": name, "geometry": geometry})
    return result


def _scene_pose_digest(snapshot: dict) -> str:
    """对实际送入 cuRobo 的 world pose 签名，不依赖可选诊断 records。"""
    rows = []
    for kind, obstacles in sorted(snapshot["scene_model"].items()):
        for name, raw in sorted(obstacles.items()):
            rows.append({"kind": kind, "name": name, "pose": raw.get("pose")})
    return _digest(rows)


def _gripper_reuse_settings(settings: dict) -> dict:
    """仅将有来源校验的动态 pad 球值移出结构身份；保留原 arm 球及锁关节。"""
    gripper = settings.get("gripper_snapshot") or {}
    if gripper.get("mode") == "aloha_parallel":
        from .profiles.aloha import gripper_reuse_identity
        return gripper_reuse_identity(settings)
    if gripper.get("mode") != "parallel_slide_proxy":
        return settings
    from .gripper import validate_gripper_snapshot, _digest as exact_digest
    validate_gripper_snapshot(gripper)
    result = copy.deepcopy(settings)
    spheres = result["robot_config"]["kinematics"]["collision_spheres"]
    if exact_digest(spheres) != gripper["merged_sphere_sha256"]:
        raise ValueError("动态夹爪球与快照不一致")
    counts = {}
    for link, added in gripper["added_spheres_by_link"].items():
        count = len(added)
        if not count or spheres[link][-count:] != added:
            raise ValueError("动态 pad 球不是已校验的追加几何")
        spheres[link] = spheres[link][:-count]
        counts[link] = count
    result["gripper_snapshot"] = {
        key: gripper.get(key) for key in (
            "schema_version", "group", "mode", "source_hashes", "source_robot_config_sha256",
        )
    }
    result["gripper_snapshot"]["dynamic_sphere_counts"] = counts
    return result


def _geometry_reuse_settings(settings):
    """持物球只有中心可动态更新；半径、顺序/来源、物体和接触link保持身份。"""
    result=copy.deepcopy(_gripper_reuse_settings(settings))
    held=result.get('held_object')
    if held is not None:
        spheres=np.asarray(held['spheres_link'],dtype=float)
        if spheres.ndim!=2 or spheres.shape[1]!=4 or not len(spheres) or not np.isfinite(spheres).all() or np.any(spheres[:,3]<=0):
            raise ValueError('动态持物球必须是有限正半径[N,4]')
        held['spheres_link']=[[0.,0.,0.,float(r)] for r in spheres[:,3]]
    return result


def _planner_key(settings: dict, snapshot: dict, planner_options: dict,
                 *, reuse_gripper_geometry: bool = False, reuse_contact_policy: bool = False) -> str:
    # 完整有效 robot_config 覆盖 Pika lock_joints；只看 collision sphere
    # 哈希会漏掉不同夹爪开口对应的锁定关节值。
    identity_settings = dict(_geometry_reuse_settings(settings) if reuse_gripper_geometry else settings)
    # 老入口将attachment放kwargs，新入口放settings。按实际配置规范身份，
    # full的省略值与显式值等价；不同球的全部几何仍参与签名。
    identity_options = dict(planner_options)
    mode = resolve_collision_mode(identity_options.get("collision_mode", snapshot.get("collision_mode")))
    identity_options["collision_mode"] = mode.value
    if "attachment_spheres_tool" in identity_settings and "attachment_spheres_tool" in identity_options:
        raise ValueError("attachment_spheres_tool不能在settings与options重复提供")
    spheres = identity_options.pop("attachment_spheres_tool", identity_settings.get("attachment_spheres_tool"))
    identity_settings["attachment_spheres_tool"] = None if mode is CollisionMode.ROBOT_ONLY else spheres
    gripper = dict(identity_settings.get("gripper_snapshot") or {})
    if gripper and not ({"dynamic_sphere_counts", "dynamic_lock_joints"} & gripper.keys()):
        # 快照在导出时已把 FK 数值抖动规范到 1e-12；保留内容哈希可确保
        # solver 球与结果中声明的 gripper 身份严格一致。
        identity_settings["gripper_snapshot"] = {
            key: gripper.get(key) for key in (
                "schema_version", "group", "mode", "snapshot_id",
                "merged_sphere_sha256", "width_m",
                "joint_positions_m", "source_hashes", "source_robot_config_sha256",
            )
        }
    identity = {
        "settings": identity_settings,
        "planner_options": identity_options,
        "scene_structure": _scene_structure(snapshot),
        "scene_scope": snapshot.get("scope"),
        "mounting_contact_policy": {} if reuse_contact_policy else snapshot.get("mounting_contact_policy") or {},
    }
    return _digest(identity)


@dataclass(frozen=True)
class PlannerLease:
    planner: Any
    cache_hit: bool
    setup_duration_s: float
    rebuild_reason: str
    cache_key: str
    acquisition_id: str = field(default_factory=lambda: uuid4().hex)

    def to_dict(self) -> dict:
        """运行记录只保存缓存证据，不序列化 GPU planner。"""
        return {"cache_hit": self.cache_hit, "setup_duration_s": self.setup_duration_s,
                "rebuild_reason": self.rebuild_reason, "cache_key": self.cache_key,
                "acquisition_id": self.acquisition_id}


@dataclass
class _Entry:
    planner: Any
    pose_digest: str
    snapshot_sha256: str
    gripper_snapshot_id: str | None
    held_digest: str | None = None
    policy_digest: str | None = None


class CuroboPlannerPool:
    """小型 LRU；结构相同只更新 world，任何更新异常都丢弃旧实例。"""

    def __init__(self, *, planner_factory: Callable[..., Any], max_entries: int = 4,
                 reuse_gripper_geometry: bool = True, reuse_contact_policy: bool = True):
        if max_entries < 1:
            raise ValueError("max_entries 必须为正整数")
        self.planner_factory = planner_factory
        self.max_entries = int(max_entries)
        self.reuse_gripper_geometry = bool(reuse_gripper_geometry)
        self.reuse_contact_policy = bool(reuse_contact_policy and getattr(planner_factory, "supports_contact_policy_update", False))
        self._entries: OrderedDict[str, _Entry] = OrderedDict()
        self._stage_owner = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.clear()

    def clear(self) -> None:
        self._stage_owner = None
        for entry in self._entries.values():
            self._dispose(entry.planner)
        self._entries.clear()

    @staticmethod
    def _dispose(planner: Any) -> None:
        """显式释放 cuRobo/CUDA 资源；测试替身也只需实现 close。"""
        cleanup = getattr(planner, "close", None) or getattr(planner, "destroy", None)
        if cleanup is not None:
            cleanup()

    def acquire(self, settings: dict, snapshot: dict, **planner_options) -> PlannerLease:
        if self._stage_owner is not None:
            raise RuntimeError("pool已有活跃stage，不能在阶段中切换共享求解器")
        started = time.perf_counter()
        mode = resolve_collision_mode(planner_options.get("collision_mode", snapshot.get("collision_mode")))
        if mode is CollisionMode.ROBOT_ONLY:
            if snapshot.get("collision_mode") != mode.value:
                raise ValueError("robot_only 必须使用直接准备的机器人快照，不能沿用完整场景")
            settings = {k: v for k, v in settings.items() if k != "attachment_spheres_tool"}
            planner_options = {**planner_options, "collision_mode": mode.value}
        key = _planner_key(settings, snapshot, planner_options,
                           reuse_gripper_geometry=self.reuse_gripper_geometry,
                           reuse_contact_policy=self.reuse_contact_policy)
        gripper_id = (settings.get("gripper_snapshot") or {}).get("snapshot_id")
        held_digest = hashlib.sha256(json.dumps(settings.get("held_object"),sort_keys=True,allow_nan=False).encode()).hexdigest()
        pose_digest = _scene_pose_digest(snapshot)
        policy = snapshot.get("mounting_contact_policy") or {}
        policy_digest = _digest(policy)
        entry = self._entries.get(key)
        if entry is not None:
            self._entries.move_to_end(key)
            held_changed = entry.held_digest != held_digest
            if entry.gripper_snapshot_id != gripper_id or held_changed:
                try:
                    if held_changed:
                        entry.planner.update_held_geometry(robot_config=settings['robot_config'],
                            gripper_snapshot=settings.get('gripper_snapshot'),held_object=settings.get('held_object'))
                    else:
                        entry.planner.update_gripper_geometry(
                            robot_config=settings["robot_config"],gripper_snapshot=settings["gripper_snapshot"])
                    # 同一 seed 的冷 solver 从采样序列开头求解。复用图和
                    # buffer 时重置采样，避免执行路径因缓存历史发生漂移。
                    entry.planner.reset_seed()
                except Exception:
                    self._entries.pop(key, None)
                    self._dispose(entry.planner)
                    planner = self._create(settings, snapshot, planner_options)
                    self._entries[key] = _Entry(planner, pose_digest, snapshot["snapshot_sha256"], gripper_id, held_digest, policy_digest)
                    self._trim()
                    return PlannerLease(planner, False, time.perf_counter() - started,
                                        "held_geometry_update_failed" if held_changed else "gripper_update_failed", key)
                entry.gripper_snapshot_id = gripper_id
                entry.held_digest = held_digest
            policy_changed = entry.policy_digest != policy_digest
            if policy_changed:
                try:
                    entry.planner.update_contact_policy(policy)
                    entry.planner.reset_seed()
                except Exception:
                    self._entries.pop(key, None)
                    self._dispose(entry.planner)
                    planner = self._create(settings, snapshot, planner_options)
                    self._entries[key] = _Entry(planner, pose_digest, snapshot["snapshot_sha256"], gripper_id, held_digest, policy_digest)
                    self._trim()
                    return PlannerLease(planner, False, time.perf_counter() - started,
                                        "contact_policy_update_failed", key)
                entry.policy_digest = policy_digest
            if entry.pose_digest != pose_digest:
                try:
                    entry.planner.update_scene(
                        scene_model=snapshot["scene_model"],
                        scene_scope=snapshot["scope"],
                        scene_snapshot_sha256=snapshot["snapshot_sha256"],
                        mounting_contact_policy=snapshot.get("mounting_contact_policy"),
                        gripper_snapshot=settings.get("gripper_snapshot"),
                    )
                except Exception:
                    # update_world 可能已经修改部分 solver；整个 wrapper 必须作废。
                    self._entries.pop(key, None)
                    self._dispose(entry.planner)
                    planner = self._create(settings, snapshot, planner_options)
                    self._entries[key] = _Entry(planner, pose_digest, snapshot["snapshot_sha256"], gripper_id, held_digest, policy_digest)
                    self._trim()
                    return PlannerLease(
                        planner, False, time.perf_counter() - started,
                        "scene_update_failed", key,
                    )
                entry.pose_digest = pose_digest
                entry.snapshot_sha256 = snapshot["snapshot_sha256"]
            elif entry.snapshot_sha256 != snapshot["snapshot_sha256"]:
                # snapshot hash 还包含运行 provenance 和机器人状态。场景几何
                # pose 完全相同时不重复 update_world，只刷新结果追踪元数据。
                entry.planner.refresh_scene_metadata(
                    scene_scope=snapshot["scope"],
                    scene_snapshot_sha256=snapshot["snapshot_sha256"],
                    gripper_snapshot=settings.get("gripper_snapshot"),
                )
                entry.snapshot_sha256 = snapshot["snapshot_sha256"]
            return PlannerLease(
                entry.planner, True, time.perf_counter() - started,
                "contact_policy_updated" if policy_changed else "cache_hit", key,
            )

        planner = self._create(settings, snapshot, planner_options)
        self._entries[key] = _Entry(planner, pose_digest, snapshot["snapshot_sha256"], gripper_id, held_digest, policy_digest)
        self._trim()
        return PlannerLease(
            planner, False, time.perf_counter() - started,
            "new_structure", key,
        )

    def _create(self, settings: dict, snapshot: dict, planner_options: dict):
        if planner_options.get("collision_mode") == CollisionMode.ROBOT_ONLY.value:
            planner_options = {**planner_options, "robot_protection_scene": snapshot["scene_model"]}
        return self.planner_factory(
            **settings,
            scene_model=snapshot["scene_model"],
            scene_scope=snapshot["scope"],
            scene_snapshot_sha256=snapshot["snapshot_sha256"],
            mounting_contact_policy=snapshot.get("mounting_contact_policy"),
            **planner_options,
        )

    def _trim(self) -> None:
        while len(self._entries) > self.max_entries:
            _, entry = self._entries.popitem(last=False)
            self._dispose(entry.planner)


class CuroboPlanningSession:
    """应用装配入口；每次只绑定一个冻结stage，复用原有有界pool。"""

    def __init__(self, robot: dict, *, seed: int = 123,
                 planner_mode: PlannerMode | str = PlannerMode.CUDA_GRAPH_BENCHMARK,
                 collision_mode: CollisionMode | str | None = None, max_entries: int = 4,
                 planner_pool: CuroboPlannerPool | None = None):
        self.robot = robot
        self.seed = int(seed)
        self.planner_mode = PlannerMode(planner_mode)
        self.collision_mode = resolve_collision_mode(collision_mode)
        if collision_mode is not None and CollisionMode(collision_mode) is not self.collision_mode:
            raise ValueError("显式session碰撞模式与外层motion配置冲突")
        self._owns_pool = planner_pool is None
        self._pool = planner_pool if planner_pool is not None else CuroboPlannerPool(
            planner_factory=SynchronousPlanner, max_entries=max_entries)
        self._closed = False
        self._active = None

    def __enter__(self):
        if self._closed:
            raise RuntimeError("规划session已关闭")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        if not self._closed:
            self._closed = True
            if self._active is not None and self._pool._stage_owner is self._active:
                self._pool._stage_owner = None
            self._active = None
            if self._owns_pool:
                self._pool.clear()

    @contextmanager
    def stage(self, *, snapshot: dict, start: JointState, group: str,
              scene_revision: int, collision_policy: CollisionPolicy | None = None,
              attachment_spheres_tool=None, restart_sampling: bool = False, diagnostic_options=None, request_output_dir=None):
        """进入才获取solver；退出使所有已导出的阶段句柄失效。"""
        if self._closed or self._active is not None:
            raise RuntimeError("session已关闭或已有活跃stage；不允许嵌套")
        if resolve_collision_mode(self.collision_mode) is not self.collision_mode:
            raise ValueError("活跃motion配置与session碰撞模式冲突")
        if CollisionMode(snapshot.get("collision_mode", "full")) is not self.collision_mode:
            raise ValueError("snapshot必须按session的实际collision_mode准备")
        names = self.robot.get("joint_groups", {}).get(group)
        if names is None or snapshot.get("active_group", group) != group:
            raise ValueError("机器人组与snapshot不一致")
        if not isinstance(start, JointState) or start.joint_names != tuple(names):
            raise ValueError("阶段起始关节名称/顺序与机器人组不一致")
        if (snapshot.get("world_frame") is not None and self.robot.get("world_frame") is not None
                and snapshot["world_frame"] != self.robot["world_frame"]):
            raise ValueError("snapshot与机器人world frame不一致")
        gripper = snapshot.get("gripper_snapshot")
        if gripper and gripper.get("group", group) != group:
            raise ValueError("gripper snapshot与机器人组不一致")
        context = PlanningContext(group, start, PlanningSceneRef(
            scene_revision, snapshot["snapshot_sha256"], snapshot["scope"]),
            seed=self.seed, planner_mode=self.planner_mode,
            collision_policy=collision_policy if collision_policy is not None else CollisionPolicy())
        if attachment_spheres_tool is not None:
            spheres = np.asarray(attachment_spheres_tool, dtype=float)
            if (spheres.ndim != 2 or spheres.shape[1] != 4 or len(spheres) == 0
                    or not np.isfinite(spheres).all() or np.any(spheres[:, 3] <= 0)):
                raise ValueError("attachment必须为[N,4]有限正半径工具坐标球")
        settings = load_motion_config(group, self.robot, gripper_snapshot=gripper,
                                      legacy_loader=load_realman_motion_config)
        if snapshot.get('attachment') and self.collision_mode is CollisionMode.FULL:
            settings['held_object']=snapshot['attachment']
        # robot_only不创建或缓存物体附着几何；显式参数仍做基本输入校验。
        if attachment_spheres_tool is not None and self.collision_mode is CollisionMode.FULL:
            settings = {**settings, "attachment_spheres_tool": spheres.tolist()}
        lease = self._pool.acquire(settings, snapshot, planner_mode=self.planner_mode,
                                   random_seed=self.seed, collision_mode=self.collision_mode.value)
        from .diagnostics import DiagnosticOptions
        diagnostic = DiagnosticOptions(**diagnostic_options) if diagnostic_options else DiagnosticOptions(mode="off")
        lease.planner.diagnostic_options = diagnostic
        lease.planner.diagnostic_snapshot = snapshot
        if restart_sampling:
            lease.planner.reset_seed()
        token = object()
        self._active = token
        self._pool._stage_owner = token

        def ensure_active():
            if self._closed or self._active is not token or self._pool._stage_owner is not token:
                raise RuntimeError("阶段已结束，须用当前状态建立新stage")

        try:
            yield StageMotion(CuroboPlannerAdapter(lease.planner, group=group,
                                                  scene_revision=scene_revision, request_output_dir=request_output_dir), context,
                              ensure_active=ensure_active, cache_evidence=lease.to_dict())
        finally:
            if self._pool._stage_owner is token:
                self._pool._stage_owner = None
            if self._active is token:
                self._active = None


__all__ = ["CuroboPlannerPool", "PlannerLease", "CuroboPlanningSession"]
