"""固定GPU缓冲区的同步复用引擎；兼容性不足时明确拒绝，不沿用旧几何。"""

import copy
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from .core import SynchronousPlanner, add_tcp_tool_frame, add_attached_collision_spheres
from .session import _gripper_reuse_settings
from .gripper import gripper_result_metadata
from ..configuration import collision_mode_metadata

CAPACITY = {"cuboid": 192, "mesh": 16}
ATTACHED = 32


def exact_digest(value):
    def encode(item):
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(type(item).__name__)

    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
            default=encode,
        ).encode()
    ).hexdigest()


def scene_hash(settings):
    return (
        settings.get("scene_snapshot_sha256")
        or hashlib.sha256(
            json.dumps(settings["scene_model"], sort_keys=True).encode()
        ).hexdigest()
    )


def validate_settings(settings):
    # 动态补偿尚未接入固定球布局；不能把本地独立padding语义静默丢掉。
    if settings.get("collision_padding_semantics", "backend_native") != "backend_native":
        raise ValueError("single_reuse当前仅支持backend_native padding")
    scene = settings.get("scene_model", {})
    if set(scene) - {"cuboid", "mesh"}:
        raise ValueError("single_reuse仅支持cuboid/mesh场景")
    names = [name for objects in scene.values() for name in objects]
    if len(names) != len(set(names)):
        raise ValueError("碰撞对象名称必须全局唯一")
    if any(len(scene.get(kind, {})) > capacity for kind, capacity in CAPACITY.items()):
        raise ValueError("世界容量不足")
    from ..configuration import resolve_collision_mode, CollisionMode

    if resolve_collision_mode(settings.get("collision_mode")) is not CollisionMode.FULL:
        raise ValueError("single_reuse要求full碰撞")
    if settings.get("robot_config", {}).get("kinematics", {}).get("num_envs", 1) != 1:
        raise ValueError("single_reuse仅支持单环境")


def padded_config(s):
    if s.get("held_object") is not None:
        raise ValueError("single_reuse暂不支持held_object接线，请使用baseline")
    if s["robot_config"]["kinematics"].get("lock_joints"):
        raise ValueError("single_reuse不支持锁关节变化，请使用baseline")
    c = add_tcp_tool_frame(
        s["robot_config"],
        s["tool_frame"],
        np.asarray(s["link_from_tcp"]),
        tcp_frame="gene_tcp",
    )
    attached = s.get("attachment_spheres_tool")
    count = 0
    if attached is not None:
        c = add_attached_collision_spheres(c, s["tool_frame"], attached)
        count = len(attached)
    if count > ATTACHED:
        raise ValueError("持物槽容量不足")
    c["kinematics"]["collision_spheres"][s["tool_frame"]].extend(
        {"center": [0.0, 0.0, 0.0], "radius": -100.0} for _ in range(ATTACHED - count)
    )
    return c


def identity(s):
    g = _gripper_reuse_settings(
        {k: s.get(k) for k in ["robot_config", "gripper_snapshot"]}
    )
    return exact_digest(
        {
            **g,
            **{
                k: s.get(k)
                for k in [
                    "joint_names",
                    "planner_joint_names",
                    "world_from_base",
                    "link_from_tcp",
                    "tool_frame",
                ]
            },
            "num_trajopt_seeds": s.get("num_trajopt_seeds", 4),
            "collision_padding_semantics": s.get("collision_padding_semantics", "backend_native"),
            "random_seed": s.get("random_seed", 123),
            "mode": s.get("planner_mode", "cuda_graph_benchmark"),
            "collision_mode": s.get("collision_mode", "full"),
            "optimizer_collision_activation_distance": s.get("optimizer_collision_activation_distance", .01),
        }
    )


class ReusablePlanner(SynchronousPlanner):
    optimization_mode = "single_reuse"

    def __init__(self, settings):
        validate_settings(settings)
        s = copy.deepcopy(settings)
        if s.get("gripper_snapshot", {}).get("mode") != "parallel_slide_proxy" or s.get(
            "joint_names"
        ) != [f"right_joint_{i}" for i in range(1, 8)]:
            raise ValueError("single_reuse当前仅验证Realman右臂并联夹爪")
        if s.get("planner_mode", "cuda_graph_benchmark") != "cuda_graph_benchmark":
            raise ValueError("single_reuse要求CUDA Graph模式")
        super().__init__(**s)
        if self.collision_mode.value != "full":
            raise ValueError("single_reuse要求full碰撞")
        self.fixed_identity = identity(s)
        self.robot_config = padded_config(s)
        from curobo.kinematics import Kinematics, KinematicsCfg

        self.kinematics = Kinematics(
            KinematicsCfg.from_robot_yaml_file(copy.deepcopy(self.robot_config))
        )
        self.collision_cache = CAPACITY.copy()
        self._geometry_id = None
        self._world_id = None
        self.switches = []
        self._mesh_history = {}
        self.activate(s)

    def _ik_batch_capacity(self, count):
        from .optimization import ik_bucket_size

        return ik_bucket_size(count)

    def models(self):
        return [self.kinematics] + [
            x.kinematics
            for x in [
                self._planner,
                self._ik,
                *self._ik_batches.values(),
                self._collision_checker,
            ]
            if x is not None
        ]

    def buffers(self):
        out = {}
        for model in self.models():
            p = model.config.kinematics_config
            out[(id(p), "spheres")] = p.link_spheres.data_ptr()
            c = model.get_self_collision_config()
            out[(id(c), "pairs")] = c.collision_pairs.data_ptr()
        for solver in [
            self._planner,
            self._ik,
            *self._ik_batches.values(),
            self._collision_checker,
        ]:
            if solver is None:
                continue
            world = (
                solver.scene_model
                if solver is self._collision_checker
                else solver.scene_collision_checker
            )
            for storage in world.data.get_valid_data():
                for name, value in vars(storage).items():
                    if isinstance(value, self.torch.Tensor):
                        out[(id(storage), name)] = value.data_ptr()
        return out

    def activate(self, s):
        validate_settings(s)
        started = time.perf_counter()
        pointers = self.buffers()
        if identity(s) != self.fixed_identity:
            raise ValueError("机器人源模型/静态几何/坐标/采样配置不兼容")
        if any(len(s["scene_model"].get(k, {})) > v for k, v in CAPACITY.items()):
            raise ValueError("世界容量不足")
        cfg = padded_config(s)
        gid = exact_digest(cfg)
        geometry_changed = gid != self._geometry_id
        if geometry_changed:
            kin = cfg["kinematics"]
            rows = []
            buffer = kin.get("collision_sphere_buffer", 0.0)
            for link in kin["collision_link_names"]:
                offset = buffer.get(link, 0.0) if isinstance(buffer, dict) else buffer
                rows.extend(
                    [*v["center"], v["radius"] + offset]
                    for v in kin["collision_spheres"][link]
                )
            tensor = self.torch.tensor(
                rows, device="cuda", dtype=self.torch.float32
            ).unsqueeze(0)
            # 初次严格核对手工排列等于cuRobo loader的排列，之后结构身份固定。
            if self._geometry_id is None:
                if not self.torch.equal(
                    tensor, self.kinematics.config.kinematics_config.link_spheres
                ):
                    raise ValueError("手工球布局与cuRobo加载布局不一致")
            seen = set()
            for model in self.models():
                p = model.config.kinematics_config
                if id(p) in seen:
                    continue
                seen.add(id(p))
                if p.link_spheres.shape != tensor.shape:
                    raise ValueError("球shape变化")
                ptr = p.link_spheres.data_ptr()
                p.link_spheres.copy_(tensor)
                p.reference_link_spheres.copy_(tensor)
                if p.link_spheres.data_ptr() != ptr:
                    raise RuntimeError("球更新改变GPU地址")
            self.robot_config = cfg
            self._geometry_id = gid
        from .world_update import mesh_content_hashes, pose_updates
        from .batch_world import apply_batch as apply_pose_updates, update_solver_world
        from curobo._src.geom.types import SceneCfg

        scene = s["scene_model"]
        hashes = mesh_content_hashes(scene)
        for name, value in scene.get("mesh", {}).items():
            if hashes.get(name) is None:
                raise ValueError("mesh内容身份缺失")
            mesh_id = exact_digest(
                {
                    "hash": hashes[name],
                    "format": Path(value["file_path"]).suffix.lower(),
                    "geometry": {
                        k: v for k, v in value.items() if k not in {"pose", "file_path"}
                    },
                }
            )
            if name in self._mesh_history and self._mesh_history[name] != mesh_id:
                raise ValueError("历史同名mesh几何变化")
            self._mesh_history[name] = mesh_id
        canonical = copy.deepcopy(scene)
        for name, value in canonical.get("mesh", {}).items():
            value.pop("file_path", None)
            value["content_sha256"] = hashes[name]
            value["loader_format"] = Path(scene["mesh"][name]["file_path"]).suffix.lower()
        wid = exact_digest(
            {"scene": canonical, "policy": s.get("mounting_contact_policy")}
        )
        world_changed = wid != self._world_id
        if world_changed:
            updates = pose_updates(
                self.scene_model,
                scene,
                previous_mesh_hashes=self._scene_mesh_hashes,
                current_mesh_hashes=hashes,
            )
            solvers = [
                x
                for x in [
                    self._planner,
                    self._ik,
                    *self._ik_batches.values(),
                    self._collision_checker,
                ]
                if x is not None
            ]
            if updates is None:
                # 同名mesh几何变化须显式处理缓存，不静默继续。
                for name, h in hashes.items():
                    if (
                        name in self._scene_mesh_hashes
                        and h != self._scene_mesh_hashes[name]
                    ):
                        raise ValueError("同名mesh几何变化，须额外处理")
                scene_cfg = SceneCfg.create(copy.deepcopy(scene))
                for solver in solvers:
                    update_solver_world(solver, scene_cfg)
            else:
                names = {name for objects in scene.values() for name in objects}
                worlds = [
                    (
                        (
                            x.scene_model
                            if x is self._collision_checker
                            else x.scene_collision_checker
                        ),
                        names,
                    )
                    for x in solvers
                ]
                apply_pose_updates(worlds, updates)
            self.scene_model = copy.deepcopy(scene)
            self._scene_mesh_hashes = hashes
            self._world_id = wid
        self.scene_model = copy.deepcopy(scene)
        self._scene_mesh_hashes = hashes
        self._source_robot_config = copy.deepcopy(s["robot_config"])
        self._gripper_snapshot = copy.deepcopy(s.get("gripper_snapshot"))
        self.attachment_spheres_tool = copy.deepcopy(s.get("attachment_spheres_tool"))
        self.held_object = None
        self.scene_scope = s["scene_scope"]
        self.scene_snapshot_sha256 = scene_hash(s)
        self.gripper_metadata = {
            **gripper_result_metadata(self._gripper_snapshot),
            **collision_mode_metadata(self.collision_mode),
        }
        # 保留现版软避障距离与旧cuRobo mesh SDF兼容证据。
        from .mesh_sdf import ensure_mesh_sdf_compatibility
        self.gripper_metadata["optimizer_collision_activation_distance_m"] = self.optimizer_collision_activation_distance
        self.gripper_metadata["mesh_sdf_query"] = ensure_mesh_sdf_compatibility()
        self.gripper_metadata["optimization_mode"] = "single_reuse"
        from .core import infer_failure_visualization_dir
        self.diagnostic_output_dir = (Path(s["diagnostic_output_dir"]).resolve()
            if s.get("diagnostic_output_dir") is not None else infer_failure_visualization_dir(scene))
        if world_changed:
            self.mounting_contact_policy = copy.deepcopy(
                s.get("mounting_contact_policy") or {}
            )
            from gene_agent.motion.collision import MountingCollisionValidator

            self._validator = (
                MountingCollisionValidator(
                    self.kinematics, self.scene_model, self.mounting_contact_policy
                )
                if (self.mounting_contact_policy.get("obstacle_names")
                    or self.mounting_contact_policy.get("scene_collision_ignore_links"))
                else None
            )
        self._start_validation_cache.clear()
        if self.buffers() != pointers:
            raise RuntimeError("固定buffer地址改变")
        self.switches.append(
            {
                "wall_s": time.perf_counter() - started,
                "geometry": geometry_changed,
                "world": world_changed,
                "stable_buffers": len(pointers),
            }
        )
