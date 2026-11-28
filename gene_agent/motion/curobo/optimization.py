"""显式、同步的 cuRobo 复用作用域；默认保持原规划器路径。"""

from contextlib import contextmanager
from contextvars import ContextVar
import copy
import threading
import time

_OWNER = ContextVar("curobo_optimization_owner", default=None)


def current_optimization():
    owner = _OWNER.get()
    if owner is not None:
        owner.check_usable()
    return owner


def ik_bucket_size(count):
    if count < 1:
        raise ValueError("IK批次必须非空")
    # 去重后的 120～128 个目标共享图容量，不为每个剩余数量捕获新图。
    for capacity in (25, 128, 160):
        if count <= capacity:
            return capacity
    return ((count + 127) // 128) * 128


class ReuseOwner:
    """一个同步任务拥有一个引擎；句柄不拥有GPU资源，结束统一释放。"""

    def __init__(self):
        from .reusable import ReusablePlanner

        self._engine_factory = ReusablePlanner
        self.engine = None
        self.active = None
        self.closed = False
        self.poisoned = False
        self.thread_id = threading.get_ident()
        self.activations = 0
        self.created = 0
        self.mesh_cache = {}
        self.mesh_export_cache = {}
        self.mesh_asset_dir = None
        self.mesh_export_hits = self.mesh_export_misses = 0
        self.mesh_file_cache = None
        self.world_sync_requests = self.world_data_syncs = 0
        self.mesh_hits = 0
        self.mesh_misses = 0
        self.started = time.perf_counter()
        self.calls = []

    def check_usable(self):
        self.check_thread()
        if self.closed or self.poisoned:
            raise RuntimeError("复用作用域已结束或失效")

    def ensure(self, handle):
        if self.closed or self.poisoned:
            raise RuntimeError("复用作用域已结束或更新失败，禁止继续使用")
        if threading.get_ident() != self.thread_id:
            raise RuntimeError("复用规划器仅支持拥有线程内的同步调用")
        marker = (handle, handle.revision)
        if marker != self.active:
            self.active = None
            try:
                if self.engine is None:
                    self.engine = self._engine_factory(handle.settings)
                    self.created += 1
                else:
                    self.engine.activate(handle.settings)
                self.engine.reset_seed()
            except BaseException:
                self.poisoned = True
                raise
            self.active = marker
            self.activations += 1
        defaults = {
            "diagnostic_options": None,
            "diagnostic_snapshot": {},
            "capture_solver_repr": False,
        }
        for name, default in defaults.items():
            setattr(self.engine, name, handle.__dict__.get(name, default))
        return self.engine

    def check_thread(self):
        if threading.get_ident() != self.thread_id:
            raise RuntimeError("复用作用域不能跨线程操作")

    def close(self):
        self.check_thread()
        if not self.closed:
            self.closed = True
            self.active = None
            try:
                if self.engine is not None:
                    self.engine.close()
            finally:
                self.mesh_cache.clear()
                self.mesh_export_cache.clear()
                self.engine = None

    def to_dict(self):
        return dict(
            mode="single_reuse",
            planner_instances=self.created,
            activations=self.activations,
            closed=self.closed,
            poisoned=self.poisoned,
            mesh_cache_hits=self.mesh_hits,
            mesh_cache_misses=self.mesh_misses,
            world_sync_requests=self.world_sync_requests,
            world_data_syncs=self.world_data_syncs,
            mesh_export_cache_hits=self.mesh_export_hits,
            mesh_export_cache_misses=self.mesh_export_misses,
            mesh_file_hash_cache={} if self.mesh_file_cache is None else self.mesh_file_cache.stats(),
            calls=list(self.calls),
        )


class PlannerHandle:
    supports_contact_policy_update = True
    planner_variant = "v2"
    optimization_mode = "single_reuse"

    def __init__(self, owner, settings):
        self.owner = owner
        self.settings = copy.deepcopy(settings)
        self.revision = 0

    def __getattr__(self, name):
        # 诊断配置属于句柄，缺省时保持原planner的hasattr语义。
        # 否则adapter会把引擎上的None误认为已经配置，跳过现场诊断接线。
        if name in {"diagnostic_options", "diagnostic_snapshot", "capture_solver_repr"}:
            raise AttributeError(name)
        engine = self.owner.ensure(self)
        value = getattr(engine, name)
        if not callable(value):
            return value

        def invoke(*args, **kwargs):
            engine = self.owner.ensure(self)
            started = time.perf_counter()
            try:
                return getattr(engine, name)(*args, **kwargs)
            finally:
                if name in (
                    "plan",
                    "solve_ik_batch",
                    "plan_linear_tcp",
                    "plan_pivot_tcp",
                ):
                    self.owner.calls.append(
                        dict(
                            operation=name,
                            duration_s=time.perf_counter() - started,
                            returned_s=time.perf_counter() - self.owner.started,
                        )
                    )

        return invoke

    def close(self):
        # pool淘汰仅释放句柄；GPU资源归外层任务作用域。
        pass

    def _update(self, **kwargs):
        self.owner.check_thread()
        if self.owner.closed or self.owner.poisoned:
            raise RuntimeError("不能更新失效规划句柄")
        self.settings.update(copy.deepcopy(kwargs))
        self.revision += 1

    def update_gripper_geometry(self, **kwargs):
        self._update(**kwargs)

    def update_held_geometry(self, **kwargs):
        self._update(**kwargs)

    def update_scene(self, **kwargs):
        self._update(**kwargs)

    def update_contact_policy(self, policy):
        self._update(mounting_contact_policy=policy)

    def refresh_scene_metadata(self, **kwargs):
        self._update(**kwargs)


def create_planner(*, _baseline_factory=None, **settings):
    if settings.get("path_orientation_constraint") is not None:
        from .orientation_constraint import OrientationConstrainedPlanner
        return OrientationConstrainedPlanner(**settings)
    owner = current_optimization()
    if owner is None:
        from .core import SynchronousPlanner

        return (_baseline_factory or SynchronousPlanner)(**settings)
    return PlannerHandle(owner, settings)


create_planner.supports_contact_policy_update = True


@contextmanager
def optimization_scope(mode="baseline"):
    """仅显式single_reuse启用，默认不改变原版；不隐式fallback。"""
    if mode not in ("baseline", "single_reuse"):
        raise ValueError(f"未知cuRobo优化模式: {mode}")
    from .world_update import mesh_file_cache_scope
    with mesh_file_cache_scope() as file_cache:
        owner = ReuseOwner() if mode == "single_reuse" else None
        if owner is not None:
            owner.mesh_file_cache = file_cache
        from .scene_mesh_cache import mesh_asset_cache_scope
        with mesh_asset_cache_scope(owner):
            token = _OWNER.set(owner)
            try:
                yield owner
            finally:
                try:
                    if owner is not None:
                        owner.close()
                finally:
                    _OWNER.reset(token)
