"""同次观测并行读取独立设备；同一 SDK 连接始终串行，离开前等待全部读取。"""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, ExitStack
from threading import Lock
import time

from gene_agent.realworld.gripper_state import endpoint_overrides, retained_width_source


class RobotStateSampler:
    def __init__(self, robots, grippers):
        if set(robots) != {"left", "right"} or set(grippers) != set(robots):
            raise ValueError("状态采集需要两臂及各自夹爪")
        self.robots, self.grippers = dict(robots), dict(grippers)
        self.pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="robot-state")
        self.locks = {}
        self.group_locks = {}
        for group in robots:
            devices = {id(getattr(obj, "sdk", obj)) for obj in (robots[group], grippers[group])}
            self.group_locks[group] = [self.locks.setdefault(k, Lock()) for k in sorted(devices)]
        self.busy = False
        self.closed = False

    def _read(self, group, overrides):
        with ExitStack() as stack:
            for lock in self.group_locks[group]:
                stack.enter_context(lock)
            started = time.monotonic()
            started_ms = time.time() * 1000.
            state = self.robots[group].read_joint_state()
            tcp = self.robots[group].tcp_pose_from_sdk_fk(state.positions_rad)
            width = (float(overrides[group]) if group in overrides
                     else float(self.grippers[group].read_width_m()))
            return {"joint_state": state, "tcp_pose_sdk": tcp.tolist(), "width_m": width,
                    "width_source": (retained_width_source(self.grippers[group], width)
                                     if group in overrides else "sdk_live_read"),
                    "started_monotonic_s": started, "ended_monotonic_s": time.monotonic(),
                    "started_at_unix_ms": started_ms, "sampled_at_unix_ms": time.time() * 1000.}

    @contextmanager
    def sample(self, overrides=None):
        if self.closed or self.busy:
            raise RuntimeError("状态采集器已关闭或已有采集正在执行")
        overrides = endpoint_overrides(self.grippers, overrides)
        if set(overrides) - set(self.robots):
            raise ValueError("未知的夹爪开口复用组")
        self.busy = True
        futures = {}
        try:
            for group in self.robots:
                futures[group] = self.pool.submit(self._read, group, overrides)
            class Batch:
                def result(self):
                    return {group: future.result() for group, future in futures.items()}
            yield Batch()
        finally:
            # 视觉异常也必须收齐设备读取，禁止后台查询跨入下一次运动。
            for future in futures.values():
                try:
                    future.result()
                except Exception:
                    pass  # 正常路径在 Batch.result 原样抛出；异常路径保留首个视觉异常。
            self.busy = False

    def close(self):
        if not self.closed:
            self.closed = True
            self.pool.shutdown(wait=True, cancel_futures=True)
