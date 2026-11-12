"""Task/Skill 同步调用链的规划配置；作用域结束即恢复，不使用环境变量。"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from enum import Enum
from functools import wraps
import inspect
import json
from pathlib import Path
from typing import Mapping


class CollisionMode(str, Enum):
    FULL = "full"
    ROBOT_ONLY = "robot_only"


@dataclass(frozen=True)
class MotionConfig:
    collision_mode: CollisionMode = CollisionMode.FULL

    def to_dict(self) -> dict:
        return {"collision_mode": self.collision_mode.value}


_CONFIG = ContextVar("motion_configuration", default=MotionConfig())


def current_motion_config() -> MotionConfig:
    return _CONFIG.get()


def resolve_collision_mode(value: CollisionMode | str | None = None) -> CollisionMode:
    resolved = CollisionMode(value) if value is not None else current_motion_config().collision_mode
    # 顶层快速模式覆盖旧阶段的 full 默认值，避免阶段偷偷重新构建 world。
    if current_motion_config().collision_mode is CollisionMode.ROBOT_ONLY:
        return CollisionMode.ROBOT_ONLY
    return resolved


@contextmanager
def motion_configuration(config: Mapping | MotionConfig | None = None):
    """接收 motion 字段内容；未指定继承外层，非法字段尽早报错。"""
    if isinstance(config, MotionConfig):
        value = config.collision_mode
    else:
        raw = dict(config or {})
        unknown = set(raw) - {"backend", "collision_mode"}
        if unknown:
            raise ValueError(f"未知 motion 配置字段：{sorted(unknown)}")
        if raw.get("backend", "curobo") != "curobo":
            raise ValueError("当前规划后端仅接入 curobo")
        value = raw.get("collision_mode")
    resolved = MotionConfig(resolve_collision_mode(value))
    token = _CONFIG.set(resolved)
    try:
        yield resolved
    finally:
        _CONFIG.reset(token)


def configured_motion(config_argument: str, *, override_argument: str | None = None):
    """让已有公开入口读取配置的 motion 段，内部算法不再解析站点配置。"""
    def decorate(function):
        signature = inspect.signature(function)
        @wraps(function)
        def run(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            config = bound.arguments.get(config_argument) or {}
            if isinstance(config, (str, Path)):
                config = json.loads(Path(config).read_text(encoding="utf-8"))
            options = dict(config.get("motion") or {})
            override = bound.arguments.get(override_argument) if override_argument else None
            if override is not None:
                options["collision_mode"] = override
            with motion_configuration(options):
                return function(*args, **kwargs)
        return run
    return decorate


def collision_mode_metadata(mode: CollisionMode | str | None = None) -> dict:
    # 已存在 solver 的记录必须反映它实际创建时的模式，不能受外层新作用域改写。
    resolved = CollisionMode(mode) if mode is not None else resolve_collision_mode()
    full = resolved is CollisionMode.FULL
    return {"collision_mode": resolved.value, "self_collision_check": True,
            "environment_collision_enabled": full, "planning_attachment_enabled": full}
