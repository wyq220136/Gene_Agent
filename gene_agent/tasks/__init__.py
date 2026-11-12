"""与场景和执行后端无关的同步 Task 组合接口。"""

from .checks import CheckResult, resolve_unknown
from .runner import TaskRun, run_steps

__all__ = ["CheckResult", "TaskRun", "resolve_unknown", "run_steps"]
