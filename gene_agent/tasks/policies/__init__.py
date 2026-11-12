"""任务题目与策略的显式边界；导入包不启动仿真或设备。"""
from pathlib import Path
import hashlib


def identity(task_name: str, *, variant: str = "default") -> dict:
    """记录实际策略/适配源码版本，不导入执行器或初始化设备。"""
    if Path(task_name).name != task_name or not task_name.endswith("_task"):
        raise ValueError("策略目录必须使用 <name>_task")
    directory = Path(__file__).resolve().parent / task_name
    # 操作/适配已按职责拆到子包；身份记录也须覆盖真正执行的嵌套源码。
    paths = sorted(directory.rglob("*.py"))
    if not (directory / "policy.py").is_file():
        raise ValueError(f"策略入口不存在：{task_name}")
    return {"id": f"{task_name}/{variant}", "version": 1,
            "sources_sha256": {str(p.relative_to(directory.parent)): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in paths}}
