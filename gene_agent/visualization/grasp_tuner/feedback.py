"""把当前 Knowledge 调参界面状态写给截图与自动检查工具。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def write_live_state(path: str | Path | None, payload: dict) -> None:
    """一次性替换状态文件，截图进程永远只会读到完整 JSON。"""
    if path is None:
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(target)


def read_tuner_state(path: str | Path, *, tool: str) -> dict:
    """接受实时状态 JSON 或截图清单，并只恢复有确定候选的有效状态。"""
    source = Path(path)
    document = json.loads(source.read_text(encoding="utf-8"))
    state = document.get("state", document)
    if not isinstance(state, dict) or state.get("tool") != tool or state.get("schema_version") != 1:
        raise ValueError(f"{source}: 不是 {tool} 的 Knowledge 调参状态")
    if not state.get("input_valid"):
        raise ValueError(f"{source}: 输入无效的截图只记录上一次有效画面，不能当作当前参数恢复")
    return state


__all__ = ["canonical_sha256", "write_live_state", "read_tuner_state"]
