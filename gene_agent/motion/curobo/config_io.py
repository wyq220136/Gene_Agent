"""只缓存静态机器人 YAML 解析；每次读取内容并返回独立副本。"""
from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import hashlib
from pathlib import Path
import yaml


@lru_cache(maxsize=16)
def _parse_robot_yaml(_path: str, content: bytes) -> dict:
    # 路径和完整内容共同作键；不能只按 mtime 或删掉源模型身份。
    return yaml.safe_load(content)


def load_robot_yaml(path: str | Path) -> tuple[dict, str]:
    source = Path(path).resolve()
    content = source.read_bytes()
    return deepcopy(_parse_robot_yaml(str(source), content)), hashlib.sha256(content).hexdigest()
