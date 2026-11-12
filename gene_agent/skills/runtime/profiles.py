"""操作参数的最终值和来源；物理材料、设备连接参数保持各自配置。"""
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path


@dataclass(frozen=True)
class ResolvedProfile:
    values: dict
    sources: dict
    sha256: str

    def to_dict(self):
        return {'values': dict(self.values), 'sources': dict(self.sources), 'sha256': self.sha256}


def resolve_operation_profile(base, object_profile=None, environment_profile=None, overrides=None):
    values = dict(base)
    sources = {key: 'task_default' for key in values}
    for name, layer in [('object_profile', object_profile), ('environment_profile', environment_profile),
                        ('explicit_override', overrides)]:
        layer = layer or {}
        if set(layer) - set(base):
            raise ValueError(f'未知操作参数：{set(layer) - set(base)}')
        for key, value in layer.items():
            values[key] = value
            sources[key] = name
    encoded = json.dumps({'values': values, 'sources': sources}, sort_keys=True, allow_nan=False)
    return ResolvedProfile(values, sources, hashlib.sha256(encoded.encode()).hexdigest())


