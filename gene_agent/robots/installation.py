"""从冻结的机器人安装配置计算 SDK 与规划 base 变换。"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from gene_agent.knowledge.geometry.transforms import validate_pose


@dataclass(frozen=True)
class InstallationTransforms:
    sdk_to_curobo: dict[str, np.ndarray]
    body_from_sdk: dict[str, np.ndarray]

    @classmethod
    def from_file(cls, path: str | Path, installation: str = "mounted") -> "InstallationTransforms":
        data = json.loads(Path(path).read_text(encoding="utf-8"))[installation]
        return cls(
            sdk_to_curobo={side: validate_pose(pose, field_name=f"sdk_to_curobo.{side}")
                           for side, pose in data["sdk_to_curobo"].items()},
            body_from_sdk={side: validate_pose(pose, field_name=f"body_from_sdk.{side}")
                           for side, pose in data["body_from_sdk"].items()},
        )

    def sdk_from_sdk(self, source: str, target: str) -> np.ndarray:
        """T_target_sdk_source_sdk = inv(T_body_target_sdk) @ T_body_source_sdk。"""
        return np.linalg.inv(self.body_from_sdk[target]) @ self.body_from_sdk[source]

    def reference_from_curobo(self, side: str, *, reference_side: str = "right") -> np.ndarray:
        """由 body 中的安装关系组合两个规划 base。"""
        body_from_side = self.body_from_sdk[side] @ np.linalg.inv(self.sdk_to_curobo[side])
        body_from_reference = self.body_from_sdk[reference_side] @ np.linalg.inv(
            self.sdk_to_curobo[reference_side]
        )
        return np.linalg.inv(body_from_reference) @ body_from_side
