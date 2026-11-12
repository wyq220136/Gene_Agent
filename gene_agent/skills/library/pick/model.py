"""抓取输入、阶段与设置；候选和设备由适配绑定。"""
from __future__ import annotations
from dataclasses import dataclass, field
import json
from pathlib import Path
import numpy as np
from gene_agent.skills.runtime.planning import DEFAULT_STAGE_OPERATIONS

GRASP_STAGE_ORDER = ("open", "pregrasp", "approach", "close")
PICK_STAGE_ORDER = GRASP_STAGE_ORDER + ("lift", "hold")
PICK_STAGE_OPERATIONS = DEFAULT_STAGE_OPERATIONS

@dataclass(frozen=True)
class GraspSettings:
    """任务侧参数。距离按目标 TCP 局部 Z 定义，单位米。"""

    approach_distance_m: float = 0.08
    lift_distance_m: float = 0.08
    open_width_m: float = 0.08
    close_width_m: float = 0.0
    hold_duration_s: float = 0.5
    max_grasp_candidates: int = 40
    pregrasp_approximate_ik_budget: dict[str, float] | None = None
    candidate_selection: str = "plan_candidate"
    max_pregrasp_joint_travel_rad: float | None = None
    max_approach_joint_travel_rad: float | None = None

    def __post_init__(self) -> None:
        values = (self.approach_distance_m, self.lift_distance_m, self.open_width_m,
                  self.hold_duration_s)
        if any(not np.isfinite(value) or value <= 0 for value in values):
            raise ValueError("距离和夹爪开口必须为正有限值")
        if not np.isfinite(self.close_width_m) or self.close_width_m < 0:
            raise ValueError("close_width_m 必须为非负有限值")
        if self.max_grasp_candidates < 1:
            raise ValueError("max_grasp_candidates 至少为 1")
        if self.candidate_selection not in {"plan_candidate", "legacy_ik"}:
            raise ValueError("candidate_selection 只支持 plan_candidate 或 legacy_ik")
        if (self.max_pregrasp_joint_travel_rad is not None
                and (not np.isfinite(float(self.max_pregrasp_joint_travel_rad))
                     or float(self.max_pregrasp_joint_travel_rad) <= 0)):
            raise ValueError("pregrasp 最大单关节行程必须是正有限弧度")
        if self.max_pregrasp_joint_travel_rad is not None:
            object.__setattr__(self, "max_pregrasp_joint_travel_rad",
                               float(self.max_pregrasp_joint_travel_rad))
        if (self.max_approach_joint_travel_rad is not None
                and (not np.isfinite(float(self.max_approach_joint_travel_rad))
                     or float(self.max_approach_joint_travel_rad) <= 0)):
            raise ValueError("approach 最大单关节行程必须是正有限弧度")
        if self.max_approach_joint_travel_rad is not None:
            object.__setattr__(self, "max_approach_joint_travel_rad",
                               float(self.max_approach_joint_travel_rad))
        if self.pregrasp_approximate_ik_budget is not None:
            budget = dict(self.pregrasp_approximate_ik_budget)
            keys = {"max_position_error_m", "max_rotation_error_rad"}
            if set(budget) != keys or any(not np.isfinite(float(budget[key])) or float(budget[key]) <= 0 for key in keys):
                raise ValueError("pregrasp 近似 IK 预算必须包含正的有限平移/转角上限")
            object.__setattr__(self, "pregrasp_approximate_ik_budget", budget)

def load_grasp_settings(path: str | Path, profile: str) -> GraspSettings:
    """从任务目录读取 Skill 参数，不在 Knowledge profile 中掺入执行策略。"""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("不支持的 skill_profiles schema_version")
    return GraspSettings(**payload[profile])

@dataclass(frozen=True)
class PickRequest:
    object_id: str
    group: str
    settings: GraspSettings = field(default_factory=GraspSettings)
    lift: bool = True

    def __post_init__(self):
        if not self.object_id or not self.group:
            raise ValueError('Pick 需要明确的对象和机器人组')
        if not isinstance(self.settings, GraspSettings) or type(self.lift) is not bool:
            raise ValueError('Pick 参数必须为 GraspSettings，lift 必须为 bool')
