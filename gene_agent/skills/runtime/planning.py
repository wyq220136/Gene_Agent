"""共享阶段规划事实；与实际执行结果分开。"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
from gene_agent.motion.model import JointState, MotionRequest, MotionResult, PlanningSceneRef, PoseTarget

DEFAULT_STAGE_OPERATIONS = {
    "open": "gripper_command", "pregrasp": "plan",
    "approach": "linear_tcp", "close": "gripper_command",
    "lift": "linear_tcp", "hold": "hold",
}



@dataclass(frozen=True)
class SkillContext:
    group: str
    start: JointState
    scene: PlanningSceneRef
    object_id: str
    seed: int = 0

    def __post_init__(self) -> None:
        if not self.group or not self.object_id:
            raise ValueError("group 和 object_id 必须非空")

@dataclass(frozen=True)
class SkillStage:
    name: str
    status: str
    target: PoseTarget | None = None
    request: MotionRequest | None = None
    result: MotionResult | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in {"planned", "failed", "unsupported", "not_executed"}:
            raise ValueError("非法 Skill 阶段状态")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "target": None if self.target is None else self.target.to_dict(),
            "request": None if self.request is None else self.request.to_dict(),
            "result": None if self.result is None else self.result.to_dict(),
            "details": self.details,
        }

@dataclass(frozen=True)
class SkillPlan:
    skill: str
    selected_candidate_id: str | None
    stages: tuple[SkillStage, ...]
    attempts: tuple[dict[str, Any], ...]

    @property
    def success(self) -> bool:
        """只表示所有阶段已有规划或指令，绝非物理成功。"""
        return bool(self.stages) and all(stage.status == "planned" for stage in self.stages)

    def stage(self, name: str) -> SkillStage:
        return next(stage for stage in self.stages if stage.name == name)

    def to_dict(self) -> dict[str, Any]:
        def operation(stage: SkillStage) -> str:
            """运动阶段取实际请求；状态阶段由 Skill 显式声明动作。"""
            if stage.request is not None:
                return stage.request.operation.value
            configured = stage.details.get("operation")
            if configured is not None:
                return str(configured)
            if stage.name in DEFAULT_STAGE_OPERATIONS:
                return DEFAULT_STAGE_OPERATIONS[stage.name]
            raise ValueError(f"阶段 {stage.name} 未声明 operation")

        return {
            "skill": self.skill,
            "claim": ("规划阶段结果；不代表夹持或物理抬升成功"
                      if self.skill in {"pick", "grasp"} else
                      "规划阶段结果；不代表物体实际接触或放置成功"),
            "success": self.success,
            "selected_candidate_id": self.selected_candidate_id,
            "stage_contract": [
                {"name": stage.name, "operation": operation(stage)}
                for stage in self.stages
            ],
            "stages": [stage.to_dict() for stage in self.stages],
            "attempts": list(self.attempts),
        }
