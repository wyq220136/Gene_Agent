"""微波炉 Task1 的同步纯视觉编排；只接受连续的真实阶段结果。"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Callable, Mapping

import numpy as np

from gene_agent.knowledge.geometry import validate_pose
from gene_agent.tasks.runner import run_steps


# 旧 MicrowaveTask1 把两轮开关门包在各自 DoorCycleTask 内，并在中间
# 注入放入或取出 Task；展开后共有七个状态交接边界。
TASK1_STEP_ORDER = (
    "open_door_first", "insert_bowl", "close_door_first", "rotate_knob",
    "open_door_second", "extract_place_bowl", "close_door_second",
)


@dataclass(frozen=True, eq=False)
class TaskState:
    """跨 Skill 交接的全部可见状态，连同资产来源一起核对。"""

    model_sha256: str
    joint_names: tuple[str, ...]
    joint_positions_rad: tuple[float, ...]
    gripper_width_m: float
    door_angle_rad: float
    knob_angle_rad: float
    object_poses_world: Mapping[str, np.ndarray] = field(default_factory=dict)
    attachments: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        names = tuple(self.joint_names)
        positions = tuple(float(value) for value in self.joint_positions_rad)
        if not self.model_sha256 or not names or len(names) != len(set(names)):
            raise ValueError("任务状态必须指定同一资产及具名机器人关节")
        if len(positions) != len(names) or not all(math.isfinite(value) for value in positions):
            raise ValueError("机器人关节状态数量不符或含非有限值")
        scalar = (self.gripper_width_m, self.door_angle_rad, self.knob_angle_rad)
        if not all(math.isfinite(float(value)) for value in scalar) or self.gripper_width_m < 0:
            raise ValueError("夹爪及门/旋钮关节状态必须有效")
        poses = {str(name): validate_pose(pose, field_name=name).copy()
                 for name, pose in self.object_poses_world.items()}
        attachments = tuple(sorted(set(self.attachments)))
        if any(name not in poses for name in attachments):
            raise ValueError("附着对象必须有世界位姿")
        object.__setattr__(self, "joint_names", names)
        object.__setattr__(self, "joint_positions_rad", positions)
        object.__setattr__(self, "object_poses_world", poses)
        object.__setattr__(self, "attachments", attachments)

    def discontinuity(self, next_state: TaskState, *, tolerance: float = 1e-5) -> str | None:
        """比较下段实际规划起点；任何资产或视觉状态差异都不能隐藏。"""
        if self.model_sha256 != next_state.model_sha256:
            return "asset_frame_changed"
        if self.joint_names != next_state.joint_names:
            return "joint_names_changed"
        if np.max(np.abs(np.asarray(self.joint_positions_rad) - next_state.joint_positions_rad)) > tolerance:
            return "robot_joint_reset"
        for name in ("gripper_width_m", "door_angle_rad", "knob_angle_rad"):
            if abs(float(getattr(self, name)) - float(getattr(next_state, name))) > tolerance:
                return f"{name}_changed"
        if set(self.object_poses_world) != set(next_state.object_poses_world):
            return "object_set_changed"
        for name in self.object_poses_world:
            if not np.allclose(self.object_poses_world[name], next_state.object_poses_world[name],
                               atol=tolerance, rtol=0.0):
                return f"object_pose_changed:{name}"
        if self.attachments != next_state.attachments:
            return "attachment_changed"
        return None

    def to_dict(self) -> dict:
        return {
            "model_sha256": self.model_sha256,
            "joint_names": list(self.joint_names),
            "joint_positions_rad": list(self.joint_positions_rad),
            "gripper_width_m": float(self.gripper_width_m),
            "door_angle_rad": float(self.door_angle_rad),
            "knob_angle_rad": float(self.knob_angle_rad),
            "object_poses_world": {name: pose.tolist()
                                   for name, pose in self.object_poses_world.items()},
            "attachments": list(self.attachments),
        }


@dataclass(frozen=True)
class TaskStepResult:
    """适配器交还的实际起终态；失败结果不得声称成功轨迹。"""

    step: str
    start_state: TaskState
    end_state: TaskState
    success: bool
    stages: tuple[dict, ...]
    failure: dict | None = None

    def __post_init__(self) -> None:
        if not self.step:
            raise ValueError("Task 阶段名称不能为空")
        object.__setattr__(self, "stages", tuple(self.stages))
        if self.success and not self.stages:
            raise ValueError("成功阶段必须有真实适配器阶段记录")
        if not self.success and not self.failure:
            raise ValueError("失败阶段必须保留诊断")


@dataclass(frozen=True)
class Task1VisualReport:
    success: bool
    steps: tuple[dict, ...]
    final_state: TaskState
    first_failure: dict | None
    configuration: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "schema_version": 1, "task": "microwave_task1",
            "source_format": "planned_visual_task", "success": self.success,
            "step_order": list(TASK1_STEP_ORDER),
            "steps": _json_safe(self.steps),
            "final_state": self.final_state.to_dict(),
            "first_failure": _json_safe(self.first_failure),
            "configuration": _json_safe(self.configuration),
            "claim": "仅记录真实规划和运动学视觉状态；不表示物理接触成功",
        }


def _json_safe(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def write_task1_report(path: str | Path, report: Task1VisualReport) -> Path:
    """在失败点也保存完整前缀和最后一个已验证状态。"""
    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return destination


def run_task1_visual(
    initial_state: TaskState,
    adapters: Mapping[str, Callable[[TaskState], TaskStepResult]],
) -> Task1VisualReport:
    """沿用同一组子任务 adapter，实际状态由上一段交给下一段。"""
    run = run_steps(TASK1_STEP_ORDER, initial_state, adapters)
    return Task1VisualReport(
        run.success, run.steps, run.final_state, run.first_failure,
    )


def plan_microwave_task1_visual(
    output_dir: str | Path, *,
    variant: str = "task1_visual_explicit",
    knob_profile_variant: str = "default",
    microwave_xy_m: tuple[float, float] | None = None,
) -> Task1VisualReport:
    """CLI 与在线 Viewer 使用同一增量会话和成功前缀终态。"""
    from .visual_session import TASK1_SELECTION, Task1VisualSession
    from gene_agent.visualization.microwave.catalog import ROOT

    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = {"variant": variant, "knob_profile_variant": knob_profile_variant}
    if microwave_xy_m is not None:
        xy = np.asarray(microwave_xy_m, dtype=float)
        if xy.shape != (2,) or not np.isfinite(xy).all():
            raise ValueError("微波炉 XY 摆位必须是两个有限数")
        source = json.loads((ROOT / "data/task_1/skills/door_task.json").read_text(encoding="utf-8"))
        microwave = np.asarray(source["microwave_pose_world"], dtype=float).copy()
        microwave[:2, 3] = xy
        config["object_poses_world"] = {"microwave": microwave.tolist()}
    config_path = output / "task1_cli_config.json"
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    session = Task1VisualSession(output_dir=output, config_path=config_path)
    session.begin_plan(TASK1_SELECTION)
    while True:
        result = session.plan_step()
        if result["done"]:
            break
    payload = json.loads((session.catalog.output_dir / "task1_visual_result.json").read_text(
        encoding="utf-8"))
    report = Task1VisualReport(
        bool(payload["success"]), tuple(payload["steps"]),
        TaskState(**payload["final_state"]), payload["first_failure"],
        dict(payload["configuration"]),
    )
    # 保留原 CLI 的稳定报告路径；revision 目录同时保存逐阶段实轨迹。
    write_task1_report(output / "task1_visual_result.json", report)
    return report


__all__ = ["TASK1_STEP_ORDER", "TaskState", "TaskStepResult", "Task1VisualReport",
           "run_task1_visual", "write_task1_report", "plan_microwave_task1_visual"]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="规划微波炉 Task1 纯视觉连续阶段")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", choices=("task1_visual_explicit", "legacy_viser"),
                        default="task1_visual_explicit")
    parser.add_argument("--knob-profile", choices=("default", "tip_x_5_azimuth_240"),
                        default="default")
    parser.add_argument("--microwave-xy", type=float, nargs=2, metavar=("X", "Y"))
    arguments = parser.parse_args()
    result = plan_microwave_task1_visual(
        arguments.output, variant=arguments.variant,
        knob_profile_variant=arguments.knob_profile,
        microwave_xy_m=None if arguments.microwave_xy is None else tuple(arguments.microwave_xy),
    )
    print(json.dumps({
        "success": result.success,
        "completed_steps": sum(row["status"] == "planned" for row in result.steps),
        "first_failure": result.first_failure,
        "report": str(arguments.output.resolve() / "task1_visual_result.json"),
    }, ensure_ascii=False))
