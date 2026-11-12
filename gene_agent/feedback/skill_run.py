"""把规划和执行产物索引为同一种只读Skill阶段记录。"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class StageView:
    stage_id: str
    name: str
    operation: str | None
    planning_status: str
    execution_status: str
    request_ref: str | None
    result_ref: str | None
    collision_snapshot_ref: str | None
    execution_ref: str | None
    start_state_id: str | None
    end_state_id: str | None
    start_state_ref: str | None = None
    end_state_ref: str | None = None
    samples_ref: str | None = None
    path_kind: str | None = None
    attachment_ref: str | None = None
    actual_sample_format: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return vars(self).copy()


@dataclass(frozen=True)
class SkillRunView:
    skill: str
    instance_id: str
    selected_candidate_id: str | None
    source_format: str
    scene_snapshot_sha256: str | None
    stages: tuple[StageView, ...]
    video_manifest_ref: str | None = None
    model_ref: str | None = None
    physical_success: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1, "skill": self.skill,
            "instance_id": self.instance_id,
            "selected_candidate_id": self.selected_candidate_id,
            "source_format": self.source_format,
            "scene_snapshot_sha256": self.scene_snapshot_sha256,
            "video_manifest_ref": self.video_manifest_ref,
            "model_ref": self.model_ref,
            "physical_success": self.physical_success,
            "stages": [stage.to_dict() for stage in self.stages],
        }


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"运行报告必须为JSON对象：{path}")
    return value


def _existing(root: Path, relative: str) -> str | None:
    return relative if (root / relative).is_file() else None


def load_skill_run(root: str | Path, *, skill: str | None = None) -> SkillRunView:
    """按产物格式读取；旧probe与新Skill规划的能力边界分别保留。"""
    run = Path(root).resolve()
    from gene_agent.feedback.microwave_actual_run import load_microwave_actual_run
    recorded = load_microwave_actual_run(run, skill=skill)
    if recorded is not None:
        return recorded[0]
    from gene_agent.feedback.replay_adapters import load_tabletop_actual_run
    recorded = load_tabletop_actual_run(run, skill=skill)
    if recorded is not None:
        return recorded[0]
    if skill not in {None, "pick", "insert"}:
        raise ValueError(f"当前运行目录不支持的 Skill 选择：{skill}")
    insert_path = run / "skill_insert_bowl_result.json"
    if skill == "insert":
        if not insert_path.is_file():
            raise FileNotFoundError(f"当前场景缺少放入 Skill 规划报告：{insert_path}")
        report = _read(insert_path)
        if report.get("schema_version") != 1 or report.get("skill") != "insert":
            raise ValueError("不支持的放入 Skill 规划报告")
        contracts = {row["name"]: row["operation"] for row in report["stage_contract"]}
        stages = []
        for index, row in enumerate(report["stages"]):
            request, result = row.get("request") or {}, row.get("result") or {}
            ref = row.get("collision_snapshot_ref")
            if ref is not None:
                snapshot_path = (run / ref).resolve()
                if not snapshot_path.is_relative_to(run) or not snapshot_path.is_file():
                    raise FileNotFoundError(f"阶段 {row['name']} 缺少碰撞快照：{ref}")
                snapshot = _read(snapshot_path)
                if request and request.get("scene", {}).get("snapshot_sha256") != snapshot.get("snapshot_sha256"):
                    raise ValueError(f"阶段 {row['name']} 的请求与碰撞快照不一致")
            attachment_ref = row.get("attachment_ref")
            if attachment_ref is not None:
                path = (run / attachment_ref).resolve()
                if not path.is_relative_to(run) or not path.is_file():
                    raise FileNotFoundError(f"阶段 {row['name']} 缺少附着状态：{attachment_ref}")
            operation = request.get("operation") or contracts[row["name"]]
            if request and result and request.get("operation") != result.get("operation"):
                raise ValueError(f"阶段 {row['name']} 的请求/结果操作类型不一致")
            stages.append(StageView(
                f"{index:02d}_{row['name']}", row["name"], operation,
                row["status"], "not_executed",
                "skill_insert_bowl_result.json" if request else None,
                "skill_insert_bowl_result.json", ref,
                None, None, None, path_kind=operation,
                attachment_ref=attachment_ref,
            ))
        return SkillRunView(
            "insert", report["instance_id"], report.get("selected_candidate_id"),
            "skill_plan", None, tuple(stages),
            model_ref=report.get("model_ref") or _existing(run, "source_scene/scene.xml"),
        )
    articulated_name = next((name for name in (
        "skill_rotate_knob_0_result.json", "skill_open_door_result.json",
        "skill_close_door_result.json") if (run / name).is_file()), None)
    if articulated_name is not None:
        articulated = run / articulated_name
        report = _read(articulated)
        expected_skill = articulated_name.removeprefix("skill_").removesuffix("_result.json")
        if report.get("schema_version") != 1 or report.get("skill") != expected_skill:
            raise ValueError("不支持的关节 Skill 规划报告")
        stages = []
        for index, row in enumerate(report["stages"]):
            collision_ref = row.get("collision_snapshot_ref")
            if not collision_ref or not (run / collision_ref).is_file():
                raise FileNotFoundError(f"关节阶段 {row['name']} 缺少碰撞快照")
            snapshot = _read(run / collision_ref)
            request, result = row.get("request") or {}, row.get("result") or {}
            if request and request.get("scene", {}).get("snapshot_sha256") != snapshot.get("snapshot_sha256"):
                raise ValueError(f"关节阶段 {row['name']} 的请求与碰撞快照不一致")
            if request and result and request.get("operation") != result.get("operation"):
                raise ValueError(f"关节阶段 {row['name']} 的规划操作类型不一致")
            stages.append(StageView(
                f"{index:02d}_{row['name']}", row["name"], row["operation"],
                row["status"], "not_executed",
                articulated_name if request else None,
                articulated_name, collision_ref,
                None, None, None, path_kind=(
                    "axis_waypoint" if row["name"].startswith(("rotate_", "swing_")) else row["operation"]
                ),
            ))
        model_ref = report.get("model_ref")
        if not model_ref or not (run / model_ref).is_file():
            raise FileNotFoundError("关节 Skill 缺少 MuJoCo 场景模型")
        return SkillRunView(
            report["skill"], report["instance_id"], report.get("selected_candidate_id"),
            "skill_plan", None, tuple(stages), model_ref=model_ref,
        )
    planning = run / "skill_pick_bowl_result.json"
    if planning.is_file():
        report = _read(planning)
        if report.get("schema_version") != 1 or report.get("skill") != "pick":
            raise ValueError("不支持的Skill规划报告")
        scene_hash = report.get("snapshot_sha256")
        snapshot_ref = _existing(run, "collision/collision_snapshot.json")
        if snapshot_ref is None:
            raise FileNotFoundError("Skill规划报告缺少碰撞快照")
        snapshot = _read(run / snapshot_ref)
        if scene_hash != snapshot.get("snapshot_sha256"):
            raise ValueError("Skill规划报告与碰撞快照哈希不一致")
        attachment = report.get("visual_attachment") or {}
        attached_ref = None
        attached_hash = None
        if attachment:
            path = Path(str(attachment.get("collision_snapshot_ref", ""))).resolve()
            if not path.is_relative_to(run) or not path.is_file():
                raise ValueError("Pick 附着碰撞快照必须位于当前运行目录")
            attached_ref = str(path.relative_to(run))
            attached_hash = _read(path).get("snapshot_sha256")
            if attached_hash != attachment.get("snapshot_sha256"):
                raise ValueError("Pick 附着碰撞快照哈希不一致")
        attachment_state_ref = (
            _existing(run, "visual_attachment/attachment_state.json")
            if attached_ref else None
        )
        if attachment_state_ref:
            attachment_state = _read(run / attachment_state_ref)
            if attachment_state.get("object_instance_id") != report.get("instance_id"):
                raise ValueError("Pick 附着物实例与 Skill 报告不一致")
        contract_operations = {
            item["name"]: item["operation"]
            for item in report.get("stage_contract", [])
            if isinstance(item, dict) and item.get("name") and item.get("operation")
        }
        # 旧规划报告没有阶段契约；非运动阶段仍应显示其实际动作类型。
        default_operations = {
            "open": "gripper_command", "close": "gripper_command",
            "lift": "linear_tcp", "hold": "hold",
        }
        stages = []
        for index, row in enumerate(report["stages"]):
            request, result = row.get("request") or {}, row.get("result") or {}
            after_close = row["name"] in {"lift", "hold"}
            if request and result and request.get("operation") != result.get("operation"):
                raise ValueError(f"阶段{row['name']}的请求/结果操作类型不一致")
            operation = (request.get("operation") or contract_operations.get(row["name"])
                         or default_operations.get(row["name"]))
            stage_snapshot_ref = row.get("collision_snapshot_ref")
            if stage_snapshot_ref is None:
                # 兼容旧报告：尚未逐阶段导出夹爪开口及目标接触快照。
                if row["name"] == "close":
                    stage_snapshot_ref = attached_ref or snapshot_ref
                elif request:
                    stage_snapshot_ref = attached_ref if after_close else snapshot_ref
            if stage_snapshot_ref is not None:
                stage_path = (run / stage_snapshot_ref).resolve()
                if not stage_path.is_relative_to(run) or not stage_path.is_file():
                    raise FileNotFoundError(f"阶段 {row['name']} 缺少碰撞快照：{stage_snapshot_ref}")
                stage_snapshot_ref = str(stage_path.relative_to(run))
                stage_snapshot = _read(stage_path)
                if request and request.get("scene", {}).get("snapshot_sha256") != stage_snapshot.get("snapshot_sha256"):
                    raise ValueError(f"阶段 {row['name']} 的请求与碰撞快照不一致")
            elif request:
                raise FileNotFoundError(f"阶段 {row['name']} 缺少规划碰撞快照")
            stages.append(StageView(
                f"{index:02d}_{row['name']}", row["name"], operation,
                row["status"], "not_executed", "skill_pick_bowl_result.json" if request else None,
                "skill_pick_bowl_result.json" if result or row["name"] in {"open", "close", "hold"} else None,
                stage_snapshot_ref,
                None, None, None,
                path_kind=operation,
                attachment_ref=(attachment_state_ref or attached_ref)
                if row["name"] in {"close", "lift", "hold"} and attached_ref else None,
            ))
        instance_id = report.get("instance_id")
        if instance_id is None and report.get("task") == "pick_bowl_skill_planning":
            instance_id = "bowl"  # 兼容旧报告；新写入器显式保存实例ID。
        if not isinstance(instance_id, str) or not instance_id:
            raise ValueError("Skill报告缺少物体实例ID")
        return SkillRunView("pick", instance_id, report.get("selected_candidate_id"),
                            "skill_plan", scene_hash, tuple(stages),
                            model_ref=(_existing(run, "source_scene/scene.xml")
                                       or _existing(run, "model/scene.xml")))

    report_path = run / "result.json"
    if not report_path.is_file():
        raise FileNotFoundError(f"找不到Skill运行报告：{run}")
    report = _read(report_path)
    source_format = {
        "temporary_pick_bowl_operation_interface_probe": "legacy_execution_probe",
        "synchronous_pick_skill": "skill_execution",
    }.get(report.get("mode"))
    if source_format is None:
        raise ValueError("不支持的Skill执行报告格式")
    stages = []
    state_names = list(report.get("stage_state_ids", {}))
    active_attachment_ref = None
    for index, (name, row) in enumerate(report["stages"].items()):
        plan_ref = _existing(run, f"plans/{name}.json")
        execution_ref = _existing(run, f"executions/{name}.json")
        if execution_ref is None and name in {"open", "close"}:
            execution_ref = "result.json"
        collision_ref = _existing(run, f"collision/{name}/collision_snapshot.json")
        plan = _read(run / plan_ref) if plan_ref else {}
        execution = (_read(run / execution_ref) if execution_ref and execution_ref != "result.json"
                     else row if execution_ref else {})
        summary_execution = row.get("execution", row)
        if execution_ref:
            for key in ("start_state_id", "end_state_id"):
                if execution.get(key) != summary_execution.get(key):
                    raise ValueError(f"阶段{name}{'起点' if key.startswith('start') else '终点'}状态ID与汇总报告不一致")
        operation = plan.get("operation") if plan_ref else ("hold" if name == "hold" else "gripper_command")
        active_attachment_ref = _existing(run, f"attachments/{name}.json") or active_attachment_ref
        if plan_ref and collision_ref is None:
            raise FileNotFoundError(f"阶段{name}缺少实际规划碰撞快照")
        if plan_ref and plan.get("scene_snapshot_sha256") != _read(run / collision_ref).get("snapshot_sha256"):
            raise ValueError(f"阶段{name}规划与碰撞快照不一致")
        start_name = state_names[index] if index < len(state_names) else None
        end_name = state_names[index + 1] if execution_ref and index + 1 < len(state_names) else None
        start_observation_ref = f"stages/{start_name}/observation.json" if start_name else None
        end_observation_ref = f"stages/{end_name}/observation.json" if end_name else None
        for boundary, relative, state_id in (
            ("起点", start_observation_ref, execution.get("start_state_id") if execution_ref else report["stage_state_ids"].get(start_name)),
            ("终点", end_observation_ref, execution.get("end_state_id") if execution_ref else None),
        ):
            if boundary == "终点" and not execution_ref:
                continue
            if relative is None or not (run / relative).is_file():
                raise FileNotFoundError(f"阶段{name}缺少{boundary}状态观测")
            if _read(run / relative).get("state_id") != state_id:
                raise ValueError(f"阶段{name}{boundary}状态ID与执行结果不一致")
        replay_samples = execution.get("replay_samples") or []
        if replay_samples and end_observation_ref:
            end_observation = _read(run / end_observation_ref)
            final_sample = replay_samples[-1]
            if (final_sample.get("state_id") != end_observation["state_id"]
                    or abs(float(final_sample.get("sim_time_s", -1))
                           - float(end_observation["sim_time_s"])) > 1e-8):
                raise ValueError(f"阶段{name}完整采样终点与状态边界不一致")
        stages.append(StageView(
            f"{index:02d}_{name}", name, operation,
            "planned" if plan.get("success") else ("not_applicable" if not plan_ref else "failed"),
            ("completed" if execution.get("success") else "failed") if execution_ref else "not_executed",
            plan_ref, plan_ref, collision_ref, execution_ref,
            execution.get("start_state_id") if execution_ref else report["stage_state_ids"].get(start_name),
            execution.get("end_state_id") if execution_ref else None,
            start_observation_ref, end_observation_ref,
            execution_ref if replay_samples or execution.get("samples") else None,
            operation,
            active_attachment_ref,
            ("complete_actual_state" if replay_samples else
             "partial_actual_sample" if execution.get("samples") else None),
        ))
    for before, after in zip(stages, stages[1:]):
        if before.end_state_id is not None and before.end_state_id != after.start_state_id:
            raise ValueError(f"执行状态链不连续：{before.name}→{after.name}")
    video = _existing(run, "visual/manifest.json")
    video_data = _read(run / video) if video else {}
    instance_id = report.get("instance_id") or video_data.get("target_body")
    if not isinstance(instance_id, str) or not instance_id:
        raise ValueError("旧运行报告缺少目标实例ID")
    return SkillRunView("pick", instance_id, report.get("candidate_id"),
                        source_format, None, tuple(stages), video,
                        _existing(run, "model/scene.xml"),
                        report.get("physical_pick_success"))


def write_skill_run_index(root: str | Path) -> Path:
    """导出小索引；轨迹、执行采样和碰撞资产仍使用原始文件。"""
    run = Path(root).resolve()
    output = run / "skill_run_index.json"
    output.write_text(
        json.dumps(load_skill_run(run).to_dict(), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output
