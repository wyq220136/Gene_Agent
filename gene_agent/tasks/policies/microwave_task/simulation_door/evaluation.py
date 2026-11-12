"""从同一实际检查点顺序比较少量参数；所有尝试都进入分母。"""

from __future__ import annotations

import json
from pathlib import Path
import re
import time
import traceback

from gene_agent.tasks.policies.microwave_task.simulation_door.session import ExperimentSession


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2,
                               allow_nan=False) + "\n", encoding="utf-8")


def _contacts(run_dir: Path, stages: list[str]) -> dict:
    """把逐阶段接触归类汇总，保留具体 geom 对以便比较失败原因。"""
    counts: dict[str, int] = {}
    pairs: dict[str, int] = {}
    for stage_id in stages:
        path = run_dir / "stages" / f"{stage_id.replace('/', '_')}.json"
        stage = json.loads(path.read_text(encoding="utf-8"))
        for name, value in stage.get("stage_contact_pairs", {}).items():
            steps = int(value["step_count"])
            classification = value["classification"]
            counts[classification] = counts.get(classification, 0) + steps
            pairs[name] = pairs.get(name, 0) + steps
    return {"step_counts_by_class": counts, "step_counts_by_geom_pair": pairs}


def evaluate_parameter_cases(
    *, selection: str, config: Path, seed: int, checkpoint: Path,
    output_dir: Path, until_stage: str, cases: list[dict],
) -> dict:
    """每组补丁从同一物理状态独立运行；失败和无效补丁也必须留痕。"""
    if not cases or len(cases) > 20:
        raise ValueError("一次对照须提供 1–20 个候选，避免无限占用 GPU")
    names = [case.get("name") for case in cases if isinstance(case, dict)]
    if len(names) != len(cases) or any(
        not isinstance(name, str) or re.fullmatch(r"[A-Za-z0-9_-]{1,60}", name) is None
        for name in names
    ):
        raise ValueError("每个候选须有唯一的简短 ASCII name")
    if len(set(names)) != len(cases):
        raise ValueError("候选 name 不能重复")
    if not until_stage:
        raise ValueError("小规模对照必须指定 --until-stage，防止意外运行整项任务")
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    checkpoint = Path(checkpoint).resolve()
    config = Path(config).resolve()
    summary = {
        "schema_version": 1,
        "selection": selection,
        "config_ref": str(config),
        "checkpoint_ref": str(checkpoint),
        "seed": int(seed),
        "until_stage": until_stage,
        "case_count": len(cases),
        "evaluated_count": 0,
        "goal_success_count": 0,
        "reached_until_stage_count": 0,
        "results": [],
    }
    for case in cases:
        name = case["name"]
        patch = case.get("patch", {})
        run_dir = destination / name
        started = time.perf_counter()
        phase = "initialization"
        try:
            session = ExperimentSession(selection, "physics", config, seed,
                                        output_dir=run_dir)
            _write_json(run_dir / "case_input.json", {"name": name, "patch": patch})
            phase = "restore"
            session.restore_checkpoint(checkpoint)
            remaining = [item.phase_id for item in session.planning.phases[
                session.planning.phase_index:session.planning.stop_index
            ]]
            if until_stage not in remaining:
                raise ValueError(f"停止阶段不在检查点之后的 Skill 中：{until_stage}")
            if patch:
                phase = "parameters"
                session.apply_parameters(patch)
            phase = "execution"
            report = session.run(until_stage=until_stage)
            result = {
                "name": name,
                "status": report["status"],
                "goal_success": bool(report["goal_success"]),
                "reached_until_stage": bool(
                    report["stages"] and report["stages"][-1] == until_stage
                    and report["status"] in {"partial", "completed"}
                ),
                "planning_success": bool(report["planning_success"]),
                "execution_success": bool(report["execution_success"]),
                "failure_stage": report["failure_stage"],
                "failure_category": report["failure_category"],
                "actual_door_angle_deg": round(
                    report["actual_final"]["door_angle_rad"] * 180 / 3.141592653589793,
                    3,
                ),
                "stages": report["stages"],
                "contact": _contacts(run_dir, report["stages"]),
                "run_ref": str(run_dir),
            }
        except Exception as error:
            run_dir.mkdir(parents=True, exist_ok=True)
            category = "invalid_parameters" if phase == "parameters" else "experiment_error"
            _write_json(run_dir / "case_error.json", {
                "phase": phase, "category": category,
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            })
            result = {
                "name": name, "status": category, "goal_success": False,
                "reached_until_stage": False,
                "planning_success": False, "execution_success": False,
                "failure_stage": None, "failure_category": category,
                "actual_door_angle_deg": None, "stages": [],
                "contact": {"step_counts_by_class": {}, "step_counts_by_geom_pair": {}},
                "run_ref": str(run_dir),
            }
        result["wall_time_s"] = round(time.perf_counter() - started, 3)
        summary["results"].append(result)
        summary["evaluated_count"] = len(summary["results"])
        summary["goal_success_count"] = sum(
            item["goal_success"] for item in summary["results"]
        )
        summary["reached_until_stage_count"] = sum(
            item["reached_until_stage"] for item in summary["results"]
        )
        # 每轮结束立即保存。进程中断时，已经完成的失败样本也不会丢失。
        _write_json(destination / "summary.json", summary)
    return summary


__all__ = ["evaluate_parameter_cases"]
