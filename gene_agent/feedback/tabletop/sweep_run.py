"""从清扫 run 的实际 MuJoCo 采样建立 Skill/Task 共用回放。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


PICK_STAGES = ("pregrasp", "approach", "lift")
SWEEP_STAGES = ("move_behind_cube", "lower_to_contact", "sweep", "retreat")
TASK_STAGES = (*PICK_STAGES, *SWEEP_STAGES, "release", "settle")
SELECTION_STAGES = {
    "pick_brush": PICK_STAGES,
    "pick_brush_long": PICK_STAGES,
    "sweep_cube": SWEEP_STAGES,
    "sweep_long_brush": SWEEP_STAGES,
    "sweep_cube_into_dustpan": TASK_STAGES,
    "sweep_long": TASK_STAGES,
}


def selected_stage_names(report: dict, selection: str) -> tuple[str, ...]:
    """按实际执行顺序选择清扫阶段；支持任意数量的目标方块。"""
    if selection not in SELECTION_STAGES:
        raise ValueError(f"不支持的清扫回放：{selection}")
    names = tuple(report.get("stages", {}))
    if selection in ("sweep_cube_into_dustpan", "sweep_long"):
        return names
    if selection in ("pick_brush", "pick_brush_long"):
        return tuple(name for name in names if name in PICK_STAGES)
    return tuple(name for name in names if name in SWEEP_STAGES or
                 any(name.endswith("__" + stage) for stage in SWEEP_STAGES))


def collect_replay_samples(report: dict, stages: tuple[str, ...]) -> list[dict]:
    """只读取实际执行的 qpos；同一时刻保留后一个阶段的状态。"""
    unique = {}
    for name in stages:
        stage = report.get("stages", {}).get(name, {})
        execution = stage.get("execution", stage)
        for sample in execution.get("replay_samples", ()):
            if "qpos" in sample and "sim_time_s" in sample:
                time_s = float(sample["sim_time_s"])
                unique[round(time_s, 7)] = {
                    "stage": name, "sim_time_s": time_s, "qpos": sample["qpos"]}
    return sorted(unique.values(), key=lambda row: row["sim_time_s"])


def find_latest_compatible_run(scene_xml: str | Path,
                               runs_root: str | Path = "runs/sweep_cube") -> Path:
    """默认选取当前场景哈希匹配的最新成功运行，避免回放旧尺度数据。"""
    digest = hashlib.sha256(Path(scene_xml).read_bytes()).hexdigest()
    candidates = sorted(Path(runs_root).glob("*/result.json"),
                        key=lambda path: path.stat().st_mtime, reverse=True)
    for path in candidates:
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("success") and report.get("scene_xml_sha256") == digest:
            return path.parent.resolve()
    raise FileNotFoundError("没有与当前清扫场景匹配的成功 run；请先执行 run_sweep_cube.py")


def load_replay(run_dir: str | Path, selection: str) -> tuple[dict, list[dict], Path]:
    """核对场景版本与实际 qpos，返回指定 Skill/Task 的记录。"""
    if selection not in SELECTION_STAGES:
        raise ValueError(f"不支持的清扫回放：{selection}")
    run = Path(run_dir).resolve()
    report = json.loads((run / "result.json").read_text(encoding="utf-8"))
    scene_xml = Path(report["scene_xml_path"]).resolve()
    digest = hashlib.sha256(scene_xml.read_bytes()).hexdigest()
    if digest != report["scene_xml_sha256"]:
        raise ValueError("run 的场景哈希与当前 scene.xml 不同，不能用旧尺度回放")
    samples = collect_replay_samples(report, selected_stage_names(report, selection))
    if not samples:
        raise ValueError(f"{selection} 缺少实际 MuJoCo 执行采样")
    return report, samples, scene_xml
