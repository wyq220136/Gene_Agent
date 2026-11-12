"""杯子 Task 的 MuJoCo 实际状态回放数据。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from gene_agent.feedback.tabletop.spoon_run import stage_samples


CUP_SKILLS = {
    "cup_pick": "杯把抓取 · MuJoCo 实际回放",
    "cup_pour": "杯子倾倒 · MuJoCo 实际回放",
    "cup_place": "杯子放回 · MuJoCo 实际回放",
}
CUP_TASKS = {"cup_pour_into_bowl": "杯子抓取、倾倒并放回 · MuJoCo 完整任务"}


def find_latest_successful_run(base: Path = Path("runs/cup_pour_task")) -> Path:
    for path in sorted(base.glob("*/task_result.json"),
                       key=lambda item: item.stat().st_mtime, reverse=True):
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
            scene = Path(report["scene_xml_path"])
            if (report.get("success") and report.get("until_skill") == "place" and
                    scene.is_file() and hashlib.sha256(scene.read_bytes()).hexdigest() ==
                    report["scene_xml_sha256"]):
                return path.parent.resolve()
        except (KeyError, OSError, ValueError):
            continue
    raise FileNotFoundError(f"{base} 下没有成功且场景版本匹配的杯子完整 Task run")


def load_replay(run_dir: str | Path, selection: str) -> tuple[dict, list[dict], Path]:
    if selection not in {*CUP_SKILLS, *CUP_TASKS}:
        raise ValueError(f"未知杯子回放选择：{selection}")
    run = Path(run_dir).resolve()
    task = json.loads((run / "task_result.json").read_text(encoding="utf-8"))
    pick = json.loads((run / "pick/result.json").read_text(encoding="utf-8"))
    if not pick.get("success"):
        raise ValueError("杯把抓取 Skill 尚未成功")
    if selection == "cup_pour" and not task.get("skills", {}).get("pour", {}).get("success"):
        raise ValueError("杯子倾倒 Skill 尚未成功")
    if selection == "cup_place" and not task.get("skills", {}).get("place", {}).get("success"):
        raise ValueError("杯子放回 Skill 尚未成功")
    if selection in CUP_TASKS and not task.get("success"):
        raise ValueError("杯子完整 Task 尚未成功")
    stages = []
    if selection in {"cup_pick", *CUP_TASKS}:
        stages.extend((name, pick["stages"].get(name, {}))
                      for name in ("pregrasp", "approach", "close", "lift", "hold"))
    if selection in {"cup_pour", *CUP_TASKS}:
        stages.extend((name, value) for name, value in task["stages"].items()
                      if name in {"carry_raise", "carry_above", "carry_hold",
                                   "pour_approach"} or
                      name.startswith("tilt_") or name.startswith("pour_lower_") or
                      name in {"hold_50_after_lower", "hold_90_after_lower",
                               "pour_settle"})
    if selection in {"cup_place", *CUP_TASKS}:
        stages.extend((name, value) for name, value in task["stages"].items()
                      if name.startswith("upright_") or name in {
                          "pour_raise", "clear_bowl", "return_turn_upright",
                          "return_above", "place_descend", "release_open",
                          "retreat", "retreat_low", "retreat_high", "place_settle"})
    frames = sorted({round(row["sim_time_s"], 7): row
                     for name, value in stages for row in stage_samples(name, value)}.values(),
                    key=lambda row: row["sim_time_s"])
    if len(frames) < 2:
        raise ValueError("run 缺少实际 MuJoCo qpos 采样")
    scene = Path(task["scene_xml_path"])
    if hashlib.sha256(scene.read_bytes()).hexdigest() != task["scene_xml_sha256"]:
        raise ValueError("运行场景 XML 已改变")
    return task, frames, scene
