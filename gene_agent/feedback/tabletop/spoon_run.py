"""抓勺、盛球、倾倒运行记录的实际 MuJoCo 状态回放。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


PICK_STAGES = ("pregrasp", "approach", "close", "lift", "hold")
SCOOP_STAGES = ("source_above", "source_pitch", "source_entry", "source_sweep",
                "source_lift_low", "source_level", "source_hold")
POUR_STAGES = ("target_above", "target_pretilt_hold", "target_tilt_30",
               "target_tilt_45", "target_tilt_60", "target_tilt_75",
               "target_settle")
SPOON_SKILLS = {"spoon_pick": "抓勺 · MuJoCo 实际回放",
                "spoon_scoop": "盛球 · MuJoCo 实际回放",
                "spoon_pour": "倾倒 · MuJoCo 实际回放"}
SPOON_TASKS = {"spoon_scoop_into_bowl": "勺子盛球倒入另一碗 · MuJoCo 完整任务"}
_SEGMENT = {"spoon_pick": "pick", "spoon_scoop": "scoop",
            "spoon_pour": "pour", "spoon_scoop_into_bowl": "full"}


def stage_samples(stage: str, value: dict) -> list[dict]:
    execution = value.get("execution") or {}
    rows = execution.get("replay_samples") or value.get("replay_samples") or ()
    return [{"sim_time_s": float(row["sim_time_s"]), "qpos": row["qpos"],
             "stage": stage} for row in rows if "sim_time_s" in row and "qpos" in row]


def find_latest_successful_run(
        base: Path = Path("runs/spoon_scoop_task"),
) -> Path:
    candidates = sorted(base.glob("*/task_result.json"),
                        key=lambda path: path.stat().st_mtime, reverse=True)
    for path in candidates:
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
            pick_run = json.loads((path.parent / "result.json").read_text(encoding="utf-8"))
            source_scene = Path(pick_run["source_scene_xml_path"])
            scene = Path(report["scene_xml_path"])
            if (report.get("success") and
                    pick_run.get("task") in {"scoop_orange", "scoop_pink"} and
                    source_scene.is_file() and
                    hashlib.sha256(source_scene.read_bytes()).hexdigest() ==
                    pick_run["source_scene_xml_sha256"] and
                    scene.is_file() and (
                    hashlib.sha256(scene.read_bytes()).hexdigest() ==
                    report["scene_xml_sha256"])):
                return path.parent.resolve()
        except (KeyError, OSError, ValueError):
            continue
    raise FileNotFoundError(f"{base} 下没有场景版本匹配的成功勺子 Task run")


def load_replay(run_dir: str | Path, selection: str) -> tuple[dict, list[dict], Path]:
    if selection not in _SEGMENT:
        raise ValueError(f"未知勺子回放选择：{selection}")
    run = Path(run_dir).resolve()
    task = json.loads((run / "task_result.json").read_text(encoding="utf-8"))
    pick = json.loads((run / "pick/result.json").read_text(encoding="utf-8"))
    if not pick.get("success"):
        raise ValueError("抓勺 Skill 尚未成功")
    segment = _SEGMENT[selection]
    for name in (() if segment == "pick" else ("scoop",)):
        if not task.get("skills", {}).get(name, {}).get("success"):
            raise ValueError(f"{name} Skill 尚未成功")
    if segment in {"pour", "full"} and not task.get("skills", {}).get("pour", {}).get("success"):
        raise ValueError("倾倒 Skill 尚未成功")
    stages = []
    if segment in {"pick", "full"}:
        stages.extend((name, pick["stages"].get(name, {})) for name in PICK_STAGES)
    if segment in {"scoop", "full"}:
        stages.extend((name, task["stages"].get(name, {})) for name in SCOOP_STAGES)
    if segment in {"pour", "full"}:
        stage_names = (POUR_STAGES if "target_tilt_30" in task["stages"] else
                       ("target_above", "target_pretilt_hold", "target_tilt",
                        "target_settle"))
        stages.extend((name, task["stages"].get(name, {})) for name in stage_names)
    unique = {round(row["sim_time_s"], 7): row
              for name, value in stages for row in stage_samples(name, value)}
    frames = sorted(unique.values(), key=lambda row: row["sim_time_s"])
    if len(frames) < 2:
        raise ValueError("run 缺少实际 MuJoCo qpos 采样")
    scene = Path(task["scene_xml_path"])
    if hashlib.sha256(scene.read_bytes()).hexdigest() != task["scene_xml_sha256"]:
        raise ValueError("运行场景 XML 已改变")
    return task, frames, scene
