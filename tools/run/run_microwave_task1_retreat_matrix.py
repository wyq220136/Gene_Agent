"""并行复跑固定失败 seed，比较释放后撤退净空策略。"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import threading
import time


ROOT = Path(__file__).resolve().parents[2]
PYTHON = Path("/data1/yuquan/miniconda3/envs/gene_agent/bin/python")
ENTRYPOINT = ROOT / "tools/run/run_microwave_task1_physical.py"
DEFAULT_CONFIG = ROOT / "data/task_1/simulation/mujoco_task1_physical_left_lower_high_variance50.json"
VARIANTS = (
    ("dynamic_2mm", .002, 0.),
    ("dynamic_5mm", .005, 0.),
    ("minimum_lift_20mm", .002, .020),
    ("preplace_lift_40mm", .002, .040),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def append_jsonl(path: Path, value, lock: threading.Lock) -> None:
    with lock, path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def read_summary(run_dir: Path) -> dict:
    path = run_dir / "summary.json"
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--gpus", default="1,2,4,5,6,7,8,9")
    parser.add_argument("--seeds", default="9836,9840")
    args = parser.parse_args()

    base_config_path = args.config.resolve()
    gpus = tuple(value.strip() for value in args.gpus.split(",") if value.strip())
    seeds = tuple(int(value.strip()) for value in args.seeds.split(",") if value.strip())
    jobs = [(variant, margin, minimum, seed)
            for variant, margin, minimum in VARIANTS for seed in seeds]
    if len(gpus) < len(jobs):
        parser.error(f"需要至少 {len(jobs)} 张 GPU，当前只给了 {len(gpus)} 张")
    if not base_config_path.is_file() or not PYTHON.is_file() or not ENTRYPOINT.is_file():
        parser.error("基础配置、解释器或 Task1 入口不存在")

    run_root = args.run_root.resolve()
    allowed_bootstrap_files = {"launcher.log", "launcher.pid"}
    if run_root.exists() and any(
            path.name not in allowed_bootstrap_files for path in run_root.iterdir()):
        parser.error(f"运行目录必须不存在或仅含启动元数据：{run_root}")
    run_root.mkdir(parents=True, exist_ok=True)
    config_dir = run_root / "configs"
    config_dir.mkdir()
    base_config = json.loads(base_config_path.read_text(encoding="utf-8"))
    variant_configs = {}
    for variant, margin, minimum in VARIANTS:
        config = dict(base_config)
        config["insert_retreat_clearance_margin_m"] = margin
        config["insert_retreat_minimum_lift_m"] = minimum
        path = config_dir / f"{variant}.json"
        write_json(path, config)
        variant_configs[variant] = path

    job_specs = []
    for index, ((variant, margin, minimum, seed), gpu) in enumerate(zip(jobs, gpus), 1):
        job_specs.append({
            "run": f"run_{index:03d}", "variant": variant, "seed": seed,
            "gpu": gpu, "clearance_margin_m": margin,
            "minimum_lift_m": minimum,
            "config": str(variant_configs[variant]),
        })
    manifest = {
        "schema_version": 1,
        "experiment": "released_retreat_hypothesis_matrix",
        "started_at": utc_now(),
        "requested_trials": len(job_specs),
        "seeds": [row["seed"] for row in job_specs],
        "jobs": len(job_specs),
        "gpus": [row["gpu"] for row in job_specs],
        "config": str(base_config_path),
        "jobs_spec": job_specs,
        "fresh_reset_each_trial": True,
        "requested_scope": "insert_bowl",
        "preparation": "prefix",
        "video": True,
        "comparison_contract": (
            "Only released-retreat clearance height changes; collision checks, "
            "failure thresholds, retreat_2 and downstream public stage target remain unchanged."
        ),
        "excluded_already_falsified_hypotheses": [
            "fixed-orientation horizontal strict line after lift",
            "segmented 20mm fixed-orientation horizontal waypoint",
        ],
    }
    write_json(run_root / "batch_manifest.json", manifest)
    lock = threading.Lock()
    completed = []

    def launch(spec: dict) -> dict:
        run_dir = run_root / spec["run"]
        terminal = run_root / f".{spec['run']}.terminal.log"
        status_base = {key: spec[key] for key in ("run", "variant", "seed", "gpu")}
        append_jsonl(run_root / "batch_status.jsonl", {
            **status_base, "queued_at": utc_now(), "status": "queued"}, lock)
        command = [
            str(PYTHON), str(ENTRYPOINT),
            "--subtask", "insert_bowl", "--preparation", "prefix",
            "--config", spec["config"], "--seed", str(spec["seed"]),
            "--randomize", "--video", "--output", str(run_dir),
        ]
        environment = os.environ.copy()
        environment.update({
            "CUDA_VISIBLE_DEVICES": spec["gpu"], "MUJOCO_GL": "egl",
            "PYOPENGL_PLATFORM": "egl", "PYTHONNOUSERSITE": "1",
            "PYTHONPATH": str(ROOT),
        })
        started = utc_now()
        append_jsonl(run_root / "batch_status.jsonl", {
            **status_base, "started_at": started, "status": "running"}, lock)
        wall_start = time.monotonic()
        with terminal.open("w", encoding="utf-8") as stream:
            stream.write(
                f"started_at={started}\nvariant={spec['variant']}\n"
                f"seed={spec['seed']}\ngpu={spec['gpu']}\n"
                f"command={json.dumps(command, ensure_ascii=False)}\n")
            stream.flush()
            process = subprocess.run(
                command, cwd=ROOT, env=environment,
                stdout=stream, stderr=subprocess.STDOUT)
        run_dir.mkdir(parents=True, exist_ok=True)
        terminal.replace(run_dir / "terminal.log")
        summary = read_summary(run_dir)
        failure = summary.get("failure") or {}
        row = {
            **status_base, "started_at": started, "finished_at": utc_now(),
            "wall_s": round(time.monotonic() - wall_start, 3),
            "returncode": process.returncode,
            "summary_present": bool(summary),
            "status": summary.get("status"),
            "operation_success": summary.get("operation_success"),
            "goal_success": summary.get("goal_success"),
            "failure_subtask": summary.get("failure_subtask"),
            "failure_stage": failure.get("stage"),
            "exception": summary.get("exception"),
            "video_present": (run_dir / "video/task.mp4").is_file(),
        }
        append_jsonl(run_root / "batch_completed.jsonl", row, lock)
        append_jsonl(run_root / "batch_status.jsonl", {
            **status_base, "finished_at": row["finished_at"],
            "status": "completed" if process.returncode == 0 else "failed"}, lock)
        with lock:
            completed.append(row)
            ordered = sorted(completed, key=lambda item: item["run"])
            write_json(run_root / "batch_results.json", {
                "requested_trials": len(job_specs),
                "completed_trials": len(ordered),
                "goal_successes": sum(item.get("operation_success") is True for item in ordered),
                "task_failures": sum(item.get("operation_success") is False and item.get("summary_present") for item in ordered),
                "infrastructure_failures": sum(not item.get("summary_present") for item in ordered),
                "runs": ordered,
            })
        return row

    with ThreadPoolExecutor(max_workers=len(job_specs), thread_name_prefix="retreat") as pool:
        futures = [pool.submit(launch, spec) for spec in job_specs]
        for future in as_completed(futures):
            future.result()
    manifest["finished_at"] = utc_now()
    write_json(run_root / "batch_manifest.json", manifest)


if __name__ == "__main__":
    main()
