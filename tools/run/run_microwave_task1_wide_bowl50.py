"""在扩展桌面上并行执行 50 轮 wide-bowl Task1，并增量保存完整诊断索引。"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
PYTHON = Path("/data1/yuquan/miniconda3/envs/gene_agent/bin/python")
DEFAULT_CONFIG = ROOT / "data/task_1/simulation/mujoco_task1_physical_wide_bowl50.json"
ENTRYPOINT = ROOT / "tools/run/run_microwave_task1_physical.py"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def append_jsonl(path: Path, value, lock: threading.Lock) -> None:
    line = json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n"
    with lock, path.open("a", encoding="utf-8") as stream:
        stream.write(line)
        stream.flush()
        os.fsync(stream.fileno())


def source_hashes(config: Path) -> dict[str, str]:
    roots = (ROOT / "gene_agent", ROOT / "benchmarks", ROOT / "config")
    files = [config, ENTRYPOINT, Path(__file__).resolve()]
    for base in roots:
        files.extend(path for path in base.rglob("*") if path.is_file() and path.suffix in {".py", ".json", ".yaml", ".yml", ".toml"})
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(set(files))}


def summarize_run(run_dir: Path, returncode: int) -> dict:
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
    return {
        "returncode": returncode,
        "summary_present": summary_path.is_file(),
        "status": summary.get("status"),
        "goal_success": summary.get("goal_success"),
        "task_success": summary.get("task_success"),
        "failure_subtask": summary.get("failure_subtask"),
        "exception": summary.get("exception"),
        "planner_backend": summary.get("planner_backend"),
        "video_present": (run_dir / "video/task.mp4").is_file(),
        "diagnostic_refs": {
            "summary": str(summary_path.relative_to(run_dir.parent)) if summary_path.is_file() else None,
            "placement": str((run_dir / "placement.json").relative_to(run_dir.parent)) if (run_dir / "placement.json").is_file() else None,
            "terminal": str((run_dir / "terminal.log").relative_to(run_dir.parent)),
            "task_exception": str((run_dir / "task_exception.txt").relative_to(run_dir.parent)) if (run_dir / "task_exception.txt").is_file() else None,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--gpus", default="0,4,6,7")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--seed-start", type=int, default=2101)
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()
    config = args.config.resolve()
    if args.trials < 1 or args.seed_start < 0 or args.seed_start + args.trials > 2**31:
        parser.error("trials 必须为正，且 seed 范围必须落在 0..2^31-1")
    seeds = tuple(range(args.seed_start, args.seed_start + args.trials))
    gpus = tuple(value.strip() for value in args.gpus.split(",") if value.strip())
    if not 1 <= args.jobs <= len(gpus):
        parser.error("jobs 必须在 1 与 GPU 数量之间")
    if not PYTHON.is_file() or not config.is_file() or not ENTRYPOINT.is_file():
        parser.error("解释器、配置或 Task1 入口不存在")
    run_root = args.run_root.resolve()
    run_root.mkdir(parents=True, exist_ok=False)
    hashes = source_hashes(config)
    manifest = {
        "schema_version": 1, "started_at": utc_now(), "requested_trials": args.trials,
        "seeds": list(seeds), "jobs": args.jobs, "gpus": list(gpus[:args.jobs]),
        "config": str(config), "entrypoint": str(ENTRYPOINT),
        "fresh_reset_each_trial": True, "video": not args.no_video,
        "planner_required": "curobo", "source_sha256_at_start": hashes,
        "sampling_note": (
            f"No IK/task-success filtering; seeds {seeds[0]}..{seeds[-1]} "
            "are fixed before execution."
        ),
    }
    write_json(run_root / "batch_manifest.json", manifest)
    lock = threading.Lock()
    gpu_locks = {gpu: threading.Lock() for gpu in gpus[:args.jobs]}
    completed: list[dict] = []

    def launch(index: int, seed: int) -> dict:
        gpu = gpus[(index - 1) % args.jobs]
        name = f"run_{index:03d}"
        run_dir = run_root / name
        terminal = run_root / f".{name}.terminal.log"
        queued = utc_now()
        append_jsonl(run_root / "batch_status.jsonl", {
            "run": name, "seed": seed, "gpu": gpu, "queued_at": queued, "status": "queued"
        }, lock)
        command = [str(PYTHON), str(ENTRYPOINT), "--config", str(config),
                   "--output", str(run_dir), "--randomize", "--seed", str(seed)]
        if not args.no_video:
            command.append("--video")
        environment = os.environ.copy()
        environment.update({
            "MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl",
            "PYTHONNOUSERSITE": "1", "PYTHONPATH": str(ROOT),
            "CUDA_VISIBLE_DEVICES": gpu,
        })
        # ThreadPool 的下一个任务不保证由同一 worker 接手；按 GPU 加锁，
        # 避免某个较快 lane 抢到另一 lane 的编号后在同卡重叠两个 cuRobo。
        with gpu_locks[gpu], terminal.open("w", encoding="utf-8") as stream:
            started = utc_now()
            wall_start = time.monotonic()
            append_jsonl(run_root / "batch_status.jsonl", {
                "run": name, "seed": seed, "gpu": gpu, "queued_at": queued,
                "started_at": started, "status": "running"
            }, lock)
            stream.write(f"started_at={started}\nseed={seed}\ngpu={gpu}\ncommand={json.dumps(command)}\n")
            stream.flush()
            process = subprocess.run(command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT)
        run_dir.mkdir(parents=True, exist_ok=True)
        terminal.replace(run_dir / "terminal.log")
        row = {
            "run": name, "seed": seed, "gpu": gpu, "started_at": started,
            "finished_at": utc_now(), "wall_s": round(time.monotonic() - wall_start, 3),
            **summarize_run(run_dir, process.returncode),
        }
        append_jsonl(run_root / "batch_completed.jsonl", row, lock)
        with lock:
            completed.append(row)
            ordered = sorted(completed, key=lambda item: item["run"])
            write_json(run_root / "batch_results.json", {
                "requested_trials": args.trials, "completed_trials": len(ordered),
                "goal_successes": sum(item.get("goal_success") is True for item in ordered),
                "task_failures": sum(item.get("goal_success") is False and item.get("summary_present") for item in ordered),
                "infrastructure_failures": sum(not item.get("summary_present") for item in ordered),
                "runs": ordered,
            })
        return row

    with ThreadPoolExecutor(max_workers=args.jobs, thread_name_prefix="task1") as pool:
        futures = [pool.submit(launch, index, seed) for index, seed in enumerate(seeds, 1)]
        for future in as_completed(futures):
            future.result()
    manifest["finished_at"] = utc_now()
    manifest["source_sha256_at_end"] = source_hashes(config)
    manifest["source_unchanged_during_batch"] = manifest["source_sha256_at_start"] == manifest["source_sha256_at_end"]
    write_json(run_root / "batch_manifest.json", manifest)


if __name__ == "__main__":
    main()
