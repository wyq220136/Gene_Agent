"""Continuously write a traceable Markdown status page for a Task1 batch."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import time


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def latest_statuses(path: Path) -> dict[str, dict]:
    latest: dict[str, dict] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return latest
    for line in lines:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        run = row.get("run")
        if run:
            latest[run] = row
    return latest


def relative_link(run_root: Path, path: Path, label: str) -> str:
    if not path.is_file():
        return "-"
    return f"[{label}]({path.relative_to(run_root).as_posix()})"


def failure_refs(run_root: Path, run_dir: Path, summary: dict) -> tuple[str, str]:
    failure = summary.get("failure") or {}
    report = failure.get("report_ref")
    traceback = failure.get("traceback_ref")
    report_link = relative_link(run_root, run_dir / report, "report") if report else "-"
    traceback_link = relative_link(run_root, run_dir / traceback, "traceback") if traceback else "-"
    return report_link, traceback_link


def render(run_root: Path) -> tuple[str, bool]:
    manifest = read_json(run_root / "batch_manifest.json")
    results = read_json(run_root / "batch_results.json")
    statuses = latest_statuses(run_root / "batch_status.jsonl")
    requested = int(manifest.get("requested_trials") or results.get("requested_trials") or 0)
    seeds = manifest.get("seeds") or []
    job_specs = {row.get("run"): row for row in manifest.get("jobs_spec", [])
                 if row.get("run")}
    completed_rows = {row.get("run"): row for row in results.get("runs", []) if row.get("run")}
    completed = int(results.get("completed_trials") or len(completed_rows))
    finished = bool(manifest.get("finished_at")) or bool(requested and completed >= requested)

    stage_counts: Counter[str] = Counter()
    artifact_counts: Counter[str] = Counter()
    run_rows: list[str] = []
    total_rows = requested or max(len(seeds), len(statuses), len(completed_rows))
    for index in range(1, total_rows + 1):
        name = f"run_{index:03d}"
        run_dir = run_root / name
        summary = read_json(run_dir / "summary.json")
        completed_row = completed_rows.get(name, {})
        live = statuses.get(name, {})
        spec = job_specs.get(name, {})
        variant = completed_row.get("variant", live.get("variant", spec.get("variant", "-")))
        seed = summary.get("seed")
        if seed is None and index <= len(seeds):
            seed = seeds[index - 1]
        gpu = completed_row.get("gpu", live.get("gpu", "-"))
        is_completed = name in completed_rows
        if is_completed and summary:
            success = summary.get("goal_success")
            if success is None:
                success = summary.get("operation_success")
            state = "success" if success is True else "failed"
            failure = summary.get("failure") or {}
            stage = failure.get("stage") or summary.get("failure_subtask") or "-"
            if state == "success":
                stage_counts["success"] += 1
            else:
                stage_counts[f"{summary.get('failure_subtask') or 'unknown'}/{stage}"] += 1
        else:
            state = live.get("status", "queued" if index <= len(seeds) else "-")
            stage = "-"

        summary_link = relative_link(run_root, run_dir / "summary.json", "summary")
        report_link, traceback_link = failure_refs(run_root, run_dir, summary)
        video_link = relative_link(run_root, run_dir / "video/task.mp4", "video")
        terminal_link = relative_link(run_root, run_dir / "terminal.log", "terminal")
        placement_link = relative_link(run_root, run_dir / "placement.json", "placement")
        if is_completed:
            artifact_counts["summary"] += (run_dir / "summary.json").is_file()
            artifact_counts["terminal"] += (run_dir / "terminal.log").is_file()
            artifact_counts["placement"] += (run_dir / "placement.json").is_file()
            artifact_counts["video"] += (run_dir / "video/task.mp4").is_file()
        run_rows.append(
            f"| {name} | {variant} | {seed if seed is not None else '-'} | {gpu} | {state} | {stage} | "
            f"{summary_link} {report_link} {traceback_link} {video_link} {terminal_link} {placement_link} |"
        )

    active = Counter(
        row.get("status", "unknown")
        for run, row in statuses.items()
        if run not in completed_rows
    )
    source_state = manifest.get("source_unchanged_during_batch")
    if source_state is None:
        source_text = "pending final hash comparison"
    else:
        source_text = "yes" if source_state else "NO"
    seed_text = f"{seeds[0]}..{seeds[-1]}" if seeds else "unknown"
    lines = [
        "# Task1 Batch Live Status",
        "",
        "> This file is generated by `tools/run/monitor_microwave_task1_batch.py`. "
        "It is refreshed atomically and becomes final when the batch completes.",
        "",
        f"- Last refresh (UTC): `{utc_now()}`",
        f"- Batch state: `{'completed' if finished else 'running'}`",
        f"- Progress: `{completed}/{requested or '?'}`",
        f"- Successful outcomes: `{results.get('goal_successes', 0)}`",
        f"- Task failures: `{results.get('task_failures', 0)}`",
        f"- Infrastructure failures: `{results.get('infrastructure_failures', 0)}`",
        f"- Seeds: `{seed_text}`",
        f"- GPUs/jobs: `{','.join(manifest.get('gpus', []))}` / `{manifest.get('jobs', '-')}`",
        f"- Config: `{manifest.get('config', 'unknown')}`",
        f"- Source unchanged during batch: `{source_text}`",
        "",
        "## Failure Stage Distribution",
        "",
        "| Outcome / stage | Count |",
        "|---|---:|",
    ]
    lines.extend(f"| {key} | {value} |" for key, value in sorted(stage_counts.items()))
    if not stage_counts:
        lines.append("| pending | 0 |")
    lines.extend([
        "",
        "## Scheduler State",
        "",
        "| Last recorded state | Runs |",
        "|---|---:|",
    ])
    lines.extend(f"| {key} | {value} |" for key, value in sorted(active.items()))
    if not active:
        lines.append("| pending | 0 |")
    lines.extend([
        "",
        "## Completed-run Artifact Coverage",
        "",
        "| Artifact | Present / completed |",
        "|---|---:|",
    ])
    for key in ("summary", "terminal", "placement", "video"):
        lines.append(f"| {key} | {artifact_counts[key]} / {completed} |")
    lines.extend([
        "",
        "## Per-run Trace Index",
        "",
        "| Run | Variant | Seed | GPU | State | Failure stage | Evidence |",
        "|---|---|---:|---:|---|---|---|",
        *run_rows,
        "",
    ])
    return "\n".join(lines), finished


def write_atomic(path: Path, content: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--interval", type=float, default=30.0)
    parser.add_argument("--output", default="live_status.md")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    if not run_root.is_dir():
        parser.error(f"batch directory does not exist: {run_root}")
    if args.interval <= 0:
        parser.error("--interval must be positive")
    output = run_root / args.output
    while True:
        content, finished = render(run_root)
        write_atomic(output, content)
        if args.once or finished:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
