"""Monitor a Task1 batch, render initial top views, and write detailed Markdown."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time


ROOT = Path(__file__).resolve().parents[2]


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def atomic_text(path: Path, value: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def first_frame(run_dir: Path) -> tuple[Path, list[float]] | None:
    frames = run_dir / "live/frames.jsonl"
    try:
        with frames.open(encoding="utf-8") as stream:
            row = json.loads(stream.readline())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    model = run_dir / "live" / str(row.get("model", "model_0.mjb"))
    qpos = row.get("qpos")
    if not model.is_file() or not isinstance(qpos, list):
        return None
    return model, qpos


def render_initial_top(run_dir: Path, *, force: bool = False) -> bool:
    output = run_dir / "initial_placement_top.png"
    if output.is_file() and not force:
        return True
    source = first_frame(run_dir)
    if source is None:
        return False
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    import mujoco
    import numpy as np
    from PIL import Image, ImageDraw

    model_path, qpos = source
    model = mujoco.MjModel.from_binary_path(str(model_path))
    # The placement image is evidence for the bowl distribution. Hide both
    # robots so an arm at home cannot occlude the bowl from the top view.
    robot_body_ids = set()
    for body_id in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        if name == "robot_body" or name.startswith(("left_", "right_")):
            robot_body_ids.add(body_id)
    for geom_id in range(model.ngeom):
        if int(model.geom_bodyid[geom_id]) in robot_body_ids:
            model.geom_rgba[geom_id, 3] = 0.0
    data = mujoco.MjData(model)
    values = np.asarray(qpos, dtype=float)
    if values.shape != data.qpos.shape:
        return False
    data.qpos[:] = values
    mujoco.mj_forward(model, data)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    # Frame the sampled lower-left work region and the microwave together.
    # This keeps the bowl large enough to inspect even at the distribution edges.
    camera.lookat[:] = [0.32, 0.27, -0.04]
    camera.distance = 1.02
    camera.azimuth = 90.0
    camera.elevation = -89.0
    # The frozen Task1 model declares a 640-pixel offscreen framebuffer.
    renderer = mujoco.Renderer(model, height=480, width=640)
    try:
        renderer.update_scene(data, camera=camera)
        image = Image.fromarray(renderer.render())
    finally:
        renderer.close()
    placement = read_json(run_dir / "placement.json")
    pose = ((placement.get("sampled_poses_world") or {}).get("bowl") or [])
    xy = None
    if len(pose) >= 2 and len(pose[0]) >= 4 and len(pose[1]) >= 4:
        xy = (float(pose[0][3]), float(pose[1][3]))
    draw = ImageDraw.Draw(image)
    label = f"{run_dir.name} | seed={placement.get('seed', '-')}"
    if xy is not None:
        label += f" | bowl XY=({xy[0]:.4f}, {xy[1]:.4f}) m"
    yaw = ((placement.get("deltas") or {}).get("bowl") or {}).get("yaw_deg")
    if yaw is not None:
        label += f" | yaw={float(yaw):.1f} deg"
    draw.rectangle((10, 10, 630, 42), fill=(255, 255, 255, 220))
    draw.text((18, 18), label, fill=(10, 10, 10))
    temporary = output.with_suffix(".png.tmp")
    image.save(temporary, format="PNG")
    temporary.replace(output)
    return True


def walk(value, path: str = ""):
    if isinstance(value, dict):
        for key, item in value.items():
            child = f"{path}.{key}" if path else str(key)
            yield child, item
            yield from walk(item, child)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from walk(item, f"{path}[{index}]")


def unique_values(data: dict, keys: set[str], limit: int = 6) -> list[str]:
    values = []
    for path, value in walk(data):
        key = path.rsplit(".", 1)[-1].split("[", 1)[0]
        if key not in keys or isinstance(value, (dict, list)) or value in (None, ""):
            continue
        text = str(value).replace("\n", " ")
        if len(text) > 180:
            text = text[:177] + "..."
        if text not in values:
            values.append(text)
        if len(values) >= limit:
            break
    return values


def reason_codes(data: dict) -> list[str]:
    result = []
    for path, value in walk(data):
        key = path.rsplit(".", 1)[-1].split("[", 1)[0]
        candidates = value if key == "reason_codes" and isinstance(value, list) else [value]
        if key not in {"reason_code", "reason_codes", "failure_code"}:
            continue
        for candidate in candidates:
            if isinstance(candidate, str) and candidate and candidate not in result:
                result.append(candidate)
    return result[:8]


def classify_owner(codes: list[str], categories: list[str], statuses: list[str]) -> str:
    joined = " ".join([*codes, *categories, *statuses]).lower()
    if any(token in joined for token in ("endpoint_tracking", "stale_state", "execution_")):
        return "execution"
    if any(token in joined for token in ("invalid_pad_contact", "angle_not_stable",
            "postcondition", "tilt_exceeded", "clearance", "support", "settling")):
        return "task_postcondition"
    if any(token in joined for token in ("forbidden_contact", "mujoco_contact")):
        return "execution_contact"
    if any(code.startswith("path_") for code in codes) or "traj" in joined or "path" in joined:
        return "curobo_trajectory"
    if any(code.startswith("goal_") for code in codes) or "ik" in joined or "candidate_screen" in joined:
        return "curobo_ik_or_candidate_screen"
    if "collision" in joined:
        return "collision_postcheck"
    return "task_or_operation_check"


def symptom(subtask: str, stage: str) -> str:
    text = f"{subtask} {stage}".lower()
    if "candidate_selection" in text:
        return "未选出可执行抓取候选，夹爪没有进入有效抓取执行"
    if "pregrasp" in text or "approach" in text:
        return "夹爪未到达碗沿抓取位姿"
    if "grasp" in text or "contact" in text:
        return "夹爪没有形成有效抓持"
    if "lift" in text:
        return "抓碗后的抬升动作未完成"
    if "insert" in text or "preplace" in text:
        return "碗未完成入炉放置"
    if "retreat" in text:
        return "放置或取碗后的机械臂撤离未完成"
    if "extract" in text:
        return "碗未完成从炉内取出"
    if "open" in text or "handle" in text or "push" in text:
        return "微波炉门操作未完成"
    return "当前任务阶段未完成"


def placement_xy(run_dir: Path) -> str:
    placement = read_json(run_dir / "placement.json")
    pose = ((placement.get("sampled_poses_world") or {}).get("bowl") or [])
    if len(pose) >= 2 and len(pose[0]) >= 4 and len(pose[1]) >= 4:
        return f"({float(pose[0][3]):.4f}, {float(pose[1][3]):.4f})"
    return "-"


def placement_record(run_dir: Path, fallback_seed: object = "-") -> dict:
    placement = read_json(run_dir / "placement.json")
    pose = ((placement.get("sampled_poses_world") or {}).get("bowl") or [])
    xyz = None
    if len(pose) >= 3 and all(len(row) >= 4 for row in pose[:3]):
        xyz = tuple(float(pose[axis][3]) for axis in range(3))
    yaw = ((placement.get("deltas") or {}).get("bowl") or {}).get("yaw_deg")
    return {
        "seed": placement.get("seed", fallback_seed),
        "xyz": xyz,
        "yaw_deg": float(yaw) if yaw is not None else None,
    }


def render_placement_gallery(run_root: Path) -> str:
    manifest = read_json(run_root / "batch_manifest.json")
    requested = int(manifest.get("requested_trials") or 0)
    seeds = manifest.get("seeds") or []
    records = []
    for index in range(1, requested + 1):
        name = f"run_{index:03d}"
        run_dir = run_root / name
        fallback_seed = seeds[index - 1] if index <= len(seeds) else "-"
        record = placement_record(run_dir, fallback_seed)
        summary = read_json(run_dir / "summary.json")
        record.update({
            "name": name,
            "result": "成功" if summary.get("goal_success") is True else "失败",
            "image": run_dir / "initial_placement_top.png",
        })
        records.append(record)

    positions = [record["xyz"] for record in records if record["xyz"] is not None]
    lines = [
        "# 每轮碗初始摆放位置",
        "",
        f"- 实验目录：`{run_root.name}`",
        f"- 轮次：`{len(records)}`",
        "- 坐标系：MuJoCo world frame；坐标单位为米，yaw单位为度。",
        "- 视图：正俯视；隐藏左右机械臂，只保留桌面、微波炉和碗，以便比较初始摆位。",
        "- 图片：每张以原始 `640 x 480` 分辨率内嵌，可点击查看原图。",
    ]
    if positions:
        xs = [xyz[0] for xyz in positions]
        ys = [xyz[1] for xyz in positions]
        zs = [xyz[2] for xyz in positions]
        lines.extend([
            f"- 实际范围：X=`[{min(xs):.4f}, {max(xs):.4f}]`，Y=`[{min(ys):.4f}, {max(ys):.4f}]`，Z=`[{min(zs):.4f}, {max(zs):.4f}]`。",
        ])
    lines.extend([
        "",
        "## 坐标索引",
        "",
        "| 轮次 | Seed | 碗中心 X (m) | 碗中心 Y (m) | 碗中心 Z (m) | Yaw (deg) | 任务结果 |",
        "|---|---:|---:|---:|---:|---:|---|",
    ])
    for record in records:
        xyz = record["xyz"]
        xyz_text = [f"{value:.6f}" for value in xyz] if xyz is not None else ["-", "-", "-"]
        yaw_text = f"{record['yaw_deg']:.3f}" if record["yaw_deg"] is not None else "-"
        lines.append(
            f"| [{record['name']}](#{record['name']}) | {record['seed']} | "
            f"{xyz_text[0]} | {xyz_text[1]} | {xyz_text[2]} | {yaw_text} | {record['result']} |"
        )
    lines.extend(["", "## 每轮俯视图", ""])
    for record in records:
        xyz = record["xyz"]
        position = (f"({xyz[0]:.6f}, {xyz[1]:.6f}, {xyz[2]:.6f}) m"
                    if xyz is not None else "未记录")
        yaw_text = f"{record['yaw_deg']:.3f} deg" if record["yaw_deg"] is not None else "未记录"
        image_path = f"{record['name']}/initial_placement_top.png"
        lines.extend([
            f"### {record['name']}",
            "",
            f"- Seed：`{record['seed']}`",
            f"- 碗中心 XYZ：`{position}`",
            f"- 初始 yaw：`{yaw_text}`",
            f"- 完整任务结果：`{record['result']}`",
            "",
        ])
        if record["image"].is_file():
            lines.extend([
                f"<a href=\"{image_path}\"><img src=\"{image_path}\" "
                f"alt=\"{record['name']} 碗初始位置俯视图\" width=\"640\"></a>",
                "",
            ])
        else:
            lines.extend(["> 本轮俯视图尚未生成。", ""])
    return "\n".join(lines)


def diagnose(run_dir: Path, summary: dict) -> tuple[str, str, str]:
    failure = summary.get("failure") or {}
    stage = str(failure.get("stage") or summary.get("failure_subtask") or "unknown")
    report_ref = failure.get("report_ref")
    report = read_json(run_dir / report_ref) if report_ref else {}
    codes = reason_codes(report)
    categories = unique_values(report, {"failure_category", "category"})
    statuses = unique_values(report, {"status", "failure_type"})
    owner = classify_owner(codes, categories, statuses)
    details = []
    if categories:
        details.append("category=" + ", ".join(categories))
    if codes:
        details.append("reason_code=" + ", ".join(codes))
    if statuses:
        details.append("status=" + ", ".join(statuses))
    numbers = unique_values(report, {
        "position_error_m", "rotation_error_rad", "rotation_error_deg",
        "maximum_error_rad", "endpoint_error_rad", "threshold_rad",
        "maximum_bowl_tilt_deg", "bowl_tilt_deg", "clearance_m",
        "required_clearance_m", "first_invalid_waypoint", "collision_pair",
    }, limit=8)
    if numbers:
        details.append("关键值=" + ", ".join(numbers))
    message = unique_values(report, {"message", "exception", "error"}, limit=2)
    if message:
        details.append("message=" + ", ".join(message))
    if not details:
        details.append("结构化报告未给出更细的reason code，需沿report_ref继续人工诊断")
    statement = (
        f"问题直接表现是{symptom(str(summary.get('failure_subtask') or ''), stage)}；"
        f"其触发了{owner}层的`{stage}`失败；详细情况为" + "；".join(details) + "。"
    )
    prefix = run_dir.name
    evidence = (f"[{report_ref}]({prefix}/{report_ref})" if report_ref
                else f"[summary]({prefix}/summary.json)")
    return stage, owner, statement + " " + evidence


def render_report(run_root: Path) -> tuple[str, bool]:
    manifest = read_json(run_root / "batch_manifest.json")
    results = read_json(run_root / "batch_results.json")
    requested = int(manifest.get("requested_trials") or 0)
    rows_by_run = {row.get("run"): row for row in results.get("runs", [])}
    completed = len(rows_by_run)
    successes = 0
    failures = 0
    stage_counts = Counter()
    rows = []
    for index in range(1, requested + 1):
        name = f"run_{index:03d}"
        run_dir = run_root / name
        summary = read_json(run_dir / "summary.json")
        completed_row = rows_by_run.get(name)
        seed = summary.get("seed")
        if seed is None:
            seeds = manifest.get("seeds") or []
            seed = seeds[index - 1] if index <= len(seeds) else "-"
        top = (f"[俯视图]({name}/initial_placement_top.png)"
               if (run_dir / "initial_placement_top.png").is_file() else "-")
        if completed_row and summary:
            if summary.get("goal_success") is True:
                successes += 1
                result, stage, owner, cause = "成功", "-", "-", "-"
                stage_counts["success"] += 1
            else:
                failures += 1
                result = "失败"
                stage, owner, cause = diagnose(run_dir, summary)
                stage_counts[f"{summary.get('failure_subtask') or 'unknown'}/{stage}"] += 1
        else:
            result, stage, owner, cause = "运行中/排队", "-", "-", "-"
        rows.append(f"| {name} | {seed} | {placement_xy(run_dir)} | {result} | {stage} | {owner} | {cause} | {top} |")
    finished = bool(manifest.get("finished_at")) or bool(requested and completed >= requested)
    lines = [
        "# Microwave Task1 50轮实验结果",
        "",
        f"- 更新时间（UTC）：`{datetime.now(timezone.utc).isoformat(timespec='seconds')}`",
        f"- 状态：`{'completed' if finished else 'running'}`",
        f"- 进度：`{completed}/{requested}`",
        f"- 完整任务成功：`{successes}`",
        f"- 任务失败：`{failures}`",
        f"- 配置：`{manifest.get('config', '-')}`",
        "- 阶段锚点：`docs/01-task-skills.svg`",
        "- 初态分布：扩展桌面下半部与微波炉左侧区域的交集；每轮fresh reset，不按IK或任务结果筛样。",
        "",
        "## 失败阶段分布",
        "",
        "| 阶段 | 数量 |",
        "|---|---:|",
        *[f"| {key} | {value} |" for key, value in sorted(stage_counts.items())],
        "",
        "## 每轮结果",
        "",
        "| 轮次 | Seed | 碗初始XY (m) | 结果 | 失败phase | failure_owner | 失败原因 | 初始俯视图 |",
        "|---|---:|---|---|---|---|---|---|",
        *rows,
        "",
        "> 自动报告记录结构化直接失败原因；涉及真实运动姿态或接触过程的根因分析，仍以实际视频抽帧和接触日志复核为准。",
        "",
    ]
    return "\n".join(lines), finished


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--interval", type=float, default=20.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    while True:
        for run_dir in sorted(run_root.glob("run_[0-9][0-9][0-9]")):
            try:
                render_initial_top(run_dir)
            except Exception as error:
                print(f"{run_dir.name}: initial top render failed: {type(error).__name__}: {error}", flush=True)
        report, finished = render_report(run_root)
        atomic_text(run_root / "experiment_results.md", report)
        atomic_text(run_root / "initial_placements.md", render_placement_gallery(run_root))
        print(f"updated {run_root / 'experiment_results.md'}", flush=True)
        if args.once or finished:
            return
        time.sleep(max(args.interval, 2.0))


if __name__ == "__main__":
    main()
