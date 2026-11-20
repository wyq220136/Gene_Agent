#!/usr/bin/env python3
"""Render every failed microwave bowl grasp candidate from saved run evidence.

The renderer is deliberately independent of the physical Task entrypoint so it
can be attached to an active batch without changing the code loaded by later
trials.  It visualizes requested TCP poses and recorded finite-search outcomes;
it never invents a robot path for a failed IK request.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

FORK_SEGMENTS_M = (
    ((0.0, 0.0, -0.052), (0.0, 0.0, -0.012)),
    ((-0.028, 0.0, -0.012), (0.028, 0.0, -0.012)),
    ((-0.028, 0.0, -0.012), (-0.028, 0.0, 0.046)),
    ((0.028, 0.0, -0.012), (0.028, 0.0, 0.046)),
)

STATUS_STYLE = {
    "final_ik_no_solution": ((220, 55, 55), "final IK finite search found no solution"),
    "pregrasp_ik_no_solution": ((245, 132, 31), "final IK passed; no pregrasp IK branch passed"),
    "pregrasp_joint_travel_exceeded": ((197, 62, 168), "pregrasp IK passed; current-state joint limit rejected it"),
    "pregrasp_plan_failed": ((224, 176, 38), "pregrasp path planning failed"),
    "pregrasp_path_joint_travel_exceeded": ((173, 82, 190), "pregrasp path exceeded current-state joint limit"),
    "approach_failed": ((106, 76, 190), "pregrasp passed; linear approach failed"),
    "approach_joint_travel_exceeded": ((65, 105, 225), "approach exceeded configured joint limit"),
    "feasible": ((45, 160, 82), "candidate passed recorded prefix checks"),
    "unknown": ((128, 136, 148), "recorded attempts do not identify a later stage"),
}

VIEW_SPECS = {
    "overview": (-18.0, -25.0),
    "side": (90.0, -20.0),
    "top": (90.0, -89.0),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def safe_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
    return slug or "candidate"


def pose_from_attempt(attempt: dict) -> np.ndarray | None:
    try:
        pose = np.asarray(attempt["request"]["target"]["matrix"], dtype=float)
    except (KeyError, TypeError, ValueError):
        return None
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        return None
    return pose


def result_success(attempt: dict) -> bool:
    return bool((attempt.get("result") or {}).get("success"))


def solver_error_score(attempt: dict) -> float:
    diagnostic = (((attempt.get("result") or {}).get("failure") or {}).get("details") or {}).get(
        "solver_diagnostic", {}
    )
    return float(diagnostic.get("position_error_m", 1e6)) + float(
        diagnostic.get("rotation_error_rad", 1e6)
    )


def choose_pregrasp(attempts: list[dict]) -> tuple[np.ndarray | None, dict | None]:
    planned = [row for row in attempts if row.get("stage") == "pregrasp"]
    if planned:
        row = next((item for item in planned if result_success(item)), planned[0])
        return pose_from_attempt(row), row

    ik_rows = [row for row in attempts if row.get("stage") == "pregrasp_ik"]
    posture = {
        row.get("request_id"): row.get("result") or {}
        for row in attempts if row.get("stage") == "pregrasp_posture_check"
    }
    successful = [row for row in ik_rows if result_success(row)]
    if successful:
        row = min(
            successful,
            key=lambda item: float(
                posture.get((item.get("request") or {}).get("request_id"), {}).get(
                    "maximum_joint_travel_rad", 1e6
                )
            ),
        )
        return pose_from_attempt(row), row
    if ik_rows:
        row = min(ik_rows, key=solver_error_score)
        return pose_from_attempt(row), row
    return None, None


def classify_candidate(attempts: list[dict]) -> str:
    final_rows = [row for row in attempts if row.get("stage") == "final_grasp_ik"]
    if not final_rows or not result_success(final_rows[0]):
        return "final_ik_no_solution"

    approach = [row for row in attempts if row.get("stage") == "approach"]
    approach_posture = [
        row for row in attempts if row.get("stage") == "approach_posture_check"
    ]
    if approach:
        if not result_success(approach[-1]):
            return "approach_failed"
        if approach_posture and not bool(
            (approach_posture[-1].get("result") or {}).get("success")
        ):
            return "approach_joint_travel_exceeded"
        return "feasible"

    preplan = [row for row in attempts if row.get("stage") == "pregrasp"]
    preplan_posture = [
        row for row in attempts if row.get("stage") == "pregrasp_posture_check"
        and str(row.get("request_id", "")).endswith(":pregrasp")
    ]
    if preplan:
        if not result_success(preplan[-1]):
            return "pregrasp_plan_failed"
        if preplan_posture and not bool(
            (preplan_posture[-1].get("result") or {}).get("success")
        ):
            return "pregrasp_path_joint_travel_exceeded"

    pre_ik = [row for row in attempts if row.get("stage") == "pregrasp_ik"]
    if pre_ik:
        if not any(result_success(row) for row in pre_ik):
            return "pregrasp_ik_no_solution"
        rejected = [
            row for row in attempts
            if row.get("stage") == "pregrasp_posture_check"
            and not bool((row.get("result") or {}).get("success"))
        ]
        if rejected:
            return "pregrasp_joint_travel_exceeded"
    return "unknown"


def candidate_records(selection: dict) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for attempt in (selection.get("plan") or {}).get("attempts") or []:
        candidate_id = attempt.get("candidate_id")
        if candidate_id:
            grouped[str(candidate_id)].append(attempt)

    records = []
    for index, candidate_id in enumerate(selection.get("candidate_order") or []):
        attempts = grouped[str(candidate_id)]
        final = next(
            (row for row in attempts if row.get("stage") == "final_grasp_ik"), None
        )
        if final is None or pose_from_attempt(final) is None:
            continue
        pregrasp_pose, pregrasp_attempt = choose_pregrasp(attempts)
        status = classify_candidate(attempts)
        final_result = final.get("result") or {}
        failure = final_result.get("failure") or {}
        records.append({
            "index": index,
            "candidate_id": str(candidate_id),
            "status": status,
            "status_description": STATUS_STYLE[status][1],
            "final_target_world": pose_from_attempt(final),
            "best_pregrasp_target_world": pregrasp_pose,
            "best_pregrasp_request_id": None if pregrasp_attempt is None else (
                (pregrasp_attempt.get("request") or {}).get("request_id")
            ),
            "final_ik_success": result_success(final),
            "final_ik_failure_kind": failure.get("kind"),
            "final_ik_failure_summary": failure.get("summary"),
            "attempt_stage_counts": dict(Counter(
                str(row.get("stage")) for row in attempts
            )),
        })
    return records


def fork_segments(pose: np.ndarray):
    rotation, origin = pose[:3, :3], pose[:3, 3]
    for start, end in FORK_SEGMENTS_M:
        yield (
            rotation @ np.asarray(start, dtype=float) + origin,
            rotation @ np.asarray(end, dtype=float) + origin,
        )


def append_pose(renderer, mujoco, pose: np.ndarray, color: tuple[int, int, int], *,
                alpha: float = 0.92, radius: float = 0.0028) -> None:
    rgba = np.asarray([*np.asarray(color, dtype=float) / 255.0, alpha], dtype=np.float32)
    for start, end in fork_segments(pose):
        if renderer.scene.ngeom >= renderer.scene.maxgeom:
            raise RuntimeError("MuJoCo renderer has no room for candidate overlays")
        geom = renderer.scene.geoms[renderer.scene.ngeom]
        mujoco.mjv_initGeom(
            geom, mujoco.mjtGeom.mjGEOM_CAPSULE,
            np.zeros(3), np.zeros(3), np.eye(3).reshape(-1), rgba,
        )
        mujoco.mjv_connector(
            geom, mujoco.mjtGeom.mjGEOM_CAPSULE, radius, start, end,
        )
        geom.category = mujoco.mjtCatBit.mjCAT_DECOR
        renderer.scene.ngeom += 1


def append_target_marker(renderer, mujoco, position: np.ndarray,
                         color: tuple[int, int, int]) -> None:
    if renderer.scene.ngeom >= renderer.scene.maxgeom:
        raise RuntimeError("MuJoCo renderer has no room for candidate marker")
    geom = renderer.scene.geoms[renderer.scene.ngeom]
    rgba = np.asarray([*np.asarray(color, dtype=float) / 255.0, 0.9], dtype=np.float32)
    mujoco.mjv_initGeom(
        geom, mujoco.mjtGeom.mjGEOM_SPHERE,
        np.asarray([0.008, 0.008, 0.008]), np.asarray(position, dtype=float),
        np.eye(3).reshape(-1), rgba,
    )
    geom.category = mujoco.mjtCatBit.mjCAT_DECOR
    renderer.scene.ngeom += 1


def font(size: int):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def annotate(pixels: np.ndarray, title: str, lines: list[str],
             color: tuple[int, int, int]) -> Image.Image:
    source = Image.fromarray(pixels)
    header = 100
    canvas = Image.new("RGB", (source.width, source.height + header), (18, 22, 30))
    canvas.paste(source, (0, header))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, 9, header), fill=color)
    draw.text((20, 10), title[:92], fill=(245, 247, 250), font=font(15))
    y = 34
    for line in lines[:3]:
        draw.text((20, y), line[:104], fill=(220, 225, 232), font=font(12))
        y += 18
    return canvas


def load_model_and_state(run_dir: Path):
    import mujoco
    from gene_agent.feedback.state import restore_state

    binary = run_dir / "live/model_0.mjb"
    scene = run_dir / "scene/source_scene/scene.xml"
    if binary.is_file():
        model = mujoco.MjModel.from_binary_path(str(binary))
        model_source = str(binary.relative_to(run_dir))
    elif scene.is_file():
        model = mujoco.MjModel.from_xml_path(str(scene))
        model_source = str(scene.relative_to(run_dir))
    else:
        raise FileNotFoundError("saved MuJoCo model was not found")
    data = mujoco.MjData(model)
    state_path = run_dir / "failure_state.npz"
    if not state_path.is_file():
        raise FileNotFoundError("failure_state.npz is required for exact candidate state")
    restore_state(model, data, state_path)
    mujoco.mj_forward(model, data)
    return mujoco, model, data, model_source


def actual_tcp_pose(mujoco, model, data) -> np.ndarray | None:
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_tcp")
    if site_id < 0:
        return None
    pose = np.eye(4)
    pose[:3, :3] = np.asarray(data.site_xmat[site_id]).reshape(3, 3)
    pose[:3, 3] = np.asarray(data.site_xpos[site_id])
    return pose


def camera_specs(records: list[dict]) -> dict[str, dict]:
    points = np.asarray([row["final_target_world"][:3, 3] for row in records])
    lower, upper = points.min(axis=0), points.max(axis=0)
    lookat = (lower + upper) / 2.0
    lookat[2] += 0.025
    span = float(np.max(upper - lower))
    distance = min(0.95, max(0.48, 0.38 + 2.3 * span))
    return {
        name: {
            "azimuth": azimuth,
            "elevation": elevation,
            "distance": distance * (1.12 if name == "top" else 1.0),
            "lookat": lookat.copy(),
        }
        for name, (azimuth, elevation) in VIEW_SPECS.items()
    }


def render_scene(renderer, mujoco, data, camera: dict, overlays: list[dict],
                 actual_tcp: np.ndarray | None) -> np.ndarray:
    view = mujoco.MjvCamera()
    view.type = mujoco.mjtCamera.mjCAMERA_FREE
    view.azimuth = float(camera["azimuth"])
    view.elevation = float(camera["elevation"])
    view.distance = float(camera["distance"])
    view.lookat[:] = camera["lookat"]
    renderer.update_scene(data, camera=view)
    if actual_tcp is not None:
        append_pose(renderer, mujoco, actual_tcp, (245, 245, 245), alpha=0.70, radius=0.0021)
    for overlay in overlays:
        append_pose(
            renderer, mujoco, overlay["pose"], overlay["color"],
            alpha=overlay.get("alpha", 0.92), radius=overlay.get("radius", 0.0028),
        )
        if overlay.get("marker", False):
            append_target_marker(renderer, mujoco, overlay["pose"][:3, 3], overlay["color"])
    return renderer.render()


def make_contact_sheet(records: list[dict], output: Path) -> None:
    columns = 4
    thumb_width, thumb_height = 320, 290
    rows = math.ceil(len(records) / columns)
    sheet = Image.new("RGB", (columns * thumb_width, rows * thumb_height), (18, 22, 30))
    draw = ImageDraw.Draw(sheet)
    for position, record in enumerate(records):
        path = output / record["images"]["overview"]
        image = Image.open(path).convert("RGB")
        image.thumbnail((thumb_width - 12, thumb_height - 54))
        x = (position % columns) * thumb_width + 6
        y = (position // columns) * thumb_height + 6
        sheet.paste(image, (x, y))
        color = STATUS_STYLE[record["status"]][0]
        draw.rectangle((x, y + thumb_height - 42, x + 12, y + thumb_height - 28), fill=color)
        draw.text(
            (x + 18, y + thumb_height - 44),
            f"{record['index']:02d} {record['candidate_id']}",
            fill=(245, 247, 250), font=font(12),
        )
        draw.text(
            (x + 18, y + thumb_height - 27), record["status"],
            fill=(205, 211, 220), font=font(11),
        )
    sheet.save(output / "contact_sheet.png")


def write_readme(output: Path, run_dir: Path, records: list[dict], counts: Counter) -> None:
    lines = [
        "# Candidate Selection Visualizations",
        "",
        f"- Run: `{run_dir.name}`",
        f"- Candidates: `{len(records)}`",
        "- White fork: actual TCP at the recorded failure state.",
        "- Colored fork/sphere: requested final grasp TCP pose.",
        "- Cyan fork: one representative recorded pregrasp request, when available.",
        "- A failed IK image is a requested pose plus finite-search result, not a cuRobo trajectory or proof of global unreachability.",
        "",
        "## Status Counts",
        "",
    ]
    lines.extend(f"- `{name}`: {count}" for name, count in sorted(counts.items()))
    lines.extend([
        "",
        "## Candidates",
        "",
        "| # | Candidate | Status | Overview | Side | Top |",
        "|---:|---|---|---|---|---|",
    ])
    for row in records:
        lines.append(
            f"| {row['index']} | `{row['candidate_id']}` | `{row['status']}` | "
            f"[overview]({row['images']['overview']}) | "
            f"[side]({row['images']['side']}) | [top]({row['images']['top']}) |"
        )
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_run(run_dir: Path, *, force: bool = False) -> dict:
    run_dir = run_dir.resolve()
    selection_path = run_dir / "pick_insert/candidate_selection.json"
    result_path = run_dir / "pick_insert/result.json"
    if not selection_path.is_file() or not result_path.is_file():
        raise FileNotFoundError("candidate_selection.json/result.json is missing")
    selection = read_json(selection_path)
    result = read_json(result_path)
    if result.get("failure_stage") != "candidate_selection":
        raise ValueError("run did not fail at candidate_selection")
    if selection.get("selected_candidate_id") is not None:
        raise ValueError("candidate_selection has a selected candidate")

    records = candidate_records(selection)
    if not records:
        raise ValueError("no renderable final_grasp_ik candidate request was recorded")
    destination = run_dir / "pick_insert/candidate_visualizations"
    if destination.is_dir() and (destination / "index.json").is_file() and not force:
        return read_json(destination / "index.json")

    mujoco, model, data, model_source = load_model_and_state(run_dir)
    actual_tcp = actual_tcp_pose(mujoco, model, data)
    cameras = camera_specs(records)
    parent = destination.parent
    temporary = Path(tempfile.mkdtemp(prefix=".candidate_visualizations_", dir=parent))
    try:
        model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), 640)
        model.vis.global_.offheight = max(int(model.vis.global_.offheight), 480)
        renderers = {
            name: mujoco.Renderer(model, height=480, width=640)
            for name in cameras
        }
        try:
            for row in records:
                color = STATUS_STYLE[row["status"]][0]
                candidate_dir = temporary / "candidates" / (
                    f"{row['index']:02d}_{safe_slug(row['candidate_id'])}"
                )
                candidate_dir.mkdir(parents=True, exist_ok=True)
                overlays = [{
                    "pose": row["final_target_world"], "color": color,
                    "marker": True, "radius": 0.0031,
                }]
                if row["best_pregrasp_target_world"] is not None:
                    overlays.append({
                        "pose": row["best_pregrasp_target_world"],
                        "color": (0, 195, 215), "alpha": 0.65, "radius": 0.0020,
                    })
                images = {}
                for view_name, camera in cameras.items():
                    pixels = render_scene(
                        renderers[view_name], mujoco, data, camera, overlays, actual_tcp,
                    )
                    image = annotate(
                        pixels,
                        f"{row['index']:02d} {row['candidate_id']} [{row['status']}]",
                        [
                            row["status_description"],
                            "colored=requested final TCP; cyan=representative pregrasp; white=actual TCP",
                            "finite-search evidence only; no unrecorded path is shown",
                        ],
                        color,
                    )
                    path = candidate_dir / f"{view_name}.png"
                    image.save(path)
                    images[view_name] = str(path.relative_to(temporary))
                row["images"] = images

            all_overlays = [
                {
                    "pose": row["final_target_world"],
                    "color": STATUS_STYLE[row["status"]][0],
                    "marker": False,
                    "radius": 0.0024,
                }
                for row in records
            ]
            counts = Counter(row["status"] for row in records)
            overview_images = {}
            legend = ", ".join(f"{name}={count}" for name, count in sorted(counts.items()))
            for view_name, camera in cameras.items():
                pixels = render_scene(
                    renderers[view_name], mujoco, data, camera, all_overlays, actual_tcp,
                )
                image = annotate(
                    pixels, f"All {len(records)} requested grasp candidates",
                    [legend, "status colors match index.json; white fork is actual TCP",
                     "requested poses only; failed IK does not imply global unreachability"],
                    (75, 135, 210),
                )
                path = temporary / f"overview_{view_name}.png"
                image.save(path)
                overview_images[view_name] = path.name
        finally:
            for renderer in renderers.values():
                renderer.close()

        make_contact_sheet(records, temporary)
        serializable = []
        for row in records:
            serializable.append({
                **{key: value for key, value in row.items()
                   if key not in {"final_target_world", "best_pregrasp_target_world"}},
                "final_target_world": row["final_target_world"].tolist(),
                "best_pregrasp_target_world": (
                    None if row["best_pregrasp_target_world"] is None
                    else row["best_pregrasp_target_world"].tolist()
                ),
            })
        index = {
            "schema_version": 1,
            "generated_at": utc_now(),
            "run": run_dir.name,
            "source_candidate_selection": "../candidate_selection.json",
            "source_failure_state": "../../failure_state.npz",
            "source_model": f"../../{model_source}",
            "candidate_count": len(records),
            "status_counts": dict(sorted(counts.items())),
            "overview_images": overview_images,
            "contact_sheet": "contact_sheet.png",
            "evidence_semantics": {
                "colored_fork": "requested final grasp TCP pose",
                "cyan_fork": "one recorded representative pregrasp request",
                "white_fork": "actual TCP in failure_state.npz",
                "failed_ik": "finite search found no solution; not proof of global unreachability",
                "trajectory_claim": "no path is drawn unless a path was recorded; these images show requested poses",
            },
            "candidates": serializable,
        }
        write_json(temporary / "index.json", index)
        write_readme(temporary, run_dir, serializable, counts)
        if destination.exists():
            if not force:
                raise FileExistsError(str(destination))
            shutil.rmtree(destination)
        temporary.replace(destination)
        return index
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def candidate_failure_runs(batch_root: Path) -> list[Path]:
    result = []
    for run_dir in sorted(batch_root.glob("run_[0-9][0-9][0-9]")):
        selection = run_dir / "pick_insert/candidate_selection.json"
        pick_result = run_dir / "pick_insert/result.json"
        state = run_dir / "failure_state.npz"
        if not (selection.is_file() and pick_result.is_file() and state.is_file()):
            continue
        try:
            if (read_json(pick_result).get("failure_stage") == "candidate_selection"
                    and read_json(selection).get("selected_candidate_id") is None):
                result.append(run_dir)
        except (OSError, json.JSONDecodeError):
            continue
    return result


def batch_finished(batch_root: Path) -> bool:
    manifest = batch_root / "batch_manifest.json"
    if not manifest.is_file():
        return False
    try:
        return bool(read_json(manifest).get("finished_at"))
    except (OSError, json.JSONDecodeError):
        return False


def process_batch(batch_root: Path, *, force: bool = False) -> dict:
    rows = []
    errors = []
    for run_dir in candidate_failure_runs(batch_root):
        try:
            index = render_run(run_dir, force=force)
            rows.append({
                "run": run_dir.name,
                "status": "rendered",
                "candidate_count": index["candidate_count"],
                "index_ref": f"{run_dir.name}/pick_insert/candidate_visualizations/index.json",
            })
        except Exception as error:
            errors.append({
                "run": run_dir.name,
                "type": type(error).__name__,
                "message": str(error),
            })
    status = {
        "schema_version": 1,
        "updated_at": utc_now(),
        "batch_finished": batch_finished(batch_root),
        "candidate_selection_failures_seen": len(candidate_failure_runs(batch_root)),
        "rendered": rows,
        "errors": errors,
    }
    write_json(batch_root / "candidate_visualization_status.json", status)
    return status


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="one run directory or a batch root")
    parser.add_argument("--watch", action="store_true", help="keep scanning an active batch")
    parser.add_argument("--interval", type=float, default=30.0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    target = args.path.resolve()
    if args.interval <= 0:
        parser.error("--interval must be positive")
    if (target / "pick_insert/candidate_selection.json").is_file():
        print(json.dumps(render_run(target, force=args.force), ensure_ascii=False))
        return
    if not (target / "batch_manifest.json").is_file():
        parser.error("path is neither a run directory nor a batch root")
    while True:
        status = process_batch(target, force=args.force)
        print(json.dumps(status, ensure_ascii=False), flush=True)
        if not args.watch or status["batch_finished"]:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
