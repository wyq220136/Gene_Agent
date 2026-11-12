#!/usr/bin/env python3
"""从实际 MuJoCo qpos/pose 记录量化勺子倾倒轨迹与小球脱离。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from gene_agent.knowledge.geometry import load_concept


def analyze(run_dir: Path) -> dict:
    run = run_dir.resolve()
    task = json.loads((run / "task_result.json").read_text(encoding="utf-8"))
    scene_path = Path(task["scene_xml_path"])
    if hashlib.sha256(scene_path.read_bytes()).hexdigest() != task["scene_xml_sha256"]:
        raise ValueError("运行场景 XML 与记录的 hash 不一致")
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    scene = json.loads((Path(task["scene_xml_path"]).parent / "result.json").read_text())
    source_scene = json.loads((Path(scene["source_scene_xml_path"]).parent / "scene.json").read_text())
    tool = task["tool_name"]
    local_center = load_concept(source_scene["concept_paths"][tool],
                                part_name="scoop_bowl").concept.pose_in_object[:3, 3]
    stages = [name for name in task["stages"]
              if name == "target_tilt" or name.startswith("target_tilt_")]
    rows = [sample for name in stages
            for sample in (task["stages"][name].get("execution") or {}).get(
                "replay_samples", ())]
    if len(rows) < 2:
        raise ValueError("没有实际执行的倾倒轨迹")
    unique = {round(float(row["sim_time_s"]), 7): row for row in rows}
    rows = sorted(unique.values(), key=lambda row: row["sim_time_s"])
    time = np.asarray([row["sim_time_s"] for row in rows], dtype=float)
    tcp = np.asarray([row["robots"]["left"]["T_world_tcp"] for row in rows])
    spoon = np.asarray([row["objects"][tool]["T_world_object"] for row in rows])
    centers = spoon[:, :3, 3] + np.einsum("nij,j->ni", spoon[:, :3, :3], local_center)
    tcp_xyz = tcp[:, :3, 3]
    angles = np.asarray([np.linalg.norm(Rotation.from_matrix(
        matrix @ spoon[0, :3, :3].T).as_rotvec()) for matrix in spoon[:, :3, :3]])
    cargo = task["skills"].get("scoop", {}).get("cargo_ball_ids", [])
    settle_rows = (task["stages"].get("target_settle", {}).get("execution") or {}).get(
        "replay_samples", ())
    arrival_rows = sorted({round(float(row["sim_time_s"]), 7): row
                           for row in (*rows, *settle_rows)}.values(),
                          key=lambda row: row["sim_time_s"])
    bowl_adr = model.joint(f"{task['target_bowl']}_free").qposadr[0]
    releases = {}
    for name in cargo:
        adr = model.joint(f"{name}_free").qposadr[0]
        positions = np.asarray([row["qpos"][adr:adr + 3] for row in rows])
        local = np.einsum("nij,nj->ni", spoon[:, :3, :3].transpose(0, 2, 1),
                          positions - spoon[:, :3, 3])
        # 球心越过侧壁外侧并留出半个球径，视为离开勺槽。
        outside = np.flatnonzero(np.abs(local[:, 1]) > .027)
        first = int(outside[0]) if len(outside) else None
        if first is None:
            releases[name] = None
            continue
        arrival = None
        for row in arrival_rows:
            if row["sim_time_s"] < time[first]:
                continue
            qpos = np.asarray(row["qpos"], dtype=float)
            relative = qpos[adr:adr + 3] - qpos[bowl_adr:bowl_adr + 3]
            if np.linalg.norm(relative[:2]) <= .035 and .007 <= relative[2] <= .11:
                arrival = float(row["sim_time_s"])
                break
        releases[name] = {
            "time_from_tilt_start_s": float(time[first] - time[0]),
            "spoon_rotation_deg": float(np.degrees(angles[first])),
            "local_ball_center_m": local[first].tolist(),
            "arrival_time_after_release_s": (
                None if arrival is None else arrival - float(time[first])),
        }
    result = {
        "run": str(run), "planner_backend": task["planner_backend"],
        "simulation_assistance": task["simulation_assistance"],
        "tilt_stages": stages, "duration_s": float(time[-1] - time[0]),
        "tcp_endpoint_displacement_m": float(np.linalg.norm(tcp_xyz[-1] - tcp_xyz[0])),
        "tcp_max_excursion_m": float(np.max(np.linalg.norm(tcp_xyz - tcp_xyz[0], axis=1))),
        "tcp_path_length_m": float(np.sum(np.linalg.norm(np.diff(tcp_xyz, axis=0), axis=1))),
        "tcp_max_speed_m_s": float(np.max(np.linalg.norm(np.diff(tcp_xyz, axis=0), axis=1)
                                          / np.diff(time))),
        "scoop_center_max_excursion_m": float(np.max(np.linalg.norm(
            centers - centers[0], axis=1))),
        "spoon_final_rotation_deg": float(np.degrees(angles[-1])),
        "cargo_release": releases,
        "cargo_in_target_after_settle": task["skills"].get("pour", {}).get(
            "transferred_ball_ids", []),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    result = analyze(args.run)
    destination = args.run / "pour_motion_analysis.json"
    destination.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
