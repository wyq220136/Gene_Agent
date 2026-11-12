"""Knowledge 候选的单臂 IK screening，不代表完整场景可达或轨迹安全。"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from gene_agent.knowledge import bowl_grasp_candidates, knob_grasp_candidates
from gene_agent.motion import SynchronousPlanner, load_realman_motion_config
from gene_agent.simulation import build_scene

SCOPE = "单臂IK screening：保留原自碰撞球与ignore，仅有远方诊断cuboid；未同步另一臂、本体、桌面、碗、微波炉。不是任务可达性或轨迹安全验证。"


def body_pose(model, data, name):
    body = data.body(name)
    pose = np.eye(4)
    pose[:3, :3] = body.xmat.reshape(3, 3)
    pose[:3, 3] = body.xpos
    return pose


def screen(planner, names, positions, candidates):
    """每个候选都保留结果，只有后端成功且FK误差达标才算IK通过。"""
    records = []
    for candidate in candidates:
        target = np.asarray(candidate["T_world_tcp"])
        result = planner.solve_ik(names, positions, target)
        category = result.get("failure_category", "backend_exception") if result.get("raw_error") else ("collision_postcheck_rejected" if result.get("failure_category") == "collision_postcheck_rejected" else "ik_solver_rejected")
        if result["success"]:
            actual = planner.fk(names, result["positions_rad"][0])
            position_error = float(np.linalg.norm(actual[:3, 3]-target[:3, 3]))
            orientation_error = float(Rotation.from_matrix(actual[:3, :3].T@target[:3, :3]).magnitude())
            result.update(fk_position_error_m=position_error, fk_orientation_error_rad=orientation_error)
            category = "ik_verified" if position_error < .005 and orientation_error < .05 else "fk_tolerance_failed"
        records.append({**candidate, "ik_screening_pass": category == "ik_verified", "failure_category": None if category == "ik_verified" else category,
                        "classification": category, "solver_result": result,
                        "full_scene_collision_checked": False, "trajectory_checked": False})
    return records


def summarize(records):
    return {"total": len(records), "ik_verified": sum(record["ik_screening_pass"] for record in records),
            "counts_by_classification": {category: sum(r["classification"] == category for r in records)
                                         for category in sorted(set(r["classification"] for r in records))}}


def run(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {"mode": "ik_screening_only", "collision_scope": SCOPE, "group": "right", "world_frame": "right_sdk_base",
              "cases": {}, "solver": {"ik_num_seeds": 16, "use_cuda_graph": False, "self_collision_check": True,
              "random_seed": 123, "candidate_order": "nominal bowl40, knob6; seed17 bowl40, knob6; optional fixed-budget parent search",
              "position_tolerance_m": .005, "orientation_tolerance_rad": .05}, "settle_time_s": 3.0}
    import torch
    report["versions"] = {"torch": torch.__version__, "cuda": torch.version.cuda, "mujoco": mujoco.__version__, "gpu": torch.cuda.get_device_name()}
    report["code_sha256"] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in
                             [Path(__file__), Path(__file__).resolve().parents[2]/"gene_agent/motion/curobo/core.py", Path(__file__).resolve().parents[2]/"gene_agent/knowledge/candidates.py"]}
    planner = None
    nominal_pose = None
    nominal_positions = None
    for label, seed, perturb in [("nominal", 0, False), ("perturbed_seed17", 17, True)]:
        model, data, metadata = build_scene(output_dir / label, seed=seed, perturb=perturb)
        for _ in range(round(3.0/model.opt.timestep)):
            mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        poses = {name: body_pose(model, data, name) for name in ["bowl", "microwave"]}
        names = metadata["robot"]["joint_groups"]["right"]
        positions = np.array([data.qpos[model.joint(name).qposadr[0]] for name in names])
        if planner is None:
            settings = load_realman_motion_config("right", metadata["robot"])
            scene_model = {"cuboid": {"diagnostic_far_box": {"dims": [.1, .1, .1], "pose": [3, 3, 3, 1, 0, 0, 0]}}}
            planner = SynchronousPlanner(**settings, scene_model=scene_model, scene_scope=SCOPE)
            report["resolved_group_config"] = settings
            report["scene_model_in_urdf_base"] = scene_model
            nominal_pose, nominal_positions = poses["microwave"].copy(), positions.copy()
        case = {"seed": seed, "perturb": perturb, "sim_time_s": data.time, "GT_poses": {k: v.tolist() for k,v in poses.items()},
                "joint_names": names, "start_joint_positions_rad": positions.tolist(), "scene_metadata": metadata,
                "simulation_warning_counts": [int(item.number) for item in data.warning],
                "qpos_snapshot": data.qpos.tolist(), "qvel_snapshot": data.qvel.tolist(), "ctrl_snapshot": data.ctrl.tolist()}
        for name, candidates in [("bowl", bowl_grasp_candidates(poses["bowl"])), ("knob", knob_grasp_candidates(poses["microwave"],concept=metadata["objects"]["knob_concept"]))]:
            expected_count = 40 if name == "bowl" else 6
            if len(candidates) != expected_count:
                raise ValueError(f"{name} 候选数发生变化: {len(candidates)}")
            case[name] = screen(planner, names, positions, candidates)
            print(label, name, summarize(case[name]), flush=True)
        case["summary"] = {name: summarize(case[name]) for name in ["bowl", "knob"]}
        report["cases"][label] = case
        (output_dir / "screening.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    # 仅当两个初始父位姿均无可用旋钮IK，按事先声明的有限集合搜索；每个结果都保留。
    if all(case["summary"]["knob"]["ik_verified"] == 0 for case in report["cases"].values()):
        search = []
        # 平移是world增量；yaw绕父物体自身原点左乘world Z旋转，不改变Concept局部安装。
        translations = [(-.10, 0, 0), (-.20, 0, 0), (0, .10, 0), (-.10, .10, 0)]
        rotations = [0., np.deg2rad(30), np.deg2rad(-30)]
        report["search_budget"] = {"translations_world_m": translations, "yaw_world_rad": rotations, "parent_poses": 12, "candidates_per_pose": 6}
        for i, delta in enumerate(translations):
            for j, yaw in enumerate(rotations):
                parent = nominal_pose.copy()
                parent[:3, 3] += delta
                parent[:3, :3] = Rotation.from_euler("z", yaw).as_matrix() @ parent[:3, :3]
                records = screen(planner, names, nominal_positions, knob_grasp_candidates(parent, pose_source="explicit_parent_pose_search",concept=report["cases"]["nominal"]["scene_metadata"]["objects"]["knob_concept"]))
                record = {"search_id": f"parent_{i}_{j}", "delta_translation_world_m": delta, "yaw_world_rad": yaw,
                          "T_world_microwave": parent.tolist(), "summary": summarize(records), "candidates": records,
                          "placement_validated_in_simulation": False}
                search.append(record)
                print(record["search_id"], record["summary"], flush=True)
        report["parent_search"] = search
        eligible = [item for item in search if item["summary"]["ik_verified"] > 0]
        report["placement_suggestions"] = [{k: item[k] for k in ["search_id", "T_world_microwave", "summary"]} for item in eligible]
    report["completed"] = True
    (output_dir / "screening.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return {label: case["summary"] for label, case in report["cases"].items()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("runs/ik-screening"))
    args = parser.parse_args()
    try:
        print(json.dumps(run(args.output), ensure_ascii=False))
    except Exception:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "raw_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
