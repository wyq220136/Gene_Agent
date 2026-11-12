#!/usr/bin/env python3
"""在固定 seed 列表上从 fresh reset 评测三方块清扫，不跳过失败样本。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

# 与现有直接运行的 CLI 一致，从源码根目录加载 benchmark 包。
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import time
import traceback

from benchmarks.sweep_cube_task.summary import summarize_evaluation
from gene_agent.tasks.policies.sweep_cube_task.simulation import SweepCubeRun


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-dir", type=Path,
                        default=Path("data/realworld/brush_cube_dustpan_0"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True,
                        help="固定 seed 清单；每个 seed 计入成功率分母")
    parser.add_argument("--diagnostic-cpu-ik", action="store_true",
                        help="CPU 运动规划诊断；接触候选另用 GPU cuRobo 纯 IK 筛选，不作为正式 cuRobo 评测")
    parser.add_argument("--grasp-tilt-deg", type=float, default=-20.,
                        help="抓取刷柄的倾角（度），用于比较夹持姿态对后续可达性的影响")
    parser.add_argument("--robot-gravity-compensation", action="store_true",
                        help="仅机器人本体重力补偿，落盘有效仿真XML；操作物体不补偿")
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds 不能重复")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    scene_dir = args.scene_dir.resolve()
    rows = []
    for seed in args.seeds:
        run_dir = output / f"seed_{seed}"
        started = time.monotonic()
        try:
            result = SweepCubeRun(scene_dir, run_dir, randomize=True, seed=seed,
                                  diagnostic_cpu_ik=args.diagnostic_cpu_ik, grasp_tilt_deg=args.grasp_tilt_deg,
                                  robot_gravity_compensation=args.robot_gravity_compensation).run()
            row = {"seed": seed, "success": result["success"],
                   "failure_stage": result["failure_stage"],
                   "run_dir": str(run_dir),
                   "placement_valid": result["placement"]["valid_initial_state"],
                   "elapsed_wall_s": time.monotonic() - started}
        except Exception as exc:
            row = {"seed": seed, "success": False,
                   "failure_stage": "setup_exception", "run_dir": str(run_dir),
                   "error": f"{type(exc).__name__}: {exc}",
                   "traceback": traceback.format_exc(),
                   "elapsed_wall_s": time.monotonic() - started}
        rows.append(row)
        print(f"seed={seed} success={row['success']} failure={row['failure_stage']}",
              flush=True)
    report = summarize_evaluation(rows)
    report.update(schema_version=1, scene_dir=str(scene_dir),
                  robot_gravity_compensation=args.robot_gravity_compensation,
                  planner_backend="mujoco_cpu_pose_ik" if args.diagnostic_cpu_ik else "curobo",
                  evaluation_scope=("diagnostic_cpu_ik_only" if args.diagnostic_cpu_ik else
                                    "curobo_stage_collision_planning_with_full_mujoco_postcheck"),
                  knowledge_candidate_screen=("curobo_pure_pose_ik_then_mujoco_cpu_path_diagnostic"
                                              if args.diagnostic_cpu_ik else "curobo_full_trajectory"),
                  scene_xml_sha256=hashlib.sha256((scene_dir / "scene.xml").read_bytes()).hexdigest(),
                  requested_seeds=args.seeds,
                  evidence_mode="actual_mujoco_execution_with_contact_verified_assisted_brush_grip")
    (output / "evaluation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(f"成功 {report['successes']}/{report['episodes']} = {report['success_rate']:.1%}")
    print(output / "evaluation.json")


if __name__ == "__main__":
    main()
