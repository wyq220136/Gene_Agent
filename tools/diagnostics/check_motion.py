"""实际 GPU FK/IK/轨迹基线；诊断场景明确不等于完整任务场景。"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import traceback
import hashlib
import platform
import statistics

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import mujoco
import numpy as np
from gene_agent.robots import build_robot
from gene_agent.motion import PlannerMode, load_realman_motion_config, SynchronousPlanner
from gene_agent.motion.configuration import CollisionMode


def measured_calls(call, *, warmup_runs: int, measure_runs: int):
    """固定输入重复调用，分开记录 CUDA Graph 预热与稳定测量。"""
    initialization = call()
    warmup = [call() for _ in range(warmup_runs)]
    measured = [call() for _ in range(measure_runs)]
    durations = [float(value["duration_s"]) for value in measured]
    return measured[-1], {
        "initialization_duration_s": float(initialization["duration_s"]),
        "warmup_duration_s": [float(value["duration_s"]) for value in warmup],
        "measured_duration_s": durations,
        "mean_s": statistics.fmean(durations),
        "median_s": statistics.median(durations),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("runs/motion-baseline"))
    parser.add_argument("--skip-plan", action="store_true")
    parser.add_argument("--planner-mode", choices=[value.value for value in PlannerMode],
                        default=PlannerMode.CUDA_GRAPH_BENCHMARK.value)
    parser.add_argument("--random-seed", type=int, default=123)
    parser.add_argument("--collision-mode", choices=[m.value for m in CollisionMode], default="full")
    parser.add_argument("--warmup-runs", type=int, default=3)
    parser.add_argument("--measure-runs", type=int, default=10)
    args = parser.parse_args()
    if args.warmup_runs < 0 or args.measure_runs < 1:
        parser.error("warmup-runs 必须非负，measure-runs 必须至少为 1")
    mode = PlannerMode(args.planner_mode)
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"scope": "单臂自碰撞加独立诊断障碍；不含另一臂、本体、桌面和任务物体，禁止作为任务执行轨迹"}
    try:
        import torch
        report["versions"] = {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda, "mujoco": mujoco.__version__, "gpu": torch.cuda.get_device_name()}
        report["code_sha256"] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__), Path(__file__).resolve().parents[2]/"gene_agent/motion/curobo/core.py"]}
        report["solver_config"] = {"planner_mode": mode.value, "self_collision_check": True,
            "use_cuda_graph": mode is PlannerMode.CUDA_GRAPH_BENCHMARK,
            "ik_num_seeds": 16, "trajopt_num_seeds": 4, "max_attempts": 1,
            "interpolation_dt_s": .025, "random_seed": args.random_seed,
            "fk_seeds": [2901, 2902, 2903, 2904]}
        report["performance"] = {"note": "同一进程、固定输入；首次初始化、额外预热和稳定测量分别记录"}
        info = build_robot(args.output / "robot")
        model = mujoco.MjModel.from_xml_path(info["xml_path"])
        data = mujoco.MjData(model)
        fk_results = {}
        report["resolved_group_configs"] = {}
        for group in info["joint_groups"]:
            settings = load_realman_motion_config(group, info)
            report["resolved_group_configs"][group] = settings
            # 一个明确远离机器人的 cuboid 用于净空正例；不从完整场景删除对象。
            scene = {"cuboid": {"diagnostic_far_box": {"dims": [.1, .1, .1], "pose": [3, 3, 3, 1, 0, 0, 0]}}}
            planner = SynchronousPlanner(**settings, scene_model=scene, scene_scope=report["scope"],
                                         collision_mode=args.collision_mode,
                                         planner_mode=mode, random_seed=args.random_seed)
            names = info["joint_groups"][group]
            home = np.array([info["home_joint_positions"][name] for name in names])
            errors = []
            for index in range(5):
                q = home if index == 0 else home + np.random.default_rng(2900+index).uniform(-.15, .15, len(names))
                for name, value in zip(names, q):
                    data.qpos[model.joint(name).qposadr[0]] = value
                mujoco.mj_forward(model, data)
                expected = data.site(info["tcp_sites"][group])
                actual = planner.fk(names, q)
                errors.append({"position_m": float(np.linalg.norm(actual[:3, 3]-expected.xpos)),
                               "rotation_matrix_max_abs": float(np.max(np.abs(actual[:3, :3]-expected.xmat.reshape(3, 3))))})
            fk_results[group] = errors
            if group == "right":
                target_q = home.copy()
                target_q[0] += .08
                target = planner.fk(names, target_q)
                report["scene_model_in_urdf_base"] = scene
                report["request"] = dict(joint_names=names, start_rad=home.tolist(), target_joint_witness_rad=target_q.tolist(), target_world_tcp=target.tolist(), target_frame="world_tcp")
                ik_call = lambda: planner.solve_ik(names, home, target)
                if mode is PlannerMode.CUDA_GRAPH_BENCHMARK:
                    report["ik"], report["performance"]["ik"] = measured_calls(
                        ik_call, warmup_runs=args.warmup_runs, measure_runs=args.measure_runs)
                else:
                    report["ik"] = ik_call()
                if report["ik"]["success"]:
                    actual = planner.fk(names, report["ik"]["positions_rad"][0])
                    report["ik"]["fk_position_error_m"] = float(np.linalg.norm(actual[:3, 3]-target[:3, 3]))
                if not args.skip_plan:
                    plan_call = lambda: planner.plan(names, home, target)
                    if mode is PlannerMode.CUDA_GRAPH_BENCHMARK:
                        report["plan"], report["performance"]["plan"] = measured_calls(
                            plan_call, warmup_runs=args.warmup_runs, measure_runs=args.measure_runs)
                    else:
                        report["plan"] = plan_call()
                    if report["plan"]["success"]:
                        actual = planner.fk(names, report["plan"]["positions_rad"][-1])
                        report["plan"]["start_joint_max_error_rad"] = float(np.max(np.abs(np.asarray(report["plan"]["positions_rad"])[0]-home)))
                        report["plan"]["fk_position_error_m"] = float(np.linalg.norm(actual[:3, 3]-target[:3, 3]))
                # 独立负例：整个机械臂处于一个 4m 方箱中，保持原自碰撞与场景检查。
                blocked = SynchronousPlanner(**settings, scene_model={"cuboid": {"diagnostic_enclosure": {"dims": [4, 4, 4], "pose": [0, 0, 0, 1, 0, 0, 0]}}}, scene_scope="故意包住整个机械臂的已知碰撞负例", planner_mode=mode, random_seed=args.random_seed, collision_mode=args.collision_mode)
                report["blocked_ik"] = blocked.solve_ik(names, home, target)
                if args.collision_mode == "robot_only":
                    # 固定种子，保留抽样与诊断；只用于找已知自碰撞负例。
                    samples = np.random.default_rng(730).uniform(-2., 2., (256, len(names)))
                    validity = planner._configuration_validity(samples)
                    hits = [i for i, valid in enumerate(validity) if not valid]
                    for i in hits:
                        diagnostic = planner._collision_diagnostic(samples[i])
                        if diagnostic["self_collision"] and not diagnostic["joint_limit_violation"]:
                            report["self_collision_negative"] = {"q_rad": samples[i].tolist(), "diagnostic": diagnostic,
                                "result": planner.solve_ik(names, samples[i], target)}
                            break
                blocked.close()
            planner.close()
        report["fk"] = fk_results
        report["fk_pass"] = all(e["position_m"] < 2e-6 and e["rotation_matrix_max_abs"] < 2e-6 for values in fk_results.values() for e in values)
        report["ik_verified"] = report["ik"]["success"] and report["ik"].get("fk_position_error_m", float("inf")) < 1e-4
        if not args.skip_plan:
            report["plan_verified"] = report["plan"]["success"] and report["plan"].get("fk_position_error_m", float("inf")) < 1e-4 and report["plan"].get("start_joint_max_error_rad", float("inf")) < 1e-5 and len(report["plan"].get("positions_rad", [])) > 1
        positive_collision_checks = report["ik"].get("collision_verification", {}).get("valid") is True
        if not args.skip_plan:
            positive_collision_checks = positive_collision_checks and report["plan"].get("collision_verification", {}).get("valid") is True
        report["success"] = (
            report["fk_pass"] and report["ik_verified"]
            and (args.skip_plan or report["plan_verified"])
            and report["ik"]["success"] and positive_collision_checks
            and (report["blocked_ik"]["success"] if args.collision_mode == "robot_only" else not report["blocked_ik"]["success"])
            and report["blocked_ik"]["raw_error"] is None
            and (report.get("self_collision_negative", {}).get("result", {}).get("failure_category") == "start_collision"
                 if args.collision_mode == "robot_only" else report["blocked_ik"].get("failure_category") == "start_collision")
            and (args.skip_plan or report["plan"]["success"])
        )
    except Exception:
        report.update(success=False, raw_error=traceback.format_exc())
    (args.output / "motion_result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: report.get(key) for key in ("success", "fk_pass")}, ensure_ascii=False))
    for operation in ("ik", "plan", "blocked_ik"):
        if operation in report:
            print(operation, report[operation]["success"], report[operation]["status"])
    return 0 if report["success"] else 1

if __name__ == "__main__":
    raise SystemExit(main())
