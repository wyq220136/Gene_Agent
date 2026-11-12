"""Motion新接口的可运行离线示例；只生成规划记录，不连接任何设备。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from gene_agent.motion import JointState, MotionOperation, PivotLimits, PoseTarget, motion_configuration
from gene_agent.motion.curobo import CuroboPlanningSession
from gene_agent.motion.curobo.scene import export_collision_scene
from gene_agent.robots import build_robot


def plan_approach(motion, target):
    """Skill只描述目标与路径类型；这一函数无需导入cuRobo或机器人型号。"""
    return motion.plan_linear(target, request_id="example/approach")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--collision-mode", choices=("full", "robot_only"), default="full")
    parser.add_argument("--measure-runs", type=int, default=3)
    args = parser.parse_args()
    if args.measure_runs < 1:
        parser.error("measure-runs必须至少为1")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    robot = build_robot(output / "robot", profile="realman_single_pika")
    model = mujoco.MjModel.from_xml_path(robot["xml_path"])
    data = mujoco.MjData(model)
    group = next(iter(robot["joint_groups"]))
    names = tuple(robot["joint_groups"][group])
    # 已验证的诊断姿态，关节单位rad；不是实测机器人初态。
    q = (-.3604, -.265, -.2368, 1.8918, .0837, 1.2068, -.6022)
    start = JointState(names, q)
    for name, value in zip(names, q):
        data.qpos[model.joint(name).qposadr[0]] = value
    mujoco.mj_forward(model, data)
    report = {"claim": "离线诊断规划；无物理执行或真机运动", "collision_mode": args.collision_mode,
              "seed": 123, "planner_mode": "cuda_graph_benchmark",
              "budget_test_scope": "严格可解目标的预算透传＋不可达拒绝；不证明近似接受分支",
              "start": start.to_dict(), "operations": {}}
    with motion_configuration({"collision_mode": args.collision_mode}):
        snapshot = export_collision_scene(model, data, robot, group, output / "scene")
        with CuroboPlanningSession(robot, seed=123) as session:
            def stage(**kwargs):
                return session.stage(snapshot=snapshot, start=start, group=group,
                                     scene_revision=0, **kwargs)
            with stage() as motion:
                tcp = motion.fk()
                report["initialization_cache"] = motion.cache_evidence
            report["snapshot_sha256"] = snapshot["snapshot_sha256"]
            shifted = tcp.copy()
            shifted[2, 3] += .01
            target = PoseTarget("world_tcp", shifted)
            turned = tcp.copy()
            turned[:3, :3] = tcp[:3, :3] @ Rotation.from_rotvec([0, 0, .04]).as_matrix()
            pivot = PoseTarget("world_tcp", turned)
            budget = {"max_position_error_m": .01, "max_rotation_error_rad": .05}
            operations = {
                "ik": lambda motion: motion.solve_ik(target, request_id="example/ik"),
                "plan": lambda motion: motion.plan_to(target, request_id="example/plan"),
                "linear": lambda motion: plan_approach(motion, target),
                "pivot": lambda motion: motion.plan_pivot(pivot, request_id="example/pivot",
                                                          limits=PivotLimits(.01, .04)),
                "batch_ik": lambda motion: motion.evaluate_ik(
                    [PoseTarget("world_tcp", tcp), target], request_ids=["example/b0", "example/b1"]),
                "budget_plan": lambda motion: motion.plan_to(
                    target, request_id="example/budget", approximate_ik_budget=budget),
            }
            for name, call in operations.items():
                samples = []
                for repeat in range(1 + args.measure_runs):
                    with stage(restart_sampling=True) as motion:
                        if name == "batch_ik":
                            requests = [motion.request(MotionOperation.IK, value, request_id=identifier)
                                        for value, identifier in ((PoseTarget("world_tcp", tcp), "example/b0"),
                                                                  (target, "example/b1"))]
                        else:
                            operation = {"ik": MotionOperation.IK, "plan": MotionOperation.PLAN,
                                         "linear": MotionOperation.LINEAR_TCP, "pivot": MotionOperation.PIVOT_TCP,
                                         "budget_plan": MotionOperation.PLAN}[name]
                            identifier = {"linear": "approach", "budget_plan": "budget"}.get(name, name)
                            requests = [motion.request(operation, pivot if name == "pivot" else target,
                                request_id=f"example/{identifier}",
                                approximate_ik_budget=budget if name == "budget_plan" else None,
                                pivot_limits=PivotLimits(.01, .04) if name == "pivot" else None)]
                        began = time.perf_counter()
                        result = call(motion)
                        elapsed = time.perf_counter() - began
                        results = result if isinstance(result, tuple) else (result,)
                        samples.append({"call_duration_s": elapsed,
                                        "cache": motion.cache_evidence,
                                        "requests": [request.to_dict() for request in requests],
                                        "results": [item.to_dict() for item in results]})
                report["operations"][name] = {
                    "first_call_s": samples[0]["call_duration_s"],
                    "warm_median_s": statistics.median(s["call_duration_s"] for s in samples[1:]),
                    "success": all(item["success"] for s in samples for item in s["results"]),
                    "samples": samples,
                }
            unreachable = tcp.copy()
            unreachable[:3, 3] = [10., 10., 10.]
            with stage(restart_sampling=True) as motion:
                report["unreachable_request"] = motion.request(
                    MotionOperation.PLAN, PoseTarget("world_tcp", unreachable),
                    request_id="example/unreachable", approximate_ik_budget=budget).to_dict()
                rejected = motion.plan_to(PoseTarget("world_tcp", unreachable),
                                          request_id="example/unreachable", approximate_ik_budget=budget)
            report["unreachable"] = rejected.to_dict()

    # 独立明确的碰撞负例，覆盖完整碰撞后检；不删减任务场景来制造成功。
    with motion_configuration({"collision_mode": "full"}):
        blocked = dict(snapshot, collision_mode="full", scope="diagnostic_blocked",
                       scene_model={"cuboid": {"blocked_box": {"dims": [3., 3., 3.],
                                                               "pose": [0, 0, 0, 1, 0, 0, 0]}}},
                       records=[])
        blocked["snapshot_sha256"] = hashlib.sha256(
            json.dumps(blocked["scene_model"], sort_keys=True).encode()).hexdigest()
        with CuroboPlanningSession(robot, seed=123, collision_mode="full") as session:
            with session.stage(snapshot=blocked, start=start, group=group, scene_revision=1) as motion:
                report["known_collision_request"] = motion.request(
                    MotionOperation.PLAN, target, request_id="example/known_collision").to_dict()
                report["known_collision_scene"] = blocked["scene_model"]
                rejected = motion.plan_to(target, request_id="example/known_collision")
                report["known_collision"] = rejected.to_dict()
    report["success"] = (all(item["success"] for item in report["operations"].values())
                         and not report["unreachable"]["success"]
                         and report["known_collision"]["failure"]["kind"] == "start_collision")
    (output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"success": report["success"], "output": str(output),
                      "operations": {k: {x: v[x] for x in ("success", "first_call_s", "warm_median_s")}
                                     for k, v in report["operations"].items()}}, ensure_ascii=False), flush=True)
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
