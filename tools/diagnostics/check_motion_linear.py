"""在实际抓碗场景中规划一段净空TCP直线，并保存可复查结果。"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import traceback

import numpy as np

from gene_agent.tasks.policies.microwave_task.planning.bowl import prepare_pick_bowl_scene, run_pick_bowl_planning
from gene_agent.tasks.policies.microwave_task.planning.grasp import run_pick_bowl_skill_planning
from gene_agent.motion import (
    CuroboPlannerAdapter, JointState, MotionOperation, MotionRequest, PlanningSceneRef,
    PoseTarget, SynchronousPlanner, load_realman_motion_config,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    output = args.output.resolve()
    try:
        case = prepare_pick_bowl_scene(output)
        run_pick_bowl_planning(case, seed=args.seed, ik_mode="batch")
        skill = run_pick_bowl_skill_planning(case, seed=args.seed)
        pre = next(stage for stage in skill["stages"] if stage["name"] == "pregrasp")
        if pre["status"] != "planned":
            raise RuntimeError("当前场景没有可用的pregrasp基准")
        start_payload = pre["result"]["goal_state"]
        start = JointState(tuple(start_payload["joint_names"]), tuple(start_payload["positions_rad"]))
        snapshot = case.snapshot
        settings = load_realman_motion_config(
            "right", case.metadata["robot"], gripper_snapshot=snapshot.get("gripper_snapshot"),
        )
        planner = SynchronousPlanner(
            **settings, scene_model=snapshot["scene_model"], scene_scope=snapshot["scope"],
            scene_snapshot_sha256=snapshot["snapshot_sha256"],
            mounting_contact_policy=snapshot["mounting_contact_policy"], random_seed=args.seed,
        )
        origin = planner.fk(start.joint_names, start.positions_rad)
        target = origin.copy()
        target[:3, 3] -= .03 * origin[:3, 2]  # 从pregrasp向远离碗的方向做净空验证。
        request = MotionRequest(
            "pick_bowl:linear_clearance_probe", MotionOperation.LINEAR_TCP, "right", start,
            PoseTarget("world_tcp", target),
            PlanningSceneRef(case.revision, snapshot["snapshot_sha256"], snapshot["scope"]),
            seed=args.seed,
        )
        result = CuroboPlannerAdapter(planner, group="right", scene_revision=case.revision).plan(request)
        path = ([planner.fk(result.trajectory.joint_names, q)[:3, 3].tolist()
                 for q in result.trajectory.positions_rad] if result.trajectory else [])
        payload = {
            "schema_version": 1, "probe": "pick_bowl_pregrasp_clearance",
            "selected_candidate_id": skill["selected_candidate_id"],
            "scene_snapshot_sha256": snapshot["snapshot_sha256"],
            "request": request.to_dict(), "result": result.to_dict(),
            "tcp_path_world_m": path,
        }
        (output / "linear_probe_result.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({
            "run": str(output), "success": result.success,
            "failure": None if result.failure is None else result.failure.to_dict(),
            "metrics": result.metrics.get("linear_path_metrics"),
        }, ensure_ascii=False))
        if not result.success:
            raise SystemExit(1)
        # 已知碰撞负例：在同一真实场景快照的复制品中加一个具名小盒。
        # 只用于证明起点预检会点名障碍物，不作为可执行规划结果。
        blocker_world = origin[:3, 3] - .115 * origin[:3, 2]
        blocker_base = np.linalg.inv(np.asarray(settings["world_from_base"])) @ np.r_[blocker_world, 1.]
        blocked_scene = copy.deepcopy(snapshot["scene_model"])
        blocked_scene["cuboid"]["test_linear_blocker"] = {
            "dims": [.02, .02, .02],
            "pose": [*blocker_base[:3].tolist(), 1., 0., 0., 0.],
        }
        blocked_planner = SynchronousPlanner(
            **settings, scene_model=blocked_scene, scene_scope="test_linear_blocker_negative",
            mounting_contact_policy=snapshot["mounting_contact_policy"], random_seed=args.seed,
        )
        blocked_target = origin.copy()
        blocked_target[:3, 3] -= .15 * origin[:3, 2]
        blocked = blocked_planner.plan_linear_tcp(
            start.joint_names, start.positions_rad, blocked_target,
        )
        (output / "linear_blocked_probe.json").write_text(json.dumps({
            "diagnostic_only": True,
            "blocker": blocked_scene["cuboid"]["test_linear_blocker"],
            "blocker_name": "test_linear_blocker",
            "target_world_tcp": blocked_target.tolist(),
            "result": blocked,
        }, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        if blocked.get("success") or blocked.get("failure_category") != "start_collision":
            raise RuntimeError("已知障碍负例未被判为起点碰撞")
        print(json.dumps({"blocked_probe": blocked["failure_category"],
                          "blocking_obstacle": "test_linear_blocker"}, ensure_ascii=False))
    except Exception:
        output.mkdir(parents=True, exist_ok=True)
        (output / "linear_probe_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
