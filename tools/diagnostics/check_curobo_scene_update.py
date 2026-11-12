"""验证复用中的 cuRobo planner 与冷启动 planner 对移动障碍物给出相同结论。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gene_agent.motion import SynchronousPlanner, load_realman_motion_config
from gene_agent.robots import build_robot


def _scene(position: list[float]) -> dict:
    # 同一个 4 m 方箱只改变 pose：远处是净空正例，原点是确定的起点碰撞负例。
    return {
        "cuboid": {
            "movable_enclosure": {
                "dims": [4.0, 4.0, 4.0],
                "pose": [*position, 1.0, 0.0, 0.0, 0.0],
            }
        }
    }


def _brief(result: dict) -> dict:
    return {
        "success": bool(result["success"]),
        "status": result.get("status"),
        "failure_category": result.get("failure_category"),
        "start_state_valid": (result.get("start_state_check") or {}).get("valid"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/curobo-scene-update-equivalence.json"),
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    robot_dir = args.output.parent / "scene-update-robot"
    robot = build_robot(robot_dir)
    settings = load_realman_motion_config("right", robot)
    names = robot["joint_groups"]["right"]
    home = np.asarray([robot["home_joint_positions"][name] for name in names], float)

    far_scene = _scene([5.0, 5.0, 5.0])
    blocked_scene = _scene([0.0, 0.0, 0.0])
    reused = SynchronousPlanner(
        **settings, scene_model=far_scene, scene_scope="可移动方箱场景更新等价性检查"
    )
    target_q = home.copy()
    target_q[0] += 0.08
    target = reused.fk(names, target_q)

    hot_far_before = reused.solve_ik(names, home, target)
    reused.update_scene(
        scene_model=blocked_scene,
        scene_scope="可移动方箱场景更新等价性检查",
        scene_snapshot_sha256="blocked",
    )
    hot_blocked = reused.solve_ik(names, home, target)
    reused.update_scene(
        scene_model=far_scene,
        scene_scope="可移动方箱场景更新等价性检查",
        scene_snapshot_sha256="far-again",
    )
    hot_far_after = reused.solve_ik(names, home, target)

    cold_blocked_planner = SynchronousPlanner(
        **settings, scene_model=blocked_scene, scene_scope="冷启动碰撞对照"
    )
    cold_blocked = cold_blocked_planner.solve_ik(names, home, target)

    checks = {
        "far_before_success": bool(hot_far_before["success"]),
        "hot_and_cold_blocked_match": (
            hot_blocked["success"] == cold_blocked["success"]
            and hot_blocked.get("failure_category") == cold_blocked.get("failure_category")
        ),
        "blocked_is_start_collision": (
            not hot_blocked["success"]
            and hot_blocked.get("failure_category") == "start_collision"
        ),
        "far_after_success": bool(hot_far_after["success"]),
    }
    report = {
        "schema_version": 1,
        "purpose": "证明 update_scene 会更新已有 IK/collision solver，且恢复场景后结果随之恢复",
        "results": {
            "hot_far_before": _brief(hot_far_before),
            "hot_blocked": _brief(hot_blocked),
            "cold_blocked": _brief(cold_blocked),
            "hot_far_after": _brief(hot_far_after),
        },
        "checks": checks,
        "success": all(checks.values()),
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    reused.close()
    cold_blocked_planner.close()
    print(json.dumps({"success": report["success"], "checks": checks}, ensure_ascii=False))
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
