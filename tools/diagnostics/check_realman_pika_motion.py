"""核对 Pika 单臂的 MuJoCo/cuRobo TCP 与一次小幅目标规划。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from gene_agent.motion import SynchronousPlanner, load_realman_motion_config
from gene_agent.motion.curobo.scene import export_collision_scene
from gene_agent.robots import build_robot


def run(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    robot = build_robot(output / "robot", profile="realman_single_pika")
    model = mujoco.MjModel.from_xml_path(robot["xml_path"])
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    snapshot = export_collision_scene(model, data, robot, "left", output / "collision")
    settings = load_realman_motion_config("left", robot,
                                          gripper_snapshot=snapshot["gripper_snapshot"])
    planner = SynchronousPlanner(
        **settings, scene_model=snapshot["scene_model"], scene_scope=snapshot["scope"],
        scene_snapshot_sha256=snapshot["snapshot_sha256"],
        mounting_contact_policy=snapshot["mounting_contact_policy"],
    )
    names = robot["joint_groups"]["left"]
    samples = ([0.0] * 7, [.1, -.1, .1, -.1, .1, -.1, .1],
               [-.2, .15, -.1, .15, -.1, .1, -.1], [.25, .1, -.15, 0, .2, 0, -.1])
    fk_checks = []
    for positions in samples:
        for name, value in zip(names, positions):
            data.qpos[model.joint(name).qposadr[0]] = value
        mujoco.mj_forward(model, data)
        expected = np.eye(4)
        site = data.site(robot["tcp_sites"]["left"])
        expected[:3, :3] = site.xmat.reshape(3, 3)
        expected[:3, 3] = site.xpos
        actual = planner.fk(names, positions)
        delta = np.linalg.inv(expected) @ actual
        fk_checks.append({
            "joint_positions_rad": list(positions),
            "translation_error_m": float(np.linalg.norm(delta[:3, 3])),
            "rotation_error_rad": float(np.arccos(np.clip((np.trace(delta[:3, :3]) - 1) / 2, -1, 1))),
        })
    zero = samples[0]
    target_joints = list(zero)
    target_joints[0] = .06
    plan = planner.plan(names, zero, planner.fk(names, target_joints))
    trajectory = np.asarray(plan.get("positions_rad") or [], dtype=float)
    endpoint_error = None
    if plan["success"] and trajectory.ndim == 2 and len(trajectory):
        achieved = planner.fk(names, trajectory[-1])
        endpoint_error = float(np.linalg.norm(achieved[:3, 3] - planner.fk(names, target_joints)[:3, 3]))
    plan_summary = {
        "success": bool(plan["success"]), "status": plan.get("status"),
        "message": plan.get("message"),
        "trajectory_points": len(trajectory),
        "endpoint_position_error_m": endpoint_error,
        "collision_verification": plan.get("collision_verification"),
    }
    # 两个确定性场景探针：左指内侧的外物必须被拒绝，远处外物必须不影响零位。
    for name in names:
        data.qpos[model.joint(name).qposadr[0]] = 0.0
    mujoco.mj_forward(model, data)
    finger_position = data.site("pika_left_inner").xpos.copy()
    collision_probes = {}
    for label, center, expected_valid in (
        ("at_left_finger", finger_position, False),
        ("far_clearance", np.array([5.0, 5.0, 5.0]), True),
    ):
        probe = SynchronousPlanner(
            **settings,
            scene_model={"cuboid": {label: {"dims": [.01, .01, .01],
                                               "pose": [*center.tolist(), 1, 0, 0, 0]}}, "mesh": {}},
            scene_scope=f"Pika {label} collision probe",
        )
        collision_probes[label] = {
            "zero_configuration_valid": probe._configuration_validity([zero])[0],
            "expected_valid": expected_valid,
        }
    (output / "plan_diagnostic.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    translation_error = max(row["translation_error_m"] for row in fk_checks)
    rotation_error = max(row["rotation_error_rad"] for row in fk_checks)
    report = {
        "fk_translation_error_m": translation_error,
        "fk_rotation_error_rad": rotation_error,
        "fk_checks": fk_checks,
        "small_motion_plan": plan_summary,
        "collision_probes": collision_probes,
        "gripper_snapshot_id": snapshot["gripper_snapshot"]["snapshot_id"],
        "planner_sphere_count": sum(len(values) for values in
                                    settings["robot_config"]["kinematics"]["collision_spheres"].values()),
        "pass": translation_error < .001 and rotation_error < .01
                and plan_summary["success"] and len(trajectory) > 1
                and plan_summary["collision_verification"].get("valid") is True
                and endpoint_error is not None and endpoint_error < .001
                and all(row["zero_configuration_valid"] == row["expected_valid"]
                        for row in collision_probes.values()),
    }
    (output / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Pika FK ({len(samples)} poses): max position error {translation_error * 1000:.2f} mm; "
          f"max rotation error {np.degrees(rotation_error):.2f} deg; "
          f"small motion plan {plan_summary['status']}; "
          f"{'PASS' if report['pass'] else 'FAIL'}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/realman-single-pika-motion-check"))
    raise SystemExit(0 if run(parser.parse_args().output.resolve())["pass"] else 1)
