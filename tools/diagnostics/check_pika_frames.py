"""离线核对 Pika frame；不创建机器人连接或发送运动命令。"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# 子进程使用装有厂家 SDK 的环境，只调用 Algo，不实例化 RoboticArm。
SDK_CODE = '''
import json, sys
from Robotic_Arm.rm_robot_interface import Algo, rm_robot_arm_model_e, rm_force_type_e
p = json.load(sys.stdin)
a = Algo(getattr(rm_robot_arm_model_e, p["model"]), rm_force_type_e.RM_MODEL_RM_B_E)
print("PIKA_AUDIT_JSON=" + json.dumps([a.rm_algo_forward_kinematics(q, flag=1) for q in p["joints_deg"]]))
'''


def run(args) -> dict:
    import hashlib
    import mujoco
    import numpy as np
    from scipy.spatial.transform import Rotation
    from gene_agent.robots import build_robot
    from gene_agent.robots.pika_frames import read_reference_transform, model_frames, pose_error

    output = args.output.resolve()
    if (output / "report.json").exists():
        raise FileExistsError(f"已有报告，请使用新的 --output：{output}")
    output.mkdir(parents=True, exist_ok=True)
    reference = read_reference_transform(args.reference)
    robot = build_robot(output / "robot", profile="realman_single_pika")
    model = mujoco.MjModel.from_xml_path(robot["xml_path"])
    data = mujoco.MjData(model)
    joints = [[0]*7, [.2,-.3,.1,-.4,.2,.3,-.1], [-.4,.25,-.3,.5,.1,-.2,.4],
              [.7,-.5,.4,-.7,-.3,.4,-.5]]
    samples = [model_frames(model, data, robot, q, width)
               for q in joints for width in (.08, .04)]
    sdk = {"status": "not_requested"}
    if args.sdk_python:
        env = dict(os.environ, PYTHONPATH="", PYTHONNOUSERSITE="1")
        result = subprocess.run([str(args.sdk_python), "-c", SDK_CODE],
                                input=json.dumps({"model": args.sdk_model,
                                                  "joints_deg": np.rad2deg(joints).tolist()}),
                                capture_output=True, text=True, check=True, timeout=40, env=env)
        lines = [line.removeprefix("PIKA_AUDIT_JSON=") for line in result.stdout.splitlines()
                 if line.startswith("PIKA_AUDIT_JSON=")]
        if len(lines) != 1:
            raise ValueError("厂家离线 FK 没有返回唯一 JSON 数据")
        poses = json.loads(lines[0])
        if len(poses) != len(joints):
            raise ValueError("厂家 FK 返回数量不匹配")
        rows = []
        zero_link_from_sdk = None
        for index, pose in enumerate(poses):
            if len(pose) != 6 or not np.isfinite(pose).all():
                raise ValueError("厂家 FK 未返回有限 6D pose")
            flange = np.eye(4)
            flange[:3, :3] = Rotation.from_euler("xyz", pose[3:]).as_matrix()
            flange[:3, 3] = pose[:3]
            link = np.asarray(samples[index * 2]["frames"]["urdf_link7"])
            # 仅检查“同 base、同零位下固定 tool 变换”假设；不据此写回标定。
            if zero_link_from_sdk is None:
                zero_link_from_sdk = np.linalg.inv(link) @ flange
            check = pose_error(link @ zero_link_from_sdk, flange)
            rows.append({"joints_rad": joints[index], "sdk_pose_xyz_euler_rad": pose,
                         "T_urdf_link7_sdk_nominal_flange": (np.linalg.inv(link) @ flange).tolist(),
                         "zero_pose_offset_hypothesis_error": check})
            for sample in samples[index*2:index*2+2]:
                sample["frames"]["sdk_nominal_flange"] = flange.tolist()
                sample["frames"]["reference_control"] = (
                    flange @ np.asarray(reference["T_sdk_flange_reference_control"])).tolist()
        sdk = {"status": "nominal_offline_only", "model": args.sdk_model,
               "python": str(args.sdk_python), "samples": rows,
               "base_alignment": "identity assumption for diagnosis; not physically verified",
               "zero_pose_offset_hypothesis_max_error_m": max(
                   r["zero_pose_offset_hypothesis_error"]["translation_m"] for r in rows)}

    gpu = {"status": "not_requested"}
    if args.curobo:
        from gene_agent.motion import SynchronousPlanner, load_realman_motion_config
        from gene_agent.motion.curobo.scene import export_collision_scene
        model_frames(model, data, robot, joints[0], .08)
        snapshot = export_collision_scene(model, data, robot, "left", output / "collision")
        settings = load_realman_motion_config("left", robot,
                                              gripper_snapshot=snapshot["gripper_snapshot"])
        planner = SynchronousPlanner(**settings, scene_model=snapshot["scene_model"],
                                     scene_scope=snapshot["scope"],
                                     scene_snapshot_sha256=snapshot["snapshot_sha256"],
                                     mounting_contact_policy=snapshot["mounting_contact_policy"])
        rows = []
        for sample in samples[::2]:
            actual = planner.fk(robot["joint_groups"]["left"], sample["joints_rad"])
            rows.append(pose_error(np.asarray(sample["frames"]["simulation_tcp"]), actual))
        gpu = {"status": "real_gpu_fk_only", "samples": rows,
               "max_translation_m": max(r["translation_m"] for r in rows),
               "max_rotation_deg": max(r["rotation_deg"] for r in rows),
               "internal_fk_matches": all(r["translation_m"] < 1e-5 and r["rotation_deg"] < .001
                                          for r in rows),
               "comparison_limits": {"translation_m": 1e-5, "rotation_deg": .001},
               "motion_planned": False}
    offsets = {}
    for name, pose in samples[0]["frames"].items():
        offsets[name] = (np.linalg.inv(samples[0]["frames"]["pika_model_origin"]) @ pose).tolist()
    report = {"schema_version": 1, "mode": "offline_geometry_and_fk_audit",
              "real_robot_connected": False, "accepted_for_realworld_execution": False,
              "robot_xml": robot["xml_path"], "reference": reference,
              "source_hashes": {**robot["source_hashes"],
                                str(Path(__file__).resolve()): hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
              "sdk": sdk, "curobo": gpu, "zero_pose_T_model_origin_frame": offsets,
              "samples": samples,
              "limitations": ["Reference control point identity relative to physical fingertips is unresolved",
                              "Pad sites and distal mesh point are geometric diagnostics, not calibrated contact points",
                              "Nominal SDK model is not actual robot calibrated DH",
                              "Production TCP and mounting configurations were not changed"]}
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"报告：{output / 'report.json'}；{len(samples)} 个姿态/开口组合；SDK={sdk['status']}；cuRobo={gpu['status']}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/pika_frames/audit"))
    parser.add_argument("--reference", type=Path, default=Path.home() / "Documents/yixuan/umi_data_train/src/umi_data_train/realman/robot.py")
    parser.add_argument("--sdk-python", type=Path, help="独立调用厂家 Algo 的 Python；不会连接设备")
    parser.add_argument("--sdk-model", default="RM_MODEL_RM_75_E", help="名义型号，不代表已确认真机型号")
    parser.add_argument("--curobo", action="store_true", help="额外执行 GPU cuRobo FK 对照；不规划运动")
    run(parser.parse_args())
