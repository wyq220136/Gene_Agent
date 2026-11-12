"""场景检查与人工验收入口，使用同步 MuJoCo 执行。"""

import argparse
from datetime import datetime
import json
import math
import os
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["check-scene", "view-scene"])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--perturb", action="store_true")
    parser.add_argument("--robot-profile", choices=("realman_dual_standard", "realman_single_pika"),
                        default="realman_dual_standard")
    parser.add_argument("--collision-overlay", action="store_true", help="额外保存碰撞代理和TCP坐标轴叠图")
    parser.add_argument("--gripper-width-mm", type=float, help="夹爪目标内开口，毫米；通过驱动执行而非直接改qpos")
    parser.add_argument("--gripper-group", help="指定夹爪所属组；不指定则应用于模型中所有夹爪")
    parser.add_argument("--settle-seconds", type=float, default=3.0)
    parser.add_argument("--restore", type=Path, help="相同场景模型的完整积分状态文件")
    args = parser.parse_args()
    if not math.isfinite(args.settle_seconds) or args.settle_seconds < 0:
        parser.error("--settle-seconds 必须有限且非负")
    if args.gripper_width_mm is not None and (not math.isfinite(args.gripper_width_mm) or args.gripper_width_mm < 0):
        parser.error("--gripper-width-mm 必须有限且非负")
    if args.gripper_group is not None and args.gripper_width_mm is None:
        parser.error("--gripper-group 需配合 --gripper-width-mm")
    # GL 后端必须在 import mujoco 前选择；桌面 viewer 保留默认 GLFW。
    if args.command == "check-scene":
        os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco
    import numpy as np
    from gene_agent.feedback import RunRecorder, snapshot, save_state, restore_state, fingerprint_files
    from gene_agent.simulation import build_scene
    from gene_agent.visualization import render_views, open_viewer

    output = (args.output or Path("runs") / datetime.now().strftime("scene-%Y%m%d-%H%M%S-%f")).resolve()
    recorder = RunRecorder(output, config={
        "seed": args.seed, "perturb": args.perturb, "settle_seconds": args.settle_seconds,
        "robot_profile": args.robot_profile,
        "command": args.command, "restore": str(args.restore) if args.restore else None,
        "collision_overlay": args.collision_overlay,
        "gripper_width_mm": args.gripper_width_mm, "gripper_group": args.gripper_group,
    })
    stage = "scene.build"
    try:
        model, data, metadata = build_scene(output / "model", seed=args.seed, perturb=args.perturb,
                                            robot_profile=args.robot_profile)
        _json(output / "scene_metadata.json", metadata)
        root = Path(__file__).resolve().parents[1]
        sources = [*root.glob("gene_agent/**/*.py"), *root.glob("config/**/*.json"),
                   *root.glob("assets/**/*")]
        _json(output / "source_hashes.json", fingerprint_files(p for p in sources if p.is_file()))
        recorder.event(stage, "passed", nq=model.nq, nv=model.nv, world_frame=metadata["world_frame"])
        stage = "scene.initialize"
        if args.restore:
            restore_state(model, data, args.restore)
        if args.gripper_width_mm is not None:
            from gene_agent.robots import set_gripper_width
            groups = ([args.gripper_group] if args.gripper_group
                      else list(metadata["robot"].get("grippers", {})))
            if not groups:
                raise ValueError("当前模型没有可驱动夹爪")
            for group in groups:
                set_gripper_width(model, data, metadata["robot"], group, args.gripper_width_mm / 1000)
        initial = snapshot(model, data)
        _json(output / "initial.json", initial)
        stage = "scene.settle"
        start = time.perf_counter()
        for _ in range(round(args.settle_seconds / model.opt.timestep)):
            mujoco.mj_step(model, data)
        if args.command == "view-scene":
            stage = "scene.viewer"
            with open_viewer(model, data) as viewer:
                while viewer.is_running():
                    tick = time.perf_counter()
                    mujoco.mj_step(model, data)
                    viewer.sync()
                    time.sleep(max(0, model.opt.timestep - (time.perf_counter() - tick)))
        # 人工 viewer 也推进物理；最终快照必须在关闭窗口后采集，避免摘要落后于看到的现场。
        stage = "scene.validate"
        final = snapshot(model, data)
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            raise RuntimeError("MuJoCo 状态出现非有限数值")
        _json(output / "state.json", final)
        save_state(model, data, output / "state.npz")
        contacts = final["contacts"]
        bowl_contacts = [c for c in contacts if "bowl" in c["body_names"]]
        bowl_z = final["bodies"]["bowl"]["T_world_body"][2][3]
        if bowl_z < metadata["objects"]["table"]["top_z_m"] - 0.1:
            raise RuntimeError("碗已落到支撑面以下，检查初态/接触几何")
        joint_error = {name: float(data.qpos[model.jnt_qposadr[model.joint(name).id]] - value)
                       for name, value in metadata["robot"]["home_joint_positions"].items()
                       if name in metadata["robot"]["joint_names"]}
        # 指关节是米、机械臂关节是弧度；不能把指位移塞进 joint_home_errors_rad。
        from gene_agent.robots import measure_gripper_width
        grippers = {group: measure_gripper_width(model, data, metadata["robot"], group)
                    for group, spec in metadata["robot"].get("grippers", {}).items()
                    if spec.get("mode") in {"parallel_slide_proxy", "pika_urdf"}}
        recorder.event(stage, "passed", sim_time=data.time, wall_seconds=time.perf_counter()-start,
                       contact_count=len(contacts), bowl_contact_count=len(bowl_contacts))
        if args.command == "check-scene":
            stage = "scene.render"
            views = render_views(model, data, output / "images",
                                 collision_overlay=args.collision_overlay,
                                 site_axes=args.collision_overlay,
                                 view_names=("overview", "top", "side", "contact", "task", "knob"))
            recorder.event(stage, "passed", state_id=views["state_id"], views=list(views["views"]))
        recorder.finish("passed", scope="scene_inspection_only", sim_time=final["sim_time"],
                        bowl_contacts=len(bowl_contacts),
                        max_home_error_rad=max(map(abs, joint_error.values()), default=0),
                        joint_home_errors_rad=joint_error,
                        grippers=grippers,
                        limitations=metadata["robot"]["limitations"] + metadata["objects"]["limitations"])
        print(f"PASS 场景检查（不代表抓取/可达性通过）\n结果：{output}")
        return 0
    except Exception as error:
        recorder.record_error(stage, error)
        recorder.finish("failed", stage=stage, error=f"{type(error).__name__}: {error}")
        print(f"FAIL {stage}: {error}\n完整记录：{output}")
        return 1


def _json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
