"""从 MuJoCo 执行记录检查被夹物体是否跟随 TCP，并定位首次失去双侧接触的帧。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from gene_agent.feedback.grasp_contact import classify_grasp_contacts
from gene_agent.simulation import MujocoRuntime
from gene_agent.simulation.diagnostics import query_diagnostics


def inspect_replay(*, scene_run: Path, execution_dir: Path, stages: list[str],
                   object_name: str = "bowl", group: str = "right",
                   target_center_m: tuple[float, float, float] | None = None) -> dict:
    """检查采样状态；接触点由回放姿态重新计算，因此只是帧级诊断。"""
    scene = scene_run / "source_scene"
    model = mujoco.MjModel.from_xml_path(str(scene / "scene.xml"))
    data = mujoco.MjData(model)
    robot = json.loads((scene / "robot/robot_metadata.json").read_text(encoding="utf-8"))
    runtime = MujocoRuntime(model, data, {"robot": robot, "world_frame": robot["world_frame"]})
    samples = []
    reference_relative = None
    for stage in stages:
        execution = json.loads((execution_dir / "executions" / f"{stage}.json").read_text(
            encoding="utf-8"))
        if not execution.get("replay_samples"):
            raise ValueError(f"{stage} 缺少回放状态")
        for index, sample in enumerate(execution["replay_samples"]):
            if samples and stage != stages[0] and index == 0:
                continue  # 阶段交界处同一帧只计算一次。
            data.qpos[:] = sample["qpos"]
            data.qvel[:] = 0.0
            data.time = sample["sim_time_s"]
            mujoco.mj_forward(model, data)
            tcp = np.asarray(data.site(robot["tcp_sites"][group]).xpos).copy()
            tcp_rotation = np.asarray(data.site(robot["tcp_sites"][group]).xmat).reshape(3, 3)
            center = np.asarray(data.body(object_name).xpos).copy()
            # 世界系相对位移会随着腕部旋转改变，即使碗牢牢夹住也会产生假漂移。
            # 必须变换到当帧 TCP 坐标系，才可比较夹持关系是否变化。
            relative = tcp_rotation.T @ (center - tcp)
            if reference_relative is None:
                reference_relative = relative.copy()
            contact = classify_grasp_contacts(
                query_diagnostics(runtime, body_names=(object_name,))["contacts"],
                robot["grippers"][group], object_body=object_name,
            )
            obstacles = []
            for contact_index in range(data.ncon):
                item = data.contact[contact_index]
                first = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(item.geom1)) or ""
                second = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(item.geom2)) or ""
                if object_name in first + second and "microwave_" in first + second:
                    obstacles.append({"geoms": [first, second],
                                      "penetration_m": max(0.0, -float(item.dist))})
            samples.append({
                "stage": stage, "sample_index": index,
                "sim_time_s": round(float(data.time), 3),
                "object_center_world_m": center.tolist(),
                "tcp_center_world_m": tcp.tolist(),
                "relative_position_drift_m": float(np.linalg.norm(relative - reference_relative)),
                "bilateral_pad_contact": bool(contact["success"]),
                "microwave_contacts": obstacles,
            })
    last = samples[-1]
    first_lost = next((row for row in samples if not row["bilateral_pad_contact"]), None)
    first_microwave = next((row for row in samples if row["microwave_contacts"]), None)
    error = None if target_center_m is None else float(np.linalg.norm(
        np.asarray(last["object_center_world_m"]) - np.asarray(target_center_m)))
    return {
        "schema_version": 1,
        "claim": "物体帧级回放诊断；接触由保存的 qpos 重算，不等于动态冲量记录",
        "stages": stages, "object_name": object_name,
        "sample_count": len(samples), "first_lost_bilateral_contact": first_lost,
        "first_microwave_contact": first_microwave,
        "final_object_center_world_m": last["object_center_world_m"],
        "final_target_error_m": error,
        "max_relative_position_drift_m": max(row["relative_position_drift_m"] for row in samples),
        "bilateral_contact_all_samples": all(row["bilateral_pad_contact"] for row in samples),
        "samples": samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-run", type=Path, default=Path("runs/open-door-physical-15mm-plan-20261008"))
    parser.add_argument("--execution-dir", type=Path, required=True)
    parser.add_argument("--stage", action="append", dest="stages", required=True)
    parser.add_argument("--object", default="bowl")
    parser.add_argument("--group", default="right")
    parser.add_argument("--target-center", type=float, nargs=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = inspect_replay(
        scene_run=args.scene_run, execution_dir=args.execution_dir,
        stages=args.stages, object_name=args.object, group=args.group,
        target_center_m=tuple(args.target_center) if args.target_center else None,
    )
    output = args.output or args.execution_dir / "held_object_replay_check.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                      encoding="utf-8")
    summary = {k: result[k] for k in (
        "sample_count", "bilateral_contact_all_samples", "final_object_center_world_m",
        "final_target_error_m", "max_relative_position_drift_m")}
    for key in ("first_lost_bilateral_contact", "first_microwave_contact"):
        row = result[key]
        summary[key] = None if row is None else {
            "stage": row["stage"], "sample_index": row["sample_index"],
            "sim_time_s": row["sim_time_s"],
            "microwave_geoms": sorted({name for pair in row["microwave_contacts"]
                                        for name in pair["geoms"] if name.startswith("microwave_")}),
        }
    print(json.dumps(summary, ensure_ascii=False))
    print(f"详细报告：{output.resolve()}")


if __name__ == "__main__":
    main()
