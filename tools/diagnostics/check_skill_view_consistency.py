"""同一 Skill 帧在 Viser 与 MuJoCo viewer 中恢复的状态一致性检查。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import viser

from gene_agent.simulation.skill_viewer import SkillMujocoViewer
from gene_agent.visualization.motion_debug.capture import _free_port
from gene_agent.visualization.skill_inspector import SkillInspectorApp


def compare(run: Path, stage: str | None, time_s: float, *, mode: str = "actual") -> dict:
    server = viser.ViserServer(host="127.0.0.1", port=_free_port())
    try:
        app = SkillInspectorApp(server, run, stage_id=stage, mode=mode,
                                initial_time_s=time_s)
        viser_frame = app.timeline.seek(time_s)
        viewer = SkillMujocoViewer(run)
        viewer.select(stage, mode=mode)
        mujoco_frame = viewer.seek(time_s)
        if viser_frame.qpos is None or mujoco_frame.qpos is None:
            raise ValueError("比较需要完整 qpos；部分旧采样无法复原物体状态")
        if viser_frame.state_id != mujoco_frame.state_id:
            raise AssertionError("两个 viewer 选择了不同状态")
        qpos_error = float(np.max(np.abs(app.data.qpos - viewer.data.qpos)))
        recorded_qpos_error = float(np.max(np.abs(app.data.qpos - viser_frame.qpos)))
        instance = app.view.instance_id
        recorded_pose = np.asarray(viser_frame.object_poses_world[instance], dtype=float)
        visible_pose = np.eye(4)
        visible_pose[:3, :3] = app.data.body(instance).xmat.reshape(3, 3)
        visible_pose[:3, 3] = app.data.body(instance).xpos
        mujoco_pose = np.eye(4)
        mujoco_pose[:3, :3] = viewer.data.body(instance).xmat.reshape(3, 3)
        mujoco_pose[:3, 3] = viewer.data.body(instance).xpos
        object_error = float(np.max(np.abs(visible_pose - mujoco_pose)))
        recorded_object_error = float(np.max(np.abs(visible_pose - recorded_pose)))
        attachment = (viser_frame.attachment or {}).get("object_instance_id")
        assert attachment == (mujoco_frame.attachment or {}).get("object_instance_id")
        assert viser_frame.collision_snapshot_ref == mujoco_frame.collision_snapshot_ref
        assert qpos_error < 1e-10 and recorded_qpos_error < 1e-10, (qpos_error, recorded_qpos_error)
        assert object_error < 1e-10 and recorded_object_error < 1e-8, (object_error, recorded_object_error)
        return {
            "selection": stage or "all", "stage_id": viser_frame.stage_id,
            "mode": mode, "instance_id": instance,
            "global_time_s": time_s, "sample_time_s": viser_frame.sample_time_s,
            "sim_time_s": viser_frame.sim_time_s, "state_id": viser_frame.state_id,
            "sample_quality": viser_frame.sample_quality,
            "qpos_length": len(viser_frame.qpos),
            "qpos_max_abs_error": qpos_error,
            "recorded_qpos_max_abs_error": recorded_qpos_error,
            "object_pose_max_abs_error": object_error,
            "recorded_object_pose_max_abs_error": recorded_object_error,
            "attachment_object_id": attachment,
            "collision_snapshot_ref": viser_frame.collision_snapshot_ref,
            "result": "PASS",
        }
    finally:
        server.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--stage")
    parser.add_argument("--mode", choices=("planned", "actual"), default="actual")
    parser.add_argument("--time", type=float, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = compare(args.run.resolve(), args.stage, args.time, mode=args.mode)
    payload = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
