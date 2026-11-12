#!/usr/bin/env python3
"""检查旧 Bowl 固定 keypose 与当前微波炉 URDF 的 Viser 几何叠图。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import mujoco
import numpy as np
from PIL import Image

from gene_agent.scene.microwave_frame import estimate_legacy_microwave_frame
from gene_agent.tasks.policies.microwave_task.operations.bowl import (
    PROFILE_NAME, load_object_operation_profile,
)
from gene_agent.tasks.policies.microwave_task.planning.bowl import load_pick_bowl_case
from gene_agent.visualization.microwave_object_keypose_app import (
    EXTRACT_LABELS, PLACE_LABELS, MicrowaveObjectKeyposeApp, projected_keyposes,
)
from gene_agent.visualization.viser_sim.capture import capture_viser_page


ROOT = Path(__file__).resolve().parents[2]


def _target_gripper_rendered(path: Path) -> bool:
    """避免浏览器仅画完柜体就保存，黄色目标夹爪必须实际入镜。"""
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.int16)
    red, green, blue = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    yellow = ((red > 175) & (green > 120) & (green < 215)
              & (blue < 110) & (red > green + 15))
    return int(np.count_nonzero(yellow)) > 1000


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "runs/pick-bowl-planning-final")
    parser.add_argument("--task-dir", type=Path, default=ROOT / "data/task_1")
    parser.add_argument("--port", type=int, default=8093)
    parser.add_argument("--target", choices=PLACE_LABELS + EXTRACT_LABELS,
                        default=PLACE_LABELS[0])
    parser.add_argument("--view", choices=("全景", "炉门正面", "夹爪侧面"),
                        default="炉门正面")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--capture", type=Path, help="截取当前 keypose 的真实 Viser 页面并退出")
    args = parser.parse_args()
    case = load_pick_bowl_case(args.run)
    joint = case.model.joint("microwave_door_joint")
    case.data.qpos[joint.qposadr[0]] = 0.0
    mujoco.mj_forward(case.model, case.data)
    alignment = estimate_legacy_microwave_frame(case.model, case.data, args.task_dir)
    profile = load_object_operation_profile(args.task_dir, "Bowl_0")
    body = case.data.body("microwave")
    T_world_calibrated = np.eye(4)
    T_world_calibrated[:3, :3] = body.xmat.reshape(3, 3)
    T_world_calibrated[:3, 3] = body.xpos
    targets, placed_object = projected_keyposes(T_world_calibrated, alignment, profile)
    source = args.task_dir / PROFILE_NAME
    report = {
        "status": "preview_only_not_accepted_for_planning",
        "task_dir": str(args.task_dir.resolve()),
        "run": str(args.run.resolve()),
        "profile_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "selected_target": args.target,
        "camera_view": args.view,
        "T_calibrated_legacy": alignment.T_calibrated_legacy.tolist(),
        "handle_center_residual_m": alignment.handle_center_residual_m,
        "door_size_max_residual_m": alignment.door_size_max_residual_m,
        "target_tcp_world": {label: pose.tolist() for label, pose in targets.items()},
        "placed_bowl_preview_world": placed_object.tolist(),
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "handle_residual_mm": round(alignment.handle_center_residual_m * 1000, 2),
        "door_size_max_residual_mm": round(alignment.door_size_max_residual_m * 1000, 2),
        "target_count": len(targets),
        "report": str(args.report.resolve()) if args.report else None,
    }, ensure_ascii=False), flush=True)
    if args.report_only:
        return
    import viser
    server = viser.ViserServer(host="127.0.0.1", port=args.port)
    try:
        MicrowaveObjectKeyposeApp(
            server, case, task_dir=args.task_dir,
            profile=profile, alignment=alignment,
            initial_target=args.target, initial_view=args.view,
        )
        print(f"几何叠图：http://127.0.0.1:{server.get_port()}", flush=True)
        if args.capture is not None:
            image = capture_viser_page(
                server, args.capture, ready_check=_target_gripper_rendered,
            )
            print(f"叠图截图：{image}", flush=True)
            return
        while True:
            time.sleep(.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()


if __name__ == "__main__":
    main()
