"""从已保存的抓碗阶段状态重新渲染诊断图，无需重跑 cuRobo。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import mujoco

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from gene_agent.feedback.state import restore_state, state_id
from gene_agent.visualization import render_views


def render_saved_stage(run: Path, stage: str, output: Path, frames_path: Path | None = None):
    run = run.resolve()
    stage_dir = run / "stages" / stage
    model = mujoco.MjModel.from_xml_path(str(run / "model" / "scene.xml"))
    data = mujoco.MjData(model)
    restore_state(model, data, stage_dir / "state.npz")
    diagnosis = json.loads((frames_path or stage_dir / "frames.json").read_text())
    if diagnosis["state_id"] != state_id(model, data):
        raise ValueError("frames.json 与阶段 state.npz 的 state_id 不一致")
    return render_views(
        model, data, output, target_body="bowl", width=640, height=480,
        view_names=("overview", "contact", "contact_side"), collision_overlay=True,
        contact_debug=True,
        debug_frames={
            name: value["T_world_frame"] for name, value in diagnosis["frames"].items()
        },
        frame_diagnostics=diagnosis,
        frame_axis_length_m=.08,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--stage", default="pregrasp")
    parser.add_argument("--frames", type=Path, help="可选的同一状态目标坐标诊断 JSON")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = render_saved_stage(args.run, args.stage, args.output, args.frames)
    print(json.dumps({"state_id": result["state_id"], "images": str(args.output.resolve())}))
