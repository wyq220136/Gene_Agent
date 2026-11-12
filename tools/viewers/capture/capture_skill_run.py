"""对只读 Skill inspector 的指定阶段和时间生成可复查 Viser 截图。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from urllib.request import urlopen

import numpy as np
from PIL import Image

from gene_agent.visualization.motion_debug.capture import (
    DEFAULT_CHROME, DEFAULT_WINDOW_SIZE, _free_port, _stop, _validate_png,
)
from gene_agent.visualization.pick_bowl.capture import (
    _devtools_page, _save_cdp_screenshot, _wait_marker,
)
from gene_agent.visualization.replay import ReplayTimeline


ROOT = Path(__file__).resolve().parents[3]


def capture_selection(timeline: ReplayTimeline, *, stage_id: str | None, mode: str,
                      time_s: float, show_collision: bool):
    """截图清单必须对应界面真正显示的阶段，而不是用户请求的空碰撞阶段。"""
    timeline.select(stage_id, mode=mode)
    frame = timeline.seek(time_s)
    if show_collision and frame.collision_snapshot_ref is None:
        first = next((stage for stage in timeline.view.stages if stage.collision_snapshot_ref), None)
        if first is not None:
            stage_id, time_s = first.stage_id, 0.
            timeline.select(stage_id, mode=mode)
            frame = timeline.seek(time_s)
    return stage_id, time_s, frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--skill", choices=("pick", "insert"))
    parser.add_argument("--stage", help="省略时截图完整 Skill 时间轴")
    parser.add_argument("--mode", choices=("planned", "actual"), required=True)
    parser.add_argument("--time", type=float, default=0.)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--show-collision", action="store_true")
    parser.add_argument("--hide-visual", action="store_true", help="截图只显示规划碰撞代理")
    args = parser.parse_args()
    timeline = ReplayTimeline.load(args.run, skill=args.skill)
    selected_stage, selected_time, frame = capture_selection(
        timeline, stage_id=args.stage, mode=args.mode, time_s=args.time,
        show_collision=args.show_collision)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    port, debug_port = _free_port(), _free_port()
    while debug_port == port:
        debug_port = _free_port()
    url = f"http://127.0.0.1:{port}/"
    log_path = output / "capture.log"
    image_path = output / f"{selected_stage or 'all'}_{args.mode}_{selected_time:.3f}.png"
    command = [sys.executable, "-u", str(ROOT / "tools/viewers/view_skill_run.py"),
               "--run", str(args.run.resolve()),
               "--mode", args.mode, "--time", str(selected_time), "--port", str(port)]
    if args.skill:
        command.extend(("--skill", args.skill))
    if selected_stage:
        command.extend(("--stage", selected_stage))
    if args.show_collision:
        command.append("--show-collision")
    if args.hide_visual:
        command.append("--hide-visual")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    with log_path.open("w", encoding="utf-8") as log:
        server = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            _wait_marker(server, log_path, "[skill-inspector] URL:", 90.)
            with urlopen(url, timeout=5.) as response:
                if response.status != 200:
                    raise RuntimeError(f"Viser HTTP {response.status}")
            with tempfile.TemporaryDirectory(prefix="skill-inspector-chrome-") as profile:
                browser = subprocess.Popen([
                    str(DEFAULT_CHROME), "--headless=new", "--no-sandbox",
                    "--ignore-gpu-blocklist", "--enable-gpu", "--use-angle=gl-egl",
                    "--disable-dev-shm-usage", "--remote-allow-origins=*",
                    "--force-device-scale-factor=1",
                    f"--window-size={DEFAULT_WINDOW_SIZE[0]},{DEFAULT_WINDOW_SIZE[1]}",
                    f"--user-data-dir={profile}", f"--remote-debugging-port={debug_port}", url,
                ], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                try:
                    websocket_url = _devtools_page(debug_port, 20.)
                    _wait_marker(browser, log_path, "[skill-inspector] scene-ready", 45.)
                    # WebGL 的纹理和 mesh 上传可能比 ready 标记晚；连续取样，
                    # 选场景区域中着色资产最多的一帧，避免把未加载完的白色模型交付。
                    scores = []
                    for index in range(4):
                        time.sleep(.8)
                        candidate = output / f".capture_{index}.png"
                        _save_cdp_screenshot(websocket_url, candidate)
                        rgb = np.asarray(Image.open(candidate).convert("RGB"))[:, :1030].astype(float)
                        saturation = (rgb.max(axis=2) - rgb.min(axis=2)) / (rgb.max(axis=2) + 1e-9)
                        scores.append((int((saturation > .15).sum()), candidate))
                    best_score, best = max(scores, key=lambda item: item[0])
                    best.replace(image_path)
                    for _, candidate in scores:
                        candidate.unlink(missing_ok=True)
                    print(f"[skill-inspector] screenshot-color-score={best_score}", flush=True)
                finally:
                    _stop(browser)
            _validate_png(image_path, DEFAULT_WINDOW_SIZE)
        finally:
            _stop(server)
    collision = (json.loads((args.run.resolve() / frame.collision_snapshot_ref).read_text())
                 if frame.collision_snapshot_ref else {})
    manifest = {
        "selection": selected_stage or "all", "requested_selection": args.stage or "all",
        "stage_id": frame.stage_id,
        "mode": args.mode, "time_s": selected_time,
        "visual_visible": not args.hide_visual, "collision_visible": args.show_collision,
        "sample_time_s": frame.sample_time_s, "sim_time_s": frame.sim_time_s,
        "state_id": frame.state_id, "scene_revision": frame.scene_revision,
        "sample_quality": frame.sample_quality,
        "qpos_sha256": (hashlib.sha256(json.dumps(frame.qpos).encode()).hexdigest()
                        if frame.qpos is not None else None),
        "object_poses_world": frame.object_poses_world,
        "attachment_object_id": (frame.attachment or {}).get("object_instance_id"),
        "collision_snapshot_ref": frame.collision_snapshot_ref,
        "collision_snapshot_sha256": collision.get("snapshot_sha256"),
        "png": image_path.name, "sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
    }
    (output / "capture_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(f"PASS {image_path}")


if __name__ == "__main__":
    main()
