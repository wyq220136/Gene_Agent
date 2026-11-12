#!/usr/bin/env python3
"""检查任意物体 mesh 与 typed Concept JSON 的几何叠加。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
from typing import Any

from gene_agent.visualization.concept_comparison import ComparisonData, build_single_mesh_comparison
from gene_agent.visualization.concept_comparison_app import ConceptComparisonApp


ROOT = Path(__file__).resolve().parents[2]


def capture_parts(server: Any, app: ConceptComparisonApp, output: Path) -> None:
    """对每个部件固定视角截图，同时记录来源、视角和图层。"""
    from gene_agent.visualization.motion_debug.capture import (
        DEFAULT_CHROME, DEFAULT_WINDOW_SIZE, _free_port, _stop, _validate_png,
    )
    from gene_agent.visualization.pick_bowl.capture import _devtools_page, _save_cdp_screenshot
    from tools.viewers.capture.capture_knowledge_viser import _wait_ui

    output.mkdir(parents=True, exist_ok=True)
    browser = None
    try:
        with tempfile.TemporaryDirectory(prefix="gene-concept-chrome-") as profile, \
                (output / "browser.log").open("w", encoding="utf-8") as log:
            chrome_port = _free_port()
            command = [str(DEFAULT_CHROME), "--headless=new", "--no-sandbox",
                       "--ignore-gpu-blocklist", "--enable-gpu", "--use-angle=gl-egl",
                       "--disable-dev-shm-usage", "--remote-allow-origins=*",
                       "--force-device-scale-factor=1",
                       f"--window-size={DEFAULT_WINDOW_SIZE[0]},{DEFAULT_WINDOW_SIZE[1]}",
                       f"--user-data-dir={profile}", f"--remote-debugging-port={chrome_port}",
                       f"http://127.0.0.1:{server.get_port()}/"]
            browser = subprocess.Popen(command, cwd=ROOT, stdout=log,
                                       stderr=subprocess.STDOUT, env=os.environ.copy())
            socket_url = _devtools_page(chrome_port, 25.)
            _wait_ui(socket_url, app.title)
            time.sleep(2.)
            records = {}
            for index, part in enumerate(app.data.parts):
                app.part.value = part
                app._refresh(focus=True)
                for client in server.get_clients().values():
                    client.flush()
                    client.get_render(height=8, width=8, transport_format="png", timeout=10.)
                image = output / f"part_{index:02d}.png"
                _save_cdp_screenshot(socket_url, image)
                _validate_png(image, DEFAULT_WINDOW_SIZE)
                records[part] = {"image": image.name,
                                 "sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
                                 "camera": app.camera,
                                 "layers": list(app.data.layer_meshes[part])}
                print(f"截图 {part}: {image}", flush=True)
            (output / "capture_manifest.json").write_text(json.dumps({
                "schema_version": 1, "source": app.data.report.get("visual_source"),
                "concept_source": app.data.report.get("concept_source"),
                "visual_sha256": app.data.report.get("visual_sha256"),
                "concept_sha256": app.data.report.get("concept_sha256"),
                "scene_snapshot_sha256": app.data.report.get("scene_snapshot_sha256"),
                "views": records,
            }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            _stop(browser)
            browser = None
    finally:
        if browser is not None:
            _stop(browser)


def serve(data: ComparisonData, *, port: int, part: str | None,
          capture_dir: Path | None = None, title: str = "Concept ↔ 参考 mesh",
          layer_labels: dict[str, str] | None = None) -> None:
    import viser

    server = viser.ViserServer(host="127.0.0.1", port=port)
    app = ConceptComparisonApp(server, data, initial_part=part, title=title,
                               layer_labels=layer_labels)
    print(f"{title}：http://127.0.0.1:{server.get_port()}", flush=True)
    if capture_dir is not None:
        try:
            capture_parts(server, app, capture_dir)
        finally:
            server.stop()
        return
    try:
        while True:
            time.sleep(.5)
    except KeyboardInterrupt:
        server.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, required=True, help="object frame 中的 OBJ/STL 等三角 mesh")
    parser.add_argument("--concept", type=Path, required=True, help="同一 object frame 的 Concept JSON")
    parser.add_argument("--part", help="初始检查部件；默认第一个")
    parser.add_argument("--port", type=int, default=8084)
    parser.add_argument("--report", type=Path, help="保存只读检查报告")
    parser.add_argument("--capture-dir", type=Path, help="每个 Concept 部件自动截图后退出")
    parser.add_argument("--inspect-only", action="store_true")
    args = parser.parse_args()
    data = build_single_mesh_comparison(args.mesh, args.concept)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(data.report, ensure_ascii=False, indent=2) + "\n",
                               encoding="utf-8")
        print(f"检查报告：{args.report}", flush=True)
    print(f"Concept 部件：{', '.join(data.parts)}", flush=True)
    if not args.inspect_only:
        serve(data, port=args.port, part=args.part, capture_dir=args.capture_dir)


if __name__ == "__main__":
    main()
