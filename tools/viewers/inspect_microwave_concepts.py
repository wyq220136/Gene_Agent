#!/usr/bin/env python3
"""微波炉场景数据适配器；通用界面见 inspect_concepts.py。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gene_agent.visualization.concept_comparison_app import ConceptComparisonApp
from gene_agent.visualization.microwave_concept_inspector import (
    LAYERS, PARTS, build_alignment_data,
)
from tools.viewers.inspect_concepts import serve


ROOT = Path(__file__).resolve().parents[2]
LAYER_LABELS = {
    "legacy_json": "旧 JSON（早期坐标）",
    "calibrated_json": "校准 JSON",
    "runtime_effective": "当前运行时使用",
}


class MicrowaveConceptInspector(ConceptComparisonApp):
    """保留旧调用入口，内部直接复用通用 Concept 界面。"""

    def __init__(self, server, data, *, initial_part="handle"):
        super().__init__(server, data, initial_part=initial_part,
                         title="微波炉 Concept ↔ URDF visual", layer_labels=LAYER_LABELS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "runs/pick-bowl-planning-final")
    parser.add_argument("--part", choices=PARTS, default="handle")
    parser.add_argument("--port", type=int, default=8084)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--capture-dir", type=Path)
    parser.add_argument("--inspect-only", action="store_true")
    args = parser.parse_args()
    data = build_alignment_data(ROOT, args.run)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(data.report, ensure_ascii=False, indent=2) + "\n",
                               encoding="utf-8")
        print(f"对齐报告：{args.report}", flush=True)
    for part in PARTS:
        metrics = data.report["parts"][part]
        summary = {layer: (None if metrics[layer] is None else
                           metrics[layer].get("center_distance_mm")) for layer in LAYERS}
        print(f"{part}: AABB 中心差(mm) {summary}", flush=True)
    if not args.inspect_only:
        serve(data, port=args.port, part=args.part, capture_dir=args.capture_dir,
              title="微波炉 Concept ↔ URDF visual", layer_labels=LAYER_LABELS)


if __name__ == "__main__":
    main()
