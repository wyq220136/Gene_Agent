#!/usr/bin/env python3
"""从 my_scene_0 的有效重建 mesh 生成清扫、倾倒、舀取场景预览。"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.tabletop import (
    SCENE_VARIANTS, build_multi_task_scenes, sample_scene_variants)
from tools.viewers.view_reconstructed_tabletop_scene import render_views


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path,
                        default=Path("data/realworld/my_scene_0"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("data/realworld/my_scene_0/scenes"))
    parser.add_argument("--seeds", type=int, nargs="*", default=[0, 7, 21],
                        help="每个场景生成的初态分布 seed")
    parser.add_argument("--tasks", nargs="+", choices=SCENE_VARIANTS,
                        help="只重新生成指定任务，保留其他任务的现有场景")
    parser.add_argument("--skip-render", action="store_true")
    args = parser.parse_args()
    scenes = build_multi_task_scenes(
        args.capture_dir, args.output_dir,
        tasks=tuple(args.tasks) if args.tasks else None)
    for task, scene in scenes.items():
        print(f"{task}: {scene['scene_xml_path']} ({scene['status']})")
        scene_dir = args.output_dir.resolve() / task
        samples = sample_scene_variants(scene_dir, tuple(args.seeds))
        if not args.skip_render:
            render_views(scene_dir, scene_dir / "renders")
            for sample in samples:
                render_views(sample, sample / "renders")


if __name__ == "__main__":
    main()
