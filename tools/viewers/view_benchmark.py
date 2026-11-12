"""独立查看 benchmark 初始场景，不执行 policy。"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MUJOCO_GL", "egl")

from benchmarks.catalog import list_benchmarks, load_definition, identity


def prepare(benchmark_id, source, output, *, seed=123,randomize=False):
    """微波炉和长刷复用正式初始化；其他任务读取既有重建场景。"""
    definition = load_definition(benchmark_id)
    source = Path(source or ROOT / definition["scene"]["input"]).resolve()
    if benchmark_id == "microwave_task":
        from benchmarks.microwave_task.scene import resolve_settings, build_initial_scene
        output.mkdir(parents=True, exist_ok=False)
        settings = resolve_settings(json.loads(source.read_text()), seed=seed,randomize=randomize)
        model, data, metadata, placement = build_initial_scene(settings, output)
        spec = {"task": benchmark_id, "robot": metadata["robot"],
                "benchmark": identity(benchmark_id), "placement": placement}
        (output / "scene.json").write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
        return output
    if benchmark_id == "sweep_long_task":
        from benchmarks.sweep_long_task.scene import prepare_long_brush_scene
        return prepare_long_brush_scene(source, output,layout_seed=seed if randomize else None)
    if not (source / "scene.xml").is_file() or not (source / "scene.json").is_file():
        raise FileNotFoundError(f"缺少已生成的 benchmark 场景：{source}")
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=list_benchmarks(), required=True)
    parser.add_argument("--input", type=Path, help="替换默认配置或已生成的 scene 目录")
    parser.add_argument("--output", type=Path, help="新建的预览目录；默认放 runs/tmp/benchmark-preview")
    parser.add_argument("--seed", type=int, default=123, help="微波炉初态 seed，规则来自配置")
    parser.add_argument("--mode", choices=("inspect", "viser", "mujoco"), default="viser")
    parser.add_argument("--port", type=int, default=8098)
    args = parser.parse_args()
    output = (args.output or ROOT / "runs/tmp/benchmark-preview" / args.benchmark /
              datetime.now().strftime("%Y%m%d-%H%M%S-%f")).resolve()
    scene = prepare(args.benchmark, args.input, output, seed=args.seed)
    from tools.viewers.view_reconstructed_tabletop_scene import load_scene, view_viser, view_mujoco
    _, model, data = load_scene(scene)
    print(json.dumps({"benchmark": args.benchmark, "scene": str(scene),
                      "nq": model.nq, "nv": model.nv, "sim_time_s": data.time}, ensure_ascii=False), flush=True)
    if args.mode == "viser":
        view_viser(scene, port=args.port)
    elif args.mode == "mujoco":
        view_mujoco(scene)


if __name__ == "__main__":
    main()
