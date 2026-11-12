#!/usr/bin/env python3
"""按物体名或 mesh 路径启动 Concept 标注器。"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from gene_agent.visualization.concept_editor.app import run_editor
from gene_agent.visualization.concept_editor.catalog import discover_concepts, resolve_object


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--object", help="已标注物体名或运行目录名；自动找 mesh 和 Concept")
    source.add_argument("--mesh", type=Path, help="直接指定新 mesh，单位米")
    parser.add_argument("--list", action="store_true", help="列出可直接打开的已标注物体")
    parser.add_argument("--concept", type=Path, help="可选；覆盖自动找到的 Concept")
    parser.add_argument("--output", type=Path, help="可选；默认另存 concept.tuned.json")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8125)
    args = parser.parse_args()
    entries = discover_concepts(ROOT) if args.list or args.mesh is None else []
    if args.list:
        for index, entry in enumerate(entries, 1):
            print(f"{index:2d}. {entry.category:<18} {entry.folder.relative_to(ROOT)}")
        return
    if args.mesh is not None:
        mesh, concept, output = args.mesh, args.concept, args.output
    else:
        if not entries:
            parser.error("没有找到已标注物体；请使用 --mesh 指定参考网格")
        if args.object:
            try:
                selected = resolve_object(entries, args.object)
            except ValueError as exc:
                parser.error(str(exc))
        else:
            if not sys.stdin.isatty():
                parser.error("非交互终端请用 --object 物体名或 --mesh 路径")
            for index, entry in enumerate(entries, 1):
                print(f"{index:2d}. {entry.category:<18} {entry.folder.relative_to(ROOT)}")
            try:
                index = int(input("选择编号：").strip())
                selected = entries[index - 1] if index > 0 else None
            except (ValueError, IndexError, EOFError):
                selected = None
            if selected is None:
                parser.error("请选择列表中的有效编号")
        mesh = selected.mesh
        concept = args.concept or selected.input_concept
        output = args.output or selected.output_concept
        print(f"已选择 {selected.category}: {selected.folder.relative_to(ROOT)}")
    run_editor(mesh_path=mesh, concept_path=concept, output_path=output,
               host=args.host, port=args.port)


if __name__ == "__main__":
    main()
