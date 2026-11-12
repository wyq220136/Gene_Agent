#!/usr/bin/env python3
"""离线生成参考 mesh 与功能性 Concept 的三个正交叠图，供 Agent 快速复核。"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import trimesh

from gene_agent.knowledge.geometry import load_concepts
from gene_agent.visualization.concept_comparison import concept_to_mesh


def render(mesh_path: Path, concept_path: Path | None, output: Path) -> None:
    reference = trimesh.load_mesh(mesh_path, process=False)
    if isinstance(reference, trimesh.Scene):
        reference = reference.dump(concatenate=True)
    parts = load_concepts(concept_path) if concept_path is not None else ()
    if concept_path is not None and not parts:
        raise ValueError("Concept 至少需要一个 part")
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    palette = ("#e04b40", "#168bd2", "#46a34a", "#a353b3")
    for ax, (horizontal, vertical) in zip(axes, ((0, 1), (2, 1), (0, 2))):
        vertices = np.asarray(reference.vertices)
        ax.scatter(vertices[:, horizontal] * 1000, vertices[:, vertical] * 1000,
                   s=0.45, c="#666666", alpha=0.16, rasterized=True)
        for index, part in enumerate(parts):
            primitive = concept_to_mesh(part.concept)
            points = np.asarray(primitive.vertices)
            color = palette[index % len(palette)]
            ax.scatter(points[:, horizontal] * 1000, points[:, vertical] * 1000,
                       s=8, c=color, alpha=0.8,
                       label=part.part_name)
            if part.concept.__class__.__name__ == "CuboidConcept":
                edge_indices = np.asarray(primitive.edges_unique)
                local = (points - part.concept.pose_in_object[:3, 3]) @ part.concept.pose_in_object[:3, :3]
                axis_changes = np.count_nonzero(
                    np.abs(local[edge_indices[:, 0]] - local[edge_indices[:, 1]]) > 1e-7,
                    axis=1,
                )
                edges = points[edge_indices[axis_changes == 1]][:, :, [horizontal, vertical]] * 1000
                ax.add_collection(LineCollection(edges, colors=color, linewidths=1.5, alpha=0.8))
        ax.set(xlabel=f"{'XYZ'[horizontal]} (mm)", ylabel=f"{'XYZ'[vertical]} (mm)")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.25)
        if parts:
            ax.legend(loc="upper right")
    fig.suptitle("Reference mesh (gray) / Concept primitives (color), object frame"
                 if parts else "Reference mesh in object frame")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(output.resolve())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--concept", type=Path, help="省略时只绘参考 mesh")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render(args.mesh, args.concept, args.output)


if __name__ == "__main__":
    main()
