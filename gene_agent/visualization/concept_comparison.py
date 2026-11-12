"""任意 typed Concept 与参考 mesh 的只读对照数据。

调用方负责把参考 mesh 和 Concept 放到同一个 object frame。本模块不依赖
MuJoCo、机器人或具体任务；只有明确的一对一部件对应关系才计算 AABB 差。
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from gene_agent.knowledge.geometry import (
    Concept, ContainerConcept, TrifoldConcept, load_concepts,
)
from gene_agent.visualization.concept_editor.model import concept_mesh


@dataclass(frozen=True)
class ComparisonData:
    """所有几何都已在相同 object frame 中，可直接交给 Viser。"""

    visual_meshes: dict[str, trimesh.Trimesh]
    layer_meshes: dict[str, dict[str, trimesh.Trimesh]]
    layer_poses: dict[str, dict[str, np.ndarray]]
    report: dict[str, Any]
    parts: tuple[str, ...]
    layers: tuple[str, ...]
    context_mesh: trimesh.Trimesh | None = None


def concept_to_mesh(concept: Concept) -> trimesh.Trimesh:
    """复用编辑器的基础几何，补齐三折结构和容器的组合几何。"""
    if isinstance(concept, TrifoldConcept):
        pieces = (concept.grip_segment, concept.positive_connector,
                  concept.negative_connector)
        meshes = []
        for piece in pieces:
            mesh = trimesh.creation.box(extents=piece.size_xyz_m)
            mesh.apply_transform(concept.pose_in_object @ piece.pose_in_trifold)
            meshes.append(mesh)
        return trimesh.util.concatenate(meshes)
    if isinstance(concept, ContainerConcept):
        meshes = []
        for wall in concept.walls:
            mesh = trimesh.creation.box(extents=wall.size_xyz_m)
            mesh.apply_transform(concept.pose_in_object @ wall.pose_in_container)
            meshes.append(mesh)
        return trimesh.util.concatenate(meshes)
    simple = concept_mesh(concept)
    return trimesh.Trimesh(vertices=simple.vertices, faces=simple.faces, process=False)


def bounds_record(reference: trimesh.Trimesh, concept: trimesh.Trimesh) -> dict[str, Any]:
    """只给有一对一对应的部件计算 AABB；它不是表面配准误差。"""
    rb, cb = np.asarray(reference.bounds), np.asarray(concept.bounds)
    rc, cc = rb.mean(axis=0), cb.mean(axis=0)
    return {
        "visual_center_object_m": rc.tolist(),
        "concept_center_object_m": cc.tolist(),
        "center_delta_object_mm": ((cc - rc) * 1000).tolist(),
        "center_distance_mm": float(np.linalg.norm(cc - rc) * 1000),
        "visual_aabb_size_mm": ((rb[1] - rb[0]) * 1000).tolist(),
        "concept_aabb_size_mm": ((cb[1] - cb[0]) * 1000).tolist(),
        "aabb_size_delta_mm": (((cb[1] - cb[0]) - (rb[1] - rb[0])) * 1000).tolist(),
    }


def build_single_mesh_comparison(mesh_path: str | Path,
                                 concept_path: str | Path) -> ComparisonData:
    """给任意 object mesh + Concept JSON 建一个可检查的对照。

    同一个完整物体 mesh 会作为各 part 的共同背景。只有一个 part 时也不
    默认认为它覆盖整个物体，因此不会报告伪精度的部件 AABB 误差。
    """
    source = Path(mesh_path).expanduser().resolve()
    loaded = trimesh.load_mesh(source, process=False)
    if isinstance(loaded, trimesh.Scene):
        # dump 会应用 Scene graph 中的节点变换；直接拼 geometry 会丢失它们。
        loaded = loaded.dump(concatenate=True)
    if not isinstance(loaded, trimesh.Trimesh) or loaded.is_empty:
        raise ValueError(f"参考 mesh 不是非空三角网格：{source}")
    concept_source = Path(concept_path).expanduser().resolve()
    parts = load_concepts(concept_source)
    if not parts:
        raise ValueError(f"Concept 文件没有部件：{concept_path}")
    visuals = {part.part_name: loaded for part in parts}
    layers = {part.part_name: {"concept": concept_to_mesh(part.concept)} for part in parts}
    poses = {part.part_name: {"concept": np.asarray(part.concept.pose_in_object).copy()}
             for part in parts}
    note = "参考 mesh 是整个物体，未提供一对一的部件 mesh；仅做视觉叠加。"
    report = {
        "schema_version": 1, "reference_frame": "object frame; metre",
        "visual_source": str(source), "concept_source": str(concept_source),
        "visual_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "concept_sha256": hashlib.sha256(concept_source.read_bytes()).hexdigest(),
        "metric_warning": "只有一对一部件参考 mesh 才计算 AABB；AABB 也不是表面配准误差。",
        "parts": {part.part_name: {"layers": ["concept"], "comparison_note": note,
                                  "concept": {"source": str(part.source_path)}} for part in parts},
    }
    return ComparisonData(visuals, layers, poses, report,
                          tuple(part.part_name for part in parts), ("concept",))
