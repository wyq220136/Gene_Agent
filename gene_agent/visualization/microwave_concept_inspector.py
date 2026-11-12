"""把微波炉 URDF visual 与多套 Concept 统一到微波炉物体坐标系比较。

这里是只读可视化适配器。旧采集 JSON、校准 JSON、当前 Knowledge 实际
使用的几何有不同来源，必须分层显示；AABB 差值只是检查线索，不是配准误差。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import trimesh

from gene_agent.tasks.policies.microwave_task.planning.bowl import load_pick_bowl_case
from gene_agent.knowledge.geometry import CuboidConcept, CylinderConcept, load_concepts
from gene_agent.visualization.concept_comparison import ComparisonData, bounds_record, concept_to_mesh
from gene_agent.visualization.grasp_tuner.microwave_scene import _body_pose, build_microwave_knowledge
from gene_agent.visualization.pick_bowl.scene import _compiled_mesh, _geom_pose


PARTS = ("handle", "door", "cavity", "knob_0", "knob_1")
LAYERS = ("legacy_json", "calibrated_json", "runtime_effective")


AlignmentData = ComparisonData


def _visual_group(body_name: str, geom_name: str) -> str | None:
    if body_name == "microwave_door":
        return "handle" if "handle" in geom_name else "door"
    if body_name in {"knob_0", "knob_1"}:
        return body_name
    if body_name == "microwave":
        return "cabinet"
    return None


def build_alignment_data(root: str | Path, run_dir: str | Path) -> AlignmentData:
    """以编译后 FK 提取真实 visual，避免重复施加 URDF origin 或 mesh scale。"""
    root, run_dir = Path(root).resolve(), Path(run_dir).resolve()
    case = load_pick_bowl_case(run_dir)
    for joint_name in ("microwave_door_joint", "knob_0_joint", "knob_1_joint"):
        case.data.qpos[case.model.joint(joint_name).qposadr[0]] = 0.
    mujoco.mj_forward(case.model, case.data)
    microwave_from_world = np.linalg.inv(_body_pose(case.data, "microwave"))
    visual_items: dict[str, list[trimesh.Trimesh]] = {}
    for geom_id in range(case.model.ngeom):
        body = case.model.body(int(case.model.geom_bodyid[geom_id])).name
        name = mujoco.mj_id2name(case.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        group = _visual_group(body, name)
        if (group is None or int(case.model.geom_group[geom_id]) == 3
                or case.model.geom_rgba[geom_id, 3] <= 0
                or int(case.model.geom_type[geom_id]) != int(mujoco.mjtGeom.mjGEOM_MESH)):
            continue
        mesh = _compiled_mesh(case.model, geom_id)
        mesh.apply_transform(microwave_from_world @ _geom_pose(case.data, geom_id))
        visual_items.setdefault(group, []).append(mesh)
    visuals = {name: trimesh.util.concatenate(items) for name, items in visual_items.items()}
    if set(PARTS) - {"cavity"} - set(visuals):
        raise ValueError("当前场景缺少微波炉门、把手或旋钮的 URDF visual")
    # 腔体没有单独的一组 URDF visual；机身仅作为背景，不能据此计算
    # “腔体 Concept 与外壳”的伪精度中心差。
    visuals["cavity"] = visuals["cabinet"]

    layer_meshes: dict[str, dict[str, trimesh.Trimesh]] = {part: {} for part in PARTS}
    layer_poses: dict[str, dict[str, np.ndarray]] = {part: {} for part in PARTS}

    def add(layer: str, part: str, concept: Any) -> None:
        layer_meshes[part][layer] = concept_to_mesh(concept)
        layer_poses[part][layer] = np.asarray(concept.pose_in_object, dtype=float).copy()

    for item in load_concepts(root / "assets/objects/microwave_concept.json"):
        if item.part_name in PARTS:
            add("legacy_json", item.part_name, item.concept)
    for item in load_concepts(root / "assets/objects/microwave_calibrated/concept.json"):
        if item.part_name in PARTS:
            add("calibrated_json", item.part_name, item.concept)
    for item in load_concepts(root / "assets/objects/microwave_calibrated/handle_concept.json"):
        if item.part_name == "microwave_handle":
            add("calibrated_json", "handle", item.concept)

    # 运行时把手尺寸沿用旧 Trifold，但根 pose 由 URDF 把手 visual 重锚定。
    # 这里直接消费装配器的输出，不在检查器里重新发明坐标变换。
    knowledge = build_microwave_knowledge(root, case, ik_scope="selected", selected_mode="handle_grasp")
    old_handle = next(item.concept for item in load_concepts(root / "assets/objects/microwave_concept.json")
                      if item.part_name == "handle")
    from dataclasses import replace
    add("runtime_effective", "handle", replace(
        old_handle, pose_in_object=microwave_from_world @ knowledge.handle_pose_world,
    ))
    knob_data = case.metadata["objects"]["knob_concept"]
    add("runtime_effective", "knob_0", CylinderConcept(
        knob_data["size"][:2], knob_data["size"][2], knob_data["pose_matrix"],
    ))
    # 推门当前不读旧 Cuboidal_Door，而从 URDF 门板 collision 生成接触姿态。
    panel_id = case.model.geom("microwave_door_panel_collision").id
    panel_pose = microwave_from_world @ _geom_pose(case.data, panel_id)
    add("runtime_effective", "door", CuboidConcept(
        2 * np.asarray(case.model.geom_size[panel_id, :3], dtype=float), panel_pose,
    ))

    report = {
        "schema_version": 1,
        "source_run": str(run_dir),
        "scene_snapshot_sha256": case.snapshot.get("snapshot_sha256"),
        "visual_source": "MuJoCo compiled visual imported from calibrated URDF; joints at zero",
        "reference_frame": "T_microwave_part; metre",
        "legacy_warning": "旧 JSON 属于早期重建资产坐标；原样叠加只用于发现 frame 差异，不能把数值直接当成物理拟合误差。",
        "metric_warning": "AABB 中心/尺寸差不是表面距离；collision proxy 与 visual 本就允许不同。",
        "parts": {},
    }
    for part in PARTS:
        record: dict[str, Any] = {
            "visual_geom_count": len(visual_items["cabinet" if part == "cavity" else part]),
            "layers": list(layer_meshes[part]),
            "comparison_note": ("腔体只有机身背景 visual，没有一对一的腔体 visual；不计算包围盒差。"
                                if part == "cavity" else None),
        }
        for layer in LAYERS:
            mesh = layer_meshes[part].get(layer)
            record[layer] = (None if mesh is None else
                             {"source": layer,
                              **({} if part == "cavity" else bounds_record(visuals[part], mesh))})
        report["parts"][part] = record
    return AlignmentData(visuals, layer_meshes, layer_poses, report, PARTS, LAYERS,
                         context_mesh=visuals["cabinet"])


__all__ = ["AlignmentData", "PARTS", "LAYERS", "build_alignment_data"]
