"""把竖立长刷场景接入同步清扫 Task，保留来源场景和显式映射。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation



def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_long_brush_scene(source_dir: Path, run_dir: Path,
                             *, layout_shift_x_m: float = -.04,
                             aligned_lane: bool = True,
                             head_proxy_half_width_y_m: float = .0861275891,
                             layout_seed: int | None = None,
                             layout_distribution: str = "wide") -> Path:
    """仅为运行生成标准对象名，物理几何、关节 pose 与支架原样保留。"""
    source_dir = Path(source_dir).resolve()
    run_dir = Path(run_dir).resolve()
    if not np.isfinite([layout_shift_x_m, head_proxy_half_width_y_m]).all():
        raise ValueError("场景调整参数必须是有限数")
    if not .05 <= head_proxy_half_width_y_m <= .0861275891:
        raise ValueError("刷头碰撞代理半宽必须在 0.05–0.0861275891 m 内")
    if layout_seed is not None and not aligned_lane:
        raise ValueError("方块扰动仅适用于 X 向通道")
    if layout_distribution not in ("small", "wide", "stress"):
        raise ValueError("方块摆位分布必须是 small、wide 或 stress")
    if run_dir.exists():
        raise FileExistsError(f"运行目录已存在: {run_dir}")
    adapter = run_dir / "task_scene"
    (adapter / "concepts").mkdir(parents=True)
    (adapter / "robot").mkdir()
    source_xml = source_dir / "scene.xml"
    source_json = source_dir / "scene.json"
    originals = json.loads(source_json.read_text())
    metadata_source = source_dir / originals["robot"]["metadata_path"]
    if not metadata_source.is_file():
        metadata_source = (Path(__file__).resolve().parents[2] /
            "data/realworld/brush_cube_dustpan_0/robot/robot_metadata.json")
    concept_source = Path(originals["concept_paths"]["brush_long"])
    tree = ET.parse(source_xml)
    rename = {
        "brush_long": "brush", "brush_long_free": "brush_free",
        "brush_long_visual": "brush_visual",
        "brush_long_head_proxy": "brush_collision_head",
        "brush_long_handle_proxy": "brush_collision_handle",
        "sim_attach_left_brush_long": "sim_attach_left_brush",
    }
    for element in tree.getroot().iter():
        for key in ("name", "body1", "body2", "joint"):
            if element.get(key) in rename:
                element.set(key, rename[element.get(key)])
        if element.get("file"):
            element.set("file", str((source_xml.parent / element.get("file")).resolve()))
        if element.get("name") == "brush_collision_head":
            # 默认保留来源碰撞尺寸；诊断时可显式缩小运行副本。
            size = [float(value) for value in element.get("size").split()]
            size[1] = head_proxy_half_width_y_m
            element.set("size", " ".join(format(value, ".12g") for value in size))
    layout_positions = {}
    perturbations = {}
    sampling_rejections = 0
    if layout_shift_x_m or aligned_lane:
        # 支架和刷子留在原位；方块排在簸箕开口前、夹爪可达的窄通道。
        source_model = mujoco.MjModel.from_xml_path(str(source_xml))
        key = tree.getroot().find("./keyframe/key")
        qpos = [float(value) for value in key.get("qpos").split()]
        pan_x = .52 + layout_shift_x_m
        lane = {"cube": (pan_x - .18, .10),
                "cube_red": (pan_x - .14, .10),
                "cube_blue": (pan_x - .10, .10),
                "dustpan": (pan_x, .10)}
        if layout_seed is not None:
            rng = np.random.default_rng(layout_seed)
            placed_xy = []
            for name in ("cube", "cube_red", "cube_blue"):
                if layout_distribution in ("wide", "stress"):
                    # 只因初态重叠而重采样；困难但有效的场景仍进入正式评测。
                    y_low, y_high = ((.075, .125) if layout_distribution == "wide"
                                     else (.06, .14))
                    for _ in range(100):
                        x, y = rng.uniform([.285, y_low], [.375, y_high])
                        if all(np.linalg.norm(np.array([x, y]) - prior) >= .03
                               for prior in placed_xy):
                            break
                        sampling_rejections += 1
                    else:
                        raise ValueError(f"seed={layout_seed} 无法采到非重叠方块初态")
                    yaw = float(rng.uniform(-60., 60.))
                    placed_xy.append(np.array([x, y]))
                    dx, dy = float(x - lane[name][0]), float(y - lane[name][1])
                    lane[name] = (float(x), float(y))
                else:
                    dx, dy = rng.uniform([-.005, -.010], [.005, .010])
                    yaw = float(rng.uniform(-10., 10.))
                    lane[name] = (lane[name][0] + float(dx), lane[name][1] + float(dy))
                perturbations[name] = {"delta_world_xy_m": [float(dx), float(dy)],
                                       "yaw_delta_deg": yaw}
        for body in tree.getroot().iter("body"):
            if body.get("name") not in ("cube", "cube_red", "cube_blue", "dustpan"):
                continue
            name = body.get("name")
            xyz = [float(value) for value in body.get("pos").split()]
            xyz[0] += layout_shift_x_m
            if aligned_lane and name in lane:
                xyz[:2] = lane[name]
            body.set("pos", " ".join(format(value, ".12g") for value in xyz))
            if aligned_lane and name == "dustpan":
                # 局部 +X 是铲口外法向；旋到世界 -X 面向方块。
                body.set("quat", "0 0 0 -1")
            layout_positions[name] = xyz
            if name != "dustpan":
                joint = source_model.joint(f"{name}_free")
                qpos[int(joint.qposadr[0]):int(joint.qposadr[0]) + 2] = xyz[:2]
                if name in perturbations:
                    address = int(joint.qposadr[0])
                    old = np.asarray(qpos[address + 3:address + 7])
                    rotation = (Rotation.from_euler("z", perturbations[name]["yaw_delta_deg"], degrees=True)
                                * Rotation.from_quat(old[[1, 2, 3, 0]]))
                    xyzw = rotation.as_quat()
                    quat = xyzw[[3, 0, 1, 2]]
                    qpos[address + 3:address + 7] = quat.tolist()
                    body.set("quat", " ".join(format(value, ".12g") for value in quat))
        key.set("qpos", " ".join(format(value, ".12g") for value in qpos))
    tree.write(adapter / "scene.xml", encoding="unicode")

    concept = json.loads(concept_source.read_text())
    concept["category"] = "brush"
    concept["source_concept_path"] = str(concept_source)
    for part in concept["conceptualization"]:
        if part["component"] == "grasp_handle":
            part["component"] = "grip_handle"
        elif part["component"] == "sweeping_face":
            part["component"] = "bristle_face"
            # 与场景的真实 box 碰撞体一致，避免原标注低估刷头长度。
            part["parameters"]["size"] = [.1302191616,
                2. * head_proxy_half_width_y_m, .0227431934]
            part["parameters"]["pose_matrix"][0][3] = .06782248
            part["parameters"]["pose_matrix"][2][3] = .0117384224
    (adapter / "concepts/brush.json").write_text(
        json.dumps(concept, ensure_ascii=False, indent=2) + "\n")
    for name in ("cube", "dustpan"):
        shutil.copy2(Path(originals["concept_paths"][name]), adapter / "concepts" / f"{name}.json")
    shutil.copy2(metadata_source, adapter / "robot/robot_metadata.json")
    objects = {}
    for name in ("brush", "cube", "cube_red", "cube_blue", "dustpan"):
        source_name = "brush_long" if name == "brush" else name
        concept_name = "cube" if name.startswith("cube") else name
        pose = originals["placement_object_poses"][source_name]["T_world_object"]
        if name in layout_positions:
            pose = [row.copy() for row in pose]
            pose[0][3], pose[1][3] = layout_positions[name][:2]
            if aligned_lane and name == "dustpan":
                pose[0][:2], pose[1][:2] = [-1., 0.], [0., -1.]
            if name in perturbations:
                matrix = np.asarray(pose, dtype=float)
                matrix[:3, :3] = (Rotation.from_euler("z", perturbations[name]["yaw_delta_deg"],
                                                   degrees=True).as_matrix() @ matrix[:3, :3])
                pose = matrix.tolist()
        objects[name] = {
            "T_world_object": pose,
            "concept_path": f"concepts/{concept_name}.json",
            "source_instance_name": source_name,
        }
    scene = {
        "schema_version": 1, "task": "sweep_long",
        "world_frame": originals["world_frame"],
        "placement_mode": "fixed_scene_keyframe",
        "layout_seed": layout_seed,
        "layout_distribution": layout_distribution if layout_seed is not None else "nominal",
        "simulation_attachment_pairs": {"left": "brush"},
        "objects": objects,
        "source_scene_xml_path": str(source_xml),
    }
    (adapter / "scene.json").write_text(json.dumps(scene, ensure_ascii=False, indent=2) + "\n")
    manifest = {
        "source_scene_xml_path": str(source_xml), "source_scene_xml_sha256": _sha256(source_xml),
        "source_scene_json_path": str(source_json), "source_scene_json_sha256": _sha256(source_json),
        "source_brush_concept_path": str(concept_source),
        "source_brush_concept_sha256": _sha256(concept_source),
        "instance_mapping": {"brush_long": "brush"},
        "part_mapping": {"grasp_handle": "grip_handle", "sweeping_face": "bristle_face"},
        "head_proxy_used_for_functional_concept": True,
        "head_collision_proxy_half_width_y_m": head_proxy_half_width_y_m,
        "head_collision_proxy_source_half_width_y_m": .0861275891,
        "layout_adjustment": {"moved_objects": list(layout_positions),
                              "nominal_positions_world_m": layout_positions,
                              "world_translation_x_m": layout_shift_x_m,
                              "aligned_lane": aligned_lane,
                              "distribution": ({"family": layout_distribution,
                                                "cube_world_x_m": "uniform[.285,.375]",
                                                "cube_world_y_m": ("uniform[.075,.125]"
                                                                   if layout_distribution == "wide"
                                                                   else "uniform[.06,.14]"),
                                                "cube_yaw_delta_deg": "uniform[-60,60]",
                                                "min_center_distance_m": .03}
                                               if layout_distribution in ("wide", "stress") else
                                               {"family": "small",
                                                "cube_dx_m": "uniform[-.005,.005]",
                                                "cube_dy_m": "uniform[-.010,.010]",
                                                "cube_yaw_delta_deg": "uniform[-10,10]"})
                              if layout_seed is not None else None,
                              "seed": layout_seed, "sampled_perturbations": perturbations,
                              "overlap_resample_count": sampling_rejections,
                              "reason": "move_sweep_path_inside_curobo_reachable_workspace"},
    }
    (adapter / "source_mapping.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return adapter


