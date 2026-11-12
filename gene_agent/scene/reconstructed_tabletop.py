"""把逐物体重建资产放到单臂 Pika 桌面场景；与感知相机外参解耦。"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import trimesh
from scipy.optimize import least_squares

from gene_agent.robots import build_robot
from gene_agent.scene.tabletop_randomization import sample_tabletop_layout
from gene_agent.visualization.style import apply_scene_style


SOURCE_ORDER = ("brush", "cube", "dustpan")
BLOCKS = ("cube", "cube_red", "cube_blue")
ORDER = ("brush", *BLOCKS, "dustpan")
LAYOUT_XY_YAW = {"brush": (.38, -.26, 0.),
                 "cube": (.32, -.09, -np.pi / 9),
                 "cube_red": (.43, -.12, np.pi / 6),
                 "cube_blue": (.38, -.02, -np.pi / 5),
                 "dustpan": (.38, .12, -np.pi / 2)}
LAYOUT_Y_M = {name: values[1] for name, values in LAYOUT_XY_YAW.items()}
COLORS = {"brush": "0.27 0.51 0.76 1", "cube": "0.13 0.72 0.22 1",
          "cube_red": "0.84 0.12 0.12 1", "cube_blue": "0.12 0.27 0.84 1",
          "dustpan": "0.84 0.78 0.66 1"}
RANDOMIZATION_RANGES_XYYAW = {
    # 经 2026-10-17 布局预览确认的 B 档；三个方块可偏离簸箕中心轴。
    "brush": (.060, .055, np.deg2rad(35.)),
    "cube": (.090, .070, np.deg2rad(90.)),
    "cube_red": (.090, .070, np.deg2rad(90.)),
    "cube_blue": (.090, .070, np.deg2rad(90.)),
    # 真实清扫任务里簸箕可视为固定；通用采样器仍允许配置它的范围。
    "dustpan": (0., 0., 0.),
}
# 纯位置 IK 选出的桌面待机姿态：TCP 靠近桌面中央上方，未作为抓取姿态验收。
STANDBY_JOINTS_RAD = (0.0, -0.56916, 0.0, 0.95157, 0.0, 1.77422, 0.0)
DUSTPAN_YAW_RAD = -np.pi / 2


def _numbers(values) -> str:
    return " ".join(format(float(value), ".9g") for value in values)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonicalize_mesh(source: str | Path, destination: str | Path) -> dict:
    """最长轴朝机器人前方 X，最薄轴朝桌面法向 Z，底点落在局部 z=0。

    SAM3D 的 mesh 局部轴不携带桌面语义；此变换是明确的宽松布局选择，
    不应当作相机估计的姿态。返回矩阵把源 mesh 顶点映射到新物体 frame。
    """
    source, destination = Path(source).resolve(), Path(destination).resolve()
    mesh = trimesh.load_mesh(source, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.vertices) < 4:
        raise ValueError(f"mesh 无有效顶点: {source}")
    if not np.isfinite(mesh.vertices).all() or np.any(mesh.extents < 0.003):
        raise ValueError(f"mesh 范围无效或过薄: {source}")
    order = np.argsort(mesh.extents)
    long_axis, middle_axis, thin_axis = int(order[2]), int(order[1]), int(order[0])
    rotation = np.zeros((3, 3), dtype=float)
    rotation[0, long_axis] = 1.0
    rotation[1, middle_axis] = 1.0
    rotation[2, thin_axis] = 1.0
    if np.linalg.det(rotation) < 0:
        rotation[1] *= -1.0
    vertices = np.asarray(mesh.vertices) @ rotation.T
    offset = np.array([-(vertices[:, 0].min() + vertices[:, 0].max()) / 2,
                       -(vertices[:, 1].min() + vertices[:, 1].max()) / 2,
                       -vertices[:, 2].min()])
    mesh.vertices = vertices + offset
    destination.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(destination)
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = offset
    return {"mesh_path": str(destination), "T_canonical_source": transform.tolist(),
            "extents_xyz_m": np.asarray(mesh.extents).tolist(),
            "source_sha256": _sha256(source), "canonical_sha256": _sha256(destination)}


def place_brush_bristles_down(canonical: dict) -> dict:
    """此刷子刷毛位于规范 mesh 的 +Y 侧；绕手柄 X 轴 -90° 使其朝下。"""
    path = Path(canonical["mesh_path"])
    mesh = trimesh.load_mesh(path, force="mesh", process=False)
    rotation = np.array([[1., 0., 0.], [0., 0., 1.], [0., -1., 0.]])
    vertices = np.asarray(mesh.vertices) @ rotation.T
    offset = np.array([0., 0., -vertices[:, 2].min()])
    mesh.vertices = vertices + offset
    mesh.export(path)
    task_transform = np.eye(4)
    task_transform[:3, :3] = rotation
    task_transform[:3, 3] = offset
    canonical["T_canonical_source"] = (
        task_transform @ np.asarray(canonical["T_canonical_source"])).tolist()
    canonical["extents_xyz_m"] = np.asarray(mesh.extents).tolist()
    canonical["canonical_sha256"] = _sha256(path)
    return canonical


def _build_brush_collision_parts(mesh_path: Path, output_dir: Path) -> list[dict]:
    """把刷子的米制 object-frame mesh 分成少量凸块，供 MuJoCo 接触使用。"""
    mesh = trimesh.load_mesh(mesh_path, force="mesh", process=False)
    try:
        decomposed = trimesh.decomposition.convex_decomposition(
            mesh, maxConvexHulls=4, resolution=100000,
            maxNumVerticesPerCH=48, asyncACD=False,
        )
    except ImportError as exc:
        raise RuntimeError("生成刷子碰撞体需要 curobo 环境中的 vhacdx") from exc
    if not isinstance(decomposed, list) or not 1 <= len(decomposed) <= 4:
        raise ValueError("刷子 VHACD 未生成 1–4 个有效凸块")
    parts = []
    for index, item in enumerate(decomposed):
        part = trimesh.Trimesh(**item, process=False)
        volume = abs(float(part.volume))
        if not np.isfinite(volume) or volume <= 1e-9:
            raise ValueError(f"刷子 VHACD 凸块 {index} 体积无效")
        path = output_dir / f"brush_collision_{index:02d}.obj"
        part.export(path)
        parts.append({"mesh_path": str(path), "geom_name": f"brush_collision_{index:02d}",
                      "volume_m3": volume, "sha256": _sha256(path)})
    return parts


def _check_layout(extents: dict[str, np.ndarray]) -> None:
    """名义布局使用与评测采样相同的桌边/物体间距规则。"""
    sample_tabletop_layout(
        LAYOUT_XY_YAW, {name: (0., 0., 0.) for name in ORDER},
        {name: extents[name][:2] for name in ORDER},
        table_bounds_xy=(-.1, .86, -.5, .5), seed=0, max_attempts=1)


def _probe_position_reachability(model: mujoco.MjModel, standby: np.ndarray,
                                 objects: dict[str, dict]) -> dict[str, dict]:
    """用 TCP 纯位置 IK 检查摆位；不代表朝向、碰撞或路径可行。"""
    data = mujoco.MjData(model)
    joints = [model.joint(f"joint_{i}") for i in range(1, 8)]
    addresses = [joint.qposadr[0] for joint in joints]
    lower = np.array([model.jnt_range[joint.id, 0] for joint in joints])
    upper = np.array([model.jnt_range[joint.id, 1] for joint in joints])
    results = {}
    for name in ORDER:
        center = np.array(objects[name]["T_world_object"])[:3, 3]
        target = center + np.array([0, 0, 0.14])

        def residual(q):
            data.qpos[addresses] = q
            mujoco.mj_forward(model, data)
            return np.r_[data.site("pika_tcp").xpos - target, 0.002 * (q - standby)]

        fit = least_squares(residual, standby, bounds=(lower, upper), max_nfev=100)
        error = float(np.linalg.norm(residual(fit.x)[:3]))
        results[name] = {
            "target_tcp_position_world_m": target.tolist(),
            "position_error_m": error,
            "joint_positions_rad": fit.x.tolist(),
            "position_only_reachable": error < 0.01,
            "orientation_checked": False,
            "collision_and_path_checked": False,
        }
        if error >= 0.01:
            raise ValueError(f"{name} 的 TCP 纯位置 IK 误差 {error:.3f} m；请调整摆位")
    return results


def build_tabletop_scene(output_dir: str | Path, *, sources: dict[str, dict]) -> dict:
    """由刷子、方块、簸箕三份 mesh 生成五物体共享场景。"""
    if set(sources) != set(SOURCE_ORDER):
        raise ValueError(f"场景需要 {SOURCE_ORDER} 三份源 mesh，当前为 {tuple(sources)}")
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    robot = build_robot(output / "robot", profile="realman_single_pika")
    root = ET.parse(robot["xml_path"]).getroot()
    root.set("model", "brush_cube_dustpan_single_pika")
    asset, world = root.find("asset"), root.find("worldbody")
    if asset is None or world is None:
        raise ValueError("机器人 MJCF 缺少 asset/worldbody")
    meshes = {}
    extents = {}
    for name in SOURCE_ORDER:
        mesh = canonicalize_mesh(sources[name]["mesh_path"], output / "assets" / f"{name}.obj")
        if name == "brush":
            mesh = place_brush_bristles_down(mesh)
        meshes[name] = mesh
        extents[name] = np.asarray(mesh["extents_xyz_m"], dtype=float)
    for name in ("cube_red", "cube_blue"):
        meshes[name] = meshes["cube"]
        extents[name] = extents["cube"].copy()
    brush_collision_parts = _build_brush_collision_parts(
        Path(meshes["brush"]["mesh_path"]), output / "assets")
    brush_collision_volume = sum(part["volume_m3"] for part in brush_collision_parts)
    _check_layout(extents)

    # 机器人 base_link 原点与桌面顶面同高，定义为本场景 world；此关系是仿真摆放，
    # 并非从未标定的实景相机推得。各块错位摆放，避免一条推扫轨迹完成任务。
    ET.SubElement(world, "geom", name="scene_table", type="box",
                  pos="0.38 0 -0.02", size="0.48 0.5 0.02",
                  rgba="0.79 0.83 0.85 1", friction="0.9 0.01 0.001")
    ET.SubElement(world, "geom", name="scene_floor", type="plane", pos="0 0 -0.045",
                  size="2 2 0.01", rgba="0.68 0.72 0.75 1")
    objects = {}
    for name in ORDER:
        mesh = meshes[name]
        size = extents[name]
        source_name = "cube" if name in BLOCKS else name
        source = sources[source_name]
        asset_name = f"{name}_visual_mesh"
        ET.SubElement(asset, "mesh", name=asset_name, file=mesh["mesh_path"])
        x, y, angle = LAYOUT_XY_YAW[name]
        z = 0.0
        body_pose = np.eye(4)
        body_pose[:3, 3] = (x, y, z)
        body_attributes = {"name": name, "pos": _numbers((x, y, z))}
        if angle:
            body_attributes["quat"] = _numbers((np.cos(angle / 2), 0, 0, np.sin(angle / 2)))
            body_pose[:3, :3] = [[np.cos(angle), -np.sin(angle), 0],
                                 [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
        body = ET.SubElement(world, "body", **body_attributes)
        if name != "dustpan":
            ET.SubElement(body, "freejoint", name=f"{name}_free")
        ET.SubElement(body, "geom", name=f"{name}_visual", type="mesh", mesh=asset_name,
                      rgba=COLORS[name], contype="0", conaffinity="0", density="0", group="1")
        if name == "dustpan":
            # 规范 mesh 的铲口为局部 +X；绕世界 Z 转 -90° 后朝向方块(-Y)。
            half_x, half_y = size[0] / 2, size[1] / 2
            ET.SubElement(body, "geom", name="dustpan_tray_collision", type="box",
                          # 铲口的物理底面顶端与桌面齐平，避免 6 mm 直立门槛把方块卡住。
                          pos=_numbers((0, 0, -0.003)), size=_numbers((half_x, half_y, 0.003)),
                          rgba="0 0 0 0", group="3", friction="0.8 0.01 0.001")
            for side in (-1, 1):
                ET.SubElement(body, "geom", name=f"dustpan_wall_{side}_collision", type="box",
                              pos=_numbers((0, side * (half_y - 0.003), 0.014)),
                              size=_numbers((half_x, 0.003, 0.014)), rgba="0 0 0 0", group="3")
            ET.SubElement(body, "geom", name="dustpan_back_collision", type="box",
                          pos=_numbers((-half_x + 0.003, 0, 0.014)),
                          size=_numbers((0.003, half_y, 0.014)), rgba="0 0 0 0", group="3")
        elif name == "brush":
            # 每块 mesh 都是凸的；显式分配质量，维持原刷子总质量 0.12 kg。
            for part in brush_collision_parts:
                asset_key = part["geom_name"] + "_mesh"
                ET.SubElement(asset, "mesh", name=asset_key, file=part["mesh_path"])
                ET.SubElement(body, "geom", name=part["geom_name"], type="mesh",
                              mesh=asset_key, rgba="0 0 0 0", group="3",
                              mass=_numbers((0.12 * part["volume_m3"] / brush_collision_volume,)),
                              friction="0.9 0.01 0.001")
        else:
            ET.SubElement(body, "geom", name=f"{name}_collision", type="box",
                          pos=_numbers((0, 0, size[2] / 2)), size=_numbers(size / 2),
                          rgba="0 0 0 0", group="3", mass="0.04",
                          friction="0.9 0.01 0.001")
        objects[name] = {
            "role": ("tool" if name == "brush" else "receptacle" if name == "dustpan"
                     else "movable_block"),
            "T_world_object": body_pose.tolist(), "mesh_path": str(Path(mesh["mesh_path"]).relative_to(output)),
            "source_mesh_path": str(Path(source["mesh_path"]).resolve()),
            "source_kind": source["source_kind"] if name == source_name else "shared_cube_asset",
            "mesh_sha256": mesh["canonical_sha256"],
            "source_sha256": mesh["source_sha256"], "T_object_source_mesh": mesh["T_canonical_source"],
            "perception": source.get("perception") if name == source_name else None,
            **({"inferred_side_m": source["inferred_side_m"],
                "scene_scale_factor": source["scene_scale_factor"]}
               if name in BLOCKS and "scene_scale_factor" in source else {}),
            "extents_xyz_m": size.tolist(),
            **({"concept_path": "concepts/cube.json", "derived_from": "cube"}
               if name in ("cube_red", "cube_blue") else {}),
            "collision": "open_tray_proxy" if name == "dustpan" else
                         "vhacd_convex_parts" if name == "brush" else "box_proxy",
            **({"collision_parts": [
                {**part, "mesh_path": str(Path(part["mesh_path"]).relative_to(output))}
                for part in brush_collision_parts],
                "collision_method": {"name": "VHACD", "max_convex_hulls": 4,
                                     "resolution": 100000, "max_vertices_per_hull": 48,
                                     "total_mass_kg": 0.12}}
               if name == "brush" else {}),
            **({"brush_face_normal_object": [0, 0, -1]} if name == "brush" else {}),
            "orientation_rule": "bristles_down_face_parallel_to_table" if name == "brush"
                                else "scoop_opening_toward_cube" if name == "dustpan"
                                else "variable_yaw_cube",
        }
    # 只在真实左右手指接触核验后由 MujocoRuntime 启用；初始绝不固定刷子。
    equality = root.find("equality")
    if equality is None:
        equality = ET.SubElement(root, "equality")
    ET.SubElement(equality, "weld", name="sim_attach_left_brush",
                  body1="pika_gripper_base_link", body2="brush", active="false")
    apply_scene_style(root)
    # MJCF 的 mesh 路径以 scene.xml 所在目录为基准，避免把构建机器的绝对路径固化。
    for mesh_asset in asset.findall("mesh"):
        mesh_file = mesh_asset.get("file")
        if mesh_file:
            mesh_asset.set("file", os.path.relpath(mesh_file, output))
    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth="1200", offheight="800")
    xml_path = output / "scene.xml"
    ET.indent(root)
    ET.ElementTree(root).write(xml_path, encoding="unicode")
    # keyframe 记录实际 qpos；修改 MJCF joint ref 会重定义零位，不可用于摆姿态。
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    for index, angle in enumerate(STANDBY_JOINTS_RAD, start=1):
        data.qpos[model.joint(f"joint_{index}").qposadr[0]] = angle
        data.ctrl[model.actuator(f"joint_{index}_position").id] = angle
    mujoco.mj_forward(model, data)
    if np.linalg.norm(data.site("pika_tcp").xpos - [0.22, 0, 0.46]) > 0.02:
        raise ValueError("Pika 待机姿态 FK 与场景布局不符")
    keyframe = root.find("keyframe")
    if keyframe is None:
        keyframe = ET.SubElement(root, "keyframe")
    ET.SubElement(keyframe, "key", name="scene_start", qpos=_numbers(data.qpos),
                  ctrl=_numbers(data.ctrl))
    ET.indent(root)
    ET.ElementTree(root).write(xml_path, encoding="unicode")
    mujoco.MjModel.from_xml_path(str(xml_path))
    reachability_probe = _probe_position_reachability(model, np.asarray(STANDBY_JOINTS_RAD), objects)
    scene = {
        "schema_version": 1, "scene_xml_path": str(xml_path),
        "world_frame": "single_robot_base", "length_unit": "m",
        "robot": {"profile": "realman_single_pika", "T_world_base": np.eye(4).tolist(),
                  "metadata_path": str((output / "robot" / "robot_metadata.json").relative_to(output)),
                  "start_keyframe": "scene_start", "standby_joint_positions_rad": list(STANDBY_JOINTS_RAD),
                  "standby_tcp_position_world_m": data.site("pika_tcp").xpos.tolist()},
        "table": {"top_z_m": 0.0, "center_xy_m": [0.38, 0.0], "size_xy_m": [0.96, 1.0]},
        "layout_mode": "task_friendly_rearrangement",
        "objects": objects,
        "randomization": {
            "distribution": "independent_uniform_xy_yaw_about_nominal",
            "frame": "world", "length_unit": "m", "angle_unit": "rad",
            "ranges_xyyaw_rad": {name: list(RANDOMIZATION_RANGES_XYYAW[name]) for name in ORDER},
            "table_bounds_xy_m": [-.1, .86, -.5, .5],
            "min_clearance_m": .015, "max_attempts": 200,
            "fixed_objects": ["dustpan"],
            "valid_initial_state": "inside_table_and_clear_at_sampling_then_settled_without_excess_displacement",
        },
        "reachability_probe": reachability_probe,
        "task_draft": {"id": "sweep_cube_into_dustpan", "status": "mujoco_skill_available",
                       "tool": "brush", "targets": list(BLOCKS), "receptacle": "dustpan"},
        "simulation_attachment_pairs": {"left": "brush"},
        "limitations": [
            "物体相对机械臂位置是可达性导向的仿真排布，不是未标定相机的严格实景复现。",
            "源 mesh 为单视角重建，collision 是近似代理；随机评测仅覆盖 scene.json 声明的 XY/yaw 分布，刷柄夹持使用接触核验后的仿真辅助 weld。",
            "Pika 法兰平移和 TCP 未经实物标定。",
        ],
    }
    (output / "scene.json").write_text(json.dumps(scene, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    task = {
        "schema_version": 1,
        "scene_path": "scene.json",
        "id": "sweep_cube_into_dustpan",
        "status": "mujoco_skill_available",
        "instruction": "使用刷子把绿、红、蓝三个方块扫入簸箕。",
        "roles": {"tool": "brush", "movable_targets": list(BLOCKS),
                  "receptacle": "dustpan"},
        "success_condition": {
            "type": "target_resting_inside_receptacle",
            "targets": list(BLOCKS), "receptacle": "dustpan",
            "implemented": True,
            "checker": "benchmarks.sweep_cube_task.evaluation.evaluate_cube_in_dustpan",
        },
        "skill_entrypoint": "tools/run/run_sweep_cube.py",
        "execution_status": "mujoco_seeded_evaluation_available",
        "evidence_mode": "physical_sweep_with_contact_verified_simulation_assisted_brush_grip",
    }
    (output / "task.json").write_text(json.dumps(task, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return scene
