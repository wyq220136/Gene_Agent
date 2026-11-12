"""从同一帧重建物体生成清扫、倾倒、舀取三种桌面场景预览。"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import trimesh
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from gene_agent.scene.reconstructed_tabletop import canonicalize_mesh, STANDBY_JOINTS_RAD
from gene_agent.perception_modules.scene_builder import observed_center_camera
from gene_agent.scene.tabletop_randomization import sample_tabletop_layout
from gene_agent.motion.mujoco_ik import MujocoPosePlanner


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "data/realworld/brush_cube_dustpan_0/scene.xml"
BOWL_SOURCE = ROOT / "data/task_1/objs/Bowl/Bowl_0/meshes/obj_mesh_simplified.obj"
RECON_NAMES = ("brush_long", "spoon_orange", "spoon_pink", "cup_green")
SCENE_VARIANTS = ("sweep_old", "sweep_long", "pour",
                  "scoop_orange", "scoop_pink")


def _numbers(values) -> str:
    return " ".join(f"{float(value):.9g}" for value in values)


def transform_container_contents_xy(points_xy: np.ndarray, *, old_xy: np.ndarray,
                                    new_xy: np.ndarray, yaw_delta_rad: float) -> np.ndarray:
    """容器改变平面位姿时，让内部小球保持相对容器的初始分布。"""
    points = np.asarray(points_xy, dtype=float)
    old = np.asarray(old_xy, dtype=float)
    new = np.asarray(new_xy, dtype=float)
    angle = float(yaw_delta_rad)
    if (points.ndim != 2 or points.shape[1] != 2 or old.shape != (2,) or
            new.shape != (2,) or not np.isfinite(points).all() or
            not np.isfinite(old).all() or not np.isfinite(new).all() or
            not np.isfinite(angle)):
        raise ValueError("容器与小球平面位姿必须为有限的 XY/yaw 数据")
    rotation = np.array([[np.cos(angle), -np.sin(angle)],
                         [np.sin(angle), np.cos(angle)]])
    return (points - old) @ rotation.T + new


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _normalize_vertical(source: Path, destination: Path, rotation: np.ndarray,
                        *, body_center_shift_xy=(0., 0.)) -> dict:
    """将杯/碗的开口轴竖立，底部贴局部 z=0；返回源 mesh 到新 frame。"""
    mesh = trimesh.load_mesh(source, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.vertices) < 4:
        raise ValueError(f"mesh 无有效三角面: {source}")
    vertices = np.asarray(mesh.vertices) @ rotation.T
    offset = np.array([-(vertices[:, 0].min() + vertices[:, 0].max()) / 2,
                       -(vertices[:, 1].min() + vertices[:, 1].max()) / 2,
                       -vertices[:, 2].min()])
    # 杯把会偏移整件包围盒中心；以杯身中心作物体原点，便于杯内球和碰撞代理对齐。
    offset[:2] += np.asarray(body_center_shift_xy)
    mesh.vertices = vertices + offset
    destination.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(destination)
    transform = np.eye(4)
    transform[:3, :3], transform[:3, 3] = rotation, offset
    return {"source_mesh_path": str(source), "source_sha256": _sha(source),
            "mesh_path": str(destination), "mesh_sha256": _sha(destination),
            "T_object_source_mesh": transform.tolist(),
            "extents_xyz_m": np.asarray(mesh.extents).tolist(),
            "body_center_shift_xy_m": list(body_center_shift_xy)}


def _prepare_assets(capture: Path, output: Path) -> dict:
    shared = output / "shared_assets"
    shared.mkdir(parents=True, exist_ok=True)
    records = {}
    for name in RECON_NAMES:
        corrected = capture / "reconstruction" / name / "functional_mesh" / "mesh.obj"
        source = corrected if name == "brush_small" and corrected.is_file() else (
            capture / "reconstruction" / name / "mesh_simplified.obj")
        mask = capture / "reconstruction" / name / "mask.png"
        if not source.is_file() or not mask.is_file():
            raise FileNotFoundError(f"缺少 {name} 的 SAM3D mesh 或 mask")
        destination = shared / f"{name}.obj"
        if source == corrected:
            mesh = trimesh.load_mesh(source, force="mesh", process=False)
            destination.write_bytes(source.read_bytes())
            row = {"source_mesh_path": str(source), "source_sha256": _sha(source),
                   "mesh_path": str(destination), "mesh_sha256": _sha(destination),
                   "T_object_source_mesh": np.eye(4).tolist(),
                   "extents_xyz_m": np.asarray(mesh.extents).tolist()}
            bristles = source.parent / "white_bristles.obj"
            bristle_out = shared / "brush_small_bristles.obj"
            bristle_out.write_bytes(bristles.read_bytes())
            row["bristles_mesh_path"] = str(bristle_out)
            blue_out = shared / "brush_small_blue_body.obj"
            blue_out.write_bytes((source.parent / "blue_body.obj").read_bytes())
            row["blue_body_mesh_path"] = str(blue_out)
        elif name == "cup_green":
            # SAM3D 杯口轴为源 +Y；Rx(+90°) 使杯口朝世界 +Z。
            cup_rotation = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
            row = _normalize_vertical(source, destination, cup_rotation,
                                      body_center_shift_xy=(0., .026))
        else:
            canonical = canonicalize_mesh(source, destination)
            row = {"source_mesh_path": str(source),
                   "source_sha256": canonical["source_sha256"],
                   "mesh_path": str(destination),
                   "mesh_sha256": canonical["canonical_sha256"],
                   "T_object_source_mesh": canonical["T_canonical_source"],
                   "extents_xyz_m": canonical["extents_xyz_m"]}
        row.update(mask_path=str(mask), mask_sha256=_sha(mask),
                   source_kind=("rgbd_functional_correction_after_sam3d_failure"
                                if source == corrected else "sam3d_single_rgbd_view"),
                   observed_center_camera_m=observed_center_camera(
                       mask, capture / "depth.png", capture / "camera.json"))
        records[name] = row
    # 旧 Bowl_0 的上方向是源 mesh 的 +Y，绕 X 轴 +90° 后为世界 +Z。
    bowl_rotation = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
    records["bowl"] = _normalize_vertical(
        BOWL_SOURCE, shared / "bowl.obj", bowl_rotation)
    records["bowl"]["source_kind"] = "existing_task_1_Bowl_0"
    (output / "asset_manifest.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return records


def _new_root(scene_dir: Path, task: str):
    root = ET.parse(TEMPLATE).getroot()
    root.set("model", f"my_scene_0_{task}")
    asset = root.find("asset")
    world = root.find("worldbody")
    assert asset is not None and world is not None
    # 模板的机器人资产留在原路径；改为相对当前 scene.xml 的路径。
    for mesh in asset.findall("mesh"):
        source = (TEMPLATE.parent / mesh.get("file")).resolve()
        mesh.set("file", os.path.relpath(source, scene_dir))
    saved = {body.get("name"): copy.deepcopy(body) for body in world.findall("body")
             if body.get("name") in {"brush", "cube", "cube_red", "cube_blue", "dustpan"}}
    for body in list(world.findall("body")):
        if body.get("name") in saved:
            world.remove(body)
    keyframe = root.find("keyframe")
    if keyframe is not None:
        root.remove(keyframe)
    equality = root.find("equality")
    if equality is not None:
        root.remove(equality)
    return root, asset, world, saved


def _mesh_asset(asset, scene_dir: Path, name: str, row: dict) -> None:
    ET.SubElement(asset, "mesh", name=f"{name}_mesh",
                  file=os.path.relpath(row["mesh_path"], scene_dir))


def _free_body(world, name: str, position, *, yaw: float = 0., mass: float = .1,
               quat=None):
    orientation = quat if quat is not None else [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
    body = ET.SubElement(world, "body", name=name, pos=_numbers(position),
                         quat=_numbers(orientation))
    ET.SubElement(body, "freejoint", name=f"{name}_free")
    ET.SubElement(body, "inertial", pos="0 0 0.03", mass=f"{mass:.6g}",
                  diaginertia="0.0003 0.0003 0.0003")
    return body


def _visual(body, name: str, rgba: str) -> None:
    ET.SubElement(body, "geom", name=f"{name}_visual", type="mesh",
                  mesh=f"{name}_mesh", rgba=rgba, group="1", density="0",
                  contype="0", conaffinity="0")


def _box(body, name: str, pos, size) -> None:
    ET.SubElement(body, "geom", name=name, type="box", pos=_numbers(pos),
                  size=_numbers(size), rgba="0 0 0 0", group="3", density="0",
                  friction="0.8 0.01 0.001")


def _tool(world, asset, scene_dir: Path, name: str, row: dict,
          position, color: str, *, kind: str, yaw: float = 0., quat=None) -> None:
    _mesh_asset(asset, scene_dir, name, row)
    extent = np.asarray(row["extents_xyz_m"])
    body = _free_body(world, name, position, yaw=yaw, quat=quat,
                      mass=.12 if kind == "brush" else .04)
    if name == "brush_small" and "blue_body_mesh_path" in row:
        ET.SubElement(asset, "mesh", name="brush_small_blue_body_mesh",
                      file=os.path.relpath(row["blue_body_mesh_path"], scene_dir))
        ET.SubElement(body, "geom", name="brush_small_blue_body_visual", type="mesh",
                      mesh="brush_small_blue_body_mesh", rgba=color, group="1",
                      density="0", contype="0", conaffinity="0")
    else:
        _visual(body, name, color)
    if name == "brush_small" and "bristles_mesh_path" in row:
        ET.SubElement(asset, "mesh", name="brush_small_bristles_mesh",
                      file=os.path.relpath(row["bristles_mesh_path"], scene_dir))
        ET.SubElement(body, "geom", name="brush_small_bristles_visual", type="mesh",
                      mesh="brush_small_bristles_mesh", rgba="0.95 0.95 0.90 1",
                      group="1", density="0", contype="0", conaffinity="0")
    if kind == "brush":
        _box(body, f"{name}_head_proxy", [extent[0] * .25, 0, extent[2] * .32],
             [extent[0] * .24, extent[1] * .49, extent[2] * .31])
        if name == "brush_long":
            # 与 grasp_handle Concept 一致：圆柄轴为局部 X，半径 8 mm。
            ET.SubElement(body, "geom", name=f"{name}_handle_proxy",
                          type="cylinder", pos="-.063 0 .018",
                          quat=_numbers(Rotation.from_euler("y", 90, degrees=True)
                                        .as_quat()[[3, 0, 1, 2]]),
                          size=".008 .0725", rgba="0 0 0 0", group="3",
                          density="0", friction="1.2 0.01 0.001")
        else:
            _box(body, f"{name}_handle_proxy", [-extent[0] * .25, 0, extent[2] * .6],
                 [extent[0] * .26, extent[1] * .19, extent[2] * .15])
    else:
        handle_half_z = max(.003, extent[2] * .2)
        _box(body, f"{name}_handle_proxy", [-extent[0] * .18, 0, handle_half_z],
             [extent[0] * .29, extent[1] * .20, handle_half_z])
        # 勺头的功能是承载球：整块实心 box 不能形成可盛球的凹槽。
        # 用底面和四段低壁表示接触，visual 仍使用 SAM3D mesh；局部 z=0 贴桌。
        scoop_x = extent[0] * .33
        half_x, half_y = extent[0] * .17, extent[1] * .46
        base_half_z = .0025
        # 侧沿顶端 8 mm，低于静置球心约 13 mm；倾斜时球可越沿滑出。
        # 背沿保留较高的推进接触面，底面仍提供承载。
        wall_half_z = .0015
        back_half_z = max(.0085, extent[2] * .40)
        wall_center_z = base_half_z * 2 + wall_half_z
        wall_half_width = .0025
        _box(body, f"{name}_scoop_base_proxy", [scoop_x, 0, base_half_z],
             [half_x, half_y, base_half_z])
        for sign, label in ((-1, "negative"), (1, "positive")):
            _box(body, f"{name}_scoop_side_{label}_proxy",
                 [scoop_x, sign * (half_y - wall_half_width), wall_center_z],
                 [half_x, wall_half_width, wall_half_z])
            end_half_z = .003 if sign > 0 else back_half_z
            end_center_z = base_half_z * 2 + end_half_z
            _box(body, f"{name}_scoop_end_{label}_proxy",
                 [scoop_x + sign * (half_x - wall_half_width), 0, end_center_z],
                 [wall_half_width, half_y - wall_half_width, end_half_z])


def _vessel(world, asset, scene_dir: Path, name: str, row: dict,
            position, color: str, *, kind: str,
            sealed_bottom: bool = False) -> None:
    _mesh_asset(asset, scene_dir, name, row)
    extent = np.asarray(row["extents_xyz_m"])
    radius = .036 if kind == "cup" else .035
    height = .095 if kind == "cup" else .075
    body = _free_body(world, name, position, mass=.18 if kind == "cup" else .25)
    _visual(body, name, color)
    if kind == "cup":
        # Concept 杯把环的上缘约位于局部 (0, +68, +94) mm。
        # 只给功能性抓取处加薄碰撞条；杯身仍由下方圆筒代理承载小球。
        _box(body, f"{name}_handle_top_proxy", [0., .068, .093],
             [.0065, .014, .004])
    # 倾倒时球从杯口以约 1 m/s 落入目标碗。旧薄底会让近侧球穿到底下；
    # 仅 pour 目标碗使用更宽、更厚的底面，不改变其他任务的容器。
    bottom_radius = .045 if sealed_bottom else radius
    bottom_half_height = .005 if sealed_bottom else .003
    ET.SubElement(body, "geom", name=f"{name}_bottom_proxy", type="cylinder",
                  pos=_numbers([0., 0., bottom_half_height]),
                  size=_numbers([bottom_radius, bottom_half_height]),
                  rgba="0 0 0 0", group="3", density="0")
    # Bowl_0 内壁近似斜面，避免球从视觉碗底侧面露出。
    count = 12 if kind == "cup" else 16
    slope = 0. if kind == "cup" else np.arctan2(.032, .072)
    wall_radius = radius if kind == "cup" else .051
    z_center = height / 2 if kind == "cup" else .039
    half_length = height / 2 if kind == "cup" else .040
    for index in range(count):
        angle = 2 * np.pi * index / count
        q = (Rotation.from_euler("z", angle) * Rotation.from_euler("y", slope)).as_quat()
        ET.SubElement(body, "geom", name=f"{name}_wall_{index}_proxy",
                      type="box", pos=_numbers([wall_radius * np.cos(angle),
                                                 wall_radius * np.sin(angle), z_center]),
                      size=_numbers([.003, wall_radius * np.sin(np.pi / count) + .004,
                                     half_length]),
                      quat=_numbers([q[3], q[0], q[1], q[2]]),
                      friction=("0.05 0.001 0.0001" if kind == "cup" else
                                "1 0.005 0.0001"),
                      rgba="0 0 0 0", group="3", density="0")


def _balls(world, prefix: str, center, *, kind: str):
    # 六角密排；杯内用 7 mm 球装到杯口附近，使不超过 50° 的慢速倾倒可出球。
    radius = .008 if kind == "bowl" else .007
    if kind == "cup":
        # 相邻层横向错开；125 颗球均在杯口下方落稳。
        spacing = .0141
        offsets = []
        for layer in range(10):
            ox, oy = ((0., 0.) if layer % 2 == 0 else (.00705, .00407))
            z = .0073 + layer * .01155
            for j in range(-4, 5):
                for i in range(-4, 5):
                    x = i * spacing + (j % 2) * spacing / 2 + ox
                    y = j * spacing * np.sqrt(3) / 2 + oy
                    if x*x + y*y <= .026**2:
                        offsets.append((x, y, z))
    else:
        layers = ((.024, .012), (.031, .026), (.038, .040),
                  (.045, .054), (.052, .068), (.052, .082),
                  (.040, .096), (.035, .110))
        offsets = []
        spacing = radius * 2.08
        for layer_index, (layer_radius, z) in enumerate(layers):
            rows = np.arange(-layer_radius, layer_radius + 1e-6,
                             spacing * np.sqrt(3) / 2)
            for row_index, dy in enumerate(rows):
                xs = np.arange(-layer_radius, layer_radius + 1e-6, spacing)
                for x in xs + (spacing / 2 if (row_index + layer_index) % 2 else 0):
                    if x*x + dy*dy <= layer_radius*layer_radius:
                        offsets.append((float(x), float(dy), z))
    for index, (dx, dy, z) in enumerate(offsets):
        body = ET.SubElement(world, "body", name=f"{prefix}_{index}",
                             pos=_numbers([center[0] + dx, center[1] + dy, z]))
        ET.SubElement(body, "freejoint", name=f"{prefix}_{index}_free")
        ET.SubElement(body, "geom", name=f"{prefix}_{index}_collision", type="sphere",
                      size=f"{radius:.6g}", mass="0.0015", rgba="0.93 0.36 0.13 1",
                      friction=("0.05 0.001 0.0001" if kind == "cup" else
                                "0.8 0.01 0.001"))


def _clone_template_body(world, saved: dict, name: str, position,
                         *, rename: str | None = None):
    body = copy.deepcopy(saved[name])
    old = body.get("name")
    if rename:
        body.set("name", rename)
        for element in list(body.iter())[1:]:
            for attribute in ("name",):
                value = element.get(attribute)
                if value and value.startswith(old):
                    element.set(attribute, rename + value[len(old):])
    body.set("pos", _numbers(position))
    world.append(body)
    return body


def _write_keyframe(root, path: Path) -> mujoco.MjModel:
    ET.indent(root)
    ET.ElementTree(root).write(path, encoding="unicode")
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    for index, value in enumerate(STANDBY_JOINTS_RAD, start=1):
        data.qpos[model.joint(f"joint_{index}").qposadr[0]] = value
    ctrl = [*STANDBY_JOINTS_RAD, 0., 0.]
    keyframe = ET.SubElement(root, "keyframe")
    ET.SubElement(keyframe, "key", name="scene_start",
                  qpos=_numbers(data.qpos), ctrl=_numbers(ctrl))
    ET.indent(root)
    ET.ElementTree(root).write(path, encoding="unicode")
    return mujoco.MjModel.from_xml_path(str(path))


def _world_pose(body) -> list[list[float]]:
    transform = np.eye(4)
    transform[:3, :3] = body.xmat.reshape(3, 3)
    transform[:3, 3] = body.xpos
    return transform.tolist()


def _position_reachability(model, object_poses: dict, names: set[str]) -> dict:
    """仅测试物体上方 14 cm 的 TCP 位置 IK；不代表抓取姿态或路径。"""
    data = mujoco.MjData(model)
    joints = [model.joint(f"joint_{index}") for index in range(1, 8)]
    addresses = [joint.qposadr[0] for joint in joints]
    lower = np.asarray([model.jnt_range[joint.id, 0] for joint in joints])
    upper = np.asarray([model.jnt_range[joint.id, 1] for joint in joints])
    standby = np.asarray(STANDBY_JOINTS_RAD)
    results = {}
    for name in sorted(names):
        target = np.asarray(object_poses[name]["T_world_object"])[:3, 3]
        target = target + np.array([0., 0., .14])

        def residual(q):
            data.qpos[addresses] = q
            mujoco.mj_forward(model, data)
            return np.r_[data.site("pika_tcp").xpos - target, .002 * (q - standby)]

        fit = least_squares(residual, standby, bounds=(lower, upper), max_nfev=100)
        results[name] = {"target_tcp_world_m": target.tolist(),
                         "position_error_m": float(np.linalg.norm(residual(fit.x)[:3])),
                         "evidence": "position_only_ik_no_orientation_or_collision"}
    return results


TOOL_GRIP_LOCAL_M = {
    "spoon_orange": (-.036, 0., .016),
    "spoon_pink": (-.037, 0., .017),
    "brush_old": (-.0625, .022, .069),
    "brush_long": (-.063, 0., .018),
    "brush_small": (0., .018, .041),
}


def _tool_grasp_probe(model: mujoco.MjModel, data: mujoco.MjData,
                      tool: str) -> dict:
    """从 standby 检查功能抓持部位与直线接近；物理夹紧另验。"""
    metadata = json.loads((TEMPLATE.parent / "robot/robot_metadata.json").read_text())
    planner = MujocoPosePlanner(model, data, metadata, "left")
    local_top_down = np.column_stack(([0, 0, -1], [0, 1, 0], [1, 0, 0]))
    # 长柄刷必须沿世界 -Z 正向下抓；其余工具沿用已有的倾斜抓取诊断。
    grip_rotation = (local_top_down if tool == "brush_long" else
                     (Rotation.from_euler("y", -20, degrees=True)
                      * Rotation.from_matrix(local_top_down)).as_matrix())
    local_grip = np.asarray(TOOL_GRIP_LOCAL_M[tool])
    world_grip = data.body(tool).xpos + data.body(tool).xmat.reshape(3, 3) @ local_grip
    stages = {}
    pregrasp_joints = None
    grasp_target = None
    for stage, height in (("pregrasp", .064), ("grasp", .002)):
        target = np.eye(4)
        target[:3, :3] = grip_rotation
        target[:3, 3] = world_grip + [0., 0., height]
        planned = planner.plan_to_pose(target, path_kind="joint", duration_s=2.)
        if stage == "pregrasp" and planned["success"]:
            pregrasp_joints = np.asarray(planned["positions_rad"][-1])
        if stage == "grasp":
            grasp_target = target
        stages[stage] = {"target_tcp_world": target.tolist(),
                         "success": bool(planned["success"]),
                         "failure_type": planned.get("failure_type"),
                         "position_error_m": planned.get("position_error_m"),
                         "rotation_error_rad": planned.get("rotation_error_rad"),
                         "forbidden_contacts": planned.get("forbidden_contacts", [])[:3]}
    if pregrasp_joints is not None and stages["grasp"]["success"]:
        approach_data = mujoco.MjData(model)
        mujoco.mj_copyData(approach_data, model, data)
        approach_data.qpos[planner.qpos_addresses] = pregrasp_joints
        mujoco.mj_forward(model, approach_data)
        approach = MujocoPosePlanner(model, approach_data, metadata, "left").plan_to_pose(
            grasp_target, path_kind="linear_tcp", duration_s=2.)
        stages["linear_approach"] = {
            "target_tcp_world": grasp_target.tolist(),
            "success": bool(approach["success"]),
            "failure_type": approach.get("failure_type"),
            "position_error_m": approach.get("position_error_m"),
            "rotation_error_rad": approach.get("rotation_error_rad"),
            "forbidden_contacts": approach.get("forbidden_contacts", [])[:3]}
    return {"success": all(row["success"] for row in stages.values()),
            "stages": stages,
            "planner_backend": "mujoco_cpu_diagnostic",
            "approach_axis_world": grip_rotation[:, 0].tolist(),
            "evidence": "full_pose_ik_joint_path_collision_and_pregrasp_to_grasp_linear_tcp_check",
            "limitations": "尚未执行夹爪闭合、接触保持和携物运动"}


def _cup_handle_grasp_probe(model: mujoco.MjModel, data: mujoco.MjData,
                            scene: dict) -> dict:
    """以杯把 Concept 检查随机初态的 pregrasp 与 grasp 可达性。"""
    from gene_agent.knowledge.grasp.cup_handle import top_down_cup_handle_grasp
    from gene_agent.knowledge.geometry import load_concept

    cup_name = scene["roles"]["source_container"]
    metadata = json.loads((TEMPLATE.parent / scene["robot"]["metadata_path"])
                          .read_text(encoding="utf-8"))
    handle = load_concept(Path(scene["concept_paths"][cup_name]),
                          part_name="grasp_handle").concept
    planner = MujocoPosePlanner(model, data, metadata, "left")
    grasp = top_down_cup_handle_grasp(np.asarray(_world_pose(data.body(cup_name))),
                                      handle.pose_in_object)
    pregrasp = grasp.copy()
    pregrasp[2, 3] += .07
    stages = {}
    for name, target in (("pregrasp", pregrasp), ("grasp", grasp)):
        plan = planner.plan_to_pose(target, path_kind="joint", duration_s=2.)
        stages[name] = {"success": bool(plan["success"]),
                        "failure_type": plan.get("failure_type"),
                        "position_error_m": plan.get("position_error_m"),
                        "rotation_error_rad": plan.get("rotation_error_rad")}
    return {"success": all(row["success"] for row in stages.values()),
            "stages": stages, "planner_backend": "mujoco_cpu_diagnostic",
            "limitations": "只筛纯 pose 与诊断路径；正式抓取仍须 cuRobo 和 MuJoCo 接触验收"}


def build_multi_task_scenes(capture_dir: str | Path, output_dir: str | Path,
                            *, tasks: tuple[str, ...] | None = None) -> dict:
    capture = Path(capture_dir).resolve()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    for name in ("rgb.png", "depth.png", "camera.json"):
        if not (capture / name).is_file():
            raise FileNotFoundError(capture / name)
    assets = _prepare_assets(capture, output)
    capture_hashes = {name: _sha(capture / name) for name in
                      ("rgb.png", "depth.png", "camera.json")}
    template_scene = json.loads((TEMPLATE.parent / "scene.json").read_text(encoding="utf-8"))
    result = {}
    chosen = SCENE_VARIANTS if tasks is None else tuple(tasks)
    if not chosen or any(task not in SCENE_VARIANTS for task in chosen):
        raise ValueError(f"tasks 必须取自 {SCENE_VARIANTS}")
    for task in chosen:
        scene_dir = output / task
        scene_dir.mkdir(parents=True, exist_ok=True)
        root, asset, world, saved = _new_root(scene_dir, task)
        if task.startswith("sweep_"):
            brush = {"sweep_old": "brush_old", "sweep_long": "brush_long",
                     "sweep_small": "brush_small"}[task]
            if brush == "brush_old":
                _clone_template_body(world, saved, "brush", (.32, -.17, 0), rename=brush)
            elif brush == "brush_long":
                # SAM3D 的局部 +X 指向刷头。绕 Y 轴转 90° 后刷头朝下、柄朝上。
                upright = Rotation.from_euler("y", 90, degrees=True).as_quat()
                _tool(world, asset, scene_dir, brush, assets[brush],
                      (.20, -.05, .135), "0.88 0.84 0.70 1", kind="brush",
                      quat=[upright[3], *upright[:3]])
                # 低矮开口支架只扶住刷头；刷子仍是 free body，可从上方拔出。
                for side, x in (("left", .195), ("right", .234)):
                    ET.SubElement(world, "geom", name=f"brush_long_stand_{side}",
                                  type="box", pos=_numbers([x, -.05, .03]),
                                  size=".004 .097 .03", rgba=".35 .35 .38 1",
                                  friction="0.8 0.01 0.001")
                # 位置伺服无重力前馈时，抓取末端会因静态关节误差偏斜数度。
                # 仅本仿真场景启用机器人重力补偿，物体仍受正常重力。
                for robot_body in world.findall(".//body"):
                    body_name = robot_body.get("name", "")
                    if body_name.startswith(("link_", "pika_gripper_")):
                        robot_body.set("gravcomp", "1")
                equality = ET.SubElement(root, "equality")
                ET.SubElement(equality, "weld", name="sim_attach_left_brush_long",
                              body1="pika_gripper_base_link", body2="brush_long",
                              active="false")
            else:
                _tool(world, asset, scene_dir, brush, assets[brush],
                      (.37, -.17, 0), "0.88 0.84 0.70 1" if brush == "brush_long"
                      else "0.24 0.70 0.82 1", kind="brush",
                      yaw=np.pi if brush == "brush_small" else 0.)
            cube_positions = (("cube", (.40, -.015, 0)),
                              ("cube_red", (.49, .030, 0)),
                              ("cube_blue", (.43, .055, 0)))
            for name, position in cube_positions:
                _clone_template_body(world, saved, name, position)
            _clone_template_body(world, saved, "dustpan", (.45, .18, 0))
            roles = {"tool": brush, "variant_family": "sweep_brush",
                     "movable_targets": ["cube", "cube_red", "cube_blue"],
                     "receptacle": "dustpan"}
        elif task == "pour":
            _vessel(world, asset, scene_dir, "cup_green", assets["cup_green"],
                    (.35, -.16, 0), "0.42 0.68 0.51 1", kind="cup")
            _vessel(world, asset, scene_dir, "bowl_target", assets["bowl"],
                    (.30, .04, 0), "0.86 0.89 0.91 1", kind="bowl",
                    sealed_bottom=True)
            _balls(world, "pour_ball", (.35, -.16), kind="cup")
            roles = {"source_container": "cup_green", "target_container": "bowl_target",
                     "contents": "pour_ball_*"}
        elif task.startswith("scoop_"):
            _vessel(world, asset, scene_dir, "bowl_source", assets["bowl"],
                    (.43, -.06, 0), "0.85 0.88 0.91 1", kind="bowl")
            # 目标碗更靠近机械臂的可达区，使勺头绕自身中心连续翻转。
            _vessel(world, asset, scene_dir, "bowl_target", assets["bowl"],
                    (.46, .125, 0), "0.88 0.86 0.83 1", kind="bowl")
            spoon = "spoon_orange" if task == "scoop_orange" else "spoon_pink"
            _tool(world, asset, scene_dir, spoon, assets[spoon],
                  (.38, -.23, 0), "0.99 0.49 0.08 1" if spoon == "spoon_orange"
                  else "0.89 0.38 0.53 1", kind="spoon")
            _balls(world, "scoop_ball", (.43, -.06), kind="bowl")
            roles = {"tool": spoon, "variant_family": "scoop_spoon",
                     "source_container": "bowl_source", "target_container": "bowl_target",
                     "contents": "scoop_ball_*"}
        path = scene_dir / "scene.xml"
        model = _write_keyframe(root, path)
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, 0)
        mujoco.mj_forward(model, data)
        object_names = {
            model.body(i).name for i in range(model.nbody)
            if model.body_jntnum[i] and model.jnt_type[model.body_jntadr[i]] ==
            mujoco.mjtJoint.mjJNT_FREE
        }
        object_names.update(name for name in (
            roles.get("receptacle"), roles.get("source_container"),
            roles.get("target_container")) if name)
        placement_poses = {
            name: {"T_world_object": _world_pose(data.body(name)),
                   "frame": "robot_base", "translation_unit": "m"}
            for name in sorted(object_names)
        }
        initial_penetrations = [
            {"geom1": model.geom(contact.geom1).name,
             "geom2": model.geom(contact.geom2).name,
             "depth_m": -float(contact.dist)}
            for contact in data.contact[:data.ncon] if contact.dist < -.003
        ]
        for _ in range(500):
            mujoco.mj_step(model, data)
        settled_poses = {name: {"T_world_object": _world_pose(data.body(name)),
                                "frame": "robot_base", "translation_unit": "m"}
                         for name in sorted(object_names)}
        # 交付的 scene_start 使用落稳状态，避免球在第一帧悬空或穿过视觉碗。
        root.find("./keyframe/key").set("qpos", _numbers(data.qpos))
        ET.indent(root)
        ET.ElementTree(root).write(path, encoding="unicode")
        nominal_poses = settled_poses
        probe_names = object_names - {name for name in object_names if "ball_" in name}
        reachability = _position_reachability(model, nominal_poses, probe_names)
        grasp_probe = None
        if roles.get("tool") in TOOL_GRIP_LOCAL_M:
            grasp_probe = _tool_grasp_probe(model, data, roles["tool"])
            if not grasp_probe["success"]:
                raise RuntimeError(f"{task} 名义工具抓取不可达: {grasp_probe['stages']}")
        old_concepts = TEMPLATE.parent / "concepts"
        concept_paths = {}
        for name in probe_names:
            if name in RECON_NAMES or name.startswith("bowl_"):
                concept_name = "bowl" if name.startswith("bowl_") else name
                candidate = capture / "concepts" / concept_name / "concept.json"
                if name == "brush_small":
                    revised = capture / "concepts_v2" / "brush_small" / "concept.json"
                    if revised.is_file():
                        candidate = revised
            else:
                concept_name = "brush" if name == "brush_old" else (
                    "cube" if name.startswith("cube") else name)
                candidate = old_concepts / f"{concept_name}.json"
            if candidate.is_file():
                concept_paths[name] = str(candidate)
        scene = {"schema_version": 1, "task": task,
                 "scene_xml_path": str(path), "world_frame": "robot_base",
                 "capture_dir": str(capture), "capture_sha256": capture_hashes,
                 "asset_manifest_path": str(output / "asset_manifest.json"),
                 "robot": template_scene["robot"], "roles": roles,
                 "concept_paths": concept_paths,
                 "placement_object_poses": placement_poses,
                 "nominal_object_poses": nominal_poses,
                 "settled_object_poses": settled_poses,
                 "reachability_probe": reachability,
                 "tool_grasp_probe": grasp_probe,
                 "spoon_collision_proxy": ("low_lip_v3"
                                           if task.startswith("scoop_") else None),
                 "settling_check": {"steps": 500,
                                    "initial_penetrations_over_3mm": initial_penetrations},
                 "status": "scene_preview_only",
                 "limitations": ["物体摆位按任务可达性设计，不等于照片中相机坐标关系。",
                                 "蓝色短刷暂不参与本轮生成和验收；其旧目录仅供历史对照。",
                                 "碰撞为简化代理；场景生成只验抓取位姿和路径，不等于物理夹持或 Task 成功。"],
                 "simulation_attachment_pairs": ({"left": "brush_long"}
                                                  if task == "sweep_long" else {}),
                 "robot_gravity_compensation": ("mujoco_builtin"
                                                 if task == "sweep_long" else "none")}
        (scene_dir / "scene.json").write_text(
            json.dumps(scene, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        task_info = {"id": task,
                     "status": ("task_implemented" if task == "pour" else
                                "scene_preview_only"),
                     "roles": roles, "simulation_backend": "mujoco",
                     "goal_evaluator_implemented": task == "pour",
                     "task_runner": ("tools/run/run_cup_pour_task.py" if task == "pour"
                                     else None)}
        (scene_dir / "task.json").write_text(
            json.dumps(task_info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        result[task] = scene
    return result


def sample_scene_variants(scene_dir: str | Path, seeds: tuple[int, ...] = (0, 7, 21)) -> list[Path]:
    """将可复现的初始分布写为独立 MJCF，并保留球相对来源容器的位置。"""
    source = Path(scene_dir).resolve()
    spec = json.loads((source / "scene.json").read_text(encoding="utf-8"))
    model = mujoco.MjModel.from_xml_path(str(source / "scene.xml"))
    nominal = spec["nominal_object_poses"]
    names = [name for name in nominal if "_ball_" not in name]
    poses = {}
    nominal_rotations = {}
    ranges = {}
    footprints = {}
    for name in names:
        transform = np.asarray(nominal[name]["T_world_object"])
        nominal_rotations[name] = Rotation.from_matrix(transform[:3, :3])
        poses[name] = (float(transform[0, 3]), float(transform[1, 3]),
                       0. if name == "brush_long" else
                       float(np.arctan2(transform[1, 0], transform[0, 0])))
        if name.startswith("brush"):
            footprints[name] = (.05, .20) if name == "brush_long" else (.28, .055)
            ranges[name] = (.015, .015, 0.) if name == "brush_long" else (
                .025, .025, np.deg2rad(12))
        elif name.startswith("spoon"):
            footprints[name] = (.16, .035)
            ranges[name] = (.01, .01, np.deg2rad(5))
        elif "bowl" in name:
            footprints[name] = (.14, .14)
            ranges[name] = ((.01, .01, np.deg2rad(5))
                            if spec["task"].startswith("scoop_") else
                            (.02, .02, np.deg2rad(8)))
        elif name.startswith("cup"):
            footprints[name] = (.09, .09)
            # 杯口局部 +Z 始终朝上；把手绕竖直轴允许一整圈初态。
            ranges[name] = (.025, .025, np.pi)
        elif name == "dustpan":
            footprints[name] = (.18, .16)
            ranges[name] = (.015, .015, np.deg2rad(7))
        else:
            footprints[name] = (.035, .035)
            ranges[name] = (.02, .02, np.deg2rad(15))
    outputs = []
    tool = spec["roles"].get("tool")
    cup_name = (spec["roles"].get("source_container")
                if spec["task"] == "pour" else None)
    probe_data = (mujoco.MjData(model)
                  if tool in TOOL_GRIP_LOCAL_M or cup_name else None)
    last_grasp_probe = None

    def sampled_quat(name: str, yaw: float) -> np.ndarray:
        delta = yaw - poses[name][2]
        quaternion = (Rotation.from_euler("z", delta) * nominal_rotations[name]).as_quat()
        return np.asarray([quaternion[3], *quaternion[:3]])

    def graspable(candidate: dict[str, np.ndarray]) -> bool:
        nonlocal last_grasp_probe
        if probe_data is None:
            return True
        mujoco.mj_resetDataKeyframe(model, probe_data, 0)
        for name, (x, y, yaw) in candidate.items():
            joint_name = f"{name}_free"
            if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name) < 0:
                continue
            adr = model.joint(joint_name).qposadr[0]
            probe_data.qpos[adr:adr+2] = [x, y]
            probe_data.qpos[adr+3:adr+7] = sampled_quat(name, yaw)
        source_name = spec["roles"].get("source_container")
        if source_name:
            balls = [name for name in nominal if "_ball_" in name]
            addresses = [model.joint(f"{name}_free").qposadr[0] for name in balls]
            if addresses:
                old_xy = np.asarray(poses[source_name][:2])
                new_xy = np.asarray(candidate[source_name][:2])
                delta_yaw = float(candidate[source_name][2] - poses[source_name][2])
                probe_data.qpos[np.asarray(addresses)[:, None] + np.arange(2)] = (
                    transform_container_contents_xy(
                        probe_data.qpos[np.asarray(addresses)[:, None] + np.arange(2)],
                        old_xy=old_xy, new_xy=new_xy, yaw_delta_rad=delta_yaw))
        mujoco.mj_forward(model, probe_data)
        last_grasp_probe = (_cup_handle_grasp_probe(model, probe_data, spec)
                            if cup_name else _tool_grasp_probe(model, probe_data, tool))
        return last_grasp_probe["success"]

    for seed in seeds:
        sample = sample_tabletop_layout(
            poses, ranges, footprints, table_bounds_xy=(-.10, .86, -.50, .50),
            seed=seed, min_clearance_m=.005, max_attempts=5000,
            candidate_validator=graspable)
        if last_grasp_probe is not None:
            sample["cup_handle_grasp_probe" if cup_name else
                   "tool_grasp_probe"] = last_grasp_probe
        output = source / "samples" / f"seed_{seed}"
        output.mkdir(parents=True, exist_ok=True)
        root = ET.parse(source / "scene.xml").getroot()
        for mesh in root.findall("./asset/mesh"):
            mesh.set("file", str((source / mesh.get("file")).resolve()))
        key = root.find("./keyframe/key")
        qpos = np.fromstring(key.get("qpos"), sep=" ")
        for name in names:
            x, y, yaw = sample["poses_xyyaw_rad"][name]
            if name == "dustpan":
                body = root.find("./worldbody/body[@name='dustpan']")
                original_z = np.fromstring(body.get("pos"), sep=" ")[2]
                body.set("pos", _numbers([x, y, original_z]))
                body.set("quat", _numbers(sampled_quat(name, yaw)))
            else:
                adr = model.joint(f"{name}_free").qposadr[0]
                qpos[adr:adr+2] = [x, y]
                qpos[adr+3:adr+7] = sampled_quat(name, yaw)
                if name == "brush_long":
                    delta_xy = np.asarray([x, y]) - np.asarray(poses[name][:2])
                    for stand in root.findall("./worldbody/geom"):
                        if stand.get("name", "").startswith("brush_long_stand_"):
                            stand_pos = np.fromstring(stand.get("pos"), sep=" ")
                            stand_pos[:2] += delta_xy
                            stand.set("pos", _numbers(stand_pos))
        source_name = spec["roles"].get("source_container")
        if source_name:
            old_xy = np.asarray(poses[source_name][:2])
            new_xy = np.asarray(sample["poses_xyyaw_rad"][source_name][:2])
            delta_yaw = float(sample["poses_xyyaw_rad"][source_name][2] -
                              poses[source_name][2])
            addresses = [model.joint(f"{name}_free").qposadr[0]
                         for name in nominal if "_ball_" in name]
            if addresses:
                indices = np.asarray(addresses)[:, None] + np.arange(2)
                qpos[indices] = transform_container_contents_xy(
                    qpos[indices], old_xy=old_xy, new_xy=new_xy,
                    yaw_delta_rad=delta_yaw)
        key.set("qpos", _numbers(qpos))
        ET.indent(root)
        xml = output / "scene.xml"
        ET.ElementTree(root).write(xml, encoding="unicode")
        sampled_model = mujoco.MjModel.from_xml_path(str(xml))
        data = mujoco.MjData(sampled_model)
        mujoco.mj_resetDataKeyframe(sampled_model, data, 0)
        mujoco.mj_forward(sampled_model, data)
        sample["sampled_object_poses"] = {
            name: {"T_world_object": _world_pose(data.body(name)), "frame": "robot_base"}
            for name in nominal}
        sample["initial_penetrations_over_3mm"] = [
            {"geom1": sampled_model.geom(c.geom1).name,
             "geom2": sampled_model.geom(c.geom2).name, "depth_m": -float(c.dist)}
            for c in data.contact[:data.ncon] if c.dist < -.003]
        for _ in range(500):
            mujoco.mj_step(sampled_model, data)
        sample["settled_object_poses"] = {
            name: {"T_world_object": _world_pose(data.body(name)), "frame": "robot_base"}
            for name in nominal}
        if tool in TOOL_GRIP_LOCAL_M:
            sample["post_settle_tool_grasp_probe"] = _tool_grasp_probe(
                sampled_model, data, tool)
            if not sample["post_settle_tool_grasp_probe"]["success"]:
                raise RuntimeError(f"{xml}: 落稳后的工具抓取不可达；"
                                   f"{sample['post_settle_tool_grasp_probe']['stages']}")
        elif cup_name:
            sample["post_settle_cup_handle_grasp_probe"] = _cup_handle_grasp_probe(
                sampled_model, data, spec)
            if not sample["post_settle_cup_handle_grasp_probe"]["success"]:
                raise RuntimeError(f"{xml}: 落稳后的杯把抓取不可达；"
                                   f"{sample['post_settle_cup_handle_grasp_probe']['stages']}")
        root.find("./keyframe/key").set("qpos", _numbers(data.qpos))
        ET.indent(root)
        ET.ElementTree(root).write(xml, encoding="unicode")
        if sample["initial_penetrations_over_3mm"]:
            raise ValueError(f"{xml}: 初态穿透 {sample['initial_penetrations_over_3mm'][:3]}")
        (output / "scene.json").write_text(json.dumps(
            {**spec, "scene_xml_path": str(xml), "distribution_sample": sample},
            ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        outputs.append(output)
    return outputs


__all__ = ["build_multi_task_scenes", "sample_scene_variants"]
