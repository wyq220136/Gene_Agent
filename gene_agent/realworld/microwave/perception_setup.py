"""真机启动前的感知准备：消费标定产物，生成可复用记录，不要求手填站点 JSON。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from uuid import uuid4
from xml.etree import ElementTree as ET

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from gene_agent.knowledge.geometry import validate_pose
from gene_agent.realworld.pose_tracking import capture_frozen_scene_frame
from gene_agent.perception_modules.camera.calibration.inspection.geometry import backproject_base
from gene_agent.perception_modules.pose_est.task_pose_est import pose_mesh_path

ROOT = Path(__file__).resolve().parents[3]
BOWL = "Bowl/Bowl_0"
MICROWAVE = "Microwave/Microwave_0"
KNOB = "Microwave/Knob_0"
FORMAT = "microwave_perception_setup_v1"


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def load_camera_calibration(path: str | Path) -> dict:
    """接受现有 manual/board-camera 的输出；不另造外参格式。"""
    path = Path(path).expanduser().resolve()
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("quality_passed") is False:
        raise ValueError("相机标定未通过质量检查")
    side = data.get("robot_group", data.get("calibrated_side", data.get("side", "right")))
    if side != "right":
        raise ValueError("当前 Task1 使用右臂 SDK base，请提供右臂固定相机外参")
    keys = [k for k in ("right_extrinsic_matrix", "extrinsic_matrix", "T_sdk_third") if k in data]
    if not keys:
        raise ValueError("标定文件缺少相机到右臂 SDK base 的外参")
    transform = validate_pose(data[keys[0]], field_name=keys[0])
    for key in keys[1:]:
        if not np.allclose(transform, validate_pose(data[key]), atol=1e-8, rtol=0):
            raise ValueError("标定文件的外参字段互相矛盾")
    serial = data.get("serial", data.get("camera_serial"))
    if not isinstance(serial, str) or not serial.strip():
        raise ValueError("标定文件缺少相机 serial，请使用标定向导正式输出")
    K = np.asarray(data.get("intrinsic_matrix"), float)
    if K.shape != (3, 3) or not np.isfinite(K).all() or min(K[0, 0], K[1, 1]) <= 0:
        raise ValueError("相机标定缺少有效 intrinsic_matrix")
    if not np.allclose(K[2], [0, 0, 1]):
        raise ValueError("相机内参最后一行无效")
    distortion = np.asarray(data.get("D", data.get("distortion_coefficients", [])), float)
    if not np.isfinite(distortion).all() or np.any(distortion != 0):
        raise ValueError("需要已对齐且去畸变的 RGB-D，不能忽略非零畸变")
    return {"path": str(path), "sha256": sha256(path), "serial": serial,
            "T_sdk_camera": transform.tolist(), "intrinsic_matrix": K.tolist(),
            "width": data.get("width"), "height": data.get("height")}


def check_capture_camera(camera: dict, calibration: dict, shape: tuple[int, int]) -> None:
    if camera.get("serial") != calibration["serial"]:
        raise ValueError("采集相机 serial 与标定不一致")
    if not np.allclose(camera["intrinsic_matrix"], calibration["intrinsic_matrix"], atol=1e-6, rtol=0):
        raise ValueError("采集内参与标定不一致，请检查分辨率和相机配置")
    height, width = shape
    if (calibration["width"] not in (None, width) or calibration["height"] not in (None, height)):
        raise ValueError("采集分辨率与标定不一致")


def task1_table_geometry() -> dict:
    """只读取已迁移的 Task1 桌面 XY 和厚度；旧 Z 不能充当本次测量。"""
    path = ROOT / "data/task_1/knowledge/legacy_extract_visual_table.json"
    table = json.loads(path.read_text(encoding="utf-8"))["legacy_virtual_table"]
    center, size = np.asarray(table["center_world_m"], float), np.asarray(table["size_xyz_m"], float)
    if (center.shape != (3,) or size.shape != (3,) or not np.isfinite(center).all()
            or not np.isfinite(size).all() or np.any(size <= 0)):
        raise ValueError("已有 Task1 桌面几何的中心或尺寸无效")
    return {"center_xy_m": center[:2].tolist(), "size_xy_m": size[:2].tolist(),
            "thickness_m": float(size[2]), "path": str(path), "sha256": sha256(path)}


def asset_geometry() -> dict:
    """复用已配对的旧感知 frame 与标定资产 frame；不修改操作 profile。"""
    scene_path = ROOT / "config/scenes/task_1.json"
    scene = json.loads(scene_path.read_text(encoding="utf-8"))
    microwave = scene["objects"]["microwave"]
    old = validate_pose(microwave["legacy_captured_pose"])
    model = validate_pose(microwave["captured_pose"])
    alignment = validate_pose(np.linalg.inv(old) @ model)
    urdf_path = ROOT / scene["microwave_asset"]["urdf"]
    urdf = ET.parse(urdf_path).getroot()
    joints = {}
    for label, name in (("door", "door_hinge"), ("knob", "power_knob_rotation")):
        joint = urdf.find(f"joint[@name='{name}']")
        origin = joint.find("origin")
        T = np.eye(4)
        T[:3, :3] = Rotation.from_euler("xyz", np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")).as_matrix()
        T[:3, 3] = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
        axis = T[:3, :3] @ np.fromstring(joint.find("axis").get("xyz"), sep=" ")
        limit = joint.find("limit")
        joints[label] = {"T_asset_joint": T.tolist(), "axis_asset": axis.tolist(),
                         "limits_rad": [float(limit.get("lower")), float(limit.get("upper"))]}
    handle_path = ROOT / "assets/objects/microwave_calibrated/handle_concept.json"
    handle = json.loads(handle_path.read_text(encoding="utf-8"))
    T_asset_handle = validate_pose(handle["conceptualization"][0]["parameters"]["pose_matrix"])
    table = task1_table_geometry()
    return {"alignment": alignment.tolist(), "T_asset_handle": T_asset_handle.tolist(),
            "joints": joints, "support": scene["microwave_support"],
            "table_geometry": {key: table[key] for key in ("center_xy_m", "size_xy_m")},
            "table_thickness_m": table["thickness_m"],
            "sources": {**{str(p): sha256(p) for p in (scene_path, urdf_path, handle_path)},
                        table["path"]: table["sha256"]}}


def export_knob_mesh(output: Path) -> Path:
    """将当前 URDF 的旋钮 visual 合并为关节局部零角 mesh，保留指针方向。"""
    import trimesh
    scene = json.loads((ROOT / "config/scenes/task_1.json").read_text(encoding="utf-8"))
    urdf_path = ROOT / scene["microwave_asset"]["urdf"]
    urdf = ET.parse(urdf_path).getroot()
    pieces = []
    for visual in urdf.findall("link[@name='power_knob']/visual"):
        spec = visual.find("geometry/mesh")
        if spec is None:
            raise ValueError("旋钮 visual 不是 mesh，需更新感知资产导出")
        mesh = trimesh.load(urdf_path.parent / spec.get("filename"), force="mesh", process=False)
        mesh.apply_scale(np.fromstring(spec.get("scale", "1 1 1"), sep=" "))
        origin = visual.find("origin")
        if origin is not None:
            T = np.eye(4)
            T[:3, :3] = Rotation.from_euler("xyz", np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")).as_matrix()
            T[:3, 3] = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
            mesh.apply_transform(T)
        pieces.append(mesh)
    if not pieces:
        raise ValueError("资产缺少 power_knob visual，不能静默假定旋钮角为零")
    path = output / "knob_mesh.obj"
    trimesh.util.concatenate(pieces).export(path)
    return path


def table_from_scene_geometry(depth_m, K, transform, geometry: dict, *, thickness_m: float) -> dict:
    """沿用 AffordToolAct：固定矩形范围内的点云计算高度，不做桌面分割或 pose estimation。

    点云只通过标定外参转换一次到右臂 SDK base。保留固定桌面范围；
    XY 范围不随相机裁剪或分割结果变化；Z 和法向使用观测平面，不强制压平。
    """
    center, size = np.asarray(geometry["center_xy_m"], float), np.asarray(geometry["size_xy_m"], float)
    thickness_m = float(thickness_m)
    if (center.shape != (2,) or size.shape != (2,) or not np.isfinite(center).all()
            or not np.isfinite(size).all() or np.any(size <= 0)
            or not np.isfinite(thickness_m) or thickness_m <= 0):
        raise ValueError("桌面固定几何的中心、尺寸或厚度无效")
    points, valid = backproject_base(depth_m, K, transform, max_depth=3.)
    points = points[valid]
    # 与旧入口 load_rgbd_scene_points 一样，在裁剪前均匀抽样最多五万个点。
    if len(points) > 50000:
        points = points[np.linspace(0, len(points) - 1, 50000, dtype=np.int64)]
    inside = np.all((points[:, :2] >= center - size / 2) & (points[:, :2] <= center + size / 2), axis=1)
    selected = points[inside]
    if len(selected) < 10:
        raise ValueError(f"固定桌面范围内有效深度点太少 ({len(selected)})，请检查深度、标定和场景位置")
    from .geometry_alignment import fitted_table_pose
    result = fitted_table_pose(selected, np.asarray(transform)[:3, 3], center, size, thickness_m)
    result["point_counts"] = {"scene": len(points), "inside_geometry": len(selected),
                              "inliers": result["inlier_count"]}
    return result


def _preview_masks(rgb, masks: dict, output: Path) -> Path:
    tiles = []
    from PIL import ImageDraw
    for name, path in masks.items():
        mask = np.asarray(Image.open(path).convert("L")) > 0
        if mask.shape != rgb.shape[:2] or not mask.any():
            raise ValueError(f"{name} mask 为空或尺寸不匹配")
        overlay = rgb.copy()
        overlay[mask] = (overlay[mask] * .45 + np.array([255, 40, 30]) * .55).astype(np.uint8)
        tile = Image.fromarray(overlay)
        ImageDraw.Draw(tile).text((10, 10), name, fill="white", stroke_width=2, stroke_fill="black")
        tiles.append(tile)
    canvas = Image.new("RGB", (rgb.shape[1] * 2, rgb.shape[0] * ((len(tiles) + 1) // 2)))
    for i, tile in enumerate(tiles):
        canvas.paste(tile, ((i % 2) * rgb.shape[1], (i // 2) * rgb.shape[0]))
    path = output / "masks_preview.png"
    canvas.save(path)
    return path


def prepare_perception(*, camera_params: str | Path, output: str | Path,
                       camera_endpoint="tcp://127.0.0.1:18082",
                       pose_server="http://127.0.0.1:18081",
                       seg_server="http://127.0.0.1:18080", prompts=None, ask=input,
                       include_knob: bool = False) -> Path:
    """先确认关闭参考与 mask，再自动注册；只调用感知服务，不连接机械臂。"""
    from gene_agent.perception_modules.pose_est.pose_client import PoseClient
    from gene_agent.perception_modules.segmentation.seg_client import segment_image, unload_model
    calibration = load_camera_calibration(camera_params)
    assets = asset_geometry()
    folder = Path(output).expanduser().resolve()
    if folder.exists() and any(folder.iterdir()):
        raise ValueError("感知准备目录已有内容，请使用新目录")
    confirmation = "CLOSED_ZERO" if include_knob else "CLOSED"
    reference_prompt = ("请关紧门、将上方旋钮置零、固定柜体和相机" if include_knob else
                        "请关紧门、固定柜体和相机（当前不感知或操作旋钮）")
    if ask(f"{reference_prompt}；确认后输入 {confirmation}: ").strip() != confirmation:
        raise ValueError("未确认本次关闭参考，不采集")
    capture = capture_frozen_scene_frame(output_dir=folder / "capture", endpoint=camera_endpoint,
                                         expected_serial=calibration["serial"])
    rgb = np.asarray(Image.open(capture / "rgb.png").convert("RGB"))
    depth = np.asarray(Image.open(capture / "depth.png"), dtype=float) * .001
    camera = json.loads((capture / "camera.json").read_text(encoding="utf-8"))
    check_capture_camera(camera, calibration, depth.shape)
    folder.joinpath("masks").mkdir()
    prompts = prompts or {"bowl": "bowl", "handle": "microwave door handle",
                          "knob": "upper microwave knob including pointer"}
    mask_names = ("bowl", "handle", "knob") if include_knob else ("bowl", "handle")
    masks = {name: folder / "masks" / f"{name}.png" for name in mask_names}
    items = [{"category": name, "prompt": prompts[name], "mask_path": str(path)} for name, path in masks.items()]
    status, response = segment_image(seg_server, capture / "rgb.png", items)
    if status != 200 or response.get("ok") is False:
        raise ValueError(f"SAM3 分割失败: {response}")
    preview = _preview_masks(rgb, masks, folder)
    table = table_from_scene_geometry(depth, camera["intrinsic_matrix"], calibration["T_sdk_camera"],
                                      assets["table_geometry"], thickness_m=assets["table_thickness_m"])
    table.update(source_frame_number=camera["frame_number"],
                 calibration_path=calibration["path"], calibration_sha256=calibration["sha256"])
    save_json(folder / "table_measurement.json", table)
    if not table["sdk_up_consistent"]:
        from .geometry_alignment import CameraSDKAxisError
        error = CameraSDKAxisError("camera_sdk_axis_inconsistent: 可见桌面上法向指向 SDK -Z；"
                                   "标定板轴与 SDK 轴约定不一致。实测平面已保存，不能翻转机器人或继续生成真机目标。")
        error.diagnostics = table
        raise error
    print(f"检查物体 mask: {preview}\n桌面计算记录: {folder / 'table_measurement.json'}\n"
          f"桌面顶高: {table['table_top_z_m']:.4f} m，固定中心 XY: {assets['table_geometry']['center_xy_m']}，"
          f"固定 XY 尺寸: {table['table_size_xy_m']}")
    if ask(f"确认 {len(masks)} 个物体 mask、标定及计算桌面与现场相符后输入 ACCEPT（不分割桌面）: ").strip() != "ACCEPT":
        raise ValueError("感知/桌面结果未确认；没有生成可运行的感知记录")
    unload_model(seg_server)
    # 与 AffordToolAct 的批量位姿入口共用 mesh 选择：把手 mask 只配把手 mesh。
    # 已有任务资产直接注册，不进入 SAM3D、Concept 或额外桌面感知。
    meshes = {name: pose_mesh_path(name.split("/")[0], ROOT / "data/task_1/objs" / name)
              for name in (BOWL, MICROWAVE)}
    if include_knob:
        meshes[KNOB] = export_knob_mesh(folder)
    mask_keys = {BOWL: "bowl", MICROWAVE: "handle", KNOB: "knob"}
    client = PoseClient(pose_server)
    token = uuid4().hex[:12]
    sessions, poses = {}, {}
    registered = []
    try:
        for name, mesh in meshes.items():
            session = f"{name}-{token}"
            result = client.register(session_id=session, rgb_path=capture / "rgb.png",
                                     depth_path=capture / "depth.png", camera_path=capture / "camera.json",
                                     mask_path=masks[mask_keys[name]], mesh_path=mesh, camera_id="0")
            if (result.get("ok") is not True or result.get("session_id") != session
                    or str(result.get("camera_id", "0")) != "0"):
                raise ValueError(f"{name} 注册失败或 session 不一致")
            registered.append(session)
            registration_path = folder / "registrations" / f"{name}.json"
            registration_path.parent.mkdir(parents=True, exist_ok=True)
            save_json(registration_path, {**result, "_capture": {
                "frame_number": camera["frame_number"], "serial": camera["serial"]}})
            poses[name] = validate_pose(np.array(calibration["T_sdk_camera"]) @ validate_pose(result["T_camera_object"])).tolist()
            sessions[name] = session
        # 把手几何存在完全等价的半转，单独注册它不能确定整台柜体方向。
        import trimesh
        from copy import deepcopy
        from .geometry_alignment import handle_halfturn, canonical_handle_pose
        handle_mesh = trimesh.load(meshes[MICROWAVE], force="mesh", process=False)
        halfturn, symmetry_residual = handle_halfturn(handle_mesh.vertices, handle_mesh.centroid)
        disambiguation = {"T_mesh_equivalent_halfturn": halfturn.tolist(),
            "T_mesh_asset": assets["alignment"], "observed_up_sdk": table["normal_world"],
            "mesh_sha256": sha256(meshes[MICROWAVE]), "symmetry_max_vertex_error_m": symmetry_residual}
        raw_reference = deepcopy(poses)
        normalized, pose_diagnostics = canonical_handle_pose(poses[MICROWAVE], disambiguation)
        poses[MICROWAVE] = normalized.tolist()
        save_json(folder / "microwave_pose_disambiguation.json", pose_diagnostics)
        # 同一冻结帧中重建机身，不新增拍照/分割/注册或现场确认。
        # 未被现有门限采纳时保留原参考，不能让可选辅助新增退出条件。
        try:
            from .door_plane import DoorPlaneObserver
            rebuilt, frame_reconstruction = DoorPlaneObserver(assets).reconstruct_closed(
                poses[MICROWAVE], capture, calibration["T_sdk_camera"],
                np.asarray(table["table_pose_world"])[:3, 2])
            poses[MICROWAVE] = rebuilt.tolist()
        except (ValueError, TypeError, KeyError, AttributeError, OSError, RuntimeError) as error:
            frame_reconstruction = {
                "method": "closed_door_plane_table_up_and_handle_center",
                "applied": False, "reason": "auxiliary_reconstruction_unavailable",
                "fallback": "original_registered_reference",
                "error_type": type(error).__name__, "detail": str(error)}
        print(f"[perception] microwave_frame_reconstruction: "
              f"applied={frame_reconstruction['applied']} reason={frame_reconstruction['reason']}")
        # CLOSED/ACCEPT 只确认门零位和 mask，不能替代碗 pose 的几何一致性。
        from .bowl_pose_check import check_bowl_pose_table, BowlPoseTableError
        try:
            consistency = check_bowl_pose_table(
                poses[BOWL], table["table_pose_world"], table["accepted_table_thickness_m"])
        except BowlPoseTableError as error:
            save_json(folder / "bowl_pose_table_consistency.json", error.diagnostics)
            raise
        save_json(folder / "bowl_pose_table_consistency.json", consistency)
    except Exception:
        for session in registered:
            try:
                client.stop(session)
            except Exception:
                pass
        raise
    sources = {**assets["sources"], calibration["path"]: calibration["sha256"]}
    sources.update({str(path): sha256(path) for path in [*meshes.values(), *masks.values(),
                   capture / "rgb.png", capture / "depth.png", capture / "camera.json"]})
    record = {"format": FORMAT, "geometry_contract_version": 2,
              "raw_closed_reference_poses_sdk": raw_reference,
              "microwave_pose_disambiguation": disambiguation,
              "microwave_frame_reconstruction": frame_reconstruction,
              "microwave_pose_diagnostics": pose_diagnostics, "accepted": True, "camera": calibration, "include_knob": bool(include_knob),
              "camera_endpoint": camera_endpoint, "pose_server": pose_server,
              "session_recovery": {name: {"seg_server": seg_server,
                  "prompt": prompts[mask_keys[name]], "category": mask_keys[name]}
                  for name in sessions},
              "sessions": sessions, "meshes": {key: str(path) for key, path in meshes.items()},
              "closed_reference_poses_sdk": poses, "reference_frame": camera,
              "assets": assets, "table": table, "input_sha256": sources,
              "reference_confirmation": confirmation, "mask_confirmation": "ACCEPT"}
    # 只有完成感知和确认才写入口文件；失败目录仅保存诊断，不被运行命令接受。
    path = folder / "perception_setup.json"
    save_json(path, record)
    return path


def load_perception_setup(path: str | Path) -> tuple[Path, dict]:
    path = Path(path).expanduser().resolve()
    if path.is_dir():
        path = path / "perception_setup.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("geometry_contract_version") != 2:
        raise ValueError("旧准备记录未测量桌面法向或消除把手半转歧义；请重新 prepare_microwave_realworld.py，旧日志仍可只读查看")
    # 兼容旧版三个 session 的记录；新记录必须明确保存所选感知范围。
    include_knob = record.get("include_knob", True)
    if not isinstance(include_knob, bool):
        raise ValueError("感知记录的 include_knob 必须为布尔值")
    expected = {BOWL, MICROWAVE, KNOB} if include_knob else {BOWL, MICROWAVE}
    if (record.get("format") != FORMAT or record.get("accepted") is not True
            or record.get("reference_confirmation") != ("CLOSED_ZERO" if include_knob else "CLOSED")
            or record.get("mask_confirmation") != "ACCEPT"):
        raise ValueError("需要 prepare_microwave_realworld.py 自动生成并确认的感知记录")
    if set(record["sessions"]) != expected or len(set(record["sessions"].values())) != len(expected):
        raise ValueError("感知 session 必须与准备时选择的对象一致，且每个对象使用独立 session")
    if set(record["meshes"]) != expected or set(record["closed_reference_poses_sdk"]) != expected:
        raise ValueError("感知 mesh、关闭参考和 session 的对象集合必须一致")
    if not record.get("input_sha256"):
        raise ValueError("感知记录缺少输入来源哈希")
    for source, digest in record["input_sha256"].items():
        if sha256(source) != digest:
            raise ValueError(f"感知或标定输入已改变，需重新准备: {source}")
    current = load_camera_calibration(record["camera"]["path"])
    if current != record["camera"]:
        raise ValueError("感知记录与相机标定不一致")
    # 同一模型的静态变换来自代码内的正式资产，不能在记录里手工替换。
    if record["assets"] != asset_geometry():
        raise ValueError("感知记录与当前任务资产不一致，请重新准备")
    table = validate_pose(record["table"]["table_pose_world"])
    correction = record["microwave_pose_disambiguation"]
    if (table[2, 2] <= 0
            or not np.allclose(table[:3, 2], correction["observed_up_sdk"], atol=1e-8)
            or not np.allclose(record["assets"]["alignment"], correction["T_mesh_asset"], atol=1e-8)
            or correction["mesh_sha256"] != sha256(record["meshes"][MICROWAVE])):
        raise ValueError("桌面上方向、资产对齐或把手消歧 mesh 来源不一致，需重新准备")
    turn = validate_pose(correction["T_mesh_equivalent_halfturn"])
    if not np.allclose(turn@turn, np.eye(4), atol=1e-8):
        raise ValueError("把手等价半转不是有效对合变换")
    for value in record["closed_reference_poses_sdk"].values():
        validate_pose(value)
    return path, record
