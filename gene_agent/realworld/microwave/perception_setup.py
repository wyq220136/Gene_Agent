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
from gene_agent.perception_modules.camera.calibration.inspection.geometry import (
    backproject_base, plane_diagnostics,
)

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
    return {"alignment": alignment.tolist(), "T_asset_handle": T_asset_handle.tolist(),
            "joints": joints, "support": scene["microwave_support"],
            "table_thickness_m": scene["table"]["thickness_m"],
            "sources": {str(p): sha256(p) for p in (scene_path, urdf_path, handle_path)}}


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


def table_from_mask(depth_m, K, transform, mask, *, thickness_m: float) -> dict:
    """用桌面分割内的 RGB-D 拟合平面与可见 footprint；结果必须人工检查。"""
    points, valid = backproject_base(depth_m, K, transform, max_depth=3.)
    mask = np.asarray(mask, bool)
    if mask.shape != valid.shape:
        raise ValueError("桌面 mask 与深度尺寸不一致")
    selected = points[valid & mask]
    plane = plane_diagnostics(selected)
    if not plane["available"] or plane["tilt_deg"] > 5. or plane["inlier_fraction"] < .7:
        raise ValueError("桌面分割内没有可靠水平平面，请检查 mask 和标定")
    normal = np.asarray(plane["normal_sdk"])
    inliers = selected[np.abs(selected @ normal + plane["offset_m"]) <= .005]
    # 桌面放置沿世界 Z；使用已验收的近水平顶面代理，不把相机裁剪范围当桌面尺寸。
    low, high = np.min(inliers[:, :2], axis=0), np.max(inliers[:, :2], axis=0)
    size = high - low
    if np.any(size < .1) or np.any(size > 3.):
        raise ValueError("桌面可见范围过小或过大，请检查完整桌面分割")
    pose = np.eye(4)
    pose[:2, 3] = (low + high) / 2
    pose[2, 3] = plane["median_height_m"] - thickness_m / 2
    return {"table_pose_world": pose.tolist(), "table_size_xy_m": size.tolist(),
            "accepted_table_thickness_m": thickness_m, "plane_diagnostics": plane,
            "source": "registered_rgbd_table_mask", "footprint_is_visible_extent": True}


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
    canvas = Image.new("RGB", (rgb.shape[1] * 2, rgb.shape[0] * 2))
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
                          "knob": "upper microwave knob including pointer", "table": "table top"}
    mask_names = ("bowl", "handle", "knob", "table") if include_knob else ("bowl", "handle", "table")
    masks = {name: folder / "masks" / f"{name}.png" for name in mask_names}
    items = [{"category": name, "prompt": prompts[name], "mask_path": str(path)} for name, path in masks.items()]
    status, response = segment_image(seg_server, capture / "rgb.png", items)
    if status != 200 or response.get("ok") is False:
        raise ValueError(f"SAM3 分割失败: {response}")
    preview = _preview_masks(rgb, masks, folder)
    table = table_from_mask(depth, camera["intrinsic_matrix"], calibration["T_sdk_camera"],
                            np.asarray(Image.open(masks["table"]).convert("L")) > 0,
                            thickness_m=assets["table_thickness_m"])
    save_json(folder / "table_measurement.json", table)
    print(f"检查 mask: {preview}\n检查桌面测量: {folder / 'table_measurement.json'}\n"
          f"桌面中心: {np.array(table['table_pose_world'])[:3, 3]}，XY尺寸: {table['table_size_xy_m']}")
    if ask(f"确认 {len(masks)} 个 mask、完整桌面边界及标定正确后输入 ACCEPT（裁剪不全则重采集）: ").strip() != "ACCEPT":
        raise ValueError("感知/桌面结果未确认；没有生成可运行的感知记录")
    unload_model(seg_server)
    meshes = {BOWL: ROOT / "data/task_1/objs/Bowl/Bowl_0/meshes/obj_mesh.obj",
              MICROWAVE: ROOT / "data/task_1/objs/Microwave/Microwave_0/meshes/handle_mesh.obj"}
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
            poses[name] = validate_pose(np.array(calibration["T_sdk_camera"]) @ validate_pose(result["T_camera_object"])).tolist()
            sessions[name] = session
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
    record = {"format": FORMAT, "accepted": True, "camera": calibration, "include_knob": bool(include_knob),
              "camera_endpoint": camera_endpoint, "pose_server": pose_server,
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
    for value in record["closed_reference_poses_sdk"].values():
        validate_pose(value)
    return path, record
