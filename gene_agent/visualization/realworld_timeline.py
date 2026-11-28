"""真机运行的 RGB-D、原始位姿、消费状态和规划索引；不请求感知服务。"""
from collections import OrderedDict
import json
from pathlib import Path

import numpy as np
from PIL import Image

from gene_agent.knowledge.geometry import validate_pose
from gene_agent.visualization.realworld_alignment import transform
from gene_agent.visualization.realworld_pick_debug import _load_trajectories


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run_root(bundle):
    bundle = Path(bundle)
    if bundle.is_file():
        bundle = bundle.parent
    return bundle.parent if bundle.name == "initial_scene" else bundle


class RealworldTimeline:
    """元数据索引常驻；点云按需解码并只缓存最近三帧。"""
    def __init__(self, bundle, scene, *, max_points=60000):
        self.bundle = Path(bundle)
        if self.bundle.is_file():
            self.bundle = self.bundle.parent
        self.root, self.scene = run_root(self.bundle), scene
        self.max_points = int(max_points)
        if self.max_points < 100:
            raise ValueError("max_points 至少为100")
        manifest = read_json(self.bundle / "manifest.json")
        self.mapping = manifest.get("file_map", {})
        self.cache, self.bounds_cache = OrderedDict(), {}
        self.world_sdk = validate_pose(scene["provenance"]["T_world_sdk"], field_name="T_world_sdk")
        self.setup = {}
        setups = [key for key in self.mapping if key.endswith("/perception_setup.json")]
        if len(setups) == 1:
            self.setup = read_json(self.resolve(setups[0]))
        self.frames, self.warnings = [], []
        self.trajectories, self.feedback = [], []
        self.report_signature = None
        self.refresh()
        self.refresh_reports()

    def resolve(self, raw):
        """优先冻结资产，再按当前 run 重定位；不拿旧主机路径的同名文件替换。"""
        if str(raw) in self.mapping:
            return self.bundle / self.mapping[str(raw)]
        text = str(raw).replace("\\", "/")
        for marker in ("/observations/", "/initial_scene/"):
            if marker in text:
                relative = text.split(marker, 1)[1]
                target = self.root / marker.strip("/") / relative
                if target.is_file():
                    return target
        direct = Path(raw)
        if not direct.is_absolute():
            direct = self.root / direct
        return direct

    def refresh(self):
        rows = []
        for path in sorted(self.root.glob("**/observations/*/capture/camera.json")):
            if "initial_scene" in path.relative_to(self.root).parts:
                continue
            folder = path.parent.parent
            try:
                camera = read_json(path)
                if not all((path.parent / name).is_file() for name in ("rgb.png", "depth.png")):
                    continue  # 实时写盘尚未完成，不把半帧加入播放器。
                provenance_path = folder / "scene/observed_scene_provenance.json"
                provenance = read_json(provenance_path) if provenance_path.is_file() else {}
                context_path = path.parent / "viewer_context.json"
                context = read_json(context_path) if context_path.is_file() else {}
                if context and context.get("expected_serial") != camera.get("serial"):
                    raise ValueError("相机序列号与本次记录上下文不一致")
                tracked = list((folder / "tracked/camera").glob("*.json"))
                extrinsic = None
                if context:
                    extrinsic = context["T_sdk_camera"]
                elif len(tracked) == 1:
                    record = read_json(tracked[0])
                    extrinsic = record.get(f"{provenance.get('source_side', camera.get('side', 'right'))}_extrinsic_matrix")
                if extrinsic is None:
                    extrinsic = self.setup.get("camera", {}).get("T_sdk_camera")
                if extrinsic is None:
                    raise ValueError("缺少已记录相机外参，不能把此帧放入统一 world")
                world_sdk = np.asarray(provenance.get("T_world_sdk", context.get("T_world_sdk", self.world_sdk)))
                if not np.allclose(world_sdk, self.world_sdk, atol=1e-9, rtol=0):
                    raise ValueError("运行中 T_world_sdk 改变，不能静默重对齐")
                pose = self.world_sdk @ validate_pose(extrinsic, field_name="T_sdk_camera")
                if self.setup.get("camera", {}).get("serial", camera.get("serial")) != camera.get("serial"):
                    raise ValueError("相机序列号变化")
                watched = [path, context_path, provenance_path]
                watched += list((folder / "capture").glob("session_recovery/**/mask.png"))
                watched += list((folder / "tracked/objs").glob("**/pose.json"))
                revision = tuple((str(p.relative_to(folder)), p.stat().st_mtime_ns, p.stat().st_size)
                                 for p in watched if p.is_file())
                rows.append(dict(folder=folder, camera=camera, provenance=provenance,
                    T_world_camera=pose, frame_number=int(camera["frame_number"]),
                    timestamp_ms=float(camera["captured_at_unix_ms"]),
                    label=str(folder.relative_to(self.root)), context=context,
                    revision=revision))
            except (OSError, KeyError, TypeError, ValueError) as error:
                message = f"{folder}: {error}"
                if message not in self.warnings:
                    self.warnings.append(message)
        rows.sort(key=lambda row: (row["timestamp_ms"], row["label"]))
        self.frames = rows
        return len(rows)

    def refresh_reports(self):
        paths = set(self.root.rglob("*_result.json")) | set(self.root.rglob("stage.json"))
        paths.update(self.root.rglob("right_gripper_preparation.json"))
        paths = sorted(p for p in paths if "initial_scene" not in p.relative_to(self.root).parts)
        signature = [(str(p), p.stat().st_mtime_ns, p.stat().st_size) for p in paths]
        if signature == self.report_signature:
            return False
        self.report_signature = signature
        self.trajectories = _load_trajectories(self.root, self.root / "_viewer_index", {})
        feedback, seen = [], set()
        def visit(value, source, context=None):
            if isinstance(value, dict):
                if value.get("source_format") == "realworld_gripper_stage" or "target_sdk_position" in value:
                    context = value
                sample = value.get("sdk_reading")
                if isinstance(sample, dict) and sample.get("sampled_at_unix_ms") is not None:
                    timestamp = float(sample["sampled_at_unix_ms"])
                    width = float(sample["width_m"])
                    group = (context or {}).get("group")
                    key = (timestamp, group, width)
                    if group in ("left", "right") and np.isfinite([timestamp, width]).all() and key not in seen:
                        seen.add(key)
                        feedback.append(dict(timestamp_ms=timestamp, width_m=width, group=group,
                            sdk_reading=sample, source=str(source),
                            joint_state=(context or {}).get("measured_joint_start")))
                for child in value.values():
                    visit(child, source, context)
            elif isinstance(value, list):
                for child in value:
                    visit(child, source, context)
        for path in paths:
            try:
                visit(read_json(path), path)
            except (OSError, ValueError, KeyError, TypeError):
                continue
        self.feedback = sorted(feedback, key=lambda row: row["timestamp_ms"])
        return True

    def cloud(self, index):
        row = self.frames[index]
        key = (str(row["folder"]), row["revision"])
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        capture, camera = row["folder"] / "capture", row["camera"]
        rgb = np.asarray(Image.open(capture / "rgb.png").convert("RGB"))
        depth = np.asarray(Image.open(capture / "depth.png"), dtype=float)
        scale = {"m": 1., "meter": 1., "meters": 1., "mm": .001,
                 "millimeter": .001, "millimeters": .001}.get(camera.get("depth_unit"))
        if scale is None:
            raise ValueError("深度单位缺失/未知，不能推断比例")
        depth *= scale
        K = np.asarray(camera["intrinsic_matrix"], dtype=float)
        if (K.shape != (3, 3) or not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0
                or depth.shape != rgb.shape[:2]):
            raise ValueError("RGB-D 尺寸或内参无效")
        v, u = np.indices(depth.shape)
        points = np.stack(((u-K[0, 2])*depth/K[0, 0], (v-K[1, 2])*depth/K[1, 1], depth), axis=-1)
        valid = np.flatnonzero(np.isfinite(depth) & (depth > .1) & (depth < 3.))
        if not len(valid):
            raise ValueError("本帧无有效深度")
        ids = valid[np.linspace(0, len(valid)-1, min(len(valid), self.max_points), dtype=int)]
        overlay = rgb.copy()
        masks = []
        for path in sorted(capture.glob("session_recovery/**/mask.png")):
            mask = np.asarray(Image.open(path).convert("L")) > 0
            if mask.shape != depth.shape:
                raise ValueError("本帧 SAM3 mask 尺寸不匹配")
            overlay[mask] = (overlay[mask]*.4 + np.array([255, 70, 30])*.6).astype(np.uint8)
            masks.append(str(path.relative_to(capture)))
        payload = dict(points=transform(points.reshape(-1, 3)[ids], row["T_world_camera"]),
                       colors=rgb.reshape(-1, 3)[ids], rgb=rgb, overlay=overlay,
                       masks=masks, ids=ids, depth_m=depth.reshape(-1)[ids])
        self.cache[key] = payload
        while len(self.cache) > 3:
            self.cache.popitem(last=False)
        return payload

    def objects(self, index):
        """原始 FoundationPose 不经门角/Concept 修正；复用状态单独列出。"""
        row = self.frames[index]
        objects = []
        for path in sorted((row["folder"] / "tracked/objs").glob("**/pose.json")):
            value = read_json(path)
            if value.get("_capture", {}).get("frame_number") != row["frame_number"]:
                raise ValueError(f"FoundationPose 与 RGB-D 帧不一致: {path}")
            timestamp = value.get("_capture", {}).get("timestamp_ms")
            if timestamp is not None and abs(float(timestamp)-row["timestamp_ms"]) > 1.:
                raise ValueError("FoundationPose 与 RGB-D 采集时间不一致")
            keys = [key for key in value if key.startswith("camera_parameters_")]
            if len(keys) != 1:
                raise ValueError("FoundationPose 相机 pose 字段缺失/不唯一")
            object_id = value["object_id"]
            pose = row["T_world_camera"] @ validate_pose(value[keys[0]][0], field_name=object_id)
            mesh = self.setup.get("meshes", {}).get(object_id)
            bounds = None
            if mesh:
                mesh_file = self.resolve(mesh)
                if mesh_file.is_file():
                    if mesh not in self.bounds_cache:
                        import trimesh
                        self.bounds_cache[mesh] = np.asarray(trimesh.load(mesh_file, force="mesh", process=False).bounds)
                    bounds = self.bounds_cache[mesh]
            objects.append(dict(object_id=object_id, pose=pose, bounds=bounds,
                                source=str(path), method="FoundationPose 当前帧"))
        return objects

    def summary(self):
        return dict(observation_count=len(self.frames), trajectory_count=len(self.trajectories), gripper_feedback_count=len(self.feedback),
                    world_frame=self.scene["snapshot"].get("world_frame"),
                    T_world_sdk=self.world_sdk.tolist(), warnings=self.warnings,
                    frames=[dict(label=row["label"], frame_number=row["frame_number"],
                                 timestamp_ms=row["timestamp_ms"],
                                 has_robot_state=bool(row["provenance"].get("joint_positions_by_group_rad")))
                            for row in self.frames])


def bbox_segments(bounds, pose):
    lo, hi = np.asarray(bounds)
    corners = np.array([[hi[k] if (i >> k) & 1 else lo[k] for k in range(3)] for i in range(8)])
    world = transform(corners, pose)
    return np.array([[world[i], world[i ^ (1 << k)]] for i in range(8) for k in range(3) if not (i >> k) & 1])


def apply_observed_robot(model, data, base_qpos, provenance):
    """离散消费状态直接设 FK；不插值为连续真机轨迹，也不裁剪反馈开口。"""
    import mujoco
    data.qpos[:] = base_qpos
    for group, positions in provenance.get("joint_positions_by_group_rad", {}).items():
        values = np.asarray(positions, dtype=float)
        if values.shape != (7,) or not np.isfinite(values).all():
            raise ValueError("观测关节角必须是七个有限弧度值")
        for i, q in enumerate(values, 1):
            data.qpos[model.joint(f"{group}_joint_{i}").qposadr[0]] = q
    for group, width in provenance.get("gripper_widths_m", {}).items():
        width = float(width)
        if not np.isfinite(width) or width < 0:
            raise ValueError("程序消费的夹爪开口无效")
        for side in ("positive", "negative"):
            data.qpos[model.joint(f"{group}_gripper_{side}_slide").qposadr[0]] = width / 2
    mujoco.mj_forward(model, data)
