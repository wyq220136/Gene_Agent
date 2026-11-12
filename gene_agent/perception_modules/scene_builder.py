"""从一帧 RGB-D 的多物体 mask 构建可导入双 viewer 的桌面场景。"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.error
import urllib.request

import numpy as np
from PIL import Image
import trimesh

from gene_agent.perception_modules.paths import PROJECT_ROOT
from gene_agent.perception_modules.recon_3d.recon_client import post_json as reconstruct_request
from gene_agent.perception_modules.recon_3d.recon_client import unload_model as unload_recon
from gene_agent.perception_modules.segmentation.seg_client import segment_image
from gene_agent.perception_modules.segmentation.seg_client import unload_model as unload_seg
from gene_agent.scene.reconstructed_tabletop import build_tabletop_scene


PYTHON_ENVS = {
    "seg": Path("/home/ubuntu/miniconda3/envs/sam3/bin/python"),
    "recon": Path("/home/ubuntu/miniconda3/envs/sam3d/bin/python"),
}
SERVICES = {
    "seg": ("gene_agent.perception_modules.segmentation.seg_server", "http://127.0.0.1:18080", "seg_server"),
    "recon": ("gene_agent.perception_modules.recon_3d.recon_server", "http://127.0.0.1:18083", "recon_3d_server"),
}
PROMPTS = {"dustpan": "a white plastic dustpan", "cube": "a small green cube"}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rgbd(mask_path: Path, depth_path: Path, camera_path: Path):
    mask = np.asarray(Image.open(mask_path).convert("L")) > 0
    depth = np.asarray(Image.open(depth_path), dtype=np.float64)
    camera = json.loads(camera_path.read_text(encoding="utf-8"))
    if camera.get("depth_unit", "millimeter") != "millimeter":
        raise ValueError("本入口要求毫米单位的 depth PNG")
    intrinsic = np.asarray(camera["intrinsic_matrix"], dtype=np.float64)
    if mask.shape != depth.shape or intrinsic.shape != (3, 3):
        raise ValueError("mask、depth 或相机内参尺寸不一致")
    valid = mask & np.isfinite(depth) & (depth > 0)
    if np.count_nonzero(valid) < 20:
        raise ValueError(f"{mask_path} 内有效深度点少于 20 个")
    return mask, depth * 0.001, intrinsic, valid


def observed_center_camera(mask_path: str | Path, depth_path: str | Path,
                           camera_path: str | Path) -> list[float]:
    """仅保存观测中心 T_camera_point，不把它误称为完整 object pose。"""
    mask, depth, intrinsic, valid = _rgbd(Path(mask_path), Path(depth_path), Path(camera_path))
    ys, xs = np.where(valid)
    z = float(np.median(depth[valid]))
    u, v = float(np.median(xs)), float(np.median(ys))
    return [float((u - intrinsic[0, 2]) * z / intrinsic[0, 0]),
            float((v - intrinsic[1, 2]) * z / intrinsic[1, 1]), z]


def make_cube_from_mask(mask_path: str | Path, depth_path: str | Path,
                        camera_path: str | Path, output_path: str | Path,
                        *, scene_scale_factor: float = 1.) -> dict:
    """以 mask 与深度估计边长；仿真缩放显式记录，保留原始估计。"""
    mask, depth, intrinsic, valid = _rgbd(Path(mask_path), Path(depth_path), Path(camera_path))
    ys, xs = np.where(mask)
    z = float(np.median(depth[valid]))
    width = float((xs.max() - xs.min() + 1) * z / intrinsic[0, 0])
    height = float((ys.max() - ys.min() + 1) * z / intrinsic[1, 1])
    side = float(np.median((width, height)))
    if not 0.01 <= side <= 0.20:
        raise ValueError(f"mask 推得方块边长 {side:.3f} m 超出合理范围")
    if not np.isfinite(scene_scale_factor) or scene_scale_factor <= 0:
        raise ValueError("scene_scale_factor 必须为有限正数")
    scene_side = side * scene_scale_factor
    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    trimesh.creation.box(extents=[scene_side] * 3).export(output)
    return {"mesh_path": str(output), "side_m": scene_side,
            "inferred_side_m": side, "scene_scale_factor": scene_scale_factor,
            "source_kind": "mask_depth_metric_box_user_scaled" if scene_scale_factor != 1.
                           else "mask_depth_metric_box", "projected_width_m": width,
            "projected_height_m": height}


class ModelService:
    """只管理本命令创建的模型服务；已有服务只调用、不终止。"""

    def __init__(self, kind: str, log_dir: Path):
        self.kind = kind
        self.module, self.url, self.service_name = SERVICES[kind]
        self.log_dir = log_dir
        self.process: subprocess.Popen | None = None
        self._log = None

    def _health(self) -> dict | None:
        try:
            with urllib.request.urlopen(self.url + "/health", timeout=1) as response:
                return json.load(response)
        except (urllib.error.URLError, TimeoutError):
            return None

    def __enter__(self):
        health = self._health()
        if health is not None:
            if health.get("service") != self.service_name:
                raise RuntimeError(f"{self.url} 已被其他服务占用: {health}")
            return self
        python = PYTHON_ENVS[self.kind]
        if not python.is_file():
            raise FileNotFoundError(f"缺少 {self.kind} 环境: {python}")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._log = (self.log_dir / f"{self.kind}_server.log").open("a", encoding="utf-8")
        env = os.environ.copy()
        env.update(PYTHONPATH="", PYTHONNOUSERSITE="1")
        self.process = subprocess.Popen(
            [str(python), "-m", self.module, "--device", "cuda"],
            cwd=PROJECT_ROOT, env=env, stdout=self._log, stderr=subprocess.STDOUT,
        )
        try:
            deadline = time.monotonic() + 120
            while time.monotonic() < deadline:
                health = self._health()
                if health is not None:
                    if health.get("service") != self.service_name:
                        raise RuntimeError(f"{self.url} 被其他服务占用")
                    return self
                if self.process.poll() is not None:
                    raise RuntimeError(f"{self.kind} 服务启动失败，详见 {self._log.name}")
                time.sleep(0.5)
            raise TimeoutError(f"{self.kind} 服务启动超时，详见 {self._log.name}")
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *_):
        if self.process is not None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        if self._log is not None:
            self._log.close()


def prepare_scene_inputs(capture_dir: str | Path, scene_dir: str | Path,
                         *, prompts: dict[str, str] | None = None,
                         allow_inference: bool = True,
                         refresh_perception: bool = False,
                         cube_scene_scale_factor: float = 1.) -> dict[str, dict]:
    """缺失的 mask/mesh 才调用模型；一阶段一个模型进程。"""
    capture = Path(capture_dir).resolve()
    scene = Path(scene_dir).resolve()
    perception = scene / "perception"
    perception.mkdir(parents=True, exist_ok=True)
    rgb, depth, camera = (capture / name for name in ("rgb.png", "depth.png", "camera.json"))
    brush_mask, brush_mesh = capture / "mask.png", capture / "meshes/obj_mesh_simplified.obj"
    for path in (rgb, depth, camera, brush_mask, brush_mesh):
        if not path.is_file():
            raise FileNotFoundError(f"缺少已有刷子资产: {path}")
    prompts = PROMPTS | (prompts or {})
    capture_hashes = {name: _sha256(path) for name, path in
                      {"rgb": rgb, "depth": depth, "camera": camera}.items()}
    old_manifest_path = perception / "manifest.json"
    if old_manifest_path.is_file():
        previous = json.loads(old_manifest_path.read_text(encoding="utf-8"))
        stale_capture = previous.get("capture_sha256") != capture_hashes
        stale_prompts = any(previous.get("objects", {}).get(name, {})
                            .get("perception", {}).get("prompt") != prompts[name]
                            for name in ("dustpan", "cube"))
        stale_masks = any(
            (perception / f"{name}_mask.png").is_file()
            and previous.get("objects", {}).get(name, {}).get("perception", {}).get("mask_sha256")
            != _sha256(perception / f"{name}_mask.png")
            for name in ("dustpan", "cube")
        )
        refresh_perception |= stale_capture or stale_prompts or stale_masks
    if refresh_perception and not allow_inference:
        raise ValueError("照片、prompt 或 mask 已变化；请允许重新推理，或改用新的场景输出目录")
    masks = {"brush": brush_mask,
             "dustpan": perception / "dustpan_mask.png",
             "cube": perception / "cube_mask.png"}
    missing = [name for name in ("dustpan", "cube")
               if refresh_perception or not masks[name].is_file()]
    if missing:
        if not allow_inference:
            raise FileNotFoundError(f"缺少 mask: {missing}")
        with ModelService("seg", scene / "logs") as service:
            items = [{"category": name, "prompt": prompts[name], "mask_path": str(masks[name])}
                     for name in missing]
            try:
                status, response = segment_image(service.url, rgb, items, timeout=300)
                if status != 200 or not response.get("success"):
                    raise RuntimeError(f"SAM3 分割失败: {response}")
            finally:
                unload_seg(service.url)
    dustpan_mesh = perception / "dustpan" / "mesh_simplified.obj"
    if refresh_perception or not dustpan_mesh.is_file():
        if not allow_inference:
            raise FileNotFoundError(f"缺少簸箕 mesh: {dustpan_mesh}")
        # 每次重建使用独立服务生命周期，避免前一次 SAM3D 推理残留显存。
        with ModelService("recon", scene / "logs") as service:
            payload = {
                "rgb_path": str(rgb), "depth_path": str(depth),
                "mask_path": str(masks["dustpan"]), "intrinsics_path": str(camera),
                "output_path": str(perception / "dustpan" / "mesh.obj"),
                "depth_scale": 0.001, "simplify_mesh": True,
                "visualize": False, "use_foundationpose": False,
            }
            payload_path = Path(payload["output_path"])
            payload_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                reconstruct_request(service.url, payload, timeout=900)
            finally:
                unload_recon(service.url)
    cube_mesh = perception / "cube" / "mesh.obj"
    # 线性尺寸修正只用于任务场景；mask/深度推算的原始尺寸仍存档。
    cube_result = make_cube_from_mask(masks["cube"], depth, camera, cube_mesh,
                                      scene_scale_factor=cube_scene_scale_factor)
    sources = {
        "brush": {"mesh_path": str(brush_mesh), "source_kind": "sam3d_existing"},
        "dustpan": {"mesh_path": str(dustpan_mesh), "source_kind": "sam3d_single_view"},
        "cube": cube_result,
    }
    manifest = {"capture_dir": str(capture), "capture_sha256": capture_hashes,
                "objects": {}}
    for name, source in sources.items():
        center = observed_center_camera(masks[name], depth, camera)
        source["perception"] = {"mask_path": str(masks[name]),
                                "mask_sha256": _sha256(masks[name]),
                                "observed_center_camera_m": center,
                                "prompt": "a brush" if name == "brush" else prompts[name]}
        manifest["objects"][name] = source
    (perception / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return sources


def build_from_capture(capture_dir: str | Path, scene_dir: str | Path,
                       *, prompts: dict[str, str] | None = None,
                       allow_inference: bool = True,
                       refresh_perception: bool = False,
                       cube_scene_scale_factor: float = 1.) -> dict:
    sources = prepare_scene_inputs(capture_dir, scene_dir, prompts=prompts,
                                   allow_inference=allow_inference,
                                   refresh_perception=refresh_perception,
                                   cube_scene_scale_factor=cube_scene_scale_factor)
    result = build_tabletop_scene(scene_dir, sources=sources)
    return result
