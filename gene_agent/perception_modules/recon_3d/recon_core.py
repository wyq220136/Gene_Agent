"""SAM3D RGB-D 重建核心逻辑。

本模块负责项目数据结构下的路径解析、RGB-D 输入读取、深度反投影
以及 SAM3D mesh 导出。重型 SAM3D 依赖只在真正执行重建时导入。
"""

from __future__ import annotations

from gene_agent.perception_modules.paths import CHECKPOINT_ROOT, PROJECT_ROOT

import json
import os
import sys
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


_MODULE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = PROJECT_ROOT
_SAM3D_DIR = _REPO_ROOT / "third_party" / "sam-3d-objects"
_DEFAULT_CONFIG = CHECKPOINT_ROOT / "SAM3D" / "checkpoints" / "pipeline.yaml"


@dataclass(frozen=True)
class ReconPaths:
    """一次重建所需的完整路径集合。"""

    object_dir: Path
    rgb: Path
    depth: Path
    mask: Path
    intrinsics: Path
    output: Path


def _as_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    return Path(value)


def _find_instance_dir(object_dir: Path, instance: str | None) -> Path:
    if (object_dir / "masks").is_dir():
        if instance is not None and instance != object_dir.name:
            raise ValueError(f"instance {instance!r} 与对象目录不匹配: {object_dir}")
        return object_dir

    if instance is not None:
        candidate = object_dir / instance
        if not (candidate / "masks").is_dir():
            raise FileNotFoundError(f"找不到 instance 的 masks 目录: {candidate / 'masks'}")
        return candidate

    candidates = sorted(
        child for child in object_dir.iterdir()
        if child.is_dir() and (child / "masks").is_dir()
    )
    if not candidates:
        raise FileNotFoundError(f"对象目录下没有找到包含 masks 的 instance: {object_dir}")
    if len(candidates) > 1:
        names = ", ".join(child.name for child in candidates)
        raise ValueError(f"对象目录下存在多个 instance，请使用 --instance 指定: {names}")
    return candidates[0]


def _find_task_dir(instance_dir: Path) -> Path:
    for parent in (instance_dir, *instance_dir.parents):
        if parent.name == "objs":
            return parent.parent
    raise ValueError(f"对象目录不符合 task/objs/category/instance 结构: {instance_dir}")


def resolve_object_paths(
    object_dir: str | Path,
    *,
    instance: str | None = None,
    rgb: str | Path | None = None,
    depth: str | Path | None = None,
    mask: str | Path | None = None,
    intrinsics: str | Path | None = None,
    output: str | Path | None = None,
) -> ReconPaths:
    """根据对象目录推断默认路径，并让显式参数覆盖默认值。"""
    object_path = Path(object_dir)
    instance_dir = _find_instance_dir(object_path, instance)
    task_dir = _find_task_dir(instance_dir)

    return ReconPaths(
        object_dir=instance_dir,
        rgb=_as_path(rgb) or task_dir / "rgbs" / "rgb_0.png",
        depth=_as_path(depth) or task_dir / "depths" / "depth_0.png",
        mask=_as_path(mask) or instance_dir / "masks" / "mask_0.png",
        intrinsics=_as_path(intrinsics) or task_dir / "camera" / "camera_parameters_0.json",
        output=_as_path(output) or instance_dir / "meshes" / "obj_mesh.obj",
    )


def validate_inputs(
    rgb: np.ndarray,
    depth: np.ndarray,
    mask: np.ndarray,
    intrinsics: np.ndarray,
) -> None:
    """校验 SAM3D RGB-D 重建所需数组的形状和有效值。"""
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise ValueError("rgb 必须是 H×W×3 或 H×W×4")
    if depth.ndim != 2 or mask.ndim != 2:
        raise ValueError("depth 和 mask 必须是 H×W")
    if rgb.shape[:2] != depth.shape or rgb.shape[:2] != mask.shape:
        raise ValueError("rgb、depth 和 mask 的尺寸不一致")
    if intrinsics.shape != (3, 3):
        raise ValueError("intrinsics 必须是 3×3 矩阵")
    if not np.isfinite(intrinsics).all() or intrinsics[0, 0] <= 0 or intrinsics[1, 1] <= 0:
        raise ValueError("intrinsics 的焦距必须是有限正数")
    if not np.any(mask > 0):
        raise ValueError("mask 中没有前景像素")
    if not np.any(np.isfinite(depth) & (depth > 0)):
        raise ValueError("depth 中没有有效正值")


def unproject_depth(
    depth: np.ndarray,
    intrinsics: np.ndarray,
    depth_scale: float = 0.001,
) -> np.ndarray:
    """将深度图反投影为米制 RGB-D 相机坐标系 pointmap。"""
    if not np.isfinite(depth_scale) or depth_scale <= 0:
        raise ValueError("depth_scale 必须是有限正数")
    if depth.ndim != 2 or intrinsics.shape != (3, 3):
        raise ValueError("depth 必须是 H×W，intrinsics 必须是 3×3")

    z = depth.astype(np.float32) * np.float32(depth_scale)
    v, u = np.indices(z.shape, dtype=np.float32)
    points = np.stack(
        (
            (u - intrinsics[0, 2]) * z / intrinsics[0, 0],
            (v - intrinsics[1, 2]) * z / intrinsics[1, 1],
            z,
        ),
        axis=-1,
    )
    points[~np.isfinite(z) | (z <= 0)] = np.nan
    return points.astype(np.float32, copy=False)


def _load_image(path: Path, *, mode: str) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as image:
        return np.asarray(image.convert(mode))


def _load_depth(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return np.asarray(np.load(path))
    return _load_image(path, mode="I")


def _load_intrinsics(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return np.asarray(np.load(path), dtype=np.float32)
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if "intrinsic_matrix" not in data:
        raise ValueError(f"相机内参 JSON 缺少 intrinsic_matrix 字段: {path}")
    return np.asarray(data["intrinsic_matrix"], dtype=np.float32)


def _to_sam3d_pointmap(points_camera: np.ndarray) -> Any:
    """将 OpenCV/RGB-D 相机坐标转换为 SAM3D 坐标约定。"""
    import torch
    from pytorch3d.transforms import Transform3d

    from sam3d_objects.pipeline.inference_pipeline_pointmap import (
        camera_to_pytorch3d_camera,
    )

    points = torch.from_numpy(points_camera)
    transform = Transform3d().rotate(
        camera_to_pytorch3d_camera(device="cpu").rotation
    )
    return transform.transform_points(points).to(dtype=torch.float32)


def _create_gt_depth_config(source_config: Path, device: str) -> Path:
    """创建关闭 MoGe 的临时配置，避免重复估计深度。"""
    from omegaconf import OmegaConf

    config = OmegaConf.load(source_config)
    config.depth_model = None
    config.device = device
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".yaml",
        prefix="gt_depth_",
        dir=source_config.parent,
        delete=False,
    ) as handle:
        output_path = Path(handle.name)
    OmegaConf.save(config, output_path)
    return output_path


def _load_inference(config_path: Path) -> Any:
    notebook_dir = _SAM3D_DIR / "notebook"
    for path in (_REPO_ROOT, _SAM3D_DIR, notebook_dir):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from inference import Inference

    return Inference(str(config_path), compile=False)


def _extract_mesh(output: dict[str, Any]) -> Any:
    mesh = output.get("glb")
    if mesh is None:
        raise RuntimeError("SAM3D 没有返回 mesh")
    return mesh


def build_output_paths(
    output_path: str | Path,
    *,
    simplify_mesh: bool = False,
    visualize: bool = False,
    use_foundationpose: bool = False,
) -> dict[str, Path | None]:
    """生成原始 mesh、简化 mesh、pose 和可视化目录的输出路径。"""
    output_path = Path(output_path)
    need_simplified = simplify_mesh or visualize or use_foundationpose
    return {
        "mesh": output_path,
        "simplified_mesh": (
            output_path.with_name(f"{output_path.stem}_simplified{output_path.suffix}")
            if need_simplified
            else None
        ),
        "pose": output_path.parent / "sam3d_pose.npz",
        "visualization_dir": output_path.parent.parent / "visualization" if visualize else None,
    }


def _pose_to_numpy(value: Any) -> np.ndarray:
    """将 SAM3D 返回的 Tensor 或数组统一转换为 NumPy。"""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _extract_pose(output: dict[str, Any]) -> dict[str, np.ndarray]:
    pose_keys = ("rotation", "translation", "scale")
    missing = [key for key in pose_keys if key not in output]
    if missing:
        raise RuntimeError(f"SAM3D 输出缺少 pose 字段: {', '.join(missing)}")
    return {key: _pose_to_numpy(output[key]) for key in pose_keys}


def _apply_mesh_scale(vertices: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """将 SAM3D scale 烘焙到 mesh 顶点，返回新的顶点数组。"""
    scale = np.asarray(scale, dtype=np.float64).reshape(-1)
    if scale.size == 1:
        scale = np.repeat(scale, 3)
    if scale.size < 3 or not np.isfinite(scale[:3]).all():
        raise ValueError("SAM3D scale 必须包含三个有限数值")
    return np.asarray(vertices) * scale[:3]


def _simplify_mesh(
    mesh_path: Path,
    simplified_path: Path,
    face_count: int = 10000,
    scale: np.ndarray | None = None,
) -> None:
    """使用 trimesh 将 mesh 简化到指定面数附近。"""
    if face_count <= 0:
        raise ValueError("face_count 必须是正整数")
    import trimesh

    loaded = trimesh.load(mesh_path, force="mesh")
    if len(loaded.faces) > face_count:
        loaded = loaded.simplify_quadric_decimation(face_count=face_count)
    if scale is not None:
        loaded.vertices = _apply_mesh_scale(loaded.vertices, scale)
    simplified_path.parent.mkdir(parents=True, exist_ok=True)
    loaded.export(simplified_path)


def _estimate_foundationpose(
    *,
    pose_server: str,
    rgb_path: str | Path,
    depth_path: str | Path,
    mask_path: str | Path,
    mesh_path: str | Path,
    intrinsics_path: str | Path,
) -> np.ndarray:
    """调用 FoundationPose server 注册当前 mesh 并返回 camera-object pose。"""
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    from gene_agent.perception_modules.pose_est.pose_client import PoseClient

    client = PoseClient(pose_server, timeout=600.0)
    session_id = f"recon_3d_{uuid.uuid4().hex}"
    registered = False
    try:
        result = client.register(
            session_id=session_id,
            rgb_path=rgb_path,
            depth_path=depth_path,
            mask_path=mask_path,
            mesh_path=mesh_path,
            camera_path=intrinsics_path,
        )
        registered = True
        pose = np.asarray(result["T_camera_object"], dtype=np.float64)
        if pose.shape != (4, 4):
            raise ValueError(f"FoundationPose 返回的 pose 不是 4×4: {pose.shape}")
        return pose
    finally:
        if registered:
            try:
                client.stop(session_id)
            except Exception as exc:
                print(f"[recon_3d] FoundationPose session 释放失败: {exc}", flush=True)


class Sam3DReconstructor:
    """常驻加载 SAM3D 模型，并复用同一个 inference pipeline。"""

    def __init__(
        self, config_path: str | Path | None = None, device: str = "cuda",
        *, load_on_init: bool = True,
    ):
        self.device = device
        self.source_config = Path(config_path) if config_path else _DEFAULT_CONFIG
        if not self.source_config.is_file():
            raise FileNotFoundError(f"SAM3D pipeline 配置不存在: {self.source_config}")

        self._inference = None
        if load_on_init:
            self._load_model()

    @property
    def is_loaded(self) -> bool:
        """返回 SAM3D inference 是否仍驻留在当前进程。"""
        return self._inference is not None

    def _load_model(self) -> None:
        """加载 SAM3D inference，并在加载完成后打印状态。"""

        print(
            f"[recon_3d] 正在加载 SAM3D 模型: config={self.source_config}, device={self.device}",
            flush=True,
        )
        temporary_config = _create_gt_depth_config(self.source_config, self.device)
        try:
            self._inference = _load_inference(temporary_config)
        finally:
            try:
                os.unlink(temporary_config)
            except FileNotFoundError:
                pass
        print("[recon_3d] SAM3D 模型加载完成，可以接收重建请求", flush=True)

    def ensure_loaded(self) -> None:
        """在模型被释放后，为下一次 SAM3D 请求重新加载模型。"""
        if not self.is_loaded:
            self._load_model()

    def unload_model(self) -> None:
        """卸载 SAM3D 模型及其编译缓存，释放当前进程占用的 GPU 显存。"""
        import gc

        # 先删除常驻 pipeline 引用，使其中的模型权重和内部模块可以析构。
        self._inference = None
        gc.collect()
        try:
            import torch

            # 即使当前入口关闭了编译，也清理可能由 SAM3D 运行期间产生的 Dynamo 缓存。
            dynamo = getattr(torch, "_dynamo", None)
            reset = getattr(dynamo, "reset", None)
            if reset is not None:
                reset()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except ImportError:
            pass
        print("[recon_3d] SAM3D 模型权重和推理缓存已卸载", flush=True)

    def clear_inference_cache(self) -> None:
        """清理本次推理临时对象和 CUDA allocator，但保留模型权重。"""
        import gc

        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except ImportError:
            pass
        print("[recon_3d] SAM3D 推理临时占用已清理，模型权重仍保留", flush=True)

    def reconstruct(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        mask: np.ndarray,
        intrinsics: np.ndarray,
        output_path: str | Path,
        *,
        rgb_path: str | Path | None = None,
        depth_path: str | Path | None = None,
        mask_path: str | Path | None = None,
        intrinsics_path: str | Path | None = None,
        depth_scale: float = 0.001,
        seed: int = 42,
        simplify_mesh: bool = False,
        visualize: bool = False,
        simplify_faces: int = 10000,
        use_foundationpose: bool = False,
        pose_server: str = "http://127.0.0.1:18081",
    ) -> dict[str, Any]:
        """使用已加载的模型完成一次 RGB-D mesh 重建。"""
        self.ensure_loaded()
        validate_inputs(rgb, depth, mask, intrinsics)
        pointmap_camera = unproject_depth(depth, intrinsics, depth_scale)
        pointmap = _to_sam3d_pointmap(pointmap_camera)
        import torch

        with torch.inference_mode():
            output = self._inference(
                rgb.astype(np.uint8, copy=False),
                mask.astype(bool, copy=False),
                seed=seed,
                pointmap=pointmap,
            )
        mesh = _extract_mesh(output)
        output_path = Path(output_path)
        output_paths = build_output_paths(
            output_path,
            simplify_mesh=simplify_mesh,
            visualize=visualize,
            use_foundationpose=use_foundationpose,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        mesh.export(output_path)

        pose = _extract_pose(output)
        pose_path = output_paths["pose"]
        assert isinstance(pose_path, Path)
        np.savez(pose_path, **pose)

        result: dict[str, Any] = {
            "output_path": str(output_path),
            "mesh_format": "obj",
            "pose_path": str(pose_path),
        }
        simplified_path = output_paths["simplified_mesh"]
        if isinstance(simplified_path, Path):
            _simplify_mesh(
                output_path,
                simplified_path,
                face_count=simplify_faces,
                scale=pose["scale"],
            )
            result["simplified_output_path"] = str(simplified_path)
            result["simplified_mesh_scale_applied"] = True

        visualization_pose: dict[str, np.ndarray] | np.ndarray = pose
        if isinstance(simplified_path, Path) and not use_foundationpose:
            visualization_pose = dict(pose)
            visualization_pose["scale"] = np.ones(3, dtype=np.float64)
        if use_foundationpose:
            foundationpose_mesh = simplified_path if isinstance(simplified_path, Path) else output_path
            if (
                rgb_path is None
                or depth_path is None
                or mask_path is None
                or intrinsics_path is None
            ):
                raise ValueError("FoundationPose 需要提供 RGB、depth、mask 和相机内参路径")
            # FoundationPose 与 SAM3D 可能竞争同一块 GPU；在调用前清理 SAM3D。
            del output, mesh, pointmap, pointmap_camera
            self.unload_model()
            try:
                foundationpose_pose = _estimate_foundationpose(
                    pose_server=pose_server,
                    rgb_path=rgb_path,
                    depth_path=depth_path,
                    mask_path=mask_path,
                    mesh_path=foundationpose_mesh,
                    intrinsics_path=intrinsics_path,
                )
            finally:
                # 无论 FoundationPose 成功还是失败，都恢复 SAM3D 服务状态。
                self.ensure_loaded()
            foundationpose_pose_path = output_path.parent / "foundationpose_pose.npy"
            np.save(foundationpose_pose_path, foundationpose_pose)
            result["foundationpose_pose_path"] = str(foundationpose_pose_path)
            visualization_pose = foundationpose_pose

        visualization_dir = output_paths["visualization_dir"]
        if isinstance(visualization_dir, Path):
            if not isinstance(simplified_path, Path):
                raise RuntimeError("可视化需要先生成简化 mesh")
            if (
                rgb_path is None
                or depth_path is None
                or mask_path is None
                or intrinsics_path is None
            ):
                raise ValueError("可视化需要提供 RGB、depth、mask 和相机内参路径")
            if str(_REPO_ROOT) not in sys.path:
                sys.path.insert(0, str(_REPO_ROOT))
            from .visualization import visualize_reconstruction

            result["visualization"] = visualize_reconstruction(
                rgb_path=rgb_path,
                depth_path=depth_path,
                intrinsics_path=intrinsics_path,
                mesh_path=simplified_path,
                pose=visualization_pose,
                output_dir=visualization_dir,
            )
        return result


def reconstruct_rgbd(
    rgb_path: str | Path,
    depth_path: str | Path,
    mask_path: str | Path,
    intrinsics_path: str | Path,
    output_path: str | Path,
    *,
    depth_scale: float = 0.001,
    seed: int = 42,
    config_path: str | Path | None = None,
    device: str = "cuda",
    simplify_mesh: bool = False,
    visualize: bool = False,
    simplify_faces: int = 10000,
    use_foundationpose: bool = False,
    pose_server: str = "http://127.0.0.1:18081",
) -> dict[str, Any]:
    """读取项目输入并使用真实 RGB-D depth 导出 SAM3D OBJ。"""
    rgb_path = Path(rgb_path)
    depth_path = Path(depth_path)
    mask_path = Path(mask_path)
    intrinsics_path = Path(intrinsics_path)
    output_path = Path(output_path)

    for path in (rgb_path, depth_path, mask_path, intrinsics_path):
        if not path.is_file():
            raise FileNotFoundError(f"输入文件不存在: {path}")

    rgb = _load_image(rgb_path, mode="RGB")
    depth = _load_depth(depth_path)
    mask = _load_image(mask_path, mode="L")
    intrinsics = _load_intrinsics(intrinsics_path)
    validate_inputs(rgb, depth, mask, intrinsics)

    reconstructor = Sam3DReconstructor(config_path=config_path, device=device)
    return reconstructor.reconstruct(
        rgb,
        depth,
        mask,
        intrinsics,
        output_path,
        rgb_path=rgb_path,
        depth_path=depth_path,
        mask_path=mask_path,
        intrinsics_path=intrinsics_path,
        depth_scale=depth_scale,
        seed=seed,
        simplify_mesh=simplify_mesh,
        visualize=visualize,
        simplify_faces=simplify_faces,
        use_foundationpose=use_foundationpose,
        pose_server=pose_server,
    )
