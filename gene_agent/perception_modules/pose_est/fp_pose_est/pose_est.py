"""FoundationPose 首帧注册和单步 tracking 的基础封装。"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Callable

import numpy as np


def _validate_rgbd(rgb, depth, intrinsic) -> None:
    """校验 FoundationPose 使用的 RGB-D 和相机内参。"""
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("rgb 必须是 HxWx3 uint8 数组")
    if depth.dtype != np.float32 or depth.shape != rgb.shape[:2]:
        raise ValueError("depth 必须是与 rgb 对齐的 HxW float32 米制数组")
    if np.asarray(intrinsic).shape != (3, 3):
        raise ValueError("intrinsic 必须是 3x3 矩阵")


class PoseSession:
    """保存单个物体的 FoundationPose estimator 状态。"""

    def __init__(self, estimator, initial_pose, *, track_iterations: int = 2):
        self.estimator = estimator
        self.pose = np.asarray(initial_pose, dtype=np.float64).reshape(4, 4)
        self.track_iterations = int(track_iterations)

    def track(self, *, rgb, depth, intrinsic) -> np.ndarray:
        """基于上一帧状态执行一次 track_one。"""
        _validate_rgbd(rgb, depth, intrinsic)
        pose = self.estimator.track_one(
            rgb=rgb,
            depth=depth,
            K=np.asarray(intrinsic),
            iteration=self.track_iterations,
        )
        self.pose = np.asarray(pose, dtype=np.float64).reshape(4, 4)
        return self.pose.copy()


class FoundationPoseBackend:
    """共享网络权重，并为每个物体创建独立 PoseSession。"""

    def __init__(
        self,
        *,
        device: str = "cuda:0",
        register_iterations: int = 5,
        track_iterations: int = 2,
        debug_dir: str = "debug/pose_est",
        estimator_factory: Callable[[object], object] | None = None,
    ):
        self.device = device
        self.register_iterations = int(register_iterations)
        self.track_iterations = int(track_iterations)
        self.debug_dir = debug_dir
        self._external_estimator_factory = estimator_factory
        self._estimator_factory = estimator_factory
        self._runtime_ready = estimator_factory is not None
        self._scorer = None
        self._refiner = None
        self._glctx = None

    @property
    def is_loaded(self) -> bool:
        return self._runtime_ready

    def unload_model(self) -> None:
        """无活动 session 时释放共享网络和 CUDA raster context。"""
        import gc

        self._estimator_factory = self._external_estimator_factory
        self._scorer = None
        self._refiner = None
        self._glctx = None
        self._runtime_ready = self._external_estimator_factory is not None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except ImportError:
            pass

    def _ensure_runtime(self) -> None:
        """首次注册时惰性加载 FoundationPose 模型和 CUDA context。"""
        if self._runtime_ready:
            return
        device_id = self.device.split(":")[-1] if ":" in self.device else "0"
        os.environ["CUDA_VISIBLE_DEVICES"] = device_id

        foundation_dir = Path(__file__).resolve().parent / "FoundationPose"
        if str(foundation_dir) not in sys.path:
            sys.path.insert(0, str(foundation_dir))

        import nvdiffrast.torch as dr
        from estimater import FoundationPose, PoseRefinePredictor, ScorePredictor

        self._scorer = ScorePredictor()
        self._refiner = PoseRefinePredictor()
        self._glctx = dr.RasterizeCudaContext()

        def create_estimator(mesh):
            return FoundationPose(
                model_pts=mesh.vertices,
                model_normals=mesh.vertex_normals,
                mesh=mesh,
                scorer=self._scorer,
                refiner=self._refiner,
                glctx=self._glctx,
                debug_dir=self.debug_dir,
            )

        self._estimator_factory = create_estimator
        self._runtime_ready = True

    def register(self, *, mesh, rgb, depth, mask, intrinsic) -> PoseSession:
        """执行首帧 register，并返回可连续 tracking 的 session。"""
        _validate_rgbd(rgb, depth, intrinsic)
        mask = np.asarray(mask)
        if mask.shape != depth.shape:
            raise ValueError("mask 尺寸必须与 depth 一致")
        mask = mask.astype(bool)
        if not mask.any():
            raise ValueError("mask 为空，不能执行 register")
        valid_depth = mask & np.isfinite(depth) & (depth >= 0.001)
        if np.count_nonzero(valid_depth) < 4:
            raise ValueError("mask 内有效深度点少于 4 个，不能执行 register")

        self._ensure_runtime()
        estimator = self._estimator_factory(mesh)
        pose = estimator.register(
            K=np.asarray(intrinsic),
            rgb=rgb,
            depth=depth,
            ob_mask=mask.astype(np.uint8),
            iteration=self.register_iterations,
            preprocess_depth=True,
        )
        return PoseSession(
            estimator,
            pose,
            track_iterations=self.track_iterations,
        )
