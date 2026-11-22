"""服务丢失临时 session 时，仅用当前冻结帧重建当前对象的跟踪状态。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from gene_agent.perception_modules.segmentation.seg_client import segment_image, unload_model


def make_session_recovery(setup):
    def recover(object_id, session_id, *, capture_dir, pose_client, camera_id):
        spec = setup.get("session_recovery", {}).get(object_id)
        if (not isinstance(spec, dict) or not spec.get("seg_server")
                or not spec.get("prompt") or not spec.get("category")):
            raise RuntimeError(
                f"{object_id} 的 tracking session 已失效，旧准备记录缺少重新分割参数；"
                "请保持 Pose server 在线，重新运行 prepare_microwave_realworld.py，"
                "并将 --perception 指向新目录。不要手工修改 session 编号。"
            )
        if setup["sessions"].get(object_id) != session_id:
            raise ValueError("恢复 session 与本次准备记录不一致")
        mesh = Path(setup["meshes"][object_id])
        if not mesh.is_file():
            raise FileNotFoundError(f"恢复 session 缺少已确认 mesh: {mesh}")
        capture = Path(capture_dir)
        # object_id 已由同帧跟踪入口校验，不能改用旧准备帧的 mask。
        folder = capture / "session_recovery" / object_id
        folder.mkdir(parents=True, exist_ok=False)
        mask = folder / "mask.png"
        status, response = segment_image(
            spec["seg_server"], capture / "rgb.png",
            [{"category": spec["category"], "prompt": spec["prompt"], "mask_path": str(mask)}],
        )
        if (status != 200 or response.get("ok") is not True
                or response.get("success") is False or not mask.is_file()
                or any(row.get("ok") is False for row in response.get("results", []))):
            raise RuntimeError(f"{object_id} 会话恢复分割失败: {response}")
        with Image.open(capture / "rgb.png") as rgb, Image.open(mask) as image:
            if image.size != rgb.size or not np.any(np.asarray(image.convert("L")) > 0):
                raise ValueError("恢复 session 的当前帧 mask 为空或尺寸不匹配")
        unload_model(spec["seg_server"])
        result = pose_client.register(
            session_id=session_id, rgb_path=capture / "rgb.png",
            depth_path=capture / "depth.png", camera_path=capture / "camera.json",
            mask_path=mask, mesh_path=mesh, camera_id=camera_id,
        )
        (folder / "registration.json").write_text(json.dumps({
            "object_id": object_id, "session_id": session_id,
            "method": "current_frame_segmentation_then_registration",
            "camera_path": str(capture / "camera.json"), "mesh_path": str(mesh),
            "prompt": spec["prompt"], "result": result,
        }, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        return result
    return recover
