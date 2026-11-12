"""单物体 RealSense → SAM3 prompt 试验 → SAM3D → FoundationPose 检查入口。"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

from gene_agent.perception_modules.camera.camera_client import CameraClient
from gene_agent.perception_modules.camera.live_rgbd_protocol import FramePacket
from gene_agent.perception_modules.paths import PROJECT_ROOT
from gene_agent.perception_modules.pose_est.pose_client import PoseClient
from gene_agent.perception_modules.recon_3d.recon_client import (
    post_json as post_reconstruction,
    unload_model,
)
from gene_agent.perception_modules.segmentation.seg_client import (
    segment_image,
    unload_model as unload_seg_model,
)


@dataclass(frozen=True)
class CapturePaths:
    object_dir: Path
    rgb: Path
    depth: Path
    camera: Path
    mask: Path
    mesh: Path
    simplified_mesh: Path
    pose: Path


@dataclass(frozen=True)
class SelectedMask:
    mask: Path
    prompt: str


def paths_for(object_dir: str | Path) -> CapturePaths:
    """同一物体的一帧输入和最终产物始终共用一个目录。"""
    directory = Path(object_dir).expanduser().resolve()
    return CapturePaths(
        object_dir=directory,
        rgb=directory / "rgb.png",
        depth=directory / "depth.png",
        camera=directory / "camera.json",
        mask=directory / "mask.png",
        mesh=directory / "meshes" / "obj_mesh.obj",
        simplified_mesh=directory / "meshes" / "obj_mesh_simplified.obj",
        pose=directory / "meshes" / "foundationpose_pose.npy",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_reused_mask(paths: CapturePaths, *, serial: str) -> str:
    """检查旧 mask 来自当前冻结帧的一次已确认 prompt。"""
    required = (paths.rgb, paths.depth, paths.camera, paths.mask)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ValueError(f"复用 mask 缺少文件: {', '.join(missing)}")
    camera = json.loads(paths.camera.read_text(encoding="utf-8"))
    if camera.get("serial") != serial:
        raise ValueError(f"旧采集帧来自 {camera.get('serial')}，与 --serial={serial} 不符")
    intrinsic = np.asarray(camera.get("intrinsic_matrix"), dtype=float)
    if intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all():
        raise ValueError("旧相机内参无效")
    with Image.open(paths.rgb) as rgb, Image.open(paths.depth) as depth, Image.open(paths.mask) as mask:
        if rgb.size != depth.size or rgb.size != mask.size:
            raise ValueError("旧 RGB、depth、mask 尺寸不一致")
        if depth.mode not in {"I;16", "I"} or not np.any(np.asarray(depth) > 0):
            raise ValueError("旧深度图无有效深度")
        if not np.any(np.asarray(mask.convert("L")) > 0):
            raise ValueError("旧 mask 没有前景")
    manifest = paths.object_dir / "prompt_trials.json"
    if not manifest.is_file():
        raise ValueError("找不到已确认 prompt 记录，不能复用 mask")
    trials = json.loads(manifest.read_text(encoding="utf-8"))
    accepted = [item for item in trials if item.get("accepted") is True]
    if len(accepted) != 1:
        raise ValueError("已确认 prompt 记录必须恰好有一项")
    trial = accepted[0]
    candidate = Path(trial["mask_path"]).resolve()
    if not candidate.is_relative_to((paths.object_dir / "prompt_trials").resolve()):
        raise ValueError("候选 mask 路径不在本物体 prompt_trials 目录")
    if not candidate.is_file() or _sha256(candidate) != _sha256(paths.mask):
        raise ValueError("最终 mask 与已确认候选 mask 不一致")
    capture_hashes = trial.get("capture_sha256")
    if capture_hashes is not None:
        for name in ("rgb", "depth", "camera"):
            if capture_hashes.get(name) != _sha256(getattr(paths, name)):
                raise ValueError(f"旧采集文件已变化: {name}")
    return str(trial["prompt"])


def _mesh_manifest(paths: CapturePaths) -> Path:
    return paths.mesh.parent / "reconstruction_inputs.json"


def save_mesh_provenance(paths: CapturePaths) -> None:
    """记录重建输入和输出哈希，供 OOM 后安全跳过 SAM3D。"""
    files = {name: getattr(paths, name) for name in
             ("rgb", "depth", "camera", "mask", "mesh", "simplified_mesh")}
    manifest = _mesh_manifest(paths)
    manifest.write_text(json.dumps({name: _sha256(path) for name, path in files.items()},
                                   ensure_ascii=False, indent=2), encoding="utf-8")


def validate_reused_mesh(paths: CapturePaths) -> None:
    manifest = _mesh_manifest(paths)
    if not manifest.is_file():
        raise ValueError("缺少 reconstruction_inputs.json，不能确认 mesh 对应当前输入")
    recorded = json.loads(manifest.read_text(encoding="utf-8"))
    for name in ("rgb", "depth", "camera", "mask", "mesh", "simplified_mesh"):
        path = getattr(paths, name)
        if not path.is_file() or recorded.get(name) != _sha256(path):
            raise ValueError(f"重建输入或 mesh 已变化: {name}")


def save_capture(packet: FramePacket, object_dir: str | Path) -> CapturePaths:
    """保存一帧已对齐 RGB-D；PNG 深度采用毫米，与现有服务默认尺度一致。"""
    packet.validate()
    paths = paths_for(object_dir)
    depth_mm = np.rint(packet.depth.astype(np.float64) * 1000.0)
    valid = np.isfinite(depth_mm) & (depth_mm >= 0)
    if np.any(depth_mm[valid] > 65535):
        raise ValueError("深度超过 uint16 毫米范围，不能无提示截断")
    depth_mm[~valid] = 0
    paths.object_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.ascontiguousarray(packet.rgb)).save(paths.rgb)
    Image.fromarray(depth_mm.astype(np.uint16)).save(paths.depth)
    paths.camera.write_text(json.dumps({
        "intrinsic_matrix": packet.intrinsic_matrix.tolist(),
        "frame_number": packet.frame_number,
        "timestamp_ms": packet.timestamp_ms,
        "serial": packet.serial,
        "depth_unit": "millimeter",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return paths


def capture_once(
    object_dir: str | Path, *, endpoint: str, expected_serial: str,
) -> CapturePaths:
    """只读一次 camera server 的 RGB-D，不在换 prompt 时重新采集。"""
    client = CameraClient(endpoint)
    try:
        info = client.info()
        if info.get("serial") != expected_serial:
            raise ValueError(
                f"相机序列号不一致: 指定 {expected_serial}，服务返回 {info.get('serial')}"
            )
        packet = client.get_frame()
    finally:
        client.close()
    if packet.serial != expected_serial:
        raise ValueError(f"采集帧来自其他相机: {packet.serial}")
    return save_capture(packet, object_dir)


def _overlay(rgb_path: Path, mask_path: Path, output_path: Path) -> int:
    rgb = np.asarray(Image.open(rgb_path).convert("RGB"), dtype=np.uint8)
    mask = np.asarray(Image.open(mask_path).convert("L")) > 0
    if mask.shape != rgb.shape[:2]:
        raise ValueError("mask 与采集 RGB 尺寸不一致")
    result = rgb.copy()
    result[mask] = np.rint(0.45 * result[mask] + 0.55 * np.array([255, 30, 30])).astype(np.uint8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(result).save(output_path)
    return int(mask.sum())


def open_preview(path: Path) -> None:
    """尝试用桌面默认图片查看器打开；无图形会话时仍保留文件供手动查看。"""
    try:
        subprocess.Popen(
            ["xdg-open", str(path)], stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, start_new_session=True,
        )
    except OSError:
        pass


def choose_mask(
    paths: CapturePaths, *, category: str, initial_prompt: str, seg_server: str,
    input_fn: Callable[[str], str] = input,
    preview_fn: Callable[[Path], None] = open_preview,
) -> SelectedMask:
    """同一张照片上反复试 prompt；用户接受后才发布最终 mask。"""
    prompt = initial_prompt.strip()
    if not prompt:
        raise ValueError("prompt 不能为空")
    trial_dir = paths.object_dir / "prompt_trials"
    trial_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = paths.object_dir / "prompt_trials.json"
    trials = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else []
    if not isinstance(trials, list):
        raise ValueError(f"prompt 记录格式错误: {manifest_path}")

    while True:
        number = len(trials) + 1
        candidate = trial_dir / f"mask_{number:03d}.png"
        overlay = trial_dir / f"overlay_{number:03d}.png"
        # 上次运行若在服务响应与记录之间中断，不能把残留 mask 当成新结果。
        candidate.unlink(missing_ok=True)
        status, response = segment_image(
            seg_server, paths.rgb,
            [{"category": category, "prompt": prompt, "mask_path": str(candidate)}],
            timeout=300.0,
        )
        good_response = (
            status == 200 and response.get("ok") and response.get("success", True)
            and all(item.get("ok", False) for item in response.get("results", []))
            and candidate.is_file()
        )
        foreground_pixels = _overlay(paths.rgb, candidate, overlay) if good_response else 0
        trial = {
            "prompt": prompt,
            "mask_path": str(candidate) if candidate.is_file() else None,
            "overlay_path": str(overlay) if good_response else None,
            "foreground_pixels": foreground_pixels,
            "accepted": False,
            "error": None if good_response and foreground_pixels else str(response),
        }
        trials.append(trial)
        manifest_path.write_text(json.dumps(trials, ensure_ascii=False, indent=2), encoding="utf-8")
        if foreground_pixels:
            print(f"预览: {overlay}；前景像素: {foreground_pixels}")
            preview_fn(overlay)
            answer = input_fn("满意输入 y；否则直接输入新 prompt（q 退出）：").strip()
            if answer.lower() in {"y", "yes"}:
                shutil.copy2(candidate, paths.mask)
                for old_trial in trials:
                    old_trial["accepted"] = False
                trial["accepted"] = True
                trial["capture_sha256"] = {
                    name: _sha256(getattr(paths, name))
                    for name in ("rgb", "depth", "camera")
                }
                manifest_path.write_text(json.dumps(trials, ensure_ascii=False, indent=2), encoding="utf-8")
                return SelectedMask(paths.mask, prompt)
        else:
            print(f"本次分割没有有效 mask：{trial['error']}")
            answer = input_fn("输入新 prompt 重试（q 退出）：").strip()
        if answer.lower() in {"q", "quit"}:
            raise KeyboardInterrupt("用户退出 prompt 试验")
        if answer and answer.lower() not in {"n", "no"}:
            prompt = answer
        else:
            prompt = input_fn("请输入新 prompt：").strip()
            if not prompt:
                raise ValueError("prompt 不能为空")


def reconstruct_mesh(paths: CapturePaths, *, server: str) -> dict:
    """只在确认 mask 后请求单视角 SAM3D；深度 PNG 单位为毫米。"""
    payload = {
        "rgb_path": str(paths.rgb),
        "depth_path": str(paths.depth),
        "mask_path": str(paths.mask),
        "intrinsics_path": str(paths.camera),
        "output_path": str(paths.mesh),
        "depth_scale": 0.001,
        "simplify_mesh": True,
        "visualize": False,
        "use_foundationpose": False,
    }
    result = post_reconstruction(server, payload, timeout=900.0)
    if not Path(result["simplified_output_path"]).is_file():
        raise RuntimeError("SAM3D 未生成带实际尺度的简化 mesh")
    return result


def estimate_pose(paths: CapturePaths, *, pose_server: str, recon_server: str | None = None) -> np.ndarray:
    """释放 SAM3D 显存后，用确认的 mask 和带尺度 mesh 注册 FoundationPose。"""
    if recon_server is not None:
        unload_model(recon_server)
    client = PoseClient(pose_server, timeout=600.0)
    session_id = f"realworld_{uuid.uuid4().hex}"
    registered = False
    try:
        response = client.register(
            session_id=session_id,
            rgb_path=paths.rgb,
            depth_path=paths.depth,
            mask_path=paths.mask,
            mesh_path=paths.simplified_mesh,
            camera_path=paths.camera,
        )
        registered = True
        pose = np.asarray(response["T_camera_object"], dtype=np.float64)
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            raise ValueError("FoundationPose 返回无效的 4×4 位姿")
        np.save(paths.pose, pose)
        return pose
    finally:
        try:
            if registered:
                client.stop(session_id)
        finally:
            client.unload_model()


def show_alignment(paths: CapturePaths, pose: np.ndarray) -> dict[str, str]:
    """在本地进程打开 Open3D，同时导出点云和按估计 pose 放置的 mesh。"""
    from gene_agent.perception_modules.recon_3d.visualization import visualize_reconstruction

    return visualize_reconstruction(
        rgb_path=paths.rgb,
        depth_path=paths.depth,
        intrinsics_path=paths.camera,
        mesh_path=paths.simplified_mesh,
        pose=pose,
        output_dir=paths.object_dir / "visualization",
        title=f"FoundationPose: {paths.object_dir.name}",
        depth_scale=0.001,
    )


def run_stages(
    paths: CapturePaths, *, seg_server: str, recon_server: str, pose_server: str,
    skip_pose: bool, reuse_mesh: bool, segmentation_used: bool = True,
) -> None:
    """按 SAM3 → SAM3D → FoundationPose 顺序运行，阶段间同步释放显存。"""
    if segmentation_used:
        unload_seg_model(seg_server)
    if reuse_mesh:
        validate_reused_mesh(paths)
    else:
        try:
            result = reconstruct_mesh(paths, server=recon_server)
        except Exception:
            # OOM 后尽力清理，但保留原始异常供用户判断是否需要重启服务。
            try:
                unload_model(recon_server)
            except Exception as cleanup_error:
                print(f"SAM3D 清理失败，需重启服务: {cleanup_error}")
            raise
        unload_model(recon_server)
        save_mesh_provenance(paths)
        print(f"SAM3D mesh: {result['simplified_output_path']}")
    if not skip_pose:
        pose = estimate_pose(paths, pose_server=pose_server)
        print(f"FoundationPose 位姿: {paths.pose}")
        print(f"Open3D 导出: {show_alignment(paths, pose)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="一台 RealSense 拍一帧，反复试 prompt 后重建单个物体")
    parser.add_argument("--object", required=True, help="物体目录名，例如 hammer_0 或 brush_0")
    parser.add_argument("--serial", required=True, help="RealSense 序列号，必须与 camera server 一致")
    parser.add_argument("--prompt", help="初始 SAM3 文本 prompt；复用已确认 mask 时可省略")
    parser.add_argument("--category", help="SAM3 类别，默认由 --object 去掉末尾编号得到")
    parser.add_argument("--output-root", default="data/realworld")
    parser.add_argument("--camera-endpoint", default="tcp://127.0.0.1:18082")
    parser.add_argument("--seg-server", default="http://127.0.0.1:18080")
    parser.add_argument("--recon-server", default="http://127.0.0.1:18083")
    parser.add_argument("--pose-server", default="http://127.0.0.1:18081")
    parser.add_argument("--reuse-capture", action="store_true", help="复用上次 RGB-D，重新试 prompt")
    parser.add_argument("--reuse-mask", action="store_true", help="跳过 SAM3，复用上次人工确认的 mask")
    parser.add_argument("--reuse-mesh", action="store_true", help="跳过 SAM3D，复用有输入哈希记录的 mesh")
    parser.add_argument("--skip-pose", action="store_true", help="只重建 mesh，暂不运行 FoundationPose")
    args = parser.parse_args(argv)
    if args.reuse_mesh:
        args.reuse_mask = True
    if args.reuse_mask:
        args.reuse_capture = True
    if not args.reuse_mask and not args.prompt:
        parser.error("运行 SAM3 分割时必须提供 --prompt")
    if not args.object or Path(args.object).name != args.object or args.object in {".", ".."}:
        parser.error("--object 只能是单个目录名")
    category = args.category or args.object.rsplit("_", 1)[0]
    root = Path(args.output_root).expanduser()
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    paths = paths_for(root / args.object)
    if args.reuse_capture:
        if not all(path.is_file() for path in (paths.rgb, paths.depth, paths.camera)):
            parser.error("--reuse-capture 需要已有 rgb.png、depth.png、camera.json")
        previous_serial = json.loads(paths.camera.read_text(encoding="utf-8")).get("serial")
        if previous_serial != args.serial:
            parser.error(f"已有采集帧来自 {previous_serial}，与 --serial 不符")
    else:
        if any(path.exists() for path in (paths.rgb, paths.depth, paths.camera)):
            parser.error("目标目录已有采集帧；复用请加 --reuse-capture，新采集请换目录名")
        paths = capture_once(paths.object_dir, endpoint=args.camera_endpoint, expected_serial=args.serial)
    if args.reuse_mask:
        prompt = validate_reused_mask(paths, serial=args.serial)
        print(f"复用已确认 prompt={prompt!r}；mask={paths.mask}")
    else:
        selected = choose_mask(paths, category=category, initial_prompt=args.prompt, seg_server=args.seg_server)
        print(f"已确认 prompt={selected.prompt!r}；mask={selected.mask}")
    try:
        run_stages(
            paths, seg_server=args.seg_server, recon_server=args.recon_server,
            pose_server=args.pose_server, skip_pose=args.skip_pose,
            reuse_mesh=args.reuse_mesh, segmentation_used=not args.reuse_mask,
        )
    except RuntimeError as exc:
        if "CUDA out of memory" in str(exc):
            stage = "--reuse-mesh" if _mesh_manifest(paths).is_file() else "--reuse-mask"
            parser.exit(2, f"GPU 显存不足；已保留采集帧和已确认 mask。请释放其他 GPU 进程、"
                            f"重启失败的模型服务，然后加 {stage} 续跑 {args.object}。\n")
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
