"""单个物体的 camera、segmentation 和 FoundationPose 流程。"""

from __future__ import annotations

from gene_agent.perception_modules.paths import DEFAULT_TEST_DIR, PROJECT_ROOT

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image


_PERCEPTION_DIR = Path(__file__).resolve().parent
_REPO_ROOT = PROJECT_ROOT
for _path in (_REPO_ROOT, _PERCEPTION_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from gene_agent.perception_modules.camera import CameraClient
from gene_agent.perception_modules.pose_est import PoseClient
from gene_agent.perception_modules.pose_est.batch_est_pose import (
    correct_pose_with_external_knowledge,
    load_camera_parameters,
    save_pose_json,
)
from gene_agent.perception_modules.pose_est.microwave_pose_postprocess import (
    correct_microwave_pose_with_door_plane,
)
from gene_agent.perception_modules.pose_est.task_pose_est import pose_mesh_path
from gene_agent.perception_modules.segmentation import segment_image


def _resolve_path(path: str | Path) -> Path:
    """把相对路径解释为仓库根目录下的路径。"""
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (_REPO_ROOT / candidate).resolve()


def _find_object_dir(
    task_dir: Path,
    category: str,
    object_dir: str | Path | None,
) -> Path:
    """获取一个明确的物体实例目录；多个实例时要求显式指定。"""
    if object_dir is not None:
        result = _resolve_path(object_dir)
    else:
        category_root = task_dir / "objs" / category
        if not category_root.is_dir():
            category_root = next(
                (
                    path
                    for path in (task_dir / "objs").iterdir()
                    if path.is_dir() and path.name.lower() == category.lower()
                ),
                category_root,
            )
        instances = sorted(path for path in category_root.iterdir() if path.is_dir())
        if len(instances) != 1:
            raise ValueError(
                f"类别 {category} 有 {len(instances)} 个实例，请通过 --object-dir 指定一个"
            )
        result = instances[0]
    if not result.is_dir():
        raise FileNotFoundError(f"找不到物体目录: {result}")
    return result


def _check_http_server(server: str, name: str) -> None:
    """检查 HTTP server 是否已经启动。"""
    try:
        with urllib.request.urlopen(server.rstrip("/") + "/health", timeout=3) as response:
            result = json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.URLError) as exc:
        raise RuntimeError(f"{name} 未就绪: {server}") from exc
    if not result.get("ok"):
        raise RuntimeError(f"{name} 未就绪: {result}")


def _capture_initial_frame(
    task_dir: Path,
    camera_endpoint: str,
    timeout_ms: int,
) -> None:
    """从 camera server 取一帧并保存到任务级初始感知路径。"""
    client = CameraClient(camera_endpoint, timeout_ms=timeout_ms)
    try:
        client.health()
        packet = client.get_frame()
    finally:
        client.close()

    rgb_path = task_dir / "rgbs" / "rgb_0.png"
    depth_path = task_dir / "depths" / "depth_0.png"
    camera_path = task_dir / "camera" / "camera_parameters_0.json"
    rgb_path.parent.mkdir(parents=True, exist_ok=True)
    depth_path.parent.mkdir(parents=True, exist_ok=True)
    camera_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(packet.rgb).save(rgb_path)
    depth_mm = np.clip(np.rint(packet.depth * 1000.0), 0, 65535).astype(np.uint16)
    Image.fromarray(depth_mm).save(depth_path)
    metadata = {
        "intrinsic_matrix": packet.intrinsic_matrix.tolist(),
        "frame_number": packet.frame_number,
        "timestamp_ms": packet.timestamp_ms,
        "serial": packet.serial,
        "depth_unit": "millimeter",
    }
    if camera_path.is_file():
        previous = json.loads(camera_path.read_text(encoding="utf-8"))
        for key in ("left_extrinsic_matrix", "right_extrinsic_matrix"):
            if key in previous:
                metadata[key] = previous[key]
    camera_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def run_single_object_perception(
    task_dir: str | Path = DEFAULT_TEST_DIR,
    *,
    category: str = "Bottle",
    object_dir: str | Path | None = None,
    prompt: str = "a bottle",
    camera_endpoint: str = "tcp://127.0.0.1:18082",
    seg_server: str = "http://127.0.0.1:18080",
    pose_server: str = "http://127.0.0.1:18081",
    camera_timeout_ms: int = 6000,
    side: str = "left",
    use_microwave_door_plane_yaw: bool = True,
    visualize: bool = False,
) -> dict:
    """只处理一个物体，不调用 task level 的 pose 批处理函数。"""
    if side not in {"left", "right"}:
        raise ValueError("side 必须是 left 或 right")
    task_path = _resolve_path(task_dir)
    selected_object = _find_object_dir(task_path, category, object_dir)
    mask_path = selected_object / "masks" / "mask_0.png"
    mesh_path = pose_mesh_path(category, selected_object)
    rgb_path = task_path / "rgbs" / "rgb_0.png"
    depth_path = task_path / "depths" / "depth_0.png"
    camera_path = task_path / "camera" / "camera_parameters_0.json"

    camera = CameraClient(camera_endpoint, timeout_ms=camera_timeout_ms)
    try:
        camera_health = camera.health()
    finally:
        camera.close()
    if not camera_health.get("ok"):
        raise RuntimeError(f"camera server 未就绪: {camera_health}")
    _check_http_server(seg_server, "segmentation server")
    _check_http_server(pose_server, "pose server")
    _capture_initial_frame(task_path, camera_endpoint, camera_timeout_ms)

    mask_path.parent.mkdir(parents=True, exist_ok=True)
    status, segmentation = segment_image(
        seg_server,
        rgb_path,
        [{
            "category": category,
            "object_dir": str(selected_object),
            "prompt": prompt,
            "mask_path": str(mask_path),
        }],
    )
    if status != 200 or not segmentation.get("ok"):
        raise RuntimeError(f"segmentation 失败: {segmentation}")
    if not mask_path.is_file():
        raise RuntimeError(f"segmentation 未生成 mask: {mask_path}")

    session_id = f"{category}/{selected_object.name}"
    client = PoseClient(pose_server)
    registered = False
    try:
        response = client.register(
            session_id=session_id,
            rgb_path=rgb_path,
            depth_path=depth_path,
            mask_path=mask_path,
            mesh_path=mesh_path,
            camera_path=camera_path,
        )
        registered = True
        raw_pose = np.asarray(response["T_camera_object"], dtype=np.float64).reshape(4, 4)
        camera_parameters = load_camera_parameters(camera_path)
        pose, correction = correct_pose_with_external_knowledge(
            category,
            raw_pose,
            camera_parameters,
            side=side,
        )
        if use_microwave_door_plane_yaw and category.lower() == "microwave":
            task_from_camera = camera_parameters.get(f"{side}_extrinsic_matrix")
            if task_from_camera is None:
                raise ValueError(
                    f"{camera_path} 缺少 {side}_extrinsic_matrix，"
                    "无法进行门板平面 yaw 修正"
                )
            pose, plane_correction = correct_microwave_pose_with_door_plane(
                pose,
                category=category,
                camera_parameters=camera_parameters,
                task_from_camera=task_from_camera,
                rgb_path=rgb_path,
                depth_path=depth_path,
                concept_path=selected_object / "conceptualization.json",
            )
            correction = {
                **correction,
                "door_plane_yaw": plane_correction,
            }
        pose_path = selected_object / "pose.json"
        save_pose_json(
            pose_path,
            camera_id="0",
            pose=pose,
        )
    finally:
        if registered:
            client.stop(session_id)

    if visualize:
        from gene_agent.perception_modules.pose_est.visualization.scene_visualizer import (
            visualize_task_scene,
        )

        visualize_task_scene(task_path, categories=[category])
    return {
        "ok": True,
        "category": category,
        "object_dir": str(selected_object),
        "mask_path": str(mask_path),
        "pose_path": str(pose_path),
        "T_camera_object": pose.tolist(),
        "correction": correction,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="单个物体的 camera、segmentation 和 pose 流程")
    parser.add_argument("--task-dir", default=DEFAULT_TEST_DIR)
    parser.add_argument("--category", default="Bottle")
    parser.add_argument("--object-dir", default=None)
    parser.add_argument("--prompt", default="a bottle")
    parser.add_argument("--camera-endpoint", default="tcp://127.0.0.1:18082")
    parser.add_argument("--seg-server", default="http://127.0.0.1:18080")
    parser.add_argument("--pose-server", default="http://127.0.0.1:18081")
    parser.add_argument("--camera-timeout-ms", type=int, default=6000)
    parser.add_argument("--side", choices=("left", "right"), default="left")
    parser.add_argument(
        "--no-microwave-door-plane-yaw",
        dest="use_microwave_door_plane_yaw",
        action="store_false",
        help="关闭 Microwave handle pose 后的门板平面 yaw 修正",
    )
    parser.set_defaults(use_microwave_door_plane_yaw=True)
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()
    result = run_single_object_perception(
        args.task_dir,
        category=args.category,
        object_dir=args.object_dir,
        prompt=args.prompt,
        camera_endpoint=args.camera_endpoint,
        seg_server=args.seg_server,
        pose_server=args.pose_server,
        camera_timeout_ms=args.camera_timeout_ms,
        side=args.side,
        use_microwave_door_plane_yaw=args.use_microwave_door_plane_yaw,
        visualize=args.visualize,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
