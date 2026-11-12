"""批量验证任务目录中物体的首帧 pose。"""

from __future__ import annotations

from gene_agent.perception_modules.paths import DEFAULT_TEST_DIR, PROJECT_ROOT

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from .batch_est_pose import correct_pose_with_external_knowledge, load_camera_parameters, save_pose_json
from .microwave_pose_postprocess import correct_microwave_pose_with_door_plane
from .pose_client import PoseClient


_POSE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = PROJECT_ROOT


def _resolve_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = _REPO_ROOT / candidate
    return candidate.resolve()


def _normalize_categories(categories: str | Iterable[str]) -> list[str]:
    if isinstance(categories, str):
        values = categories.split()
    else:
        values = [str(value) for value in categories]
    return list(dict.fromkeys(values or ["all"]))


def discover_task_objects(
    task_dir: str | Path,
    categories: str | Iterable[str] = "all",
) -> list[tuple[str, Path]]:
    """发现任务目录中指定类别的全部物体实例。"""
    object_root = _resolve_path(task_dir) / "objs"
    if not object_root.is_dir():
        raise FileNotFoundError(f"找不到物体目录: {object_root}")

    category_dirs = {
        path.name.lower(): path
        for path in object_root.iterdir()
        if path.is_dir()
    }
    requested = _normalize_categories(categories)
    if requested == ["all"]:
        selected_dirs = sorted(category_dirs.values(), key=lambda path: path.name)
    else:
        selected_dirs = []
        for name in requested:
            category_dir = category_dirs.get(name.lower())
            if category_dir is None:
                raise FileNotFoundError(f"任务中不存在类别: {name}")
            selected_dirs.append(category_dir)

    objects = []
    for category_dir in selected_dirs:
        instances = sorted(path for path in category_dir.iterdir() if path.is_dir())
        if not instances:
            raise FileNotFoundError(f"类别目录没有物体实例: {category_dir}")
        objects.extend((category_dir.name, instance) for instance in instances)
    return objects


def _required_paths(
    task_dir: Path,
    object_dir: Path,
    *,
    camera_id: str = "0",
) -> dict[str, Path]:
    return {
        "rgb_path": task_dir / "rgbs" / "rgb_0.png",
        "depth_path": task_dir / "depths" / "depth_0.png",
        "camera_path": task_dir / "camera" / f"camera_parameters_{camera_id}.json",
        "mask_path": object_dir / "masks" / "mask_0.png",
        "mesh_path": pose_mesh_path(object_dir.parent.name, object_dir),
    }


def pose_mesh_path(category: str, object_dir: str | Path) -> Path:
    """返回 FoundationPose 使用的模型 mesh。

    Microwave 的 SAM mask 只覆盖 handle，因此 pose 注册必须使用同一
    handle mesh；FoundationPose 返回的 pose 仍保存到 Microwave 目录，
    供整个 concept 的所有部件共同使用。其他类别继续使用 obj_mesh。
    """
    object_path = Path(object_dir)
    if str(category).lower() == "microwave":
        return object_path / "meshes" / "handle_mesh.obj"
    return object_path / "meshes" / "obj_mesh.obj"


def estimate_task_poses(
    task_dir: str | Path,
    categories: str | Iterable[str] = "all",
    *,
    client: PoseClient | None = None,
    server: str = "http://127.0.0.1:18081",
    camera_id: str = "0",
    side: str = "left",
    use_microwave_door_plane_yaw: bool = True,
    visualize: bool = False,
) -> list[dict]:
    """逐个请求首帧 pose，保存 pose.json，并返回成功/失败信息。"""
    if side not in {"left", "right"}:
        raise ValueError("side 必须是 left 或 right")
    task_path = _resolve_path(task_dir)
    objects = discover_task_objects(task_path, categories)
    pose_client = client or PoseClient(server)
    camera_path = task_path / "camera" / f"camera_parameters_{camera_id}.json"
    camera_parameters = load_camera_parameters(camera_path)
    results = []

    for category, object_dir in objects:
        session_id = f"{category}/{object_dir.name}"
        paths = _required_paths(task_path, object_dir, camera_id=camera_id)
        missing = [str(path) for path in paths.values() if not path.is_file()]
        if missing:
            error = f"缺少 pose 输入文件: {missing}"
            print(f"pose 失败: {session_id} - {error}")
            results.append({
                "ok": False,
                "category": category,
                "object_dir": str(object_dir),
                "error": error,
            })
            continue

        registered = False
        print(f"开始 pose 估计: {session_id}")
        try:
            response = pose_client.register(
                session_id=session_id,
                camera_id=camera_id,
                **paths,
            )
            registered = True
            raw_pose = np.asarray(
                response["T_camera_object"], dtype=np.float64
            ).reshape(4, 4)
            corrected_pose, correction = correct_pose_with_external_knowledge(
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
                corrected_pose, plane_correction = (
                    correct_microwave_pose_with_door_plane(
                        corrected_pose,
                        category=category,
                        camera_parameters=camera_parameters,
                        task_from_camera=task_from_camera,
                        rgb_path=paths["rgb_path"],
                        depth_path=paths["depth_path"],
                        concept_path=object_dir / "conceptualization.json",
                    )
                )
                correction = {
                    **correction,
                    "door_plane_yaw": plane_correction,
                }
            pose_path = object_dir / "pose.json"
            save_pose_json(
                pose_path,
                camera_id=camera_id,
                pose=corrected_pose,
            )
            results.append({
                "ok": True,
                "category": category,
                "object_dir": str(object_dir),
                "pose_path": str(pose_path),
                "camera_id": camera_id,
                "pose": corrected_pose.tolist(),
                "correction": correction,
            })
            print(f"pose 成功: {session_id} -> {pose_path}")
        except Exception as exc:
            results.append({
                "ok": False,
                "category": category,
                "object_dir": str(object_dir),
                "error": str(exc),
                "error_type": type(exc).__name__,
            })
            print(f"pose 失败: {session_id} - {exc}")
        finally:
            if registered:
                try:
                    pose_client.stop(session_id)
                except Exception as exc:
                    print(f"session 释放失败: {session_id} - {exc}")

    if visualize:
        try:
            from .visualization.scene_visualizer import visualize_task_scene
        except ImportError:
            from gene_agent.perception_modules.pose_est.visualization.scene_visualizer import visualize_task_scene
        visualize_task_scene(task_path, categories=categories)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="批量验证任务目录首帧 pose")
    parser.add_argument("--task-dir", default=DEFAULT_TEST_DIR)
    parser.add_argument("--categories", nargs="+", default=["all"])
    parser.add_argument("--server", default="http://127.0.0.1:18081")
    parser.add_argument("--camera-id", default="0")
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

    results = estimate_task_poses(
        args.task_dir,
        args.categories,
        server=args.server,
        camera_id=args.camera_id,
        side=args.side,
        use_microwave_door_plane_yaw=args.use_microwave_door_plane_yaw,
        visualize=args.visualize,
    )
    print(json.dumps(results, ensure_ascii=False, indent=2))
    raise SystemExit(0 if all(result.get("ok") for result in results) else 1)


if __name__ == "__main__":
    main()
