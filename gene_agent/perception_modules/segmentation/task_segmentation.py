"""通过 camera server 和 seg_server 处理任务目录中物体的调试工具。

该文件既可以作为命令行脚本使用，也可以被其他模块导入。默认的任务结构为::

    data/perception_test/
      objs/<Category>/<ObjectInstance>/

相机帧保存为任务级初始感知，然后以临时目录请求分割服务，最后将 mask
写入对应物体实例目录的标准 masks 子目录。
"""

from __future__ import annotations

from gene_agent.perception_modules.paths import DEFAULT_TEST_DIR, PROJECT_ROOT

import argparse
from collections.abc import Mapping
import importlib.util
import json
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


_SEGMENTATION_DIR = Path(__file__).resolve().parent
_PERCEPTION_DIR = _SEGMENTATION_DIR.parent
_REPO_ROOT = PROJECT_ROOT
if str(_PERCEPTION_DIR) not in sys.path:
    sys.path.insert(0, str(_PERCEPTION_DIR))
if str(_SEGMENTATION_DIR) not in sys.path:
    sys.path.insert(0, str(_SEGMENTATION_DIR))


@dataclass(frozen=True)
class InitialPerceptionPaths:
    """任务开始时相机数据在任务目录中的标准位置。"""

    rgb: Path
    depth: Path
    camera_parameters: Path


def initial_perception_paths(task_dir: str | Path) -> InitialPerceptionPaths:
    """返回符合项目数据格式的初始感知文件路径。"""
    task_path = _resolve_path(task_dir)
    return InitialPerceptionPaths(
        rgb=task_path / "rgbs" / "rgb_0.png",
        depth=task_path / "depths" / "depth_0.png",
        camera_parameters=task_path / "camera" / "camera_parameters_0.json",
    )


def _resolve_path(path: str | Path) -> Path:
    """将相对路径优先解释为当前目录路径，再尝试解释为仓库路径。"""
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    if candidate.exists():
        return candidate.resolve()
    return (_REPO_ROOT / candidate).resolve()


def _load_config_mapping(config_path: str | Path) -> dict:
    """读取 Python 配置中的 CATEGORY_CONFIG 或 JSON 配置映射。"""
    path = _resolve_path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"找不到类别配置: {path}")

    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as file:
            config = json.load(file)
    elif path.suffix.lower() == ".py":
        spec = importlib.util.spec_from_file_location("task_category_config", path)
        if spec is None or spec.loader is None:
            raise ImportError(f"无法加载类别配置: {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        config = getattr(module, "CATEGORY_CONFIG", None)
        if config is None:
            config = getattr(module, "CATEGORIES", None)
    else:
        raise ValueError("类别配置只支持 .py 或 .json 文件")

    if isinstance(config, list):
        config = {str(category): {} for category in config}
    if not isinstance(config, dict) or not config:
        raise ValueError(
            f"类别配置必须是非空映射，或类别名称列表: {path}"
        )
    return {str(category): value for category, value in config.items()}


def load_category_names(config_path: str | Path) -> list[str]:
    """读取配置中的类别名称，并保持配置文件中的顺序。"""
    return list(_load_config_mapping(config_path).keys())


def _normalise_requested_categories(categories: str | Iterable[str]) -> list[str]:
    """统一处理 all、空白分隔字符串和类别名称列表。"""
    if isinstance(categories, str):
        requested = categories.split()
    else:
        requested = [str(category) for category in categories]
    if not requested:
        return ["all"]
    return list(dict.fromkeys(requested))


def discover_object_dirs(
    task_dir: str | Path,
    categories: str | Iterable[str],
    configured_categories: Iterable[str],
) -> list[tuple[str, Path]]:
    """发现 task_dir/objs 下指定类别的全部物体实例目录。"""
    object_root = _resolve_path(task_dir) / "objs"
    if not object_root.is_dir():
        raise FileNotFoundError(f"找不到物体目录: {object_root}")

    configured = list(configured_categories)
    requested = _normalise_requested_categories(categories)
    result: list[tuple[str, Path]] = []
    actual_dirs = {
        path.name.lower(): path
        for path in object_root.iterdir()
        if path.is_dir()
    }
    configured_by_lower = {name.lower(): name for name in configured}
    if requested == ["all"]:
        # all 表示任务目录实际存在的物体，配置中没有出现的类别不参与分割。
        selected = [
            configured_name
            for configured_name in configured
            if configured_name.lower() in actual_dirs
        ]
    else:
        unknown = [
            name for name in requested if name.lower() not in configured_by_lower
        ]
        if unknown:
            raise ValueError(f"配置中不存在这些类别: {unknown}")
        selected = [configured_by_lower[name.lower()] for name in requested]

    for category in selected:
        category_dir = actual_dirs.get(category.lower())
        if category_dir is None:
            raise FileNotFoundError(
                f"任务目录中找不到类别目录: {object_root / category}"
            )
        instances = sorted(path for path in category_dir.iterdir() if path.is_dir())
        if not instances:
            raise FileNotFoundError(f"类别目录没有物体实例: {category_dir}")
        result.extend((category, instance) for instance in instances)
    return result


def capture_camera_frame(
    camera_endpoint: str,
    task_dir: str | Path,
    *,
    timeout_ms: int = 6000,
) -> dict:
    """从 camera server 取一帧，并保存为任务初始感知数据。"""
    from gene_agent.perception_modules.camera.camera_client import CameraClient

    paths = initial_perception_paths(task_dir)
    client = CameraClient(camera_endpoint, timeout_ms=timeout_ms)
    try:
        packet = client.get_frame()
    finally:
        client.close()

    metadata = {
        "intrinsic_matrix": packet.intrinsic_matrix.tolist(),
        "frame_number": packet.frame_number,
        "timestamp_ms": packet.timestamp_ms,
        "serial": packet.serial,
        "depth_unit": "millimeter",
    }
    if paths.camera_parameters.is_file():
        previous = json.loads(paths.camera_parameters.read_text(encoding="utf-8"))
        for key in ("left_extrinsic_matrix", "right_extrinsic_matrix"):
            if key in previous:
                metadata[key] = previous[key]
    paths.rgb.parent.mkdir(parents=True, exist_ok=True)
    paths.depth.parent.mkdir(parents=True, exist_ok=True)
    paths.camera_parameters.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(packet.rgb).save(paths.rgb)
    # 项目规范使用 PNG；以毫米保存 uint16，camera_parameters_0.json 记录单位。
    depth_mm = np.clip(np.rint(packet.depth * 1000.0), 0, 65535).astype(np.uint16)
    Image.fromarray(depth_mm).save(paths.depth)
    paths.camera_parameters.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metadata


def remove_legacy_outputs(object_dirs: Iterable[Path]) -> list[Path]:
    """删除旧调试脚本生成的非规范文件，不触碰 mesh 和概念化数据。"""
    legacy_names = ("rgb.png", "depth.npy", "camera_info.json", "mask.png", "door_mask.png")
    removed = []
    for object_dir in object_dirs:
        for name in legacy_names:
            path = object_dir / name
            if path.is_file():
                path.unlink()
                removed.append(path)
    return removed


def _category_config(config_mapping: dict, category: str) -> dict:
    """大小写不敏感地取类别配置，未配置时使用简单 prompt。"""
    for name, config in config_mapping.items():
        if name.lower() == category.lower():
            if isinstance(config, dict) and config.get("masks"):
                return config
            break
    return {"masks": [{"prompt": f"a {category}", "suffix": ""}]}


def _build_task_segmentation_items(
    object_dirs: Iterable[tuple[str, Path]],
    config_mapping: dict,
    *,
    prompt_overrides: Mapping[str, str] | None = None,
) -> list[dict]:
    """根据任务物体目录和类别配置生成显式 server item。

    ``prompt_overrides`` 只覆盖本次请求的文本 prompt，不修改类别配置文件。
    key 可以是类别名（如 ``Bowl``）或实例目录名（如 ``Bowl_0``）。
    """
    items = []
    for category, object_dir in object_dirs:
        config = _category_config(config_mapping, category)
        masks = config.get("masks", [])
        if not isinstance(masks, list) or not masks:
            raise ValueError(f"类别 {category} 没有有效 masks 配置")
        for mask_cfg in masks:
            suffix = str(mask_cfg.get("suffix", ""))
            mask_name = "mask_0.png" if not suffix else f"mask_{suffix}.png"
            prompt = str(mask_cfg["prompt"])
            if prompt_overrides:
                override = None
                for key in (category, object_dir.name):
                    for configured_key, configured_prompt in prompt_overrides.items():
                        if str(configured_key).lower() == key.lower():
                            override = configured_prompt
                            break
                    if override is not None:
                        break
                if override is not None:
                    prompt = str(override).strip()
                    if not prompt:
                        raise ValueError(
                            f"{category} 的 segmentation prompt 覆盖值不能为空"
                        )
            items.append({
                "category": category,
                "object_dir": str(object_dir),
                "prompt": prompt,
                "mask_path": str(object_dir / "masks" / mask_name),
                "suffix": suffix,
                "crop_on_fail": mask_cfg.get("crop_on_fail", False),
                "crop_boxes": mask_cfg.get("crop_boxes", []),
                "fallback_prompts": mask_cfg.get("fallback_prompts", []),
                "llm_bbox_prompts": mask_cfg.get("llm_bbox_prompts", []),
            })
    return items


def _segment_items_once(
    segment_image,
    *,
    seg_server: str,
    image_path: Path,
    items: list[dict],
    seg_timeout: float | None,
) -> list[dict]:
    """调用一次 segmentation server，并把请求异常转换成失败结果。"""
    try:
        status, response = segment_image(
            seg_server,
            image_path,
            items,
            timeout=seg_timeout,
        )
        if status != 200 or not response.get("ok"):
            raise RuntimeError(response.get("error", f"seg_server HTTP {status}"))
        return list(response.get("results", []))
    except Exception as exc:
        return [
            {
                "ok": False,
                "category": item["category"],
                "object_dir": item["object_dir"],
                "prompt": item["prompt"],
                "mask_path": item["mask_path"],
                "error": str(exc),
                "error_type": type(exc).__name__,
            }
            for item in items
        ]


def segment_task_objects(
    task_dir: str | Path,
    categories: str | Iterable[str] = "all",
    *,
    config_path: str | Path = _SEGMENTATION_DIR / "category_config.py",
    camera_endpoint: str = "tcp://127.0.0.1:18082",
    seg_server: str = "http://127.0.0.1:18080",
    camera_timeout_ms: int = 6000,
    seg_timeout: float | None = None,
    prompt_overrides: Mapping[str, str] | None = None,
) -> list[dict]:
    """获取一帧并为任务目录下的每个目标实例执行 segmentation。

    返回列表中的成功项包含 ``ok=True`` 和 ``result``；失败项包含
    ``ok=False``、``error`` 和 ``error_type``，单个物体失败不会中断后续处理。
    segmentation server 会自动处理 ``red handle`` 的失败重试；客户端只发送
    一次请求，其他分割项仍只执行一次分割流程。
    """
    task_path = _resolve_path(task_dir)
    config_mapping = _load_config_mapping(config_path)
    configured_categories = list(config_mapping)
    object_dirs = discover_object_dirs(
        task_path,
        categories,
        configured_categories,
    )
    object_paths = [object_dir for _, object_dir in object_dirs]
    removed = remove_legacy_outputs(object_paths)
    for path in removed:
        print(f"已清理旧格式文件: {path}")
    capture_camera_frame(
        camera_endpoint,
        task_path,
        timeout_ms=camera_timeout_ms,
    )

    from seg_client import segment_image

    request_items = _build_task_segmentation_items(
        object_dirs,
        config_mapping,
        prompt_overrides=prompt_overrides,
    )
    paths = initial_perception_paths(task_path)
    results = _segment_items_once(
        segment_image,
        seg_server=seg_server,
        image_path=paths.rgb,
        items=request_items,
        seg_timeout=seg_timeout,
    )
    for result in results:
        state = "成功" if result.get("ok") else "失败"
        print(f"分割{state}: {result.get('category')}/{Path(result.get('object_dir', '')).name}"
              f" - {result.get('error', result.get('mask_path', ''))}")
    return results


def _check_servers(camera_endpoint: str, seg_server: str) -> None:
    """在真正处理前检查两个常驻服务。"""
    from gene_agent.perception_modules.camera.camera_client import CameraClient

    client = CameraClient(camera_endpoint)
    try:
        health = client.health()
    finally:
        client.close()
    if not health.get("ok"):
        raise RuntimeError(f"camera server 未就绪: {health}")

    with urllib.request.urlopen(seg_server.rstrip("/") + "/health", timeout=3) as response:
        health = json.loads(response.read().decode("utf-8"))
    if not health.get("ok"):
        raise RuntimeError(f"seg_server 未就绪: {health}")


def main() -> None:
    parser = argparse.ArgumentParser(description="感知测试目录的相机取帧与物体分割工具")
    parser.add_argument("--task-dir", default=DEFAULT_TEST_DIR)
    parser.add_argument("--categories", nargs="+", default=["all"])
    parser.add_argument(
        "--config",
        default=str(_SEGMENTATION_DIR / "category_config.py"),
        help="类别配置 .py 或 .json 文件",
    )
    parser.add_argument("--camera-endpoint", default="tcp://127.0.0.1:18082")
    parser.add_argument("--seg-server", default="http://127.0.0.1:18080")
    parser.add_argument("--camera-timeout-ms", type=int, default=6000)
    args = parser.parse_args()

    _check_servers(args.camera_endpoint, args.seg_server)
    results = segment_task_objects(
        args.task_dir,
        args.categories,
        config_path=args.config,
        camera_endpoint=args.camera_endpoint,
        seg_server=args.seg_server,
        camera_timeout_ms=args.camera_timeout_ms,
    )
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
