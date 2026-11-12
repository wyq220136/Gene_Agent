"""
LLM bbox 保底分割验证脚本 — 不依赖 seg_server，直接加载 SAM3 测试。
支持从本地图片文件或 camera server 实时取帧。

用法:
    # 从本地图片测试
    conda activate sam3
    export API_KEY="your-key"
    python -m gene_agent.perception_modules.segmentation.test_llm_bbox \
        --image_path data/perception_test/rgbs/rgb_0.png \
        --category Microwave

    # 从 camera server 实时取帧测试
    python -m gene_agent.perception_modules.segmentation.test_llm_bbox \
        --use_camera --camera_endpoint tcp://127.0.0.1:18082 \
        --category Microwave

    # 取帧并保存到指定目录
    python -m gene_agent.perception_modules.segmentation.test_llm_bbox \
        --use_camera \
        --output_dir /tmp/llm_bbox_test
"""

from __future__ import annotations

from gene_agent.perception_modules.paths import CHECKPOINT_ROOT, PROJECT_ROOT

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import PIL.Image as Image
import torch


_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = PROJECT_ROOT
for _path in (_REPO_ROOT, _SCRIPT_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from gene_agent.perception_modules.segmentation.batch_seg_mask import load_model, seg_with_prompts, build_explicit_items
from gene_agent.perception_modules.segmentation.llm_bbox_seg import llm_bbox_fallback_batch


def _capture_from_camera(
    camera_endpoint: str,
    output_dir: str,
    timeout_ms: int = 6000,
) -> str:
    """从 camera server 取一帧 RGB，保存到 output_dir 并返回图片路径。"""
    from gene_agent.perception_modules.camera import CameraClient

    os.makedirs(output_dir, exist_ok=True)
    print(f"连接 camera server: {camera_endpoint}")
    client = CameraClient(camera_endpoint, timeout_ms=timeout_ms)
    try:
        health = client.health()
        if not health.get("ok"):
            raise RuntimeError(f"camera server 未就绪: {health}")
        print(f"camera server 状态: {health}")
        packet = client.get_frame()
        print(f"取帧成功: frame={packet.frame_number},"
              f" rgb={packet.rgb.shape}, timestamp_ms={packet.timestamp_ms:.0f}")
    finally:
        client.close()

    image_path = os.path.join(output_dir, "camera_rgb.png")
    Image.fromarray(packet.rgb).save(image_path)
    print(f"RGB 已保存: {image_path}")
    return image_path


def _print_separator(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def test_llm_bbox_fallback(
    image_path: str,
    category: str = "Microwave",
    mask_suffix: str = "door",
    prompt: str = "the door of microwave",
    output_dir: str = "/tmp/llm_bbox_test",
    checkpoint: str = str(CHECKPOINT_ROOT / "SAM3" / "sam3.pt"),
    device: str = "cuda",
    verbose: bool = False,
) -> dict:
    """验证 LLM bbox 保底分割流程。

    1. 加载 SAM3
    2. 先用文本 prompt 分割（可能失败）
    3. 对失败项尝试 LLM bbox fallback
    4. 输出结果和对比
    """
    os.makedirs(output_dir, exist_ok=True)
    start_time = time.time()

    # ---- 检查输入 ----
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"找不到图片: {image_path}")
    api_key = os.getenv("API_KEY", "")
    print(f"API_KEY 已设置: {'是' if api_key else '否（LLM fallback 将被跳过）'}")

    # ---- 加载 SAM3 ----
    _print_separator("加载 SAM3 模型")
    _, processor = load_model(checkpoint, device)

    # ---- 准备测试 item ----
    mask_name = f"{mask_suffix}_mask.png" if mask_suffix else "mask.png"
    mask_path = os.path.join(output_dir, mask_name)
    overlay_path = os.path.join(output_dir, f"{mask_suffix}_overlay.png" if mask_suffix else "overlay.png")

    from gene_agent.perception_modules.segmentation.category_config import CATEGORY_CONFIG
    config = CATEGORY_CONFIG.get(category, {})
    masks_config = config.get("masks", [])

    # 找到匹配 suffix 的 mask 配置
    mask_cfg = next(
        (m for m in masks_config if m.get("suffix", "") == mask_suffix),
        masks_config[0] if masks_config else {},
    )

    llm_bbox_prompts = mask_cfg.get("llm_bbox_prompts", [])
    print(f"类别: {category}, prompt: '{prompt}', suffix: '{mask_suffix}'")
    print(f"llm_bbox_prompts: {llm_bbox_prompts}")

    items = build_explicit_items([{
        "category": category,
        "prompt": prompt,
        "mask_path": mask_path,
        "llm_bbox_prompts": llm_bbox_prompts,
        "crop_on_fail": mask_cfg.get("crop_on_fail", False),
        "crop_boxes": mask_cfg.get("crop_boxes", []),
        "fallback_prompts": mask_cfg.get("fallback_prompts", []),
        "image_path": os.path.abspath(image_path),
    }])

    # ---- 加载图片 ----
    image = Image.open(image_path).convert("RGB")
    img_w, img_h = image.size
    print(f"图片: {image_path} ({img_w}x{img_h})")

    # ---- set_image ----
    _print_separator("图片编码")
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        inference_state = processor.set_image(image)
    print("图片 embedding 编码完成。")

    # ---- 第一轮：文本 prompt 分割 ----
    _print_separator("第一轮：文本 prompt 分割")
    failures = seg_with_prompts(processor, inference_state, items, device)
    text_success = len(failures) == 0

    if text_success:
        print("✓ 文本 prompt 分割成功（无需 LLM fallback）")
    else:
        for f in failures:
            print(f"  ✗ 失败: [{f['category']}] '{f['prompt']}' — {f.get('error', '')}")

    # ---- 第二轮：LLM bbox fallback ----
    if failures:
        _print_separator("第二轮：LLM bbox 保底分割")
        still_failed = llm_bbox_fallback_batch(
            processor, inference_state, image, failures, device, image_path,
            verbose=verbose,
        )
        llm_success = len(still_failed) == 0
        if llm_success:
            print("✓ LLM bbox 保底分割成功！")
        else:
            for f in still_failed:
                print(f"  ✗ 仍然失败: [{f['category']}] '{f['prompt']}' — {f.get('error', '')}")
    else:
        still_failed = []
        llm_success = None  # 不需要 fallback

    # ---- 保存 overlay 对比图 ----
    _print_separator("保存结果")
    if os.path.isfile(mask_path):
        mask_img = np.array(Image.open(mask_path).convert("L"))
        mask_bool = mask_img > 127
        mask_area = mask_bool.sum()
        print(f"mask 已保存: {mask_path}")
        print(f"mask 面积: {mask_area} px ({mask_area / (img_w * img_h) * 100:.1f}%)")

        # 生成 overlay 图
        base = image.convert("RGBA")
        color = np.zeros((img_h, img_w, 4), dtype=np.uint8)
        color[mask_bool] = [0, 220, 255, 128]  # 青色半透明
        overlay = Image.alpha_composite(base, Image.fromarray(color, mode="RGBA"))
        overlay.convert("RGB").save(overlay_path)
        print(f"overlay 已保存: {overlay_path}")
    else:
        print(f"✗ mask 未生成")

    elapsed = time.time() - start_time
    _print_separator(f"完成 ({elapsed:.1f} 秒)")

    return {
        "image_path": image_path,
        "category": category,
        "prompt": prompt,
        "mask_suffix": mask_suffix,
        "text_prompt_success": text_success,
        "llm_bbox_fallback_attempted": len(failures) > 0 if failures else False,
        "llm_bbox_fallback_success": llm_success,
        "mask_path": mask_path if os.path.isfile(mask_path) else None,
        "overlay_path": overlay_path if os.path.isfile(overlay_path) else None,
        "elapsed_seconds": elapsed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LLM bbox 保底分割验证 — 不依赖 server，直接加载 SAM3 测试"
    )
    # 图片来源：二选一
    parser.add_argument(
        "--image_path", default=None,
        help="输入 RGB 图片路径（与 --use_camera 二选一）"
    )
    parser.add_argument(
        "--use_camera", action="store_true",
        help="从 camera server 实时取帧（需要 camera server 已启动）"
    )
    parser.add_argument(
        "--camera_endpoint", default="tcp://127.0.0.1:18082",
        help="camera server 地址（默认 tcp://127.0.0.1:18082）"
    )
    parser.add_argument(
        "--camera_timeout_ms", type=int, default=6000,
        help="camera 取帧超时毫秒（默认 6000）"
    )
    # 分割参数
    parser.add_argument(
        "--category", default="Microwave",
        help="物体类别（默认 Microwave）"
    )
    parser.add_argument(
        "--mask_suffix", default="door",
        help="mask suffix，如 door / ''（默认 door）"
    )
    parser.add_argument(
        "--prompt", default="the entire front door of microwave including the handle",
        help="文本 prompt（默认 'the entire front door of microwave including the handle'）"
    )
    # 输出和模型
    parser.add_argument(
        "--output_dir", default="/tmp/llm_bbox_test",
        help="输出目录（默认 /tmp/llm_bbox_test）"
    )
    parser.add_argument(
        "--checkpoint", default=str(CHECKPOINT_ROOT / "SAM3" / "sam3.pt"),
        help="SAM3 权重路径"
    )
    parser.add_argument(
        "--device", default="cuda",
        help="计算设备（cuda / cpu）"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="打印 VLM 原始返回等详细调试信息"
    )
    args = parser.parse_args()

    # ---- 确定图片来源 ----
    if args.use_camera and args.image_path:
        parser.error("--use_camera 和 --image_path 不能同时指定")
    if not args.use_camera and not args.image_path:
        parser.error("必须指定 --image_path 或 --use_camera 之一")

    if args.use_camera:
        image_path = _capture_from_camera(
            camera_endpoint=args.camera_endpoint,
            output_dir=args.output_dir,
            timeout_ms=args.camera_timeout_ms,
        )
    else:
        image_path = args.image_path

    result = test_llm_bbox_fallback(
        image_path=image_path,
        category=args.category,
        mask_suffix=args.mask_suffix,
        prompt=args.prompt,
        output_dir=args.output_dir,
        checkpoint=args.checkpoint,
        device=args.device,
        verbose=args.verbose,
    )
    print("\n结果摘要:")
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
