"""
统一分割模块 — 单个模式和批量模式合一
========================================

SAM3 先用 set_image() 将图片编码为 embedding，然后用多个 text prompt 快速生成 mask。
本文件合并了 segment.py（单个模式）和 batch_segment.py（批量模式）的功能，
通过 --single 参数切换模式。

【批量模式】场景中有多个物体（原 batch_segment.py 功能）
    python batch_seg_mask.py --data_dir data/long_demo --categories Bottle KitchenPot Microwave
    数据结构：
    {data_dir}/
        rgb.png              ← 输入：共享的场景图
        depth.npy
        Bottle/
            mask.png         ← 输出
        KitchenPot/
            mask.png         ← 输出
        Microwave/
            mask.png         ← 输出：把手
            door_mask.png    ← 输出：门

【单个模式】场景中只有一个物体（原 segment.py 功能）
    python batch_seg_mask.py --data_dir data/single_demo/Bottle/batch_01_0002 --categories Bottle --single
    数据结构：
    {data_dir}/
        rgb.png              ← 输入
        depth.npy
        mask.png             ← 输出
        door_mask.png        ← 输出（仅 Microwave）

核心设计：
- CATEGORY_CONFIG 字典集中管理所有类别的 prompt、额外 mask、crop 策略
- 普通分割项同一张图只做一次 set_image() 并复用 embedding；
  seg_server 对失败的 red handle 会重新 set_image() 后再使用原 prompt 重试
- 第一轮全图分割 → 失败项自动 crop + fallback prompts 重试
- crop 框使用比例坐标（0~1），适配不同尺寸图片
- **每个 mask 独立配置** crop 策略和 llm_bbox 保底 prompt（per-mask，非 per-object）
"""

import os
import sys
# 确保脚本所在目录在 sys.path 中，使得无论从哪个目录运行都能正确 import
# category_config 和 llm_bbox_seg（使用 abspath 处理 __file__ 可能为相对路径的情况）
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
# 同时把父目录也加入 sys.path，以便 llm_bbox_seg 内部可以 import 依赖
_PARENT_DIR = os.path.dirname(_SCRIPT_DIR)
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)
import time
import argparse
import numpy as np
import PIL.Image as Image
import torch

from gene_agent.perception_modules.segmentation.sam3_paths import configure_sam3_import

_SAM3_PATHS = configure_sam3_import()

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

from gene_agent.perception_modules.segmentation.category_config import CATEGORY_CONFIG


def _lookup_config(category):
    """大小写不敏感地查找 CATEGORY_CONFIG，找不到则返回默认配置。

    返回:
        dict: 包含 "masks" 列表的配置字典
    """
    # 精确匹配
    if category in CATEGORY_CONFIG:
        return CATEGORY_CONFIG[category]
    # 大小写不敏感匹配
    for key, val in CATEGORY_CONFIG.items():
        if key.lower() == category.lower():
            return val
    # 兜底：生成简单描述作为单个 mask
    return {
        "masks": [
            {
                "prompt": f"a {category}",
                "suffix": "",
            },
        ],
    }


def build_output_paths(data_dir, categories, single_mode):
    """根据模式构建 item 列表，每个 item 携带完整 per-mask 配置。

    参数:
        data_dir:    数据根目录
        categories:  物体类别列表
        single_mode: True=单个模式（mask 直接写在 data_dir 下），
                     False=批量模式（mask 写在 data_dir/{category}/ 下）

    返回:
        img_path: 输入 rgb 图片路径
        items:    [item_dict, ...]，每个 item 包含:
                  prompt, save_path, category, suffix,
                  crop_on_fail, crop_boxes, fallback_prompts, llm_bbox_prompts
    """
    img_path = os.path.join(data_dir, "rgb.png")
    items = []

    for category in categories:
        config = _lookup_config(category)
        masks = config["masks"]

        for mask_cfg in masks:
            prompt = mask_cfg["prompt"]
            suffix = mask_cfg["suffix"]
            mask_name = f"{suffix}_mask.png" if suffix else "mask.png"

            if single_mode:
                save_path = os.path.join(data_dir, mask_name)
            else:
                cat_dir = os.path.join(data_dir, category)
                os.makedirs(cat_dir, exist_ok=True)
                save_path = os.path.join(cat_dir, mask_name)

            # 每个 mask 携带自己的完整配置，后续 fallback 阶段直接使用
            items.append({
                "prompt": prompt,
                "save_path": save_path,
                "category": category,
                "suffix": suffix,
                "crop_on_fail": mask_cfg.get("crop_on_fail", False),
                "crop_boxes": mask_cfg.get("crop_boxes", []),
                "fallback_prompts": mask_cfg.get("fallback_prompts", []),
                "llm_bbox_prompts": mask_cfg.get("llm_bbox_prompts", []),
            })

    return img_path, items


def build_explicit_items(items):
    """校验 server 的显式 item，并转换为分割函数使用的内部结构。

    每个 item 必须提供 category、prompt 和 mask_path；调用方可以额外传递
    crop、fallback 等字段，它们会原样保留给后续分割流程使用。
    """
    if not isinstance(items, list) or not items:
        raise ValueError("items 必须是非空列表")

    normalized = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"items[{index}] 必须是 object")
        category = item.get("category")
        prompt = item.get("prompt")
        mask_path = item.get("mask_path")
        if not isinstance(category, str) or not category.strip():
            raise ValueError(f"items[{index}].category 必须是非空字符串")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"items[{index}].prompt 必须是非空字符串")
        if not isinstance(mask_path, str) or not mask_path.strip():
            raise ValueError(f"items[{index}].mask_path 必须是非空字符串")

        normalized_item = dict(item)
        normalized_item["category"] = category
        normalized_item["prompt"] = prompt
        normalized_item["save_path"] = mask_path
        os.makedirs(os.path.dirname(mask_path) or ".", exist_ok=True)
        normalized.append(normalized_item)
    return normalized


def load_model(checkpoint_path, device="cuda"):
    """加载 SAM3 模型到 GPU（全程只调用一次）。"""
    # BPE 文件使用仓库内 third_party/sam3 的资源，不能依赖当前工作目录。
    bpe_path = str(_SAM3_PATHS.bpe_path)
    # ckpt 可能放在当前目录、仓库根目录或 segmentation/ckpts 下，依次尝试。
    if not os.path.isabs(checkpoint_path) and not os.path.exists(checkpoint_path):
        candidate_paths = [
            os.path.join(_SAM3_PATHS.repo_root, checkpoint_path),
            os.path.join(_PARENT_DIR, checkpoint_path),
            os.path.join(_SCRIPT_DIR, checkpoint_path),
        ]
        for candidate_path in candidate_paths:
            if os.path.exists(candidate_path):
                checkpoint_path = candidate_path
                break
        else:
            raise FileNotFoundError(
                "找不到 SAM3 checkpoint，请用 --checkpoint 指定权重路径。"
                f" 已尝试: {[checkpoint_path] + candidate_paths}"
            )

    print(f"正在加载 SAM3 模型 (device={device}, ckpt={checkpoint_path})...")
    model = build_sam3_image_model(
        bpe_path=bpe_path,
        checkpoint_path=checkpoint_path,
        load_from_HF=False,
    ).to(device)
    processor = Sam3Processor(model)
    print("模型加载完成。")
    return model, processor


def seg_with_prompts(processor, inference_state, items, device="cuda"):
    """对已编码的图片执行多个 text prompt 分割（复用同一个 embedding）。

    参数:
        processor:        SAM3 processor
        inference_state:  set_image() 返回的状态（图片 embedding）
        items:            [item_dict, ...] 每个 item 是包含 prompt/save_path/... 的字典
        device:           计算设备

    返回:
        failures:  [item_dict, ...] 首次分割失败的项（保留完整配置）
    """
    failures = []

    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        for item in items:
            prompt = item["prompt"]
            save_path = item["save_path"]
            category = item["category"]

            try:
                output = processor.set_text_prompt(state=inference_state, prompt=prompt)
                masks = output["masks"]
                scores = output["scores"]

                if len(scores) > 0:
                    best_idx = torch.argmax(scores).item()
                    best_mask = masks[best_idx].cpu().to(torch.float32).numpy().squeeze()
                    mask_img = Image.fromarray((best_mask * 255).astype(np.uint8))
                    mask_img.save(save_path)
                    print(f"  ✓ [{category}] '{prompt}' → {save_path}")
                else:
                    print(f"  ✗ [{category}] '{prompt}' 未识别到物体")
                    failure = dict(item)
                    failure["error"] = "未识别到物体"
                    failures.append(failure)

            except Exception as e:
                print(f"  ✗ [{category}] '{prompt}' 出错: {e}")
                failure = dict(item)
                failure["error"] = str(e)
                failures.append(failure)

    return failures


def seg_with_llm_bbox_fallback(
    processor,
    inference_state,
    image,
    failures,
    device="cuda",
    image_path="",
    verbose=False,
):
    """对文本 prompt 分割失败的项尝试 LLM bbox → SAM3 分割。"""
    from gene_agent.perception_modules.segmentation.llm_bbox_seg import llm_bbox_fallback_batch

    print(f"\n--- 第二轮：LLM bbox 保底分割 ({len(failures)} 项待重试) ---")
    still_failed = llm_bbox_fallback_batch(
        processor, inference_state, image, failures, device, image_path,
        verbose=verbose,
    )
    print(f"--- LLM bbox 保底分割完成:"
          f" 成功 {len(failures) - len(still_failed)},"
          f" 仍失败 {len(still_failed)} ---")
    return still_failed
