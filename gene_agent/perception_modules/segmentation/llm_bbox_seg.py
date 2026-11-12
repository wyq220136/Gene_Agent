"""
LLM bbox → SAM3 分割 fallback 模块。

当文本 prompt 分割失败时，如果 item 配置了 llm_bbox_prompts 且 API_KEY 已设置，
则调用 VLM 获取物体 bbox，再用 SAM3 的 add_geometric_prompt() 进行 bbox 分割。

使用方式:
    from gene_agent.perception_modules.segmentation.llm_bbox_seg import llm_bbox_segment

    # 对单个失败项尝试 LLM bbox 分割
    success = llm_bbox_segment(processor, inference_state, image, item, device)
"""

from __future__ import annotations

from gene_agent.perception_modules.paths import PROJECT_ROOT

import os
import sys
from typing import Optional

import numpy as np
import PIL.Image as Image
import torch


# 确保能 import agents.llm
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = PROJECT_ROOT
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from .vlm_client import get_bbox_from_vlm


def _bbox_norm_to_cxcywh(box_norm):
    """把归一化 xyxy bbox [x1, y1, x2, y2] 转为 SAM3 所需的 cxcywh_norm 格式。

    SAM3 的 add_geometric_prompt() 接受 [cx, cy, w, h] 归一化坐标。
    """
    x1, y1, x2, y2 = box_norm
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w = x2 - x1
    h = y2 - y1
    return [cx, cy, w, h]


def llm_bbox_segment(
    processor,
    inference_state,
    image: Image.Image,
    item: dict,
    device: str = "cuda",
    verbose: bool = False,
) -> bool:
    """对单个失败项尝试 LLM bbox → SAM3 分割。

    流程:
        1. 检查 item 是否有 llm_bbox_prompts 配置
        2. 检查 API_KEY 环境变量是否已设置
        3. 遍历 llm_bbox_prompts，调用 VLM 获取 bbox
        4. 用 bbox 调用 SAM3 add_geometric_prompt() 生成 mask
        5. 保存最佳 mask

    参数:
        processor:        SAM3 processor 实例
        inference_state:  set_image() 返回的状态（图片 embedding）
        image:            PIL Image 对象
        item:             分割项字典，需包含 llm_bbox_prompts / save_path / category / prompt
        device:           计算设备

    返回:
        True 表示 LLM bbox 分割成功并已保存 mask，False 表示失败
    """
    # 检查是否有 llm_bbox_prompts 配置
    llm_prompts = item.get("llm_bbox_prompts", [])
    if not llm_prompts:
        return False

    # 检查 API_KEY 是否可用
    api_key = os.getenv("API_KEY", "")
    if not api_key:
        print(f"   [LLM Bbox] 跳过: API_KEY 未设置")
        return False

    category = item.get("category", "unknown")
    prompt = item.get("prompt", "")
    save_path = item.get("save_path", "")
    print(f"   [LLM Bbox] [{category}] 尝试 LLM bbox 分割: '{prompt}'")

    # 获取图片路径（需要传给 VLM）
    # 从 image 对象无法直接获取路径，需要把当前帧写入临时文件
    # 或者支持传入图片路径。这里先检查 item 中有没有 image_path
    image_path = item.get("image_path", "")
    if not image_path or not os.path.isfile(image_path):
        # 如果没有图片路径，写入临时文件
        import tempfile
        tmp_path = os.path.join(tempfile.gettempdir(), f"_llm_bbox_{os.getpid()}.png")
        image.save(tmp_path)
        image_path = tmp_path
        _cleanup_tmp = True
    else:
        _cleanup_tmp = False

    try:
        img_w, img_h = image.size

        # 遍历 llm_bbox_prompts 尝试
        for bbox_prompt in llm_prompts:
            vlm_result = get_bbox_from_vlm(image_path, str(bbox_prompt), verbose=verbose)
            if vlm_result is None:
                print(f"   [LLM Bbox] VLM 未返回有效 bbox: '{bbox_prompt}'")
                continue

            box_norm = vlm_result.get("box_2d_norm")
            if not box_norm or len(box_norm) != 4:
                print(f"   [LLM Bbox] bbox 格式无效: {box_norm}")
                continue

            # 转为 SAM3 几何 prompt 格式 (cxcywh_norm)
            box_cxcywh = _bbox_norm_to_cxcywh(box_norm)
            print(f"   [LLM Bbox] VLM bbox xyxy_norm={[round(v, 4) for v in box_norm]}"
                  f" → cxcywh_norm={[round(v, 4) for v in box_cxcywh]}")

            # 调用 SAM3 bbox prompt
            try:
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    # 重置所有 prompt，只用 bbox
                    processor.reset_all_prompts(inference_state)
                    output = processor.add_geometric_prompt(
                        box=box_cxcywh,
                        label=True,
                        state=inference_state,
                    )

                masks = output.get("masks")
                scores = output.get("scores")

                if masks is not None and scores is not None and len(scores) > 0:
                    best_idx = torch.argmax(scores).item()
                    best_score = float(scores[best_idx].detach().cpu().item())
                    best_mask = masks[best_idx].cpu().to(torch.float32).numpy().squeeze()

                    # 用 bbox 约束 mask（只保留 bbox 内的部分，防止 VLM bbox 不精确时误分割）
                    x1 = int(box_norm[0] * img_w)
                    y1 = int(box_norm[1] * img_h)
                    x2 = int(box_norm[2] * img_w)
                    y2 = int(box_norm[3] * img_h)
                    x1, x2 = sorted((max(0, x1), min(img_w, x2)))
                    y1, y2 = sorted((max(0, y1), min(img_h, y2)))

                    bbox_mask = np.zeros((img_h, img_w), dtype=bool)
                    bbox_mask[y1:y2, x1:x2] = True
                    constrained_mask = np.logical_and(best_mask > 0.5, bbox_mask)

                    if constrained_mask.sum() > 0:
                        mask_img = Image.fromarray(
                            (constrained_mask.astype(np.uint8)) * 255
                        )
                        mask_img.save(save_path)
                        print(f"   [LLM Bbox] ✓ [{category}] '{prompt}'"
                              f" → bbox prompt '{bbox_prompt}'"
                              f" → {save_path} (score={best_score:.4f})")
                        return True
                    else:
                        print(f"   [LLM Bbox] ✗ bbox 约束后 mask 为空 (score={best_score:.4f})")
                else:
                    print(f"   [LLM Bbox] ✗ SAM3 未返回 mask")

            except Exception as e:
                print(f"   [LLM Bbox] ✗ SAM3 bbox 分割出错: {type(e).__name__} - {e}")
                continue

        # 所有 llm_bbox_prompts 都尝试失败
        return False

    finally:
        if _cleanup_tmp and os.path.isfile(image_path):
            try:
                os.unlink(image_path)
            except OSError:
                pass


def llm_bbox_fallback_batch(
    processor,
    inference_state,
    image: Image.Image,
    failures: list,
    device: str = "cuda",
    image_path: str = "",
    verbose: bool = False,
) -> list:
    """对一批分割失败项尝试 LLM bbox fallback。

    每个失败项独立尝试，不影响其他项。

    参数:
        processor:        SAM3 processor 实例
        inference_state:  set_image() 返回的状态
        image:            PIL Image 对象
        failures:         失败项列表 [item_dict, ...]
        device:           计算设备
        image_path:       原始图片路径（用于 VLM 调用）

    返回:
        still_failed:  [item_dict, ...] 仍然失败的项
    """
    if not failures:
        return []

    # 检查是否有任何项配置了 llm_bbox_prompts
    has_llm_config = any(
        item.get("llm_bbox_prompts") for item in failures
    )
    if not has_llm_config:
        return failures

    # 检查 API_KEY
    if not os.getenv("API_KEY", ""):
        print("   [LLM Bbox] 跳过: API_KEY 未设置")
        return failures

    # 把 image_path 注入到 item 中，便于 llm_bbox_segment 使用
    for item in failures:
        if "image_path" not in item:
            item["image_path"] = image_path

    still_failed = []
    for item in failures:
        success = llm_bbox_segment(
            processor, inference_state, image, item, device, verbose=verbose,
        )
        if not success:
            still_failed.append(item)

    return still_failed
