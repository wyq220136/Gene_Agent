"""
使用 bbox prompt 测试 SAM3 对微波炉门的分割效果。

示例：
    # 弹出可视化窗口，依次点击微波炉门 bbox 的左上角和右下角
    python segmentation/bbox_seg_microwave_door.py \
        --data_dir data/long_demo

    # bbox 使用像素 xyxy 坐标：x1 y1 x2 y2
    python segmentation/bbox_seg_microwave_door.py \
        --data_dir data/long_demo \
        --bbox 120 80 520 430

    # 单物体目录，输出到 data_dir 下
    python segmentation/bbox_seg_microwave_door.py \
        --data_dir data/single_demo/Microwave/batch_01_0002 \
        --single \
        --bbox 0.20 0.10 0.82 0.78 \
        --bbox_format xyxy_norm

输出：
    批量模式：{data_dir}/Microwave/door_bbox_mask.png
             {data_dir}/Microwave/door_bbox_overlay.png
    单个模式：{data_dir}/door_bbox_mask.png
             {data_dir}/door_bbox_overlay.png
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import PIL.Image as Image
from PIL import ImageDraw
import torch



# 保证从仓库根目录或 segmentation 目录运行时，都能 import sam3 和 batch_seg_mask。
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
_PARENT_DIR = os.path.dirname(_SCRIPT_DIR)
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)
_REPO_ROOT = str(Path(__file__).resolve().parents[3])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from gene_agent.perception_modules.segmentation.batch_seg_mask import load_model
from gene_agent.perception_modules.paths import CHECKPOINT_ROOT


def _clamp(value, low, high):
    """把数值限制在合法范围内，避免 bbox 越界。"""
    return max(low, min(high, value))


def bbox_to_xyxy_abs(bbox, bbox_format, img_w, img_h):
    """把不同格式的 bbox 统一转换为像素 xyxy 坐标。"""
    if bbox_format == "xyxy_abs":
        x1, y1, x2, y2 = bbox
    elif bbox_format == "xyxy_norm":
        x1, y1, x2, y2 = (
            bbox[0] * img_w,
            bbox[1] * img_h,
            bbox[2] * img_w,
            bbox[3] * img_h,
        )
    elif bbox_format == "cxcywh_norm":
        cx, cy, bw, bh = bbox
        x1 = (cx - bw / 2.0) * img_w
        y1 = (cy - bh / 2.0) * img_h
        x2 = (cx + bw / 2.0) * img_w
        y2 = (cy + bh / 2.0) * img_h
    else:
        raise ValueError(f"不支持的 bbox_format: {bbox_format}")

    x1, x2 = sorted((float(x1), float(x2)))
    y1, y2 = sorted((float(y1), float(y2)))
    x1 = _clamp(x1, 0.0, float(img_w))
    y1 = _clamp(y1, 0.0, float(img_h))
    x2 = _clamp(x2, 0.0, float(img_w))
    y2 = _clamp(y2, 0.0, float(img_h))

    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"bbox 无效或越界后为空: {[x1, y1, x2, y2]}")

    return [x1, y1, x2, y2]


def xyxy_abs_to_cxcywh_norm(box_xyxy, img_w, img_h):
    """SAM3 的几何 prompt 需要归一化 cxcywh 格式。"""
    x1, y1, x2, y2 = box_xyxy
    cx = ((x1 + x2) / 2.0) / img_w
    cy = ((y1 + y2) / 2.0) / img_h
    bw = (x2 - x1) / img_w
    bh = (y2 - y1) / img_h
    return [cx, cy, bw, bh]


def build_output_paths(data_dir, single_mode, output_prefix):
    """按现有批量/单物体目录约定生成输出路径。"""
    img_path = os.path.join(data_dir, "rgb.png")
    if single_mode:
        out_dir = data_dir
    else:
        out_dir = os.path.join(data_dir, "Microwave")
        os.makedirs(out_dir, exist_ok=True)

    mask_path = os.path.join(out_dir, f"{output_prefix}_mask.png")
    overlay_path = os.path.join(out_dir, f"{output_prefix}_overlay.png")
    return img_path, mask_path, overlay_path


def select_bbox_interactively(image):
    """弹出图片窗口，手动点击左上角和右下角两个点生成 bbox。"""
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except Exception as e:
        raise RuntimeError(
            "无法打开可视化选择界面，请安装 matplotlib 或改用 --bbox 手动传入坐标。"
        ) from e

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(image)
    ax.set_title("Click top-left, then bottom-right of microwave door bbox")
    ax.axis("off")

    points = []
    rect_patch = None

    def onclick(event):
        nonlocal rect_patch
        if event.inaxes != ax or event.xdata is None or event.ydata is None:
            return

        points.append((float(event.xdata), float(event.ydata)))
        ax.plot(event.xdata, event.ydata, "ro", markersize=5)

        if len(points) == 1:
            ax.set_title("Click bottom-right of bbox")
        elif len(points) == 2:
            x1, y1 = points[0]
            x2, y2 = points[1]
            left, right = sorted((x1, x2))
            top, bottom = sorted((y1, y2))
            rect_patch = Rectangle(
                (left, top),
                right - left,
                bottom - top,
                linewidth=2,
                edgecolor="red",
                facecolor="none",
            )
            ax.add_patch(rect_patch)
            ax.set_title("bbox selected, closing window")
            fig.canvas.draw_idle()
            plt.pause(0.3)
            plt.close(fig)

        fig.canvas.draw_idle()

    cid = fig.canvas.mpl_connect("button_press_event", onclick)
    plt.show()
    fig.canvas.mpl_disconnect(cid)

    if len(points) < 2:
        raise RuntimeError("没有选满两个点，已取消 bbox 选择。")

    return [points[0][0], points[0][1], points[1][0], points[1][1]]


def save_best_mask(output, mask_path):
    """选择最高分 mask 保存为二值图。"""
    masks = output["masks"]
    scores = output["scores"]

    if len(scores) == 0:
        raise RuntimeError("SAM3 没有返回任何 mask，请检查 bbox 是否覆盖了微波炉门。")

    best_idx = torch.argmax(scores).item()
    best_score = float(scores[best_idx].detach().cpu().item())
    best_mask = masks[best_idx].cpu().to(torch.float32).numpy().squeeze()
    mask_uint8 = (best_mask * 255).astype(np.uint8)
    Image.fromarray(mask_uint8).save(mask_path)
    return mask_uint8, best_score, best_idx, len(scores)


def save_overlay(image, mask_uint8, boxes_xyxy, overlay_path):
    """保存 mask 半透明叠加图，并画出输入 bbox。"""
    base = image.convert("RGBA")
    mask_bool = mask_uint8 > 127

    # 青色区域表示 SAM3 输出的门 mask。
    color = np.zeros((mask_uint8.shape[0], mask_uint8.shape[1], 4), dtype=np.uint8)
    color[mask_bool] = [0, 220, 255, 105]
    overlay = Image.alpha_composite(base, Image.fromarray(color, mode="RGBA"))

    draw = ImageDraw.Draw(overlay)
    for box in boxes_xyxy:
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline=(255, 60, 0, 255), width=4)

    overlay.convert("RGB").save(overlay_path)


def main():
    parser = argparse.ArgumentParser(
        description="使用 bbox prompt 测试 SAM3 微波炉门分割"
    )
    parser.add_argument("--data_dir", type=str, required=True, help="包含 rgb.png 的数据目录")
    parser.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        action="append",
        metavar=("X1", "Y1", "X2", "Y2"),
        help="门的 bbox；可传多次。若不传，则打开窗口点击左上角和右下角",
    )
    parser.add_argument(
        "--bbox_format",
        choices=["xyxy_abs", "xyxy_norm", "cxcywh_norm"],
        default="xyxy_abs",
        help="bbox 坐标格式，默认 xyxy_abs",
    )
    parser.add_argument(
        "--single",
        action="store_true",
        help="单个模式：mask 直接输出到 data_dir 下",
    )
    parser.add_argument(
        "--output_prefix",
        default="door_bbox",
        help="输出文件名前缀，默认 door_bbox",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(CHECKPOINT_ROOT / "SAM3" / "sam3.pt"),
        help="SAM3 模型权重路径",
    )
    parser.add_argument("--device", type=str, default="cuda", help="计算设备 (cuda / cpu)")
    parser.add_argument(
        "--text_prompt",
        type=str,
        default=None,
        help="可选：先设置文本 prompt，再叠加 bbox；不传则只用 bbox prompt",
    )

    args = parser.parse_args()
    start_time = time.time()

    img_path, mask_path, overlay_path = build_output_paths(
        args.data_dir, args.single, args.output_prefix
    )
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"找不到图片: {img_path}")

    image = Image.open(img_path).convert("RGB")
    img_w, img_h = image.size
    print("=== SAM3 bbox prompt 微波炉门分割测试 ===")
    print(f"图片: {img_path}")
    print(f"图片尺寸: {img_w}x{img_h}")

    bbox_inputs = args.bbox
    bbox_format = args.bbox_format
    if not bbox_inputs:
        selected_bbox = select_bbox_interactively(image)
        bbox_inputs = [selected_bbox]
        bbox_format = "xyxy_abs"
        print(f"手动选择 bbox xyxy_abs={np.round(selected_bbox, 2).tolist()}")

    boxes_xyxy = [
        bbox_to_xyxy_abs(bbox, bbox_format, img_w, img_h)
        for bbox in bbox_inputs
    ]
    boxes_cxcywh = [
        xyxy_abs_to_cxcywh_norm(box, img_w, img_h)
        for box in boxes_xyxy
    ]

    for idx, (box_xyxy, box_cxcywh) in enumerate(zip(boxes_xyxy, boxes_cxcywh), start=1):
        print(f"bbox {idx} xyxy_abs={np.round(box_xyxy, 2).tolist()}")
        print(f"bbox {idx} cxcywh_norm={np.round(box_cxcywh, 4).tolist()}")

    _, processor = load_model(args.checkpoint, args.device)

    with torch.autocast(device_type=args.device, dtype=torch.bfloat16):
        state = processor.set_image(image)
        if args.text_prompt:
            print(f"先设置 text prompt: {args.text_prompt}")
            state = processor.set_text_prompt(prompt=args.text_prompt, state=state)

        output = None
        for box_prompt in boxes_cxcywh:
            output = processor.add_geometric_prompt(
                box=box_prompt,
                label=True,
                state=state,
            )

    mask_uint8, best_score, best_idx, mask_count = save_best_mask(output, mask_path)
    save_overlay(image, mask_uint8, boxes_xyxy, overlay_path)

    print(f"保存 mask: {mask_path}")
    print(f"保存 overlay: {overlay_path}")
    print(f"候选 mask 数量: {mask_count}, best_idx={best_idx}, score={best_score:.4f}")
    print(f"mask 面积: {(mask_uint8 > 127).sum()} px")
    print(f"耗时: {time.time() - start_time:.2f} 秒")


if __name__ == "__main__":
    main()
