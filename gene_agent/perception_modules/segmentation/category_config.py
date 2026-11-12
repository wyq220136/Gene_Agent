"""
物体类别分割配置（per-mask 结构）
================================

每个 category 下有一个 "masks" 列表，每个 mask 独立配置：

字段说明：
  prompt:          SAM3 文本 prompt（必填）
  suffix:          输出文件名后缀，""=mask.png，"door"=door_mask.png（必填）
  crop_on_fail:    首次分割失败后是否启用 crop 重试（可选，默认 False）
  crop_boxes:      裁剪框列表，坐标使用 0~1 比例（可选，默认 []）
  fallback_prompts: crop 后尝试的替代 SAM3 prompt 列表（可选，默认 []）
  llm_bbox_prompts:     llm_bbox 保底的独立 prompt 列表，应包含空间位置描述（可选，默认 []）

Microwave 只配置 handle mask。FoundationPose 使用 handle mesh 和该 mask
估计 Microwave concept 的父 pose，门和其他 concept 部件不单独请求分割。
"""

CATEGORY_CONFIG = {
    "Bottle": {
        "masks": [
            {
                "prompt": "a bottle",
                "suffix": "",
            },
        ],
    },
    "Mug": {
        "masks": [
            {
                "prompt": "a cup",
                "suffix": "",
            },
        ],
    },
    "KitchenPot": {
        "masks": [
            {
                "prompt": "pot",
                "suffix": "",
                "crop_on_fail": True,
                "crop_boxes": [
                    (0.5, 0.0, 1.0, 0.5),  # 图片右上角 1/4 区域
                ],
                "fallback_prompts": ["pot"],
                "llm_bbox_prompts": [
                    "a kitchen pot, metallic silver color",
                ],
            },
        ],
    },
    "Knife": {
        "masks": [
            {
                "prompt": "knife",
                "suffix": "",
            },
        ],
    },
    "Spoon": {
        "masks": [
            {
                "prompt": "spoon",
                "suffix": "",
            },
        ],
    },
    "Plate": {
        "masks": [
            {
                "prompt": "green plate",
                "suffix": "",
            },
        ],
    },
    "Bowl": {
        "masks": [
            {
                "prompt": "blue bowl",
                "suffix": "",
            },
        ],
    },
    "Microwave": {
        "masks": [
            {
                # Microwave 只分割把手；FoundationPose 使用把手 mesh
                # 估计的 pose 作为整个 Microwave concept 的父 pose。
                "prompt": "red handle",
                "suffix": "",
                "crop_on_fail": True,
                "crop_boxes": [
                    (0.0, 0.0, 1.0, 1.0),     # 全图
                    (0.66, 0.0, 1.0, 1.0),   # 右 1/3
                    (0.25, 0.0, 0.5, 0.33),   # 左上 1/4 宽 × 上 1/3 高
                    (0.0, 0.0, 0.5, 0.33),    # 左半 × 上 1/3
                    (0.0, 0.0, 0.5, 0.5),     # 左上 1/4
                ],
                "fallback_prompts": [
                    "red handle of microwave door",
                    "orange handle of microwave door",
                    "orange handle",
                    "Microwave handle",
                ],
                # llm_bbox 独立 prompt：包含详细的空间位置描述，便于 VLM 生成精确描述
                "llm_bbox_prompts": [
                    "red microwave door handle"
                ],
            },
        ],
    },
    "Tabletop": {
        "masks": [
            {
                "prompt": "tabletop",
                "suffix": "",
            },
        ],
    },
}
