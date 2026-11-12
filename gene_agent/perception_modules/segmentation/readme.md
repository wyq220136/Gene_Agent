# Segmentation

默认端口：`http://127.0.0.1:18080`。

## 启动

```bash
conda activate sam3
python -m gene_agent.perception_modules.segmentation.seg_server \
  --host 127.0.0.1 --port 18080 --device cuda
```

模型在首次分割请求时加载。单物体流程确认 mask 后会调用
`POST /model/unload` 释放 SAM3 权重；下一次 `/segment` 会自动重新加载。
`GET /health` 返回 `model_loaded`。也可手动调用 `POST /model/load`。

## Python 调用

请求项需要 `category`、`prompt`、`mask_path`：

```python
from gene_agent.perception_modules.segmentation import segment_image

status, result = segment_image(
    "http://127.0.0.1:18080",
    "data/perception_test/rgbs/rgb_0.png",
    [{"category": "Bottle", "prompt": "a bottle",
      "mask_path": "data/perception_test/objs/Bottle/Bottle_0/masks/mask_0.png"}],
)
if status != 200 or not result.get("ok"):
    raise RuntimeError(result)
```

## 任务批量分割

先启动 camera server 和 segmentation server：

```bash
bash gene_agent/perception_modules/segmentation/run_task_segmentation.sh
CATEGORIES="Bottle Microwave" bash gene_agent/perception_modules/segmentation/run_task_segmentation.sh
```

脚本使用 `category_config.py` 生成 prompt，也支持通过 `CONFIG` 指定同结构的
`.py` 或 `.json`。`all` 表示任务目录中配置过的全部类别。

初始数据保存为 `rgbs/rgb_0.png`、`depths/depth_0.png` 和
`camera/camera_parameters_0.json`；mask 保存为物体目录下的 `masks/mask_0.png`。

## LLM bbox 保底分割

当文本 prompt 分割失败时，如果 item 在 `category_config.py` 中配置了
`llm_bbox_prompts` 且设置了 `API_KEY` 环境变量，系统会自动调用 VLM
（Qwen-VL）获取物体 bbox，再用 SAM3 的几何 prompt 进行分割。

### 配置方式

在 `category_config.py` 中为目标 mask 添加 `llm_bbox_prompts` 字段：

```python
"Microwave": {
    "masks": [
        {
            "prompt": "microwave handle",
            "suffix": "",
            "llm_bbox_prompts": [
                "the red handle of a microwave door"
            ],
        },
    ],
},
```

### 环境变量

```bash
export API_KEY="your-api-key"
```

默认使用 ZenMux OpenAI-compatible 接口（`qwen/qwen3-vl-plus`），可通过
`VLMBboxClient` 构造参数覆盖 base_url 和 model。

### 独立调用

```python
from gene_agent.perception_modules.segmentation import llm_bbox_fallback_batch

still_failed = llm_bbox_fallback_batch(
    processor, inference_state, image, failures, device="cuda", image_path=img_path,
)
```
