"""
VLM bbox 客户端 — 调用 VLM 从图片中检测指定物体的 bbox。

参考 third_party/doric/pipeline/gene_seg_qwen_sam3/qwenvl_sam.py 中的调用方式，
使用 OpenAI-compatible API（支持 DashScope / ZenMux 等）。

API key 从环境变量 API_KEY 读取，未配置时跳过 LLM fallback。
"""

from __future__ import annotations

import base64
import json
import os
import re
from pathlib import Path
from typing import Optional


def _image_data_url(image_path: str) -> str:
    """把本地图片编码为 OpenAI-compatible image_url（base64 data URL）。"""
    suffix = Path(image_path).suffix.lower()
    mime_type = "image/png" if suffix == ".png" else "image/jpeg"
    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime_type};base64,{image_b64}"


def _build_single_bbox_prompt(description: str) -> str:
    """构建单物体 bbox 检测的 VLM prompt。

    参数:
        description: 物体描述，如 "the glass door of a microwave oven in the center"

    返回:
        VLM prompt 字符串，要求输出 [0, 1000] 坐标系的 bbox JSON
    """
    return f"""You are a precise visual grounding assistant.
Your task is to locate a specific object in the image and output its tight bounding box.

# Object to Find:
{description}

# Guidelines:
1. The description above may include appearance hints and approximate position hints.
   **Position hints are suggestions only** — if the described object is visible anywhere
   in the image, you MUST output its bbox even if it does not match the suggested position.
2. Only set "visible" to false if the described TYPE of object is truly absent from the image.
3. Output a tightly fitting **Axis-Aligned Bounding Box (AABB)**.
4. **Coordinate Format**: integer coordinates scaled to [0, 1000].
   Format: [xmin, ymin, xmax, ymax]. Top-left is [0, 0], bottom-right is [1000, 1000].
5. The box must strictly enclose the object, do NOT include empty background.

# Output Format:
Return ONLY valid JSON:
```json
{{
  "visible": true,
  "box_2d": [xmin, ymin, xmax, ymax],
  "reasoning": "Brief description of where the object is located."
}}
```"""


def _parse_bbox_json(text: str) -> Optional[dict]:
    """解析 VLM 返回的 JSON，提取 bbox。

    参考 qwenvl_sam.py 的 parse_llm_json()，兼容 ```json ... ``` 包裹和纯 JSON。
    """
    text = text.strip()
    # 优先匹配 ```json ... ``` 包裹的 JSON
    match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        json_str = match.group(1)
    else:
        # 尝试找到 JSON 对象的起止位置
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            json_str = text[start : end + 1]
        else:
            return None

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return None


class VLMBboxClient:
    """OpenAI-compatible VLM 客户端，用于从图片中获取物体 bbox。

    使用方式:
        client = VLMBboxClient()
        bbox_norm = client.get_bbox(image_path, "a microwave door")
        # bbox_norm: [x1, y1, x2, y2] 归一化到 [0, 1]，或 None（失败时）
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://zenmux.ai/api/v1",
        model: str = "qwen/qwen3-vl-plus",
        temperature: float = 0.1,
        max_tokens: int = 800,
        timeout: float = 120.0,
    ):
        """
        参数:
            api_key: API key，默认从环境变量 API_KEY 读取
            base_url: OpenAI-compatible API 地址
            model: 模型名称
            temperature: 生成温度
            max_tokens: 最大输出 token 数
            timeout: 请求超时秒数
        """
        self._api_key = api_key or os.getenv("API_KEY", "")
        self._base_url = base_url
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._timeout = timeout
        self._client = None  # 延迟创建

    @property
    def is_available(self) -> bool:
        """API key 是否已配置，未配置时跳过 LLM fallback。"""
        return bool(self._api_key)

    def _ensure_client(self):
        """延迟创建 OpenAI client，避免 import 开销（仅 conda 环境有 openai 时可用）。"""
        if self._client is not None:
            return
        try:
            import httpx
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "需要安装 openai 和 httpx 才能使用 VLM bbox 功能。"
                f" 请检查 conda 环境。原始错误: {e}"
            )

        limits = httpx.Limits(max_connections=5, max_keepalive_connections=3)
        http_client = httpx.Client(limits=limits)
        self._client = OpenAI(
            api_key=self._api_key,
            base_url=self._base_url,
            timeout=httpx.Timeout(self._timeout, connect=10.0, read=self._timeout),
            max_retries=2,
            http_client=http_client,
        )

    def get_bbox(self, image_path: str, description: str, verbose: bool = False) -> Optional[dict]:
        """调用 VLM 获取物体 bbox。"""
        if not self.is_available:
            if verbose:
                print(f"   [VLM] 跳过: API_KEY 未设置")
            return None

        self._ensure_client()

        prompt_text = _build_single_bbox_prompt(description)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": _image_data_url(image_path)}},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]

        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )
            raw_content = resp.choices[0].message.content
            if verbose:
                print(f"   [VLM] 原始返回 ({len(raw_content)} chars):\n{raw_content[:500]}")

            parsed = _parse_bbox_json(raw_content)

            if parsed is None:
                if verbose:
                    print(f"   [VLM] JSON 解析失败，原始返回已打印")
                print(f"   [VLM] JSON 解析失败，原始返回: {raw_content[:200]}")
                return None

            if not parsed.get("visible", True):
                print(f"   [VLM] 物体不可见: {parsed.get('reasoning', '')}")
                return None

            if "box_2d" not in parsed:
                print(f"   [VLM] 返回 JSON 缺少 box_2d 字段: {parsed}")
                return None

            box = parsed["box_2d"]
            if len(box) != 4:
                print(f"   [VLM] box_2d 长度不是 4: {box}")
                return None

            # 转换 [0, 1000] → [0, 1] 归一化
            parsed["box_2d_norm"] = [
                box[0] / 1000.0,
                box[1] / 1000.0,
                box[2] / 1000.0,
                box[3] / 1000.0,
            ]
            parsed["raw_content"] = raw_content
            print(f"   [VLM] ✓ bbox={box} → norm={[round(v, 3) for v in parsed['box_2d_norm']]}")
            return parsed

        except Exception as e:
            print(f"   [VLM Error] get_bbox 失败: {type(e).__name__} - {e}")
            return None


def get_bbox_from_vlm(
    image_path: str,
    description: str,
    api_key: Optional[str] = None,
    base_url: str = "https://zenmux.ai/api/v1",
    model: str = "qwen/qwen3-vl-plus",
    verbose: bool = False,
) -> Optional[dict]:
    """便捷函数：调用 VLM 获取物体 bbox。

    参数:
        image_path: 图片路径
        description: 物体自然语言描述
        api_key: API key，默认从环境变量 API_KEY 读取
        base_url: API 地址
        model: 模型名称

    返回:
        {"box_2d_norm": [x1, y1, x2, y2]} 归一化 bbox，失败返回 None
    """
    client = VLMBboxClient(api_key=api_key, base_url=base_url, model=model)
    return client.get_bbox(image_path, description, verbose=verbose)
