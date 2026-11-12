"""
SAM3 分割常驻服务。

启动后模型会保存在当前 Python 进程里，后续请求复用同一个 processor。
请求示例:
    curl -X POST http://127.0.0.1:18080/segment \
      -H 'Content-Type: application/json' \
      -d '{"image_path":"data/perception_test/rgbs/rgb_0.png","items":[{"category":"Bottle","prompt":"a bottle","mask_path":"data/perception_test/objs/Bottle/Bottle_0/masks/mask_0.png"}]}'
"""

import argparse
import gc
import json
import os
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import PIL.Image as Image
import torch

from gene_agent.perception_modules.paths import CHECKPOINT_ROOT, PROJECT_ROOT

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.fspath(PROJECT_ROOT)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from gene_agent.perception_modules.segmentation.batch_seg_mask import (  # noqa: E402
    build_explicit_items,
    load_model,
    seg_with_llm_bbox_fallback,
    seg_with_prompts,
)
from gene_agent.perception_modules.segmentation.seg_retry import retry_failed_items_with_fresh_image  # noqa: E402


RED_HANDLE_MAX_ATTEMPTS = 10


def _is_red_handle_item(item):
    """判断是否是需要重新注册图片后重试的红色微波炉把手。"""
    return "red handle" in str(item.get("prompt", "")).strip().lower()


def _resolve_repo_path(path):
    """把相对路径统一解释为仓库根目录下的路径。"""
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(_REPO_ROOT, path))


class SegmentationService:
    """保存 SAM3 模型，并串行处理 GPU 请求。"""

    def __init__(self, default_checkpoint, default_device):
        self.default_checkpoint = default_checkpoint
        self.default_device = default_device
        self._model = None
        self._processor = None
        self._model_key = None
        self._lock = threading.Lock()

    def _get_processor(self, checkpoint, device):
        model_key = (checkpoint, device)
        if self._processor is None or self._model_key != model_key:
            # 保留 model 引用，避免只剩 processor 时模型被释放。
            self._model, self._processor = load_model(checkpoint, device)
            self._model_key = model_key
        return self._processor

    @property
    def model_loaded(self):
        return self._processor is not None

    def load_model(self):
        """在服务锁内预加载；已加载时保持幂等。"""
        with self._lock:
            self._get_processor(self.default_checkpoint, self.default_device)
            return {"ok": True, "model_loaded": self.model_loaded}

    def unload_model(self):
        """分割完成后释放权重，后续 segment 会按原配置重新加载。"""
        with self._lock:
            self._processor = None
            self._model = None
            self._model_key = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            return {"ok": True, "model_loaded": self.model_loaded}

    def segment(self, payload):
        """执行一次显式图片、prompt 和 mask 路径的分割请求。"""
        with self._lock:
            start_time = time.time()
            if not isinstance(payload, dict):
                raise ValueError("请求 JSON 必须是 object")
            if "image_path" not in payload or "items" not in payload:
                raise ValueError("请求必须包含 image_path 和 items")

            image_path = _resolve_repo_path(payload["image_path"])
            raw_items = payload["items"]
            items = build_explicit_items([
                {
                    **item,
                    "mask_path": _resolve_repo_path(item["mask_path"]),
                }
                for item in raw_items
            ])
            checkpoint = payload.get("checkpoint", self.default_checkpoint)
            device = payload.get("device", self.default_device)

            if not os.path.isfile(image_path):
                raise FileNotFoundError(f"找不到图片: {image_path}")

            print("\n=== [seg_server] 分割请求开始 ===")
            print(f"图片: {image_path}")
            print(f"分割数量: {len(items)}")

            processor = self._get_processor(checkpoint, device)
            image = Image.open(image_path).convert("RGB")
            print(f"图片尺寸: {image.size}")

            red_handle_items = [
                item for item in items if _is_red_handle_item(item)
            ]
            other_items = [
                item for item in items if not _is_red_handle_item(item)
            ]
            failures = []

            if other_items:
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    inference_state = processor.set_image(image)
                print("图片 embedding 编码完成。")

                print("\n--- 第一轮：全图分割 ---")
                other_failures = seg_with_prompts(
                    processor, inference_state, other_items, device
                )

                # 非 red handle 项继续使用原有的 LLM bbox 保底逻辑。
                if other_failures:
                    other_failures = seg_with_llm_bbox_fallback(
                        processor,
                        inference_state,
                        image,
                        other_failures,
                        device,
                        image_path,
                    )
                failures.extend(other_failures)

            if red_handle_items:
                print(
                    "\n--- Microwave red handle 分割："
                    f"每次重新 set_image，最多 {RED_HANDLE_MAX_ATTEMPTS} 次 ---"
                )

                def run_red_handle_attempt(pending):
                    with torch.autocast(
                        device_type=device, dtype=torch.bfloat16
                    ):
                        retry_state = processor.set_image(image)
                    print("[seg_server] red handle 已重新注册图片 embedding。")
                    # pending 中的 prompt 不变，只重新在新的 image state 上提交。
                    return seg_with_prompts(
                        processor, retry_state, pending, device
                    )

                def log_red_handle_retry(attempt, max_attempts, _pending):
                    print(
                        "[seg_server] red handle 分割失败，"
                        f"第 {attempt}/{max_attempts} 次重试"
                        "（重新 set_image 后使用同一 prompt）。",
                        flush=True,
                    )

                red_failures, _attempt_count = (
                    retry_failed_items_with_fresh_image(
                        red_handle_items,
                        run_red_handle_attempt,
                        max_attempts=RED_HANDLE_MAX_ATTEMPTS,
                        on_retry=log_red_handle_retry,
                    )
                )
                failures.extend(red_failures)

            failed_paths = {item["save_path"] for item in failures}
            failure_by_path = {
                item["save_path"]: item for item in failures
            }
            results = []
            for item in items:
                result = {
                    "ok": item["save_path"] not in failed_paths,
                    "category": item["category"],
                    "prompt": item["prompt"],
                    "mask_path": item["save_path"],
                }
                if "object_dir" in item:
                    result["object_dir"] = item["object_dir"]
                if not result["ok"]:
                    result["error"] = failure_by_path[item["save_path"]].get(
                        "error", "未生成有效 mask"
                    )
                results.append(result)

            elapsed = time.time() - start_time
            print(f"=== [seg_server] 分割完成，耗时 {elapsed:.2f} 秒 ===")
            return {
                "ok": True,
                "elapsed": elapsed,
                "image_path": image_path,
                "mask_count": len(items),
                "success": len(failures) == 0,
                "results": results,
                "failures": [result for result in results if not result["ok"]],
            }


class SegRequestHandler(BaseHTTPRequestHandler):
    service = None

    def _send_json(self, status, body):
        body_bytes = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body_bytes)))
        self.end_headers()
        self.wfile.write(body_bytes)

    def do_GET(self):
        if self.path == "/health":
            self._send_json(200, {"ok": True, "service": "seg_server",
                                  "model_loaded": self.service.model_loaded})
            return
        self._send_json(404, {"ok": False, "error": "unknown endpoint"})

    def do_POST(self):
        if self.path not in {"/segment", "/model/load", "/model/unload"}:
            self._send_json(404, {"ok": False, "error": "unknown endpoint"})
            return

        try:
            if self.path == "/model/load":
                self._send_json(200, self.service.load_model())
                return
            if self.path == "/model/unload":
                self._send_json(200, self.service.unload_model())
                return
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            result = self.service.segment(payload)
            self._send_json(200, result)
        except Exception as exc:
            traceback.print_exc()
            self._send_json(
                500,
                {
                    "ok": False,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )

    def log_message(self, fmt, *args):
        print(f"[seg_server] {self.address_string()} - {fmt % args}")


def main():
    parser = argparse.ArgumentParser(description="SAM3 分割常驻服务")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--checkpoint", default=str(CHECKPOINT_ROOT / "SAM3" / "sam3.pt"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    SegRequestHandler.service = SegmentationService(args.checkpoint, args.device)
    server = ThreadingHTTPServer((args.host, args.port), SegRequestHandler)
    print(f"seg_server 已启动: http://{args.host}:{args.port}")
    print("POST /segment, /model/load, /model/unload; GET /health")
    server.serve_forever()


if __name__ == "__main__":
    main()
