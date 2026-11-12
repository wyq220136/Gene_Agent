"""
SAM3 分割服务客户端。

示例:
    python segmentation/seg_client.py \
      --image_path data/perception_test/rgbs/rgb_0.png \
      --items_json data/perception_test/segmentation_items.json
"""

import argparse
import json
import urllib.error
import urllib.request


def segment_image(
    server,
    image_path,
    items,
    *,
    checkpoint=None,
    device=None,
    timeout=None,
):
    """向 segmentation server 发送显式图片、prompt 和输出路径请求。"""
    payload = {
        "image_path": str(image_path),
        "items": items,
    }
    if checkpoint is not None:
        payload["checkpoint"] = checkpoint
    if device is not None:
        payload["device"] = device
    return post_json(server.rstrip("/") + "/segment", payload, timeout=timeout)


def unload_model(server, *, timeout=60.0):
    """通知 SAM3 服务释放 GPU 模型；失败时阻止进入 SAM3D 阶段。"""
    status, result = post_json(server.rstrip("/") + "/model/unload", {}, timeout=timeout)
    if status != 200 or not result.get("ok") or result.get("model_loaded"):
        raise RuntimeError(f"SAM3 模型卸载失败: {result}")
    return result


def post_json(url, payload, *, timeout=None):
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def main():
    parser = argparse.ArgumentParser(description="向 seg_server 发送分割请求")
    parser.add_argument("--server", default="http://127.0.0.1:18080")
    parser.add_argument("--image_path", required=True)
    parser.add_argument("--items_json", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    with open(args.items_json, "r", encoding="utf-8") as file:
        items = json.load(file)
    status, result = segment_image(
        args.server,
        args.image_path,
        items,
        checkpoint=args.checkpoint,
        device=args.device,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(
        0
        if status == 200 and result.get("ok") and result.get("success", True)
        else 1
    )


if __name__ == "__main__":
    main()
