"""按项目对象目录调用 SAM3D 重建服务的客户端。"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path


_MODULE_DIR = Path(__file__).resolve().parent
_PERCEPTION_DIR = _MODULE_DIR.parent
if str(_PERCEPTION_DIR) not in sys.path:
    sys.path.insert(0, str(_PERCEPTION_DIR))

try:
    from gene_agent.perception_modules.recon_3d.recon_core import resolve_object_paths  # noqa: E402
except ImportError:
    from .recon_core import resolve_object_paths


def build_payload(
    object_dir: str | Path,
    *,
    instance: str | None = None,
    rgb: str | Path | None = None,
    depth: str | Path | None = None,
    mask: str | Path | None = None,
    intrinsics: str | Path | None = None,
    output: str | Path | None = None,
    depth_scale: float = 0.001,
    seed: int = 42,
    simplify_mesh: bool = False,
    visualize: bool = False,
    use_foundationpose: bool = False,
    pose_server: str = "http://127.0.0.1:18081",
) -> dict:
    """从对象目录生成 server 请求，显式路径优先于自动推断。"""
    paths = resolve_object_paths(
        object_dir,
        instance=instance,
        rgb=rgb,
        depth=depth,
        mask=mask,
        intrinsics=intrinsics,
        output=output,
    )
    return {
        "rgb_path": str(paths.rgb),
        "depth_path": str(paths.depth),
        "mask_path": str(paths.mask),
        "intrinsics_path": str(paths.intrinsics),
        "output_path": str(paths.output),
        "depth_scale": depth_scale,
        "seed": seed,
        "simplify_mesh": simplify_mesh,
        "visualize": visualize,
        "use_foundationpose": use_foundationpose,
        "pose_server": pose_server,
    }


def post_json(
    server: str,
    payload: dict,
    timeout: float = 600.0,
    endpoint: str = "/reconstruct",
) -> dict:
    """向重建服务发送 JSON 请求并解析 JSON 响应。"""
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        server.rstrip("/") + endpoint,
        data=body,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"recon_3d_server HTTP {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"无法连接 recon_3d_server: {exc}") from exc

    if not result.get("ok", False):
        raise RuntimeError(result.get("error", "重建服务返回失败"))
    return result


def unload_model(server: str, timeout: float = 30.0) -> dict:
    """通知 SAM3D server 卸载模型并释放 GPU 显存。"""
    return post_json(server, {}, timeout=timeout, endpoint="/model/unload")


def load_model(server: str, timeout: float = 600.0) -> dict:
    """通知 SAM3D server 加载模型。"""
    return post_json(server, {}, timeout=timeout, endpoint="/model/load")


def get_health(server: str, timeout: float = 5.0) -> dict:
    """查询 server 状态，确认 SAM3D 模型是否已经加载。"""
    request = urllib.request.Request(server.rstrip("/") + "/health", method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"recon_3d_server health HTTP {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"无法连接 recon_3d_server: {exc}") from exc
    return result


def reconstruct_object(
    object_dir: str | Path,
    *,
    server: str = "http://127.0.0.1:18083",
    instance: str | None = None,
    rgb: str | Path | None = None,
    depth: str | Path | None = None,
    mask: str | Path | None = None,
    intrinsics: str | Path | None = None,
    output: str | Path | None = None,
    depth_scale: float = 0.001,
    seed: int = 42,
    timeout: float = 600.0,
    verbose: bool = False,
    simplify_mesh: bool = False,
    visualize: bool = False,
    use_foundationpose: bool = False,
    pose_server: str = "http://127.0.0.1:18081",
) -> dict:
    """供其他模块调用：按对象目录推断路径并请求一次 mesh 重建。"""
    health = get_health(server, timeout=min(timeout, 5.0))
    if not health.get("ok"):
        raise RuntimeError("SAM3D server health 检查失败")
    if verbose:
        if health.get("model_loaded"):
            print("[recon_client] SAM3D 模型已加载，可以开始重建", flush=True)
        else:
            print("[recon_client] SAM3D 模型当前已释放，将在请求中重新加载", flush=True)

    payload = build_payload(
        object_dir,
        instance=instance,
        rgb=rgb,
        depth=depth,
        mask=mask,
        intrinsics=intrinsics,
        output=output,
        depth_scale=depth_scale,
        seed=seed,
        simplify_mesh=simplify_mesh,
        visualize=visualize,
        use_foundationpose=use_foundationpose,
        pose_server=pose_server,
    )
    return post_json(server, payload, timeout=timeout)


def main() -> None:
    parser = argparse.ArgumentParser(description="调用 SAM3D RGB-D 重建服务")
    parser.add_argument("--object-dir", default=None)
    parser.add_argument("--instance", default=None)
    parser.add_argument("--rgb", default=None)
    parser.add_argument("--depth", default=None)
    parser.add_argument("--mask", default=None)
    parser.add_argument("--intrinsics", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--server", default="http://127.0.0.1:18083")
    parser.add_argument("--depth-scale", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--simplify-mesh", action="store_true")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--use-foundationpose", action="store_true")
    parser.add_argument("--pose-server", default="http://127.0.0.1:18081")
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument("--unload-model", action="store_true")
    model_group.add_argument("--load-model", action="store_true")
    args = parser.parse_args()

    if args.unload_model or args.load_model:
        result = (
            unload_model(args.server, timeout=args.timeout)
            if args.unload_model
            else load_model(args.server, timeout=args.timeout)
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if args.object_dir is None:
        parser.error("重建请求必须指定 --object-dir；模型控制请求请使用 --load-model 或 --unload-model")

    result = reconstruct_object(
        args.object_dir,
        server=args.server,
        instance=args.instance,
        rgb=args.rgb,
        depth=args.depth,
        mask=args.mask,
        intrinsics=args.intrinsics,
        output=args.output,
        depth_scale=args.depth_scale,
        seed=args.seed,
        timeout=args.timeout,
        verbose=True,
        simplify_mesh=args.simplify_mesh,
        visualize=args.visualize,
        use_foundationpose=args.use_foundationpose,
        pose_server=args.pose_server,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
