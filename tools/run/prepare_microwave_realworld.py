"""只采集和注册 Task1 真机感知；自动保存记录，不控制机械臂。"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from gene_agent.realworld.microwave.perception_setup import prepare_perception


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-params", required=True, help="现有标定向导生成的 cam_param.json 或第三视角外参文件")
    parser.add_argument("--output", help="新的感知记录目录；省略时自动生成")
    parser.add_argument("--camera-endpoint", default="tcp://127.0.0.1:18082")
    parser.add_argument("--pose-server", default="http://127.0.0.1:18081")
    parser.add_argument("--seg-server", default="http://127.0.0.1:18080")
    parser.add_argument("--bowl-prompt", default="bowl")
    parser.add_argument("--handle-prompt", default="microwave door handle")
    parser.add_argument("--knob-prompt", default="upper microwave knob including pointer")
    parser.add_argument("--table-prompt", default="table top")
    args = parser.parse_args(argv)
    if not sys.stdin.isatty():
        raise ValueError("感知准备需要交互终端确认门/旋钮零位和 mask")
    output = args.output or ROOT / "runs/realworld" / f"perception-{datetime.now():%Y%m%d-%H%M%S-%f}"
    path = prepare_perception(camera_params=args.camera_params, output=output,
        camera_endpoint=args.camera_endpoint, pose_server=args.pose_server, seg_server=args.seg_server,
        prompts={name: getattr(args, f"{name}_prompt") for name in ("bowl", "handle", "knob", "table")})
    print(f"感知准备完成：{path.parent}\n运行入口使用 --perception {path.parent}；无需填写站点或角度 JSON。")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError) as error:
        print(f"感知准备失败：{error}", file=sys.stderr)
        raise SystemExit(2) from None
