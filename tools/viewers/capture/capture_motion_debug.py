"""读取 Motion DebugBundle 并生成固定多视角 Viser 截图。"""

from __future__ import annotations

import argparse
from pathlib import Path

from gene_agent.feedback import load_motion_debug_bundle
from gene_agent.visualization.motion_debug.capture import capture_motion_debug


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--views",
        nargs="+",
        choices=("overview", "target", "collision"),
        default=("overview", "target", "collision"),
    )
    args = parser.parse_args()
    manifest = capture_motion_debug(
        load_motion_debug_bundle(args.bundle),
        args.output,
        views=tuple(args.views),
    )
    print(f"PASS Motion Viser 截图：{args.output.resolve()}")
    print(f"views={','.join(manifest['views'])}")


if __name__ == "__main__":
    main()
