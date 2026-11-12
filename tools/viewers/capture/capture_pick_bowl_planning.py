"""对已保存的真实 pick bowl 规划场景生成可复查截图。"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from gene_agent.visualization.pick_bowl.capture import capture_pick_bowl_planning


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-id", help="固定选中的候选；可截图检查单个 IK 失败")
    parser.add_argument("--trajectory-source", choices=("抓取目标", "Skill pregrasp", "Skill approach", "TCP 直线探针"),
                        default="抓取目标", help="截图时显示哪段规划轨迹")
    args = parser.parse_args()
    manifest = capture_pick_bowl_planning(
        args.run, args.output, candidate_id=args.candidate_id,
        trajectory_source=args.trajectory_source,
    )
    print(f"PASS pick bowl 截图：{args.output.resolve()}")
    print(f"snapshot={manifest['scene_snapshot_sha256']}; views={','.join(manifest['views'])}")


if __name__ == "__main__":
    main()
