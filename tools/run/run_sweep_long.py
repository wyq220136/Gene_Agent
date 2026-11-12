#!/usr/bin/env python3
"""从 fresh reset 运行竖立长刷抓取与三方块入簸箕任务。"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# 与现有直接运行的 CLI 一致，从源码根目录加载 benchmark 包。
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gene_agent.tasks.policies.sweep_long_task.simulation import run_long_brush


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-dir", type=Path,
        default=Path("data/realworld/my_scene_0/scenes/sweep_long"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layout-shift-x-m", type=float, default=-.04,
                        help="清扫通道沿世界 X 的整体平移，默认 -4 cm（已验证可达）")
    parser.add_argument("--source-layout", action="store_true",
                        help="保留原方块横向位置；用于对照，不启用收拢通道")
    parser.add_argument("--grasp-yaw-deg", type=float, default=0.,
                        help="世界 -Z 抓取时绕竖直轴的夹爪角度，单位度")
    parser.add_argument("--head-proxy-half-width-y-m", type=float, default=.0861275891,
                        help="刷头碰撞代理局部 Y 半宽，默认保留原场景尺寸")
    parser.add_argument("--layout-seed", type=int,
                        help="对三块方块按记录的范围采样 XY/yaw；省略时使用名义摆位")
    parser.add_argument("--layout-distribution", choices=("small", "wide", "stress"),
                        default="wide",
                        help="给定 seed 时的采样范围；wide 默认，small 重现旧小扰动，stress 保留侧缘难例")
    parser.add_argument("--goal-inset-extra-m", type=float, default=.002,
                        help="比原清扫 Knowledge 深入托盘的距离，默认 2 mm")
    parser.add_argument('--collision-mode',choices=('full','robot_only'),default='full')
    parser.add_argument('--live',action='store_true')
    args = parser.parse_args()
    from gene_agent.motion.configuration import motion_configuration
    from gene_agent.feedback.live_record import live_physics_recording
    with motion_configuration({'collision_mode':args.collision_mode}), live_physics_recording(args.output_dir,enabled=args.live):
        result = run_long_brush(args.scene_dir, args.output_dir,
                                layout_shift_x_m=args.layout_shift_x_m,
                                aligned_lane=not args.source_layout,
                                grasp_yaw_deg=args.grasp_yaw_deg,
                                head_proxy_half_width_y_m=args.head_proxy_half_width_y_m,
                                layout_seed=args.layout_seed,
                                layout_distribution=args.layout_distribution,
                                goal_inset_extra_m=args.goal_inset_extra_m)
    print(f"status={result['status']} success={result['success']} "
          f"failure_stage={result['failure_stage']}")
    print(f"记录: {args.output_dir / 'result.json'}")
    if not result["success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
