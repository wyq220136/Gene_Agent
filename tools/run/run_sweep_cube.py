#!/usr/bin/env python3
"""从 fresh reset 执行 Pika 抓刷子、扫方块入簸箕的 MuJoCo Task。"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# 与现有直接运行的 CLI 一致，从源码根目录加载 benchmark 包。
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gene_agent.tasks.policies.sweep_cube_task.simulation import SweepCubeRun


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-dir", type=Path,
                        default=Path("data/realworld/brush_cube_dustpan_0"))
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="新的运行目录，不能已存在")
    parser.add_argument("--randomize", action="store_true",
                        help="按 scene.json 的合理范围采样初始 XY/yaw")
    parser.add_argument("--seed", type=int, default=0,
                        help="随机摆放 seed；相同场景与 seed 可复现")
    parser.add_argument("--diagnostic-cpu-ik", action="store_true",
                        help="CPU 运动规划诊断；接触候选另用 GPU cuRobo 纯 IK 筛选，不作为正式 cuRobo 评测")
    parser.add_argument("--grasp-tilt-deg", type=float, default=-20.,
                        help="抓取刷柄的倾角（度），用于比较夹持姿态对后续可达性的影响")
    parser.add_argument("--robot-gravity-compensation", action="store_true",
                        help="仅机器人本体重力补偿，落盘有效仿真XML；操作物体不补偿")
    parser.add_argument('--collision-mode',choices=['full','robot_only'],default='full',
                        help='cuRobo 场景碰撞；不改变 MuJoCo 物理接触')
    parser.add_argument('--live',action='store_true',help='追加实际 qpos 供在线查看')
    parser.add_argument('--skill', choices=('task','pick','sweep'), default='task',
                        help='独立 Skill 执行必要 Pick 前缀，不声明完整 Task 成功')
    parser.add_argument('--target', choices=('cube','cube_red','cube_blue'), default='cube')
    parser.add_argument('--stop-after', choices=('move_behind_cube','lower_to_contact','sweep','retreat'),
                        help='仅在 --skill sweep 时停止于指定阶段，保留当前现场')
    args = parser.parse_args()
    if args.stop_after is not None and args.skill != 'sweep':
        parser.error('--stop-after 需配合 --skill sweep')
    print(f"初始化清扫场景：{args.scene_dir}；seed={args.seed}；"
          f"planner_backend={'diagnostic_cpu_ik' if args.diagnostic_cpu_ik else 'curobo'}",
          flush=True)
    from gene_agent.motion.configuration import motion_configuration
    from gene_agent.feedback.live_record import live_physics_recording
    with motion_configuration({'collision_mode':args.collision_mode}), live_physics_recording(args.output_dir,enabled=args.live):
        result = SweepCubeRun(args.scene_dir, args.output_dir,
                          randomize=args.randomize, seed=args.seed,
                          diagnostic_cpu_ik=args.diagnostic_cpu_ik, grasp_tilt_deg=args.grasp_tilt_deg,
                                  robot_gravity_compensation=args.robot_gravity_compensation).run(
                                      selection=args.skill, target_name=args.target, stop_after=args.stop_after)
    print(f"scope={result['requested_scope']} status={result['status']} success={result['success']} "
          f"failure_stage={result['failure_stage']}")
    print(f"记录: {args.output_dir / 'result.json'}")
    if not result["success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
