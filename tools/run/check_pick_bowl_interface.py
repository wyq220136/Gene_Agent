"""独立抓碗诊断入口；正式 MuJoCo 实现位于 tasks/policies/microwave_task/simulation_bowl。"""
import argparse
import json
from pathlib import Path
import sys
import traceback
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from gene_agent.tasks.policies.microwave_task.simulation_bowl.pick import (
    run_experiment, run_probe, make_pregrasp_pose,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-id", default="bowl_ring_00_00")
    parser.add_argument("--auto-select", action="store_true",
                        help="按原 Viser Skill 从全部 Bowl 抓取与 pregrasp 候选中筛选")
    parser.add_argument("--depth-m", type=float, default=None,
                        help="自动模式默认 0.015 m（与 Viser 一致）；手动模式默认 0.012 m")
    parser.add_argument("--retreat-m", type=float, default=None,
                        help="覆盖任务 Skill profile 中的 approach_distance_m")
    parser.add_argument("--seed", type=int, default=None,
                        help="省略时使用 --scene-config 的 seed；无配置时默认为 0")
    parser.add_argument("--scene-config", type=Path,
                        help="复用 Viser 的 object_poses_world、微波炉关节和桌面尺寸")
    parser.add_argument("--arm-kp-scale", type=float, default=1.0,
                        help="仅调整本次 MuJoCo 活动臂 position actuator 刚度比例")
    parser.add_argument("--skill-profile", choices=("pick_bowl", "pick_bowl_legacy_visual"),
                        default="pick_bowl",
                        help="选用现有 Skill 参数；legacy_visual 与 Viser 已调试方案一致")
    parser.add_argument("--perturb", action="store_true")
    parser.add_argument("--record-video", action="store_true",
                        help="执行时按真实 MuJoCo 状态采帧，保存分阶段 GIF 和统一 MP4")
    parser.add_argument("--linear-approach", action="store_true",
                        help="兼容旧命令；approach 阶段始终使用 TCP 直线规划")
    parser.add_argument("--gravity-compensated-arm", action="store_true",
                        help="在生成的MuJoCo机器人XML中给本次活动臂开启本体重力补偿")
    parser.add_argument("--lift-m", type=float, default=None,
                        help="覆盖任务 Skill profile 中的抬升距离；0表示只验证接近与闭合")
    parser.add_argument("--simulation-weld-after-contact", action="store_true",
                        help="双侧指垫接触后启用显式 MuJoCo weld；不算纯物理抓取")
    parser.add_argument('--stop-after', choices=('pick', 'pregrasp', 'approach', 'grasp'),
                        default='pick', help='只执行所选阶段前缀，保留实际现场；pick 为完整请求')
    args = parser.parse_args()
    try:
        result = run_experiment(
            args.output, stop_after=args.stop_after, candidate_id="auto" if args.auto_select else args.candidate_id,
            depth_m=args.depth_m, retreat_m=args.retreat_m,
            seed=args.seed, perturb=args.perturb, record_video=args.record_video,
            linear_approach=True,
            gravity_compensated_arm=args.gravity_compensated_arm,
            lift_m=args.lift_m,
            simulation_weld_after_contact=args.simulation_weld_after_contact,
            scene_config_path=args.scene_config,
            arm_kp_scale=args.arm_kp_scale,
            skill_profile=args.skill_profile,
        )
    except Exception:
        Path(args.output).mkdir(parents=True, exist_ok=True)
        (Path(args.output) / "raw_error.txt").write_text(
            traceback.format_exc(), encoding="utf-8"
        )
        raise
    print(json.dumps({
        "success": result["success"],
        "requested_scope": result.get("requested_scope", args.stop_after),
        "failure": result.get("failure"),
        "failure_stage": result["failure_stage"],
        "output": str(Path(args.output).resolve()),
    }, ensure_ascii=False))
    raise SystemExit(0 if result["success"] else 1)
