"""启动多类型 Grasp Knowledge 实验台，使用 Concept 和参数滑块查看抓取公式。"""
from __future__ import annotations

import argparse
from pathlib import Path
import time

import viser

from gene_agent.visualization.grasp_tuner.lab_app import KINDS, KnowledgeGraspLabApp
from gene_agent.visualization.grasp_tuner.feedback import read_tuner_state
from gene_agent.tasks.policies.microwave_task.planning.bowl import load_pick_bowl_case

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind",choices=KINDS,default=None)
    parser.add_argument("--host",default="127.0.0.1")
    parser.add_argument("--port",type=int,default=8080)
    parser.add_argument("--gripper-run",type=Path,default=None,
                        help="从已验证的场景读取真实夹爪 visual；不会运行规划")
    parser.add_argument("--state-out",type=Path,default=None,help="实时界面状态 JSON；供 Codex 截取当前调参结果")
    parser.add_argument("--restore-state",type=Path,help="读取实时状态或截图清单，恢复有效调参结果")
    args=parser.parse_args()
    restored=read_tuner_state(args.restore_state,tool="lab") if args.restore_state else None
    server=viser.ViserServer(host=args.host,port=args.port)
    try:
        gripper_run = args.gripper_run or Path((restored or {}).get("gripper_run") or ROOT / "runs/pick-bowl-planning-final")
        gripper_case = load_pick_bowl_case(gripper_run) if (gripper_run / "case_meta.json").is_file() else None
        KnowledgeGraspLabApp(server,initial_kind=args.kind or (restored or {}).get("kind","ring"),
                             state_path=args.state_out,restored=restored,gripper_case=gripper_case)
        print(f"[knowledge-lab] 已就绪：http://{args.host}:{args.port}",flush=True)
        print("[knowledge-lab] 服务会持续运行；打开上面的地址，Ctrl-C 退出。",flush=True)
        while True:time.sleep(.25)
    except KeyboardInterrupt:
        print("[knowledge-lab] 已退出",flush=True)
    finally:server.stop()


if __name__=="__main__":main()
