"""纯视觉规划；不加载物理引擎，不发送设备命令。"""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task',choices=('sweep','microwave'),required=True)
    parser.add_argument('--scene',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--collision-mode',choices=('full','robot_only'),default='full')
    parser.add_argument('--skill')
    parser.add_argument('--seed',type=int,default=123)
    parser.add_argument('--randomize',action='store_true')
    parser.add_argument('--tilt',type=float,default=-20.)
    args=parser.parse_args()
    from gene_agent.tasks.planning.runner import VisualPlanningRun
    report=VisualPlanningRun(args.task,args.scene,args.output,skill=args.skill,seed=args.seed,
        randomize=args.randomize,tilt=args.tilt,collision_mode=args.collision_mode).run()
    print(f"planning_success={report['planning_success']}；task_success=未知；mujoco_imported={report['mujoco_imported']}",flush=True)
    if not report['planning_success']:raise SystemExit(1)

if __name__=='__main__':main()
