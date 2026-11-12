"""统一 Skill/Task 视频入口：共享回放、离屏 Renderer、FFmpeg 流式编码。"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
# 在 import mujoco 前选择无窗口后端；软件诊断可显式 MUJOCO_GL=osmesa。
os.environ.setdefault('MUJOCO_GL', 'egl')

from gene_agent.visualization.video_export import export_video


def record(run, output, *, stage=None, mode='actual', camera_name='task', skill=None,
           fps=24, width=960, height=720, **options):
    """保留既有 Python 入口；正式实现位于共享 visualization 模块。"""
    return export_video(run, output, stage=stage, mode=mode, cameras=(camera_name,),
                        skill=skill, fps=fps, width=width, height=height, **options)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--mode',choices=('actual','planned'),default='actual')
    parser.add_argument('--stage')
    parser.add_argument('--skill',help='保存回放中的范围，如 pick/insert/extract/place')
    parser.add_argument('--camera',help='兼容单视角，如 task/contact/gripper_side')
    parser.add_argument('--cameras',nargs='+',help='横向拼接；微波炉推荐 overview detail')
    parser.add_argument('--profile',choices=('microwave_task1','standard'),
                        help='省略时对完整微波炉 Task1 自动选机位')
    parser.add_argument('--fps',type=int,default=12)
    parser.add_argument('--width',type=int,default=640,help='每个视角宽度')
    parser.add_argument('--height',type=int,default=480)
    parser.add_argument('--full-forward',action='store_true',help='仅用于同画质耗时对照')
    parser.add_argument('--preset',default='veryfast')
    parser.add_argument('--threads',type=int,default=2)
    args=parser.parse_args()
    if args.camera and args.cameras:parser.error('--camera 与 --cameras 只能用一个')
    profile=args.profile
    summary=args.run/'summary.json'
    if profile is None and summary.is_file():
        if json.loads(summary.read_text()).get('task')=='microwave_task1':profile='microwave_task1'
    if profile=='standard':profile=None
    cameras=tuple(args.cameras or ([args.camera] if args.camera else
                  (['overview','detail'] if profile else ['task'])))
    result=export_video(args.run,args.output,mode=args.mode,cameras=cameras,profile=profile,
        fps=args.fps,width=args.width,height=args.height,stage=args.stage,skill=args.skill,
        full_forward=args.full_forward,preset=args.preset,threads=args.threads)
    print(json.dumps({k:result[k] for k in ('video','mode','evidence_mode','task_success',
        'frame_count','duration_s','source_duration_s','timing_s')},ensure_ascii=False))


if __name__=='__main__':main()
