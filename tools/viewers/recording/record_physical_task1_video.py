"""微波炉旧录制命令兼容入口；实际导出统一调用 visualization.video_export。"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
os.environ.setdefault('MUJOCO_GL','egl')

from gene_agent.visualization.replay import ReplayTimeline
from gene_agent.visualization.video_export import export_video


def collect_actual_samples(run):
    """兼容切片工具；与viewer共用索引，包含转移与松爪落稳边界。"""
    timeline=ReplayTimeline.load(Path(run)).select(mode='actual')
    indexed={}
    for segment in timeline._segments:
        for sample in segment.samples:
            if sample.qpos is None or sample.sim_time_s is None:
                raise ValueError('微波炉录像缺少完整实际 qpos/sim_time_s')
            indexed[round(sample.sim_time_s,9)]={'sim_time_s':sample.sim_time_s,
                'qpos':list(sample.qpos),'label':segment.stage.name}
    return [indexed[key] for key in sorted(indexed)]


def record_physical_task1_video(run, *, fps=12,width=640,height=480):
    run=Path(run).resolve()
    summary=json.loads((run/'summary.json').read_text())
    if summary.get('status')!='completed' or not summary.get('goal_success'):
        raise ValueError('成功视频要求完整 Task 已通过；失败前缀可用通用视频入口')
    # 全部输出先在同目录暂存；任一裁切失败都不覆盖旧交付及其manifest。
    with tempfile.TemporaryDirectory(prefix='.task_video.',dir=run) as temporary:
        staging=Path(temporary)
        movie=staging/'task_video.mp4'
        result=export_video(run,movie,cameras=('overview','detail'),profile='microwave_task1',
                            fps=fps,width=width,height=height)
        overview,detail=staging/'task_overview.mp4',staging/'task_detail.mp4'
        subprocess.run(['ffmpeg','-y','-v','error','-i',str(movie),'-filter_complex',
            f'[0:v]split=2[a][b];[a]crop={width}:{height}:0:0[o];[b]crop={width}:{height}:{width}:0[d]',
            '-map','[o]','-c:v','libx264','-preset','veryfast','-threads','2','-crf','22',str(overview),
            '-map','[d]','-c:v','libx264','-preset','veryfast','-threads','2','-crf','22',str(detail)],
            check=True,capture_output=True)
        result['video']=str(run/'task_video.mp4')
        text=json.dumps(result,ensure_ascii=False,indent=2)+'\n'
        (staging/'task_video.json').write_text(text)
        manifest={**result,'videos':{'combined':str(run/'task_video.mp4'),
            'overview':str(run/'task_overview.mp4'),'detail':str(run/'task_detail.mp4')}}
        (staging/'video_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
        for name in ('task_video.mp4','task_overview.mp4','task_detail.mp4',
                     'task_video.json','video_manifest.json'):
            (staging/name).replace(run/name)
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    parser.add_argument('--fps',type=int,default=12)
    args=parser.parse_args()
    result=record_physical_task1_video(args.run,fps=args.fps)
    print(json.dumps({'videos':result['videos'],'timing_s':result['timing_s']},ensure_ascii=False))


if __name__=='__main__':main()
