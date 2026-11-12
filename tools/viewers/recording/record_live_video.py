"""由在线实际 qpos 与冻结模型离屏导出视频；允许失败前缀，不重跑 policy。"""
from pathlib import Path
import argparse
import bisect
import json
import os
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
os.environ.setdefault('MUJOCO_GL','egl')


def record(run,output,fps=12):
    from gene_agent.feedback.live_record import JsonlTail,atomic_json
    import mujoco
    import numpy as np
    run=Path(run).resolve();output=Path(output).resolve()
    frames=JsonlTail(run/'live/frames.jsonl').read()
    if len(frames)<2:raise ValueError('缺少实际在线采样；历史记录请使用原任务导出器')
    times=[0.]
    for a,b in zip(frames,frames[1:]):times.append(times[-1]+max(0.,b['sim_time_s']-a['sim_time_s']))
    output.parent.mkdir(parents=True,exist_ok=True)
    temporary=output.with_name(output.stem+'.partial.mp4')
    encoder=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24',
        '-s','960x720','-r',str(fps),'-i','-','-c:v','libx264','-preset','veryfast','-crf','22',
        '-pix_fmt','yuv420p',str(temporary)],stdin=subprocess.PIPE,stderr=subprocess.PIPE)
    renderer=None;model_name=None
    count=max(2,int(np.ceil(times[-1]*fps))+1)
    try:
        for index in range(count):
            row=frames[max(0,bisect.bisect_right(times,min(index/fps,times[-1]))-1)]
            if row['model']!=model_name:
                if renderer:renderer.close()
                model_name=row['model']
                if Path(model_name).name!=model_name:raise ValueError('冻结模型名称无效')
                model=mujoco.MjModel.from_binary_path(str(run/'live'/model_name));data=mujoco.MjData(model)
                model.vis.global_.offwidth=max(960,model.vis.global_.offwidth);model.vis.global_.offheight=max(720,model.vis.global_.offheight)
                renderer=mujoco.Renderer(model,width=960,height=720)
                camera=mujoco.MjvCamera();camera.azimuth=145;camera.elevation=-30;camera.distance=1.05;camera.lookat[:]=[.44,0.,.12]
                option=mujoco.MjvOption();option.geomgroup[3]=0
            if len(row['qpos'])!=model.nq:raise ValueError('qpos 与冻结模型不一致')
            data.qpos[:]=row['qpos'];data.time=row['sim_time_s'];mujoco.mj_forward(model,data)
            renderer.update_scene(data,camera=camera,scene_option=option)
            encoder.stdin.write(renderer.render().tobytes())
        encoder.stdin.close();error=encoder.stderr.read().decode('utf-8',errors='replace')
        if encoder.wait()!=0:raise RuntimeError(error)
        temporary.replace(output)
    except BaseException:
        encoder.kill();encoder.wait();temporary.unlink(missing_ok=True);raise
    finally:
        if renderer:renderer.close()
    atomic_json(output.with_suffix('.manifest.json'),dict(source_run=str(run),source='actual_live_qpos',
        sample_count=len(frames),frame_count=count,duration_s=times[-1],fps=fps,
        evidence=(json.loads((run/'live/manifest.json').read_text(encoding='utf-8')))))
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();print(record(args.run,args.output))

if __name__=='__main__':main()
