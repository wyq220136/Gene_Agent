"""纯视觉 viewer 的命令适配；模块导入不含物理引擎。"""
import argparse
from pathlib import Path
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task',choices=('sweep','microwave'));p.add_argument('--scene',type=Path)
    p.add_argument('--run',type=Path);p.add_argument('--knowledge');p.add_argument('--skill');p.add_argument('--output',type=Path)
    p.add_argument('--tilt',type=float,default=-20.);p.add_argument('--port',type=int,default=8093)
    args=p.parse_args()
    if not args.run and not(args.task and args.scene and args.knowledge):p.error('需要 --run 或 --task/--scene/--knowledge')
    if args.output is None:
        from uuid import uuid4
        args.output=Path('runs/tmp/visual_planning')/('viewer_'+uuid4().hex[:10])
    import viser
    from .app import PurePlanningApp
    server=viser.ViserServer(host='127.0.0.1',port=args.port)
    try:
        app=PurePlanningApp(server,task=args.task,source=args.scene,run=args.run,knowledge=args.knowledge,output=args.output,tilt=args.tilt,skill=args.skill)
        print(f'纯视觉预览：http://127.0.0.1:{server.get_port()}；未执行物理',flush=True)
        while True:app.tick();time.sleep(.03)
    finally:server.stop()

if __name__=='__main__':main()
