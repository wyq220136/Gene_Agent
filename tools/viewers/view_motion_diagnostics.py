#!/usr/bin/env python3
"""从保存的规划快照查看/导出诊断；不运行机器人或仿真。"""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from gene_agent.visualization.motion_diagnostics import export_diagnostic_image


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path);p.add_argument('--run',type=Path);p.add_argument('--episode');p.add_argument('--stage')
    p.add_argument('--output',type=Path,help='Viser图片输出目录；matplotlib兼容模式为PNG路径');p.add_argument('--port',type=int,default=8081)
    p.add_argument('--renderer',choices=('viser','matplotlib'),default='viser')
    p.add_argument('--view',choices=('both','global','local'),default='both')
    p.add_argument('--layer',choices=('both','visual','collision'),default='both')
    p.add_argument('--context-radius',type=float,default=.25,help='局部保留几何的邻域半径，米')
    p.add_argument('--pair',type=int,default=0)
    p.add_argument('--azimuth',type=float,default=-55.)
    p.add_argument('--elevation',type=float,default=25.)
    p.add_argument('--look-at',type=float,nargs=3,metavar=('X','Y','Z'),help='world观察中心，单位米')
    p.add_argument('--distance',type=float,help='相机到观察中心的米制距离；默认自动适配')
    args=p.parse_args();path=args.bundle
    if path is None:
        if not args.run or not args.episode or not args.stage:p.error('指定--bundle或--run/--episode/--stage')
        root=args.run/'episodes'/args.episode if (args.run/'episodes').exists() else args.run
        from gene_agent.feedback.motion_diagnostics import find_diagnostic_bundles
        matches=find_diagnostic_bundles(root,stage=args.stage)
        if len(matches)!=1:p.error('没有唯一诊断工件，旧记录不可静默重建')
        path=matches[0]
    data=json.loads(path.read_text());bundle=data.get('diagnostic',data)
    if not bundle:p.error('无球级诊断现场')
    if args.renderer=='matplotlib':
        if not args.output:p.error('matplotlib仅支持--output PNG')
        print(json.dumps(export_diagnostic_image(bundle,args.output),ensure_ascii=False))
        return
    from gene_agent.visualization.motion_diagnostic_viser import export_viser_views,DiagnosticViewer
    options=dict(pair_index=args.pair,azimuth=args.azimuth,elevation=args.elevation,distance=args.distance,look_at=args.look_at,context_radius=args.context_radius)
    if args.output:
        result=export_viser_views(bundle,args.output,view=args.view,layer=args.layer,**options)
        print(json.dumps(dict(output=str(args.output),visual_status=result['visual_status'],elapsed_s=result['elapsed_s'],warnings=result['warnings']),ensure_ascii=False))
    else:
        import time
        app=DiagnosticViewer(bundle,port=args.port,view='global' if args.view=='both' else args.view,layer='collision' if args.layer=='both' else args.layer,**options)
        print(f'Diagnostic viewer: http://localhost:{args.port}',flush=True)
        try:
            while True:time.sleep(1)
        finally:app.close()


if __name__=='__main__':main()
