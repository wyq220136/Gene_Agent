"""从保存的真机 RGB-D 对照标定/桌面；不连接设备、不启动仿真。"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))


def main(argv=None):
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--run',help='运行目录或 initial_scene 场景包')
    source.add_argument('--perception',help='感知准备目录；失败准备也可检查')
    parser.add_argument('--camera-params',help='准备失败无 setup 时，指定原 cam_param.json')
    parser.add_argument('--table-roi',nargs=4,type=int,metavar=('U0','V0','U1','V1'),help='干净桌面上表面的像素矩形；不依赖标定拟合平面')
    parser.add_argument('--anchor',nargs=5,type=float,action='append',default=[],metavar=('U','V','SDK_X','SDK_Y','SDK_Z'),help='可重复；同帧可见标记像素与独立测得的右臂 SDK 坐标，单位米')
    parser.add_argument('--output',required=True,help='新目录；生成自包含 HTML 和 JSON')
    args=parser.parse_args(argv)
    from gene_agent.visualization.realworld_alignment import load_inputs,build_alignment_report
    inputs=load_inputs(run=args.run,perception=args.perception,camera_params=args.camera_params)
    path=build_alignment_report(inputs,args.output,table_roi=args.table_roi,anchors=args.anchor)
    print(path)
    return 0


if __name__=='__main__':raise SystemExit(main())
