#!/usr/bin/env python3
"""启动 Gene Agent 本地任务开发工作台。"""
import argparse
from pathlib import Path
import sys
import webbrowser

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',type=int,default=8070)
    parser.add_argument('--python',default=sys.executable,help='模型、仿真与 viewer 使用的 Python，建议 curobo 环境')
    parser.add_argument('--open',action='store_true',help='同时打开浏览器')
    args=parser.parse_args()
    from gene_agent.workbench.server import make_server
    server=make_server(ROOT,args.port,args.python)
    url=f'http://127.0.0.1:{server.server_port}'
    print(f'Gene Agent 工作台：{url}\nCtrl+C 关闭服务及本服务启动的作业。',flush=True)
    if args.open:webbrowser.open(url)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.service.jobs.close();server.server_close()

if __name__=='__main__':main()
