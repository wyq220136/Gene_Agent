"""本地一次性模型服务：仅管理自己启动的进程，阶段结束彻底回收 CUDA context。"""
from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
CONDA = Path('/home/ubuntu/miniconda3/envs')


def environment(name):
    prefix = CONDA / name
    python = prefix / 'bin/python'
    if not python.is_file():
        raise FileNotFoundError(f'缺少模型环境：{python}')
    env = {**os.environ, 'PYTHONPATH': str(ROOT), 'PYTHONNOUSERSITE': '1',
           'PYTHONUTF8': '1', 'PYTHONUNBUFFERED': '1',
           'PATH': str(prefix/'bin') + os.pathsep + os.environ.get('PATH', '')}
    return python, env


@contextmanager
def model_service(kind, output, *, startup_timeout_s=180):
    """使用空闲 localhost 端口，不占用或卸载用户的常驻服务。"""
    specs = {'seg': ('sam3', 'segmentation.seg_server', 'seg_server'),
             'pose': ('foundationpose', 'pose_est.pose_server', 'pose_server')}
    name, module, expected_service = specs[kind]
    python, env = environment(name)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    url = f'http://127.0.0.1:{port}'
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    log_path = output / f'{kind}_server.log'
    command = [str(python), '-m', f'gene_agent.perception_modules.{module}',
               '--host', '127.0.0.1', '--port', str(port)]
    with log_path.open('w', encoding='utf-8') as log:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + startup_timeout_s
            heartbeat = 0.
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f'{kind} 服务启动退出；查看 {log_path}')
                try:
                    with urllib.request.urlopen(url + '/health', timeout=2) as response:
                        health = json.load(response)
                    if health.get('service') == expected_service:
                        break
                except (OSError, ValueError):
                    pass
                if time.monotonic() >= heartbeat:
                    print(f'等待 {kind} 模型服务启动；日志：{log_path}', flush=True)
                    heartbeat = time.monotonic() + 20
                time.sleep(.3)
            else:
                raise TimeoutError(f'{kind} 服务启动超时；查看 {log_path}')
            yield url
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill(); process.wait(timeout=5)
