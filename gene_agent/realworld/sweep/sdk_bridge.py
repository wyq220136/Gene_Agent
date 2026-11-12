"""同步 JSONL 桥，SDK 与 cuRobo 可位于不同 conda 环境。"""
from __future__ import annotations

import json
import os
from pathlib import Path
import queue
import signal
import subprocess
import threading
import time

ROOT=Path(__file__).resolve().parents[3]


class SDKBridge:
    def __init__(self, config, output):
        self.config=config;self.output=Path(output);self.process=None;self.sequence=0
        self.messages=queue.Queue()

    def __enter__(self):
        self.output.mkdir(parents=True,exist_ok=True)
        path=self.output/'worker_config.json'
        path.write_text(json.dumps(self.config,ensure_ascii=False,indent=2),encoding='utf-8')
        self.log=(self.output/'sdk_worker.log').open('w',encoding='utf-8')
        self.process=subprocess.Popen([self.config['sdk_python'],'-m','gene_agent.realworld.sweep.sdk_worker',
            '--config',str(path)],cwd=ROOT,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,
            text=True,encoding='utf-8',bufsize=1,
            env={**os.environ,'PYTHONPATH':str(ROOT),'PYTHONNOUSERSITE':'1','PYTHONUTF8':'1'})
        def reader():
            for line in self.process.stdout:
                if line.startswith('GENE_RPC='):
                    try:self.messages.put(json.loads(line[len('GENE_RPC='):]))
                    except ValueError:self.messages.put({'ok':False,'error':'worker JSON损坏'})
            self.messages.put({'ok':False,'error':'SDK worker退出；查看sdk_worker.log'})
        self.reader=threading.Thread(target=reader,daemon=True);self.reader.start()
        try:
            ready=self.messages.get(timeout=40)
            if not ready.get('ready'):raise RuntimeError(str(ready))
        except BaseException:
            self.__exit__(True,None,None);raise
        return self

    def call(self, command, payload=None, timeout=600):
        self.sequence+=1;identity=self.sequence
        self.process.stdin.write(json.dumps({'id':identity,'command':command,'payload':payload or {}},allow_nan=False)+'\n')
        self.process.stdin.flush()
        try:response=self.messages.get(timeout=timeout)
        except queue.Empty as error:raise TimeoutError('SDK阶段响应超时') from error
        if response.get('id')!=identity or not response.get('ok'):
            raise RuntimeError(f'SDK worker失败：{response}')
        return response['result']

    def __exit__(self,exc_type,exc,tb):
        if self.process is not None and self.process.poll() is None:
            try:
                if exc_type is None:self.call('shutdown',timeout=5)
                else:self.process.send_signal(signal.SIGINT)
                self.process.wait(timeout=8)
            except Exception:
                self.process.terminate()
                try:self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:self.process.kill();self.process.wait()
        if hasattr(self,'log'):self.log.close()
