"""持久化本地作业；仿真、摆位渲染、查看器和助手分别调度。"""
from concurrent.futures import ThreadPoolExecutor
import copy
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time
import uuid
import fcntl


class JobManager:
    def __init__(self,root):
        self.root=Path(root).resolve();self.base=self.root/'runs/workbench/jobs'
        self.base.mkdir(parents=True,exist_ok=True)
        self.owner=(self.base/'.service.lock').open('a')
        try:fcntl.flock(self.owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            self.owner.close();raise ValueError('已有工作台管理该项目，请使用已打开的服务') from None
        self.lock=threading.RLock();self.rows={};self.processes={};self.closed=False
        self.heavy=ThreadPoolExecutor(max_workers=1);self.light=ThreadPoolExecutor(max_workers=12)
        self.chat=ThreadPoolExecutor(max_workers=1)
        # 摆位只构建初态和渲染图片，不应等待完整 cuRobo 仿真或常驻查看窗口退出。
        # 同类预览仍串行，限制同时创建的 MuJoCo/OpenGL 上下文和内存峰值。
        self.preview=ThreadPoolExecutor(max_workers=1)
        self.interactive=ThreadPoolExecutor(max_workers=8)
        for p in self.base.glob('*/job.json'):
            try:
                row=json.loads(p.read_text())
                if row['state'] in {'queued','running','stopping'}:
                    # 只回收上次保存且启动标识仍一致的进程组，避免误杀复用的 PID。
                    pid=row.get('pid')
                    if pid and row.get('pid_start') and self._pid_start(pid)==row['pid_start']:
                        try:os.killpg(pid,signal.SIGTERM)
                        except ProcessLookupError:pass
                    row.update(state='interrupted',message='服务重启；未自动重跑。',finished_at=time.time())
                self.rows[row['id']]=row;self._save(row)
            except (ValueError,KeyError,OSError):continue

    def directory(self,jid):
        if not jid or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in jid):raise ValueError('无效作业 ID')
        return self.base/jid

    def _save(self,row):
        p=self.directory(row['id']);p.mkdir(exist_ok=True)
        tmp=p/'job.tmp';tmp.write_text(json.dumps(row,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(p/'job.json')

    def submit(self,kind,inputs,commands,*,background=False,timeout=7200,job_id=None):
        with self.lock:
            if self.closed:raise ValueError('服务正在退出')
            jid=job_id or time.strftime('%Y%m%d_%H%M%S_')+uuid.uuid4().hex[:8]
            if jid in self.rows:raise ValueError('作业已存在')
            lane=('chat' if kind=='ask' else 'preview' if kind=='distribution' else
                  'interactive' if kind in {'knowledge_preview','visual_viewer','live_viewer','run_preview'} else 'light' if background else 'heavy')
            row={'id':jid,'kind':kind,'inputs':inputs,'state':'queued','created_at':time.time(),
                 'resource_queue':lane,
                 'steps':[],'commands':commands,'timeout_s':timeout,'background':background}
            self.rows[jid]=row;self._save(row)
            getattr(self,lane).submit(self._run,jid)
            return copy.deepcopy(row)

    def _run(self,jid):
        with self.lock:
            row=self.rows[jid]
            if row['state']=='canceled':return
            row.update(state='running',started_at=time.time());self._save(row)
        failed=False
        try:
            env=os.environ.copy();env.update(PYTHONPATH=str(self.root),PYTHONNOUSERSITE='1',PYTHONUNBUFFERED='1',PYTHONUTF8='1',MUJOCO_GL='egl')
            with (self.directory(jid)/'output.log').open('ab',buffering=0) as log:
                for cmd in row['commands']:
                    with self.lock:
                        if row['state']=='canceled':break
                        step={'argv':cmd,'started_at':time.time()};row['steps'].append(step);self._save(row)
                        proc=subprocess.Popen(cmd,cwd=self.root,env=env,stdout=log,stderr=subprocess.STDOUT,
                                              stdin=subprocess.DEVNULL,start_new_session=True)
                        self.processes[jid]=proc
                        row.update(pid=proc.pid,pid_start=self._pid_start(proc.pid));self._save(row)
                    try:code=proc.wait(timeout=row['timeout_s'])
                    except subprocess.TimeoutExpired:
                        self._terminate(proc);code=proc.wait();step['timeout']=True
                    with self.lock:
                        self.processes.pop(jid,None);step.update(returncode=code,finished_at=time.time())
                        row.pop('pid',None);row.pop('pid_start',None)
                        failed=failed or code!=0;self._save(row)
                    # 即使任务失败也运行下一个视频导出命令，其结果独立记录。
        except Exception as exc:
            failed=True
            with self.lock:row['error']=f'{type(exc).__name__}: {exc}'
        finally:
            with self.lock:
                if row['state']!='canceled':row['state']='failed' if failed else 'completed'
                row['finished_at']=time.time();self._save(row)

    @staticmethod
    def _pid_start(pid):
        try:return Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()[19]
        except (OSError,IndexError):return None

    @staticmethod
    def _terminate(proc):
        try:os.killpg(proc.pid,signal.SIGTERM)
        except ProcessLookupError:return
        try:proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            try:os.killpg(proc.pid,signal.SIGKILL)
            except ProcessLookupError:pass

    def cancel(self,jid):
        with self.lock:
            row=self.rows[jid]
            if row['state'] not in {'queued','running'}:return
            row.update(state='canceled',finished_at=time.time());self._save(row)
            proc=self.processes.get(jid)
        if proc:self._terminate(proc)

    def get(self,jid):
        with self.lock:return copy.deepcopy(self.rows[jid])

    def list(self):
        with self.lock:return sorted(copy.deepcopy(list(self.rows.values())),key=lambda r:r['created_at'],reverse=True)

    def log(self,jid):
        p=self.directory(jid)/'output.log'
        if not p.exists():return ''
        with p.open('rb') as f:
            f.seek(max(0,p.stat().st_size-24000));return f.read().decode('utf-8',errors='replace')

    def close(self):
        self.closed=True
        for row in self.list():
            if row['state'] in {'queued','running'}:self.cancel(row['id'])
        self.heavy.shutdown(wait=False,cancel_futures=True);self.light.shutdown(wait=False,cancel_futures=True)
        self.chat.shutdown(wait=False,cancel_futures=True)
        self.preview.shutdown(wait=False,cancel_futures=True)
        self.interactive.shutdown(wait=False,cancel_futures=True)
        fcntl.flock(self.owner,fcntl.LOCK_UN);self.owner.close()
