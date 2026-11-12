"""可选的同步规划进程边界，用于原生环境和本项目cuRobo依赖不兼容时。"""
from __future__ import annotations
from contextlib import contextmanager
import json
import os
from pathlib import Path
import selectors
import subprocess
import sys
import time
from uuid import uuid4

from ..model import motion_result_from_dict


class ProcessPlanningSession:
    """worker内持有真实CuroboPlanningSession/pool，不提供替代求解器。"""
    def __init__(self,robot,*,python=sys.executable,pythonpath=None,log_path=None,timeout_s=300.,diagnostic_options=None,request_output_dir=None):
        self.robot=robot
        self.diagnostic_options=diagnostic_options
        self.request_output_dir=str(Path(request_output_dir).resolve()) if request_output_dir else None
        self._python=python
        self._pythonpath=pythonpath
        self._log_path=Path(log_path) if log_path else None
        self._timeout=float(timeout_s)
        self._process=None
        self._log=None
        self.closed=False

    def __enter__(self):
        """只启动协议进程；solver在首个真实请求时初始化。"""
        if self.closed:raise RuntimeError('规划session已关闭')
        if self._process is not None:raise RuntimeError('规划进程已启动')
        root=Path(__file__).resolve().parents[3]
        env=os.environ.copy()
        env['PYTHONPATH']=os.pathsep.join(filter(None,[self._pythonpath,str(root)]))
        if self._log_path:
            self._log_path.parent.mkdir(parents=True,exist_ok=True);self._log=self._log_path.open('w')
        self._process=subprocess.Popen([self._python,'-u','-m','gene_agent.motion.curobo.worker'],
            cwd=root,env=env,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self._log or subprocess.DEVNULL,text=True,bufsize=1)
        return self

    def _request(self, payload):
        """记录每次等待，包括worker异常；JSONL独立于重复的阶段/候选结果。"""
        started = time.perf_counter()
        response = None
        error = None
        try:
            response = self._exchange(payload)
            return response
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
            raise
        finally:
            if self._log_path and payload.get('operation') not in {'describe', 'fk'}:
                try:
                    rows = response if isinstance(response, list) else [response] if isinstance(response, dict) else []
                    keys = {'planner_cache', 'planning_timing', 'timing', 'batch_size', 'batch_total_duration_s', 'replay_request_ref', 'request_recording_s', 'request_recording_error'}
                    results = [dict(request_id=r.get('request_id'), operation=r.get('operation'),
                                    success=r.get('success'), duration_s=r.get('duration_s'),
                                    metrics={k: v for k, v in r.get('metrics', {}).items() if k in keys}) for r in rows]
                    identifiers = payload.get('request_ids') or [payload.get('request_id')]
                    stage = 'candidate' if identifiers[0] and identifiers[0].startswith('candidate/') else (identifiers[0] or '').split('/')[-1]
                    row = dict(schema_version=1, call_id=uuid4().hex, operation=payload.get('operation'),
                               stage_id=stage, group=payload.get('group'), request_ids=identifiers,
                               scene_snapshot_sha256=payload.get('snapshot', {}).get('snapshot_sha256'),
                               roundtrip_wall_s=time.perf_counter() - started, results=results, error=error)
                    with self._log_path.with_name('planning_calls.jsonl').open('a') as stream:
                        stream.write(json.dumps(row, allow_nan=False) + '\n')
                except Exception as recording_error:
                    import warnings
                    warnings.warn(f'规划计时记录失败: {recording_error}', RuntimeWarning)

    def _exchange(self,payload):
        """一问一答，失败保留worker traceback；有时限而不是无限等待。"""
        if self.closed or self._process is None:raise RuntimeError('规划进程未运行')
        proc=self._process
        proc.stdin.write(json.dumps(payload,allow_nan=False)+'\n');proc.stdin.flush()
        with selectors.DefaultSelector() as selector:
            selector.register(proc.stdout,selectors.EVENT_READ)
            if not selector.select(self._timeout):
                self.close();raise TimeoutError('规划worker响应超时')
        line=proc.stdout.readline()
        if not line:raise RuntimeError(f'规划worker退出，code={proc.poll()}；日志 {self._log_path}')
        response=json.loads(line)
        if not response['ok']:raise RuntimeError(response['error']+'\n'+response.get('traceback',''))
        return response['result']

    def describe(self):
        """无GPU地报告实际API来源，供部署前检查两版依赖隔离。"""
        return self._request({'operation':'describe'})

    @contextmanager
    def stage(self, *, snapshot,start,group,scene_revision,collision_policy=None,**kwargs):
        """保持原stage调用形状；实际上下文和pool在worker中创建。"""
        if group not in self.robot.get('joint_groups',{}):raise ValueError('未知机器人组')
        if self.diagnostic_options is not None:
            kwargs.setdefault("diagnostic_options", self.diagnostic_options)
        if self.request_output_dir is not None:
            kwargs.setdefault("request_output_dir", self.request_output_dir)
        stage=_ProcessStage(self,dict(robot=self.robot,snapshot=snapshot,start=start.to_dict(),group=group,
                scene_revision=scene_revision,collision_policy=collision_policy.to_dict() if collision_policy else None,options=kwargs))
        try:yield stage
        finally:stage.active=False

    def close(self):
        """仅关闭本对象拥有的进程，异常路径也回收GPU资源。"""
        if self.closed:return
        self.closed=True
        if self._process is not None:
            try:
                self._process.stdin.write('{"operation":"close"}\n');self._process.stdin.flush()
                self._process.wait(timeout=5)
            except (OSError,subprocess.TimeoutExpired):
                self._process.kill();self._process.wait()
            finally:
                self._process.stdin.close();self._process.stdout.close()
        if self._log:self._log.close()

    def __exit__(self,*args):self.close()


class _ProcessStage:
    """只转交现有请求和结果；离开stage后的句柄不能继续使用。"""
    def __init__(self,session,context):self.session=session;self.context=context;self.active=True

    def _call(self,operation,target=None,request_id=None,**kwargs):
        if not self.active:raise RuntimeError('stage已结束')
        result=self.session._request({**self.context,'operation':operation,
            'target':target.to_dict() if target else None,'request_id':request_id,'request_options':kwargs})
        if operation=='fk':
            import numpy as np
            return np.asarray(result)
        return motion_result_from_dict(result)

    def fk(self):return self._call('fk')
    def evaluate_ik(self,targets,*,request_ids):
        if not self.active:raise RuntimeError('stage已结束')
        targets=list(targets);request_ids=list(request_ids)
        if len(targets)!=len(request_ids) or len(set(request_ids))!=len(request_ids):raise ValueError('batch IK请求ID须唯一且与目标对应')
        rows=self.session._request({**self.context,'operation':'evaluate_ik','targets':[t.to_dict() for t in targets],'request_ids':request_ids})
        if len(rows)!=len(targets):raise RuntimeError('batch IK结果数量不匹配')
        return tuple(motion_result_from_dict(r) for r in rows)
    def plan_to(self,target,*,request_id,**kwargs):return self._call('plan_to',target,request_id,**kwargs)
    def plan_linear(self,target,*,request_id,**kwargs):return self._call('plan_linear',target,request_id,**kwargs)
