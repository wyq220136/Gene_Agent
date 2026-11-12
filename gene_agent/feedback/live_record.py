"""可选在线记录：原子规划快照与只追加的实际 qpos；不定义任务成功。"""
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
import json
import time

_SINK=ContextVar('live_physics_sink',default=None)


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    def convert(item):
        if hasattr(item,'tolist'):return item.tolist()
        if isinstance(item,Path):return str(item)
        raise TypeError(type(item).__name__)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,default=convert,allow_nan=False),encoding='utf-8')
    temporary.replace(path)


def publish_plan(run,report):
    root=Path(run)/'live'
    atomic_json(root/'plan.json',report)
    atomic_json(root/'manifest.json',dict(schema_version=1,kind='planned',task=report['task'],
        status=report['status'],updated_at=time.time(),physics_used=False))


class JsonlTail:
    """只消费完整行；中断留下的半行不能吞掉或变成伪帧。"""
    def __init__(self,path):self.path=Path(path);self.offset=0

    def read(self):
        if not self.path.exists():return []
        rows=[]
        with self.path.open('rb') as stream:
            stream.seek(self.offset)
            while True:
                line=stream.readline()
                if not line or not line.endswith(b'\n'):break
                rows.append(json.loads(line));self.offset=stream.tell()
        return rows


class _PhysicsSink:
    def __init__(self,run):
        self.root=Path(run)/'live';self.models={};self.count=0;self.manifest=None;self.error=None

    def sample(self,runtime,row):
        # 调用处就是既有实际状态采样点，绝不额外步进、插值或重置 runtime。
        import mujoco
        key=id(runtime.model)
        if key not in self.models:
            self.root.mkdir(parents=True,exist_ok=True)
            name=f'model_{len(self.models)}.mjb'
            mujoco.mj_saveModel(runtime.model,str(self.root/name),None)
            self.models[key]=name
        payload=dict(qpos=row['qpos'],sim_time_s=row['sim_time_s'],model=self.models[key],
            simulation_attachments=row.get('simulation_attachments',[]))
        with (self.root/'frames.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(payload,ensure_ascii=False,allow_nan=False)+'\n')
        self.count+=1
        self.manifest=dict(schema_version=1,kind='actual',status='running',physics_used=True,
            model=self.models[key],sample_count=self.count,updated_at=time.time(),
            evidence_mode='actual_mujoco_qpos',simulation_assistance='见原始任务报告与逐帧 simulation_attachments')
        # 每秒最多刷新4次索引；帧流仍完整追加。
        if self.count==1 or time.monotonic()-getattr(self,'last_flush',0)>.25:
            atomic_json(self.root/'manifest.json',self.manifest);self.last_flush=time.monotonic()

    def finish(self,error=None):
        if self.manifest is None:return
        self.manifest.update(status='stopped',updated_at=time.time(),sample_count=self.count)
        for name in ('summary.json','task_result.json','result.json'):
            path=self.root.parent/name
            if path.exists():
                report=json.loads(path.read_text(encoding='utf-8'))
                self.manifest.update(status=report.get('status','stopped'),failure_stage=report.get('failure_stage',report.get('failure_subtask')),
                    task_success=report.get('goal_success',report.get('task_success',report.get('success'))))
                break
        if error is not None:
            self.manifest.update(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',
                error=f'{type(error).__name__}: {error}',task_success=None)
            self.manifest['failure_stage']=self.manifest.get('failure_stage') or 'exception'
        if self.error:self.manifest['recording_error']=self.error
        atomic_json(self.root/'manifest.json',self.manifest)


def record_actual_sample(runtime,row):
    sink=_SINK.get()
    if sink is not None and sink.error is None:
        try:sink.sample(runtime,row)
        except Exception as error:
            # 反馈故障不改变运动/任务判定；显式报错，不静默伪造空视频。
            sink.error=f'{type(error).__name__}: {error}'
            print('在线记录失败：'+sink.error,flush=True)


@contextmanager
def live_physics_recording(run,*,enabled=False):
    if not enabled:
        yield
        return
    sink=_PhysicsSink(run);token=_SINK.set(sink);error=None
    try:yield
    except BaseException as exc:error=exc;raise
    finally:
        _SINK.reset(token)
        try:sink.finish(error)
        except Exception as finish_error:print(f'在线记录结束写入失败：{finish_error}',flush=True)
