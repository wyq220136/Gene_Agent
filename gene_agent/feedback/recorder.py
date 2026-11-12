"""短摘要与完整错误分开落盘，保留稳定的调用方 stage_id。"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import platform
import re
import traceback
import mujoco


def _write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def fingerprint_files(paths):
    return {str(Path(path)): hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in paths}


class RunRecorder:
    def __init__(self, output_dir, config=None, metadata=None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if (self.output_dir / 'manifest.json').exists():
            raise FileExistsError(f'Run already exists: {self.output_dir}')
        self.sequence = 0
        self.manifest = {'schema_version': 1, 'created_at': datetime.now(timezone.utc).isoformat(),
                         'python_version': platform.python_version(), 'mujoco_version': mujoco.__version__,
                         'metadata': metadata or {}}
        _write(self.output_dir / 'manifest.json', self.manifest)
        _write(self.output_dir / 'resolved_config.json', config or {})
        (self.output_dir / 'events.jsonl').touch()
        (self.output_dir / 'logs.txt').touch()

    def event(self, stage_id, status, **details):
        if not re.fullmatch(r'[A-Za-z0-9_.-]+', stage_id) or stage_id in {'.', '..'}:
            raise ValueError('stage_id must be a stable simple identifier')
        event = {'sequence': self.sequence, 'stage_id': stage_id, 'status': status,
                 'wall_time': datetime.now(timezone.utc).isoformat(), 'details': details}
        with (self.output_dir / 'events.jsonl').open('a') as stream:
            stream.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + '\n')
        self.sequence += 1
        return event

    def log(self, message):
        with (self.output_dir / 'logs.txt').open('a') as stream:
            stream.write(str(message) + '\n')

    def record_error(self, stage_id, exc):
        self.event(stage_id, 'failed', error_type=type(exc).__name__, message=str(exc)[:300])
        folder = self.output_dir / 'tracebacks'
        folder.mkdir(exist_ok=True)
        # 使用追加方式保留同一阶段的所有失败，而非只留下最后一次重试。
        with (folder / f'{stage_id}.txt').open('a') as stream:
            stream.write(''.join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
        self.log(f'{stage_id}: {type(exc).__name__}: {exc}')

    def finish(self, status, **summary):
        value = {'status': status, **summary}
        _write(self.output_dir / 'summary.json', value)
        return value
