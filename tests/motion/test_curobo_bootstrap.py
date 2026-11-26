"""启动优化只能冻结导入对象，不能接管已有CUDA会话或已有冻结集合。"""
from types import SimpleNamespace
import pytest
from gene_agent.motion.curobo import bootstrap


class GC:
    def __init__(self, frozen=0):
        self.frozen = frozen
        self.calls = []
    def get_freeze_count(self): return self.frozen
    def collect(self): self.calls.append('collect'); return 0
    def freeze(self): self.calls.append('freeze'); self.frozen = 123
    def unfreeze(self): self.calls.append('unfreeze'); self.frozen = 0


@pytest.mark.parametrize('raises', [False, True])
def test_bootstrap_restores_collection_after_task(monkeypatch, raises):
    gc = GC()
    monkeypatch.setattr(bootstrap, 'gc', gc)
    monkeypatch.setattr(bootstrap, '_load_runtime', lambda: SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda: False)))
    def execute():
        with bootstrap.curobo_import_gc_scope() as report:
            assert report['applied'] and report['frozen_objects'] == 123
            gc.collect()  # 图初始化的显式回收仍能正常调用。
            if raises: raise RuntimeError('task failed')
    if raises:
        with pytest.raises(RuntimeError, match='task failed'): execute()
    else: execute()
    assert gc.frozen == 0
    assert gc.calls == ['collect', 'freeze', 'collect', 'unfreeze', 'collect']


@pytest.mark.parametrize('enabled,frozen,cuda,reason', [
    (False, 0, False, 'disabled'),
    (True, 7, False, 'existing_frozen_objects'),
    (True, 0, True, 'cuda_already_initialized'),
])
def test_bootstrap_does_not_modify_existing_runtime(monkeypatch, enabled, frozen, cuda, reason):
    gc = GC(frozen)
    monkeypatch.setattr(bootstrap, 'gc', gc)
    monkeypatch.setattr(bootstrap, '_load_runtime', lambda: SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda: cuda)))
    with bootstrap.curobo_import_gc_scope(enabled=enabled) as report:
        assert not report['applied'] and report['reason'] == reason
    assert gc.frozen == frozen
    assert gc.calls == []
