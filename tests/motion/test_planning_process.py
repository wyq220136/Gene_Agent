"""隔离规划进程必须保持协议、依赖来源和确定的资源关闭。"""
import sys


def test_process_describes_dependency_without_loading_simulators(tmp_path):
    from gene_agent.motion.curobo.process import ProcessPlanningSession
    session=ProcessPlanningSession({},python=sys.executable,log_path=tmp_path/'worker.log')
    with session:
        info=session.describe()
        assert info['protocol_version']==1
        assert info['python']==sys.executable
        assert info['loaded_engines']==[]
        pid=info['pid'];assert pid>0
        assert session.describe()['pid']==pid
    assert session.closed


def test_process_stage_forwards_batch_ik_without_changing_targets():
    from gene_agent.motion.curobo.process import _ProcessStage
    from gene_agent.motion import PoseTarget
    import numpy as np
    class Session:
        def _request(self,payload):
            assert payload['operation']=='evaluate_ik'
            assert len(payload['targets'])==2
            assert payload['request_ids']==['a','b']
            return []
    stage=_ProcessStage(Session(),{})
    import pytest
    with pytest.raises(RuntimeError,match='数量'):
        stage.evaluate_ik([PoseTarget('world_tcp',np.eye(4))]*2,request_ids=['a','b'])


def test_worker_exception_is_logged_as_failed_attempt_without_loading_simulator(tmp_path):
    from gene_agent.motion.curobo.process import ProcessPlanningSession
    import json,pytest
    session=ProcessPlanningSession({},python=sys.executable,log_path=tmp_path/'worker.log')
    with session:
        with pytest.raises(RuntimeError):session._request({'operation':'bad','request_id':'native/test'})
    rows=[json.loads(l) for l in (tmp_path/'planning_calls.jsonl').read_text().splitlines()]
    assert len(rows)==1 and rows[0]['error']
    assert rows[0]['roundtrip_wall_s']>=0 and rows[0]['request_ids']==['native/test']
