"""共享取出/旋钮流程必须在证据未知、失败或状态链断裂时停下。"""
import pytest

from gene_agent.skills.contracts import OperationState, StageOutcome
from gene_agent.skills.library.transfer.skill import run_extract_flow
from gene_agent.skills.library.knob.skill import run_knob_flow
from gene_agent.tasks.checks import CheckResult


class Backend:
    evidence_mode = 'execution'

    def __init__(self, failure=None, status='unknown', source='observed'):
        self.names = []
        self.failure, self.status, self.source = failure, status, source

    def observe(self):
        return OperationState('0', 'observed', {})

    def execute(self, stage, start):
        self.names.append(stage.name)
        check = CheckResult(self.status if stage.name == self.failure else 'pass',
                            self.source if stage.name == self.failure else 'observed', {})
        end = OperationState(str(len(self.names)), 'observed', {})
        return StageOutcome(stage.name, True, start, end, check)


def test_extraction_grasp_unknown_prevents_attachment():
    backend = Backend('grasp_check')
    result = run_extract_flow(backend)
    assert not result.success
    assert result.failure_stage == 'grasp_check'
    assert 'attach' not in backend.names


def test_extraction_retreat_rechecks_grasp_and_hold():
    backend = Backend('post_extract_1_hold_check', 'fail')
    result = run_extract_flow(backend, resume_after_grasp=True)
    assert backend.names == ['grasp_check', 'attach', 'post_extract_1_hold_check']
    assert result.failure_code == 'postcondition_failed'


def test_extraction_prefix_stops_after_check():
    backend = Backend()
    assert run_extract_flow(backend, stop_after='grasp').success
    assert backend.names == ['open', 'pre_extract_1', 'pre_extract_2', 'grasp', 'close', 'grasp_check']


def test_knob_grasp_unknown_prevents_rotation():
    backend = Backend('grasp_check')
    result = run_knob_flow(backend, target_names=lambda: ['open', 'pregrasp', 'approach', 'close', 'grasp_settle', 'rotate_90', 'release', 'retreat'])
    assert not result.success
    assert backend.names[-1] == 'grasp_check'
    assert 'rotate_90' not in backend.names


def test_knob_final_predicted_evidence_is_rejected():
    backend = Backend('rotation_check', 'pass', 'prediction')
    result = run_knob_flow(backend, target_names=lambda: ['open', 'pregrasp', 'approach', 'close', 'grasp_settle', 'rotate_90', 'release', 'retreat'])
    assert result.failure_code == 'predicted_check_in_execution'


def test_knob_targets_selected_only_after_open():
    backend = Backend()
    def names():
        assert backend.names == ['open']
        return ['open', 'pregrasp', 'approach', 'close', 'rotate_90', 'release', 'retreat']
    assert run_knob_flow(backend, target_names=names).success
    assert backend.names[-1] == 'rotation_check'


def test_real_knob_cannot_enable_simulation_eligibility():
    with pytest.raises(ValueError, match='混合仿真'):
        run_knob_flow(Backend(), target_names=lambda: [], grasp_mode='declared_kinematic')


def test_declared_simulation_preserves_unknown_grasp_evidence():
    backend = Backend('grasp_check')
    backend.evidence_mode = 'mujoco_mixed_knob'
    result = run_knob_flow(backend, grasp_mode='declared_kinematic', target_names=lambda: [
        'open', 'pregrasp', 'approach', 'close', 'rotate_90', 'release', 'retreat'])
    assert result.success
    assert next(s for s in result.stages if s.stage == 'grasp_check').check.status == 'unknown'
    assert 'kinematic_eligibility_check' in backend.names


def test_knob_rejects_rotation_before_grasp():
    with pytest.raises(ValueError, match='前缀'):
        run_knob_flow(Backend(), target_names=lambda: [
            'open', 'rotate_90', 'pregrasp', 'approach', 'close', 'release', 'retreat'])


@pytest.mark.parametrize('flow', [run_extract_flow, lambda backend: run_knob_flow(
    backend, target_names=lambda: ['open', 'pregrasp', 'approach', 'close',
                                  'rotate_90', 'release', 'retreat'])])
def test_shared_flows_preserve_failed_end_and_stop_on_state_discontinuity(flow):
    backend = Backend()
    def execute(stage, start):
        backend.names.append(stage.name)
        return StageOutcome(stage.name, True, OperationState('unrelated', 'observed', {}),
                            OperationState('measured_after_failure', 'observed', {}))
    backend.execute = execute
    result = flow(backend)
    assert not result.success
    assert result.failure_code == 'state_discontinuity'
    assert result.end_state.state_id == 'measured_after_failure'
    assert backend.names == ['open']
