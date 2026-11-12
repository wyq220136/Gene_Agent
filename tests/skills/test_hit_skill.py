from gene_agent.skills.contracts import OperationState,StageOutcome

def test_hit_stops_on_failed_alignment_and_passes_actual_state():
    from gene_agent.skills.library import hit
    class Backend:
        evidence_mode='native_physical'
        def __init__(self,fail):self.fail=fail;self.calls=[]
        def observe(self):return OperationState('start','actual',{})
        def execute(self,stage,start):
            self.calls.append((stage.name,start.state_id))
            return StageOutcome(stage.name,not self.fail,start,OperationState(stage.name,'actual',{}),details={'execution_status':'executed'})
    b=Backend(True);result=hit.run(b)
    assert not result.success and b.calls==[('align','start')]
    b=Backend(False);result=hit.run(b)
    assert result.success and b.calls==[('align','start'),('strike','align')]


def test_optional_withdraw_continues_from_actual_strike_and_stops_on_failure():
    from gene_agent.skills.library import hit
    class Backend:
        evidence_mode='native_physical'
        def __init__(self, fail=None):self.fail=fail;self.calls=[]
        def observe(self):return OperationState('start','actual',{})
        def execute(self,stage,start):
            self.calls.append((stage.name,start.state_id))
            return StageOutcome(stage.name,stage.name!=self.fail,start,
                OperationState(stage.name,'actual',{}),details={'execution_status':'executed'})
    b=Backend();result=hit.run(b,withdraw=True)
    assert result.success and b.calls==[('align','start'),('strike','align'),('withdraw','strike')]
    b=Backend('strike');result=hit.run(b,withdraw=True)
    assert not result.success and b.calls==[('align','start'),('strike','align')]
    b=Backend('withdraw');result=hit.run(b,withdraw=True)
    assert not result.success and result.failure_stage=='withdraw'
