import pytest
from infra.feedback_retained import sequence


def test_cost_preserves_prior_worker_rate_when_host_changes():
    from infra.feedback_session import ledger_cost
    ledger = {'hourly_usd':4.6,'ancillary_reserve_usd':15,'workers':{
        '0':{'hourly_usd':2.3,'sessions':[{'start_epoch':0,'end_epoch':3600}]},
        '1':{'sessions':[{'start_epoch':3600,'end_epoch':None}]}}}
    cost = ledger_cost(ledger, 7200)
    assert cost['worker_hours'] == 2
    assert cost['compute_allowance_usd'] == pytest.approx(6.9)
    assert cost['including_ancillary_reserve_usd'] == pytest.approx(21.9)


def test_sequential_runs_are_audited_and_backed_up_before_advancing():
    events = []
    sequence(lambda s: events.append(('execute', s)),
             lambda s: events.append(('audit', s)) or s,
             lambda s: events.append(('backup', s)),
             lambda s: events.append(('assign', s)),
             lambda s, e: events.append(('record', s)))
    assert events == [(action, slot) for slot in range(1, 6)
                      for action in ('assign', 'execute', 'audit', 'record', 'backup')]


@pytest.mark.parametrize('failure', ['execute', 'audit', 'backup'])
def test_failure_prevents_advancing_to_next_run(failure):
    assigned = []
    def operation(name):
        def call(slot):
            if name == failure:
                raise RuntimeError(name)
            return slot
        return call
    with pytest.raises(RuntimeError, match=failure):
        sequence(operation('execute'), operation('audit'), operation('backup'),
                 assigned.append, lambda s, e: None)
    assert assigned == [1]
