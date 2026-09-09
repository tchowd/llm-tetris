from pathlib import Path
from types import SimpleNamespace
import pytest
from infra.feedback_session import envelope, ledger_cost, check_quota_room


def approval():
    return {'maximum_workers':6,'hard_limit_usd':250,'session_hours':16,'hourly_usd':2.3,'ancillary_reserve_usd':15}


def test_global_envelope_includes_all_workers_and_ancillary_cost():
    assert envelope(approval()) == pytest.approx(235.8)
    with pytest.raises(ValueError):envelope({**approval(),'session_hours':18})
    with pytest.raises(ValueError):envelope({**approval(),'maximum_workers':7})


def test_progressive_launch_respects_applied_quota():
    check_quota_room(8, 0)
    check_quota_room(48, 40)
    with pytest.raises(ValueError, match='no room'):check_quota_room(8, 8)
    with pytest.raises(ValueError, match='no room'):check_quota_room(48, 48)
    with pytest.raises(ValueError, match='no room'):check_quota_room(4, 0)


def test_ledger_charges_running_stopped_and_restart_sessions():
    ledger={'hourly_usd':2.3,'ancillary_reserve_usd':15,'workers':{
        '0':{'sessions':[{'start_epoch':0,'end_epoch':3600},{'start_epoch':7200,'end_epoch':None}]},
        '1':{'sessions':[{'start_epoch':3600,'end_epoch':None}]}}}
    actual=ledger_cost(ledger,10800)
    assert actual['worker_hours']==4
    assert actual['including_ancillary_reserve_usd']==pytest.approx(24.2)


def test_worker_rejects_unapproved_budget_before_work(monkeypatch):
    import infra.feedback_worker as worker
    monkeypatch.setattr(worker,'approved_session',lambda _: {**approval(),'hard_limit_usd':251})
    monkeypatch.setattr(worker,'read',lambda p: {'slot':0} if Path(p).name=='worker.json' else {'run_order':[{}]*6})
    with pytest.raises(ValueError,match='budget'):worker.context()


def test_failed_subprocess_remains_failed(tmp_path,monkeypatch):
    import infra.feedback_worker as worker
    import sys
    monkeypatch.setattr(worker,'SESSION',tmp_path)
    with pytest.raises(RuntimeError,match='exited 3'):
        worker.phase([sys.executable,'-c','raise SystemExit(3)'],'failure-proof',5)
    assert list(tmp_path.glob('failure-proof-*.log'))


def test_timeout_terminates_child(tmp_path,monkeypatch):
    import infra.feedback_worker as worker
    import sys
    monkeypatch.setattr(worker,'SESSION',tmp_path)
    with pytest.raises(TimeoutError,match='timed out'):
        worker.phase([sys.executable,'-c','import time; time.sleep(10)'],'timeout-proof',.1)
