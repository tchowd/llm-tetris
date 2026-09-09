#!/usr/bin/env python3
"""Run registered slots 1..5 on one retained instance; science stays frozen."""
from pathlib import Path
import fcntl
import subprocess
import shutil
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from infra import feedback_worker as worker
from scripts.analyze_feedback import training_evidence, evaluation_rows, analyze
from scripts.stage6_feedback import read, validate
from tetris.rl import atomic_write_json


def sequence(execute, audit, backup, assign, record, slots=range(1, 6)):
    for slot in slots:
        assign(slot)
        execute(slot)
        evidence = audit(slot)
        record(slot, evidence)
        backup(slot)  # A failed verified backup must prevent the next run.


def main():
    a, _, _ = worker.context()
    r = validate()
    identity = read(worker.SESSION / 'worker.json')
    lock_path = worker.SESSION / '.sync.lock'
    def assign(slot):
        with lock_path.open('w') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            atomic_write_json(worker.SESSION / 'worker.json', {
                **identity, 'slot': slot, 'run_id': r['run_order'][slot]['run_id'],
                'retained_instance': True})
    def audit(slot):
        run = r['run_order'][slot]
        m, _ = training_evidence(r, run)
        rows = evaluation_rows(r, run['run_id'], m['output_adapter_sha256'])
        return {'status': 'completed', 'run_id': run['run_id'],
                'training_updates': m['completed_updates'],
                'evaluation_cases': {k: len(v) for k, v in rows.items()},
                'at_epoch': time.time()}
    def backup(slot):
        with lock_path.open('w') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            worker.sync(slot, verify=True)
    try:
        sequence(lambda slot: worker.pilot(a, r['run_order'][slot]), audit, backup,
                 assign, lambda slot, evidence: atomic_write_json(
                     worker.SESSION / f'retained-completion-{slot}.json', evidence))
        analyze()
        for name in ('report.json', 'report.md'):
            shutil.copy2(worker.ROOT / name, worker.SESSION / ('completed-pilot-' + name))
        atomic_write_json(worker.SESSION / 'retained-status.json',
                          {'status': 'completed', 'slots': list(range(1, 6)), 'at_epoch': time.time()})
        backup(5)
    except Exception as error:
        atomic_write_json(worker.SESSION / 'retained-status.json',
                          {'status': 'failed', 'error': repr(error), 'at_epoch': time.time()})
        try:
            backup(read(worker.SESSION / 'worker.json')['slot'])
        except Exception as backup_error:
            atomic_write_json(worker.SESSION / 'retained-backup-failure.json',
                              {'error': repr(backup_error), 'at_epoch': time.time()})
        raise
    finally:
        subprocess.run(['sudo', '/usr/sbin/shutdown', '-h', 'now'], check=False)


if __name__ == '__main__':
    main()
