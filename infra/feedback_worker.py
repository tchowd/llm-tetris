#!/usr/bin/env python3
"""Approved six-worker pilot execution; immutable scientific code is imported.

The independent systemd shutdown timer is the ultimate lifetime bound. This
worker writes only its assigned run and evaluation; it never provisions EC2.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tetris.rl import atomic_write_json, file_sha256
from scripts.stage6_feedback import ROOT, REGISTRATION, read, validate, training_command
from scripts.eval_feedback import approved_session

SESSION = ROOT / 'session'
APPROVAL = SESSION / 'approval.json'
BUCKET = 'llm-tetris-artifacts-566629888938-us-east-1'
PREFIX = 'runs/feedback-pilot-v1'


def context():
    a = approved_session(APPROVAL)
    slot = read(SESSION / 'worker.json')['slot']
    r = read(REGISTRATION)
    if a['maximum_workers'] != 6 or a['hard_limit_usd'] != 250 or not 0 <= slot < 6:
        raise ValueError('unregistered worker or budget')
    if a['maximum_aggregate_worker_hours'] * a['hourly_usd'] + a['ancillary_reserve_usd'] > a['hard_limit_usd']:
        raise ValueError('aggregate budget exceeds cap')
    return a, slot, r['run_order'][slot]


def sync(slot, *, verify=False):
    import boto3
    s3 = boto3.client('s3', region_name='us-east-1')
    receipt = SESSION / f'sync-worker-{slot}.json'
    previous = read(receipt).get('objects', {}) if receipt.exists() else {}
    run = read(REGISTRATION)['run_order'][slot]['run_id']
    roots = [SESSION, ROOT / 'evaluation' / run, Path('runs') / run / 'rl']
    if slot == 0:
        roots += [ROOT / 'gpu-proof-v1', ROOT / 'gpu-proof-v2', ROOT / 'evaluation' / 'sft']
    objects = dict(previous)
    for base in roots:
        for path in sorted(base.rglob('*')) if base.exists() else []:
            if not path.is_file() or any(p.startswith('.') for p in path.parts) or path.name.startswith('sync-worker-'):
                continue
            if path.suffix not in {'.json', '.jsonl', '.log', '.md', '.safetensors', '.pt', '.jinja'}:
                continue
            checkpoint = next((p for p in path.parents if p.name.startswith('checkpoint-')), None)
            if checkpoint and not (checkpoint / 'complete.json').exists():
                continue
            name = path.as_posix(); digest = file_sha256(path)
            key = f'{PREFIX}/worker-{slot}/{name}'
            if objects.get(name, {}).get('sha256') != digest:
                s3.upload_file(str(path), BUCKET, key, ExtraArgs={'ServerSideEncryption': 'AES256'})
                objects[name] = {'sha256': digest, 's3_key': key, 'bytes': path.stat().st_size}
    if verify:
        import hashlib
        for name, obj in objects.items():
            response = s3.get_object(Bucket=BUCKET, Key=obj['s3_key'])
            if response.get('ServerSideEncryption') != 'AES256':
                raise ValueError('backup encryption absent')
            digest = hashlib.sha256()
            for chunk in response['Body'].iter_chunks(chunk_size=1024*1024):
                digest.update(chunk)
            if digest.hexdigest() != obj['sha256']:
                raise ValueError(f'backup read-back differs: {name}')
    atomic_write_json(receipt, {'objects': objects, 'at_epoch': time.time(), 'readback_verified': verify})
    s3.upload_file(str(receipt), BUCKET, f'{PREFIX}/worker-{slot}/{receipt.as_posix()}', ExtraArgs={'ServerSideEncryption': 'AES256'})


def phase(arguments, name, timeout):
    if timeout <= 0:
        raise TimeoutError('no approved time remains')
    status = SESSION / 'worker-status.json'
    atomic_write_json(status, {'status': 'running', 'phase': name, 'phase_started_epoch': time.time()})
    log = SESSION / f'{name}-{time.time_ns()}.log'
    with log.open('x') as output:
        process = subprocess.Popen(arguments, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL); process.wait()
            raise TimeoutError(f'{name} timed out')
    if code:
        raise RuntimeError(f'{name} exited {code}; inspect {log}')


def remaining(a):
    from datetime import datetime
    return datetime.fromisoformat(a['absolute_deadline_utc']).timestamp() - time.time() - 3600


def proof_and_baseline(a):
    from infra.feedback_gpu_proof_v2 import ROOT as proof_root, REG as proof_registration
    phase([sys.executable, 'infra/diagnose_feedback_alignment.py'], 'alignment-diagnosis', min(600, remaining(a)))
    diagnosis = read(SESSION / 'alignment-diagnosis-v1/result.json')
    if (not diagnosis['policy_unchanged'] or not diagnosis['reference_unchanged'] or diagnosis['optimizer_steps'] != 0
            or any(max(o['single_same_input_error'], o['paired_same_input_error']) > diagnosis['original_tolerance']
                   for o in diagnosis['observations'])):
        raise ValueError('identical-input token alignment diagnosis failed')
    started = time.time()
    if (proof_root / 'workflow.json').exists():
        raise ValueError('proof already attempted; retain evidence, no silent retry')
    atomic_write_json(proof_root / 'compute-ledger.json', {
        'run_id': 'feedback-gpu-proof-v2', 'hard_limit_usd': 25,
        'deadline_epoch': min(started + 3600, started + remaining(a)),
        'charged_to_session_approval_sha256': file_sha256(APPROVAL),
        'worker': read(SESSION / 'worker.json'),
        'note': 'Disposable proof subcap within new $250 approval; no old allowance is spent.'})
    atomic_write_json(proof_root / 'workflow.json', {'started_epoch': started, 'status': 'running',
        'session_approval_sha256': file_sha256(APPROVAL), 'registration_sha256': file_sha256(proof_registration)})
    commands = [('directions', []), ('train', ['--arm', 'control']),
                ('train', ['--arm', 'resumed', '--stop-after', '2']),
                ('train', ['--arm', 'resumed', '--resume']), ('verify', [])]
    for index, (action, extra) in enumerate(commands):
        phase([sys.executable, 'infra/feedback_gpu_proof_v2.py', action, *extra],
              f'proof-{index}-{action}', min(started + 3600 - time.time(), remaining(a)))
    proof = read(proof_root / 'gpu-proof.json')
    if proof['status'] != 'passed' or not all(proof['checks'].values()):
        raise ValueError('new GPU proof failed')
    phase([sys.executable, 'scripts/eval_feedback.py', '--label', 'sft', '--approval-file', str(APPROVAL)],
          'sft-evaluation', min(7200, remaining(a)))
    metrics = proof['training_manifests'][0]['update_metrics']
    worst = max(m['seconds'] / m['turns'] for m in metrics)
    per_run_training = 120 + worst * (32 * 4 * 128) * 1.25
    evaluation = sum(read(p)['seconds'] for p in (ROOT / 'evaluation/sft').glob('*-*.json'))
    per_run_eval = max(1800, evaluation * 1.5)
    current_elapsed = time.time() - a['first_launch_epoch']
    projected_cost = (current_elapsed + 6*(per_run_training + per_run_eval + 900)) / 3600 * a['hourly_usd'] + a['ancillary_reserve_usd']
    if per_run_training > 14400 or per_run_training + per_run_eval + 1800 > remaining(a) or projected_cost > a['hard_limit_usd']:
        raise ValueError('measured complete pilot projection does not fit; do not launch all runs')
    gate = {'status': 'passed', 'registration_sha256': file_sha256(REGISTRATION),
        'approval_sha256': file_sha256(APPROVAL), 'proof_sha256': file_sha256(proof_root/'gpu-proof.json'),
        'sft_complete_sha256': file_sha256(ROOT/'evaluation/sft/complete.json'),
        'per_run_training_seconds': per_run_training, 'per_run_eval_seconds': per_run_eval,
        'projected_cost_usd': projected_cost, 'generated_epoch': time.time()}
    atomic_write_json(SESSION / 'pilot-gate.json', gate)
    sync(0, verify=True)


def pilot(a, run):
    r = validate()
    gate = read(SESSION / 'pilot-gate.json')
    if gate['status'] != 'passed' or gate['registration_sha256'] != file_sha256(REGISTRATION) or gate['approval_sha256'] != file_sha256(APPROVAL):
        raise ValueError('pilot has no matching GPU and runtime gate')
    cmd = training_command(r, run)
    cmd += ['--approval-file', str(APPROVAL), '--instance-hourly-usd', str(a['hourly_usd']),
        '--max-wall-clock-hours', '4', '--pilot-dollar-limit', str(4 * a['hourly_usd']), '--stage-dollar-limit', '250',
        '--prior-stage-spend-usd', '0']
    # Aggregate safety is enforced by the shared absolute deadline and six-slot
    # ledger. Trainer's own budget remains a stricter per-run four-hour bound.
    out = Path('runs') / run['run_id'] / 'rl'
    if (out / 'manifest.json').exists():
        m = read(out / 'manifest.json')
        if m['status'] != 'completed':
            latest = read(out / 'latest_checkpoint.json')
            checkpoint = Path(latest['path'])
            if not (checkpoint / 'complete.json').exists():
                raise ValueError('no fully committed resume checkpoint')
            cmd += ['--resume', str(checkpoint)]
        else:
            cmd = None
    if cmd:
        phase(cmd, 'training', min(15000, remaining(a)))
    if read(out / 'manifest.json')['status'] != 'completed':
        raise ValueError('pilot training incomplete')
    phase([sys.executable, 'scripts/eval_feedback.py', '--label', run['run_id'], '--approval-file', str(APPROVAL)],
          'candidate-evaluation', min(10800, remaining(a)))


def monitor():
    import fcntl
    SESSION.mkdir(parents=True, exist_ok=True)
    with (SESSION / '.sync.lock').open('w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        a, slot, _ = context()
        if remaining(a) <= 0:
            subprocess.run(['sudo', 'systemctl', 'stop', 'feedback-pilot.service'], check=False)
            try: sync(slot)
            finally: subprocess.run(['sudo', '/usr/sbin/shutdown', '-h', 'now'], check=False)
            return
        sync(slot)


def main(mode):
    a, slot, run = context()
    validate()
    if mode == 'monitor':
        monitor(); return
    if mode == 'sync':
        sync(slot, verify=True); return
    failed = None
    try:
        if mode == 'first':
            if slot != 0: raise ValueError('only slot zero performs proof and common baseline')
            proof_and_baseline(a)
        pilot(a, run)
        atomic_write_json(SESSION/'worker-status.json', {'status': 'completed', 'run_id': run['run_id'], 'at_epoch': time.time()})
    except Exception as error:
        failed = error
        atomic_write_json(SESSION/'worker-status.json', {'status': 'failed', 'error': repr(error), 'run_id': run['run_id'], 'at_epoch': time.time()})
    finally:
        # Serialize final read-back against the periodic uploader.
        import fcntl
        with (SESSION / '.sync.lock').open('w') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try: sync(slot, verify=True)
            except Exception as error:
                atomic_write_json(SESSION/'backup-failure.json', {'error': repr(error), 'at_epoch': time.time()})
        subprocess.run(['sudo', '/usr/sbin/shutdown', '-h', 'now'], check=False)
    if failed: raise failed


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['first', 'pilot', 'monitor', 'sync'])
    main(parser.parse_args().mode)
