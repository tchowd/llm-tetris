#!/usr/bin/env python3
"""Local six-slot launch, source packaging and cost/status ledger.

Only --action launch provisions resources, and only within applied quota after all
registered inputs validate. Calls never mutate the old experiments or Stage 4.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import time
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.stage6_feedback import ROOT, REGISTRATION, read, validate
from tetris.rl import atomic_write_json, file_sha256

SESSION = ROOT / 'session'
AUTHORITY = ROOT / 'budget-approval-v3-progressive.json'
REGION = 'us-east-1'
SUBNETS = ['subnet-07af2623089657057', 'subnet-011c567e0b89bdea6', 'subnet-01df3bd9aafb4de79', 'subnet-094caed03deaaae73']


def client(service, region=REGION):
    import boto3
    from botocore.config import Config
    return boto3.client(service, region_name=region,
        config=Config(retries={'max_attempts': 0}, connect_timeout=10, read_timeout=30))


def envelope(a):
    if a['maximum_workers'] != 6 or a['hard_limit_usd'] != 250:
        raise ValueError('approved six-worker/$250 scope required')
    cost = a['maximum_workers'] * a['session_hours'] * a['hourly_usd'] + a['ancillary_reserve_usd']
    if cost > a['hard_limit_usd'] or a['session_hours'] > 16:
        raise ValueError('maximum lifetime cost exceeds approval')
    return cost


def ledger_cost(ledger, current):
    hours = sum(max(0, (entry.get('end_epoch') or current)-entry['start_epoch'])/3600
        for worker in ledger['workers'].values() for entry in worker['sessions'])
    cost = sum(max(0, (entry.get('end_epoch') or current)-entry['start_epoch'])/3600
        * worker.get('hourly_usd', ledger['hourly_usd'])
        for worker in ledger['workers'].values() for entry in worker['sessions'])
    return {'worker_hours': hours, 'compute_allowance_usd': cost,
            'including_ancillary_reserve_usd': cost+ledger['ancillary_reserve_usd']}


def check_quota_room(quota, used_vcpus, requested_vcpus=8):
    if quota < used_vcpus + requested_vcpus:
        raise ValueError(f'GPU quota has no room: {used_vcpus} used + {requested_vcpus} requested > {quota}')


def used_gpu_vcpus(ec):
    response = ec.describe_instances(Filters=[{'Name':'instance-state-name','Values':['pending','running','stopping']}])
    instances = [i for r in response['Reservations'] for i in r['Instances']
                 if i['InstanceType'].startswith(('g', 'vt'))]
    kinds = sorted({i['InstanceType'] for i in instances})
    if not kinds:
        return 0
    specs = ec.describe_instance_types(InstanceTypes=kinds)['InstanceTypes']
    counts = {s['InstanceType']:s['VCpuInfo']['DefaultVCpus'] for s in specs}
    return sum(counts[i['InstanceType']] for i in instances)


def status():
    ec = client('ec2')
    a = read(AUTHORITY)
    quota = client('service-quotas').get_service_quota(ServiceCode='ec2', QuotaCode='L-DB2E81BA')['Quota']['Value']
    histories = client('service-quotas').list_requested_service_quota_change_history_by_quota(
        ServiceCode='ec2', QuotaCode='L-DB2E81BA', MaxResults=100)['RequestedQuotas']
    regions = {REGION}
    if (SESSION/'compute-ledger.json').exists():
        regions |= {w.get('region', REGION) for w in read(SESSION/'compute-ledger.json')['workers'].values()}
    instances = []
    for region in sorted(regions):
        response = client('ec2', region).describe_instances(Filters=[{'Name':'tag:RunId','Values':a['worker_run_ids']}])
        instances.extend({**i, 'PilotRegion':region} for r in response['Reservations'] for i in r['Instances'])
    result = {'at_epoch':time.time(), 'quota_vcpus':quota,
        'quota_requests':[{'id':h['Id'],'desired':h['DesiredValue'],'status':h['Status']} for h in histories if h['DesiredValue']>=48],
        'workers':[{'id':i['InstanceId'],'region':i['PilotRegion'],'state':i['State']['Name'],'ip':i.get('PublicIpAddress'),
                    'run_id':next(t['Value'] for t in i['Tags'] if t['Key']=='RunId')} for i in instances]}
    lp = SESSION / 'compute-ledger.json'
    if lp.exists():
        ledger = read(lp)
        for worker in ledger['workers'].values():
            found = next((i for i in instances if i['InstanceId']==worker['instance_id']), None)
            if found:
                worker['state'] = found['State']['Name']; worker['public_ip'] = found.get('PublicIpAddress')
                if found['State']['Name'] in ('stopped','terminated') and worker['sessions'][-1]['end_epoch'] is None:
                    # Conservative upper bound: charge through confirmed observation.
                    worker['sessions'][-1]['end_epoch'] = time.time()
                    worker['sessions'][-1]['end_basis'] = 'confirmed stopped observation; upper bound'
        ledger['cost'] = ledger_cost(ledger, time.time())
        ledger['last_observed_epoch'] = time.time()
        atomic_write_json(lp, ledger); result['ledger'] = ledger
    atomic_write_json(ROOT/'live-status.json', result)
    print(json.dumps(result, indent=2))
    return result


def source_bundle():
    validate()
    proof = read(ROOT/'gpu-proof-v1/registration.json')
    paths = set(proof['source_and_input_sha256']) | {str(ROOT/'gpu-proof-v1/registration.json'), str(AUTHORITY)}
    repaired = read(ROOT/'gpu-proof-v2/registration.json')
    paths |= set(repaired['source_and_input_sha256']) | {str(ROOT/'gpu-proof-v2/registration.json'),
        'infra/diagnose_feedback_alignment.py', str(SESSION/'alignment-diagnosis-v1/registration.json')}
    paths |= {str(p) for p in Path('runs/sft-v1/adapter').rglob('*') if p.is_file()}
    paths |= {str(p) for p in (ROOT/'prechange').glob('*.py')}
    paths |= {'infra/feedback_session.py','infra/feedback_worker.py','infra/feedback-pilot-setup.sh',
              'infra/test_feedback_session.py','experiments/stage6-feedback-v1/six-worker-launch.md'}
    files = sorted(Path(p) for p in paths)
    forbidden = {'benchmarks/stress-v1/states.jsonl', 'data/stage6-recovery-v1/validation-starts.jsonl'}
    if paths & forbidden:
        raise ValueError('held-out historical state files must not ship')
    temp = Path(tempfile.mkdtemp(prefix='llm-tetris-feedback-six-'))
    archive = temp/'source.tgz'
    with tarfile.open(archive, 'w:gz') as tar:
        for path in files:
            if not path.is_file() or path.is_symlink(): raise ValueError(f'invalid bundle input: {path}')
            tar.add(path, arcname=str(path), recursive=False)
    evidence = {'archive':str(archive),'sha256':file_sha256(archive),
                'files_sha256':{str(p):file_sha256(p) for p in files},'created_epoch':time.time()}
    SESSION.mkdir(parents=True,exist_ok=True)
    atomic_write_json(SESSION/'source-bundle.json', evidence)
    print(json.dumps({'archive':str(archive),'sha256':evidence['sha256'],'files':len(files)}))


def launch(slot, subnet):
    import fcntl
    from botocore.exceptions import ClientError
    if slot not in range(6) or subnet not in SUBNETS: raise ValueError('invalid slot/subnet')
    SESSION.mkdir(parents=True,exist_ok=True)
    with (SESSION/'.launch.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        r = validate(); a = read(AUTHORITY); envelope(a)
        bundle = read(SESSION/'source-bundle.json')
        for p, digest in bundle['files_sha256'].items():
            if file_sha256(Path(p)) != digest: raise ValueError(f'bundle source changed: {p}')
        if file_sha256(Path(bundle['archive'])) != bundle['sha256']: raise ValueError('archive changed')
        if client('sts').get_caller_identity()['Account']!='566629888938': raise ValueError('wrong AWS account')
        quota = client('service-quotas').get_service_quota(ServiceCode='ec2',QuotaCode='L-DB2E81BA')['Quota']['Value']
        ec = client('ec2')
        check_quota_room(quota, used_gpu_vcpus(ec))
        lp=SESSION/'compute-ledger.json'; ledger=read(lp) if lp.exists() else None
        if slot>0:
            gate=read(SESSION/'pilot-gate.json')
            if gate['status']!='passed' or gate['approval_sha256']!=file_sha256(SESSION/'approval.json'):
                raise ValueError('first-worker GPU/runtime gate required before other five launches')
        if ledger and str(slot) in ledger['workers']: raise ValueError('slot already launched; do not duplicate')
        active=ec.describe_instances(Filters=[{'Name':'tag:RunId','Values':a['worker_run_ids']},
            {'Name':'instance-state-name','Values':['pending','running','stopping','stopped']}])
        existing=[i for x in active['Reservations'] for i in x['Instances']]
        policy = SESSION/'execution-policy.json'
        if policy.exists():
            limit = read(policy)['maximum_concurrent_workers']
            if sum(i['State']['Name'] in ('pending','running','stopping') for i in existing) >= limit:
                raise ValueError('user concurrency limit reached')
        run=r['run_order'][slot]
        if any(next(t['Value'] for t in i['Tags'] if t['Key']=='RunId')==run['run_id'] for i in existing):
            raise ValueError('slot instance exists without ledger; reconcile its client token before continuing')
        if len(existing)>=6: raise ValueError('six-worker limit')
        deadline=ledger['deadline_epoch'] if ledger else time.time()+16*3600
        if deadline-time.time()<4*3600: raise ValueError('insufficient global time remains')
        request=read(ROOT/'six-worker-requests'/f'worker-{slot}.json')
        request['SubnetId']=subnet
        request['ClientToken']=f'feedback-six-v1-slot{slot}-{subnet}'
        calendar=datetime.fromtimestamp(deadline,timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
        request['UserData']=(f'#!/bin/bash\nset -eu\nsystemd-run --unit=feedback-pilot-bootstrap-deadline --on-calendar="{calendar}" /usr/sbin/shutdown -h now\n'
            if ledger else '#!/bin/bash\nset -eu\nsystemd-run --unit=feedback-pilot-bootstrap-deadline --on-active=16h /usr/sbin/shutdown -h now\n')
        rp=SESSION/f'launch-request-{slot}-{subnet}.json'
        if rp.exists():
            saved=read(rp)
            if saved != request:
                raise ValueError('idempotent request differs')
            request=saved
        else: atomic_write_json(rp,request)
        try: ec.run_instances(**request,DryRun=True)
        except ClientError as e:
            if e.response['Error']['Code']!='DryRunOperation':raise
        try: response=ec.run_instances(**request)
        except ClientError as e:
            atomic_write_json(SESSION/f'launch-failure-{slot}-{time.time_ns()}.json',
                {'code':e.response['Error']['Code'],'message':e.response['Error']['Message'].split('Encoded authorization')[0],
                 'request_sha256':file_sha256(rp),'at_epoch':time.time()})
            raise
        instance=response['Instances'][0];start=instance['LaunchTime'].timestamp()
        if ledger is None:
            # Capacity failures start no session. Pin the successful launch time;
            # setup replaces the boot fallback with this absolute deadline.
            deadline=start+16*3600
            session_a={**a,'absolute_deadline_utc':datetime.fromtimestamp(deadline,timezone.utc).isoformat(),
                'first_launch_epoch':start,'authority_sha256':file_sha256(AUTHORITY)}
            atomic_write_json(SESSION/'approval.json',session_a)
            ledger={'first_launch_epoch':start,'deadline_epoch':deadline,'hourly_usd':a['hourly_usd'],
                'ancillary_reserve_usd':a['ancillary_reserve_usd'],'hard_limit_usd':250,'workers':{},'operations_complete':False}
        ledger['workers'][str(slot)]={'instance_id':instance['InstanceId'],'run_id':run['run_id'],
            'sessions':[{'start_epoch':start,'end_epoch':None}],'state':instance['State']['Name'],
            'request_sha256':file_sha256(rp),'root_volume_id':None}
        atomic_write_json(lp,ledger)
        atomic_write_json(SESSION/f'worker-{slot}.json',{'slot':slot,'instance_id':instance['InstanceId'],'run_id':run['run_id']})
        print(json.dumps({'slot':slot,'instance_id':instance['InstanceId'],'deadline_epoch':deadline}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['status','bundle','launch']);p.add_argument('--slot',type=int,default=0)
    p.add_argument('--subnet',choices=SUBNETS,default=SUBNETS[0]);args=p.parse_args()
    if args.action=='status':status()
    elif args.action=='bundle':source_bundle()
    else:launch(args.slot,args.subnet)
