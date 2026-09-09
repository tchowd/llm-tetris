#!/usr/bin/env python3
"""One retained worker under the explicit continuation amendment."""
from pathlib import Path
from datetime import datetime, timezone
import fcntl
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from infra.feedback_session import SESSION, SUBNETS, client, used_gpu_vcpus, check_quota_room, ledger_cost
from scripts.stage6_feedback import validate, read, REGISTRATION
from tetris.rl import atomic_write_json, file_sha256
from botocore.exceptions import ClientError


def launch():
    with (SESSION / '.launch.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        r = validate()
        a = read(SESSION / 'approval.json')
        ledger = read(SESSION / 'compute-ledger.json')
        if '1' in ledger['workers']:
            raise ValueError('retained worker exists; inspect it, never duplicate')
        assert a['maximum_concurrent_workers'] == 1 and a['hard_limit_usd'] == 250
        deadline = datetime.fromisoformat(a['absolute_deadline_utc']).timestamp()
        assert deadline == ledger['deadline_epoch']
        if deadline - time.time() < 18 * 3600:
            raise ValueError('insufficient conservative time for all five runs')
        projected = ledger_cost(ledger, time.time())['including_ancillary_reserve_usd'] + (deadline-time.time())/3600*a['hourly_usd']
        assert projected < a['hard_limit_usd']
        assert read(SESSION/'pilot-gate.json')['approval_sha256'] == file_sha256(SESSION/'approval.json')
        bundle = read(SESSION/'source-bundle.json')
        assert file_sha256(Path(bundle['archive'])) == bundle['sha256']
        for p, digest in bundle['files_sha256'].items():
            assert file_sha256(Path(p)) == digest, p
        extra = read(SESSION/'retained-source.json')
        for p, digest in extra['files_sha256'].items():
            assert file_sha256(Path(p)) == digest, p
        assert client('sts').get_caller_identity()['Account'] == '566629888938'
        policy = read(SESSION/'execution-policy.json')
        region = policy.get('region', 'us-east-1')
        assert region in ('us-east-1', 'us-east-2', 'us-west-2')
        ec = client('ec2', region)
        quota = client('service-quotas', region).get_service_quota(ServiceCode='ec2', QuotaCode='L-DB2E81BA')['Quota']['Value']
        instance_type = policy['instance_type']
        assert instance_type in a['allowed_instance_types']
        specs = ec.describe_instance_types(InstanceTypes=[instance_type])['InstanceTypes'][0]
        assert specs['GpuInfo']['Gpus'][0]['Name'] == 'L40S'
        assert specs['GpuInfo']['Gpus'][0]['Count'] == 1
        check_quota_room(quota, used_gpu_vcpus(ec), specs['VCpuInfo']['DefaultVCpus'])
        for inspect_region in ('us-east-1', 'us-east-2', 'us-west-2'):
            found = client('ec2', inspect_region).describe_instances(Filters=[{'Name':'tag:RunId','Values':a['worker_run_ids']},
                {'Name':'instance-state-name','Values':['pending','running','stopping','stopped','shutting-down']}])
            if any(x['Instances'] for x in found['Reservations']):
                raise ValueError('existing pilot instance; reconcile before launch')
        regional = read(SESSION/f'region-{region}.json') if region != 'us-east-1' else None
        subnets = regional['subnets'] if regional else SUBNETS
        for subnet in subnets:
            request = read(Path('experiments/stage6-feedback-v1/six-worker-requests/worker-1.json'))
            request['SubnetId'] = subnet
            request['InstanceType'] = instance_type
            request['ClientToken'] = f'feedback-retained-v1-{instance_type}-{subnet}'
            if regional:
                request.update(ImageId=regional['ami'], SecurityGroupIds=[regional['security_group']], KeyName=regional['key_name'])
            calendar = datetime.fromtimestamp(deadline, timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
            request['UserData'] = f'#!/bin/bash\nset -eu\nsystemd-run --unit=feedback-pilot-bootstrap-deadline --on-calendar="{calendar}" /usr/sbin/shutdown -h now\n'
            path = SESSION / f'retained-request-{instance_type}-{subnet}.json'
            if path.exists():
                assert read(path) == request
            else:
                atomic_write_json(path, request)
            try:
                ec.run_instances(**request, DryRun=True)
            except ClientError as error:
                if error.response['Error']['Code'] != 'DryRunOperation': raise
            try:
                response = ec.run_instances(**request)
            except ClientError as error:
                code = error.response['Error']['Code']
                atomic_write_json(SESSION/f'retained-launch-failure-{time.time_ns()}.json',
                    {'at_epoch':time.time(),'code':code,'subnet':subnet,'request_sha256':file_sha256(path)})
                if code != 'InsufficientInstanceCapacity': raise
                print(subnet, code, flush=True)
                continue
            instance = response['Instances'][0]
            identity = {'slot':1,'instance_id':instance['InstanceId'],'region':region,'run_id':r['run_order'][1]['run_id'],
                        'retained_slots':list(range(1,6))}
            atomic_write_json(SESSION/'worker-1.json', identity)
            ledger['workers']['1'] = {**identity,'sessions':[{'start_epoch':instance['LaunchTime'].timestamp(),'end_epoch':None}],
                                     'state':instance['State']['Name'],'request_sha256':file_sha256(path),'root_volume_id':None}
            atomic_write_json(SESSION/'compute-ledger.json', ledger)
            print(identity, flush=True)
            return


if __name__ == '__main__':
    launch()
