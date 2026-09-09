#!/usr/bin/env python3
"""Deploy and inspect the sole retained worker using authenticated host keys."""
from pathlib import Path
import argparse
import base64
import hashlib
import shlex
import subprocess
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from infra.feedback_session import client, SESSION, read
from tetris.rl import atomic_write_json, file_sha256

REMOTE = '/home/ubuntu/llm-tetris'


def connection():
    identity = read(SESSION/'worker-1.json')
    ec = client('ec2', identity.get('region', 'us-east-1'))
    instance = ec.describe_instances(InstanceIds=[identity['instance_id']])['Reservations'][0]['Instances'][0]
    if instance['State']['Name'] != 'running':
        raise ValueError(f"worker is {instance['State']['Name']}; do not duplicate")
    assert instance['MetadataOptions']['HttpTokens'] == 'required'
    root = next(b for b in instance['BlockDeviceMappings'] if b['DeviceName'] == instance['RootDeviceName'])['Ebs']
    assert root['DeleteOnTermination']
    volume = ec.describe_volumes(VolumeIds=[root['VolumeId']])['Volumes'][0]
    assert volume['Encrypted']
    ledger = read(SESSION/'compute-ledger.json')
    ledger['workers']['1']['root_volume_id'] = root['VolumeId']
    ledger['workers']['1']['public_ip'] = instance['PublicIpAddress']
    atomic_write_json(SESSION/'compute-ledger.json', ledger)
    try:
        ec.terminate_instances(InstanceIds=[identity['instance_id']], DryRun=True)
    except Exception as error:
        if getattr(error, 'response', {}).get('Error', {}).get('Code') != 'DryRunOperation': raise
    known = SESSION/'known_hosts'
    lines = known.read_text().splitlines() if known.exists() else []
    if not any(line.startswith(identity['instance_id']+' ') for line in lines):
        console = ec.get_console_output(InstanceId=identity['instance_id'], Latest=True).get('Output', '')
        scan = subprocess.run(['ssh-keyscan','-T','10','-t','ed25519',instance['PublicIpAddress']],
                              capture_output=True, text=True, timeout=20)
        keys = [line.split() for line in scan.stdout.splitlines() if ' ssh-ed25519 ' in line and not line.startswith('#')]
        assert len(keys) == 1, 'SSH host key not available yet'
        key = keys[0]
        fingerprint = 'SHA256:'+base64.b64encode(hashlib.sha256(base64.b64decode(key[2])).digest()).decode().rstrip('=')
        assert fingerprint in console, 'host fingerprint not yet confirmed by EC2 console'
        with known.open('a') as output:
            output.write(f"{identity['instance_id']} {key[1]} {key[2]}\n")
        atomic_write_json(SESSION/'retained-host-identity.json', {
            'instance_id':identity['instance_id'],'ip':instance['PublicIpAddress'],
            'fingerprint':fingerprint,'root_volume_id':root['VolumeId'],'at_epoch':time.time()})
    options = ['-i',str(Path.home()/'.ssh/gpu-training.pem'),'-o','BatchMode=yes',
               '-o','StrictHostKeyChecking=yes','-o',f'UserKnownHostsFile={known.resolve()}',
               '-o',f"HostKeyAlias={identity['instance_id']}",'-o','ConnectTimeout=15']
    return options, 'ubuntu@'+instance['PublicIpAddress']


def main(action):
    options, host = connection()
    def ssh(command, timeout=50):
        result = subprocess.run(['ssh',*options,host,command], capture_output=True, text=True, timeout=timeout)
        print(result.stdout, end='')
        if result.returncode:
            print(result.stderr, file=sys.stderr)
            raise RuntimeError(f'SSH command failed ({result.returncode})')
        return result.stdout
    def copy(source, target):
        subprocess.run(['scp',*options,str(source),f'{host}:{target}'],check=True,timeout=60)
    session = f'{REMOTE}/{SESSION.as_posix()}'
    if action == 'setup':
        active = ssh('systemctl is-active feedback-pilot.service || true').strip()
        if active == 'active':
            raise ValueError('pilot already running; inspect only')
        ssh(f'mkdir -p {shlex.quote(session)}')
        bundle = read(SESSION/'source-bundle.json')
        assert file_sha256(Path(bundle['archive'])) == bundle['sha256']
        copy(bundle['archive'], REMOTE+'/source.tgz')
        ssh(f'cd {REMOTE} && echo {shlex.quote(bundle["sha256"]+"  source.tgz")} | sha256sum -c - && tar xzf source.tgz')
        extra = read(SESSION/'retained-source.json')
        for path, digest in extra['files_sha256'].items():
            assert file_sha256(Path(path)) == digest
            copy(path, REMOTE+'/'+path)
        for name in ['approval.json','pilot-gate.json','source-bundle.json','retained-source.json','execution-policy.json']:
            copy(SESSION/name, session+'/'+name)
        copy(SESSION/'worker-1.json', session+'/worker.json')
        check = "import json,hashlib,pathlib; manifests=['source-bundle.json','retained-source.json']; [(None if hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()==d else (_ for _ in ()).throw(ValueError(p))) for n in manifests for p,d in json.load(open('experiments/stage6-feedback-v1/session/'+n))['files_sha256'].items()]; print('All source hashes verified')"
        ssh(f'cd {REMOTE} && python3 -c {shlex.quote(check)}')
        ssh(f'sudo systemd-run --unit=feedback-retained-setup --uid=ubuntu --working-directory={REMOTE} --property=RuntimeMaxSec=1800 /bin/bash infra/feedback-pilot-setup.sh')
    elif action == 'status':
        ssh(f'systemctl show feedback-retained-setup.service feedback-pilot.service -p Id -p ActiveState -p SubState -p Result -p ExecMainStatus; sudo journalctl -u feedback-retained-setup.service -n 12 --no-pager; cd {REMOTE} && cat experiments/stage6-feedback-v1/session/worker-status.json 2>/dev/null || true')
    elif action == 'start':
        setup = ssh('systemctl show feedback-retained-setup.service -p ActiveState -p Result -p ExecMainStatus')
        assert 'ActiveState=inactive' in setup and 'Result=success' in setup and 'ExecMainStatus=0' in setup
        assert ssh('systemctl is-active feedback-pilot-monitor.timer').strip() == 'active'
        ssh('systemctl show feedback-pilot-absolute-deadline.timer -p NextElapseUSecRealtime')
        ssh(f'cd {REMOTE} && .venv-rl/bin/python -m pytest infra/test_feedback_retained.py -q')
        ssh(f'sudo systemd-run --unit=feedback-pilot --uid=ubuntu --working-directory={REMOTE} '
            '--property=RuntimeMaxSec=86400 --property=KillMode=control-group '
            f'--setenv=PYTHONPATH={REMOTE} --setenv=PYTHONUNBUFFERED=1 --setenv=OMP_NUM_THREADS=1 '
            '--setenv=TOKENIZERS_PARALLELISM=false --setenv=HF_HUB_OFFLINE=1 --setenv=TRANSFORMERS_OFFLINE=1 '
            f'{REMOTE}/.venv-rl/bin/python infra/feedback_retained.py')
        atomic_write_json(SESSION/'retained-started.json', {'at_epoch':time.time(),'worker':read(SESSION/'worker-1.json')})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['setup','status','start'])
    main(parser.parse_args().action)
