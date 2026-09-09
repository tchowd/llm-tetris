#!/usr/bin/env python3
"""Bounded operational wrapper for the separately approved feedback GPU proof."""
from __future__ import annotations
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from infra.feedback_gpu_proof import ROOT, REG, APPROVAL, read, now, write_new, validate
from tetris.rl import atomic_write_json, file_sha256

BUCKET='llm-tetris-artifacts-566629888938-us-east-1'
RUN_ID='feedback-gpu-proof-v1'

def sync():
    import boto3
    s3=boto3.client('s3',region_name='us-east-1'); uploaded=[]
    for path in sorted(ROOT.rglob('*')):
        if not path.is_file() or any(p.startswith('.') for p in path.parts):continue
        checkpoint=next((p for p in path.parts if p.startswith('checkpoint-')),None)
        if checkpoint and (checkpoint!='checkpoint-4' or 'adapter' not in path.parts):continue
        if path.suffix not in {'.json','.jsonl','.log','.md','.safetensors','.jinja'}:continue
        key='runs/'+RUN_ID+'/'+path.relative_to(ROOT).as_posix()
        s3.upload_file(str(path),BUCKET,key,ExtraArgs={'ServerSideEncryption':'AES256'})
        uploaded.append({'path':str(path),'s3_key':key,'sha256':file_sha256(path)})
    atomic_write_json(ROOT/'sync-receipt.json',{'status':'passed','completed_at':now(),
        'objects':uploaded,'optimizer_checkpoints_uploaded':False,'encryption':'AES256'})
    s3.upload_file(str(ROOT/'sync-receipt.json'),BUCKET,'runs/'+RUN_ID+'/sync-receipt.json',
                   ExtraArgs={'ServerSideEncryption':'AES256'})
    print('synced',len(uploaded),'objects',flush=True)

def run():
    r=validate();ledger=read(ROOT/'compute-ledger.json')
    started=time.time();deadline=min(started+3600,ledger['deadline_epoch']-300)
    if deadline-started < 1800:raise ValueError('less than 30 minutes proof time remains; do not start')
    state={'status':'running','started_epoch':started,'started_at':now(),'deadline_epoch':deadline,
           'registration_sha256':file_sha256(REG),'phase':'directions','pilot_started':False}
    write_new(ROOT/'workflow.json',state)
    commands=[['directions'],['train','--arm','control'],['train','--arm','resumed','--stop-after','2'],
              ['train','--arm','resumed','--resume'],['verify']]
    failure=None
    try:
        for index,arguments in enumerate(commands):
            remaining=deadline-time.time()
            if remaining<=0:raise TimeoutError('one-hour proof limit')
            state.update(phase=' '.join(arguments),phase_started_at=now())
            atomic_write_json(ROOT/'workflow.json',state)
            with (ROOT/f'phase-{index+1}.log').open('x') as log:
                process=subprocess.Popen([sys.executable,'infra/feedback_gpu_proof.py',*arguments],
                    stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                try:code=process.wait(timeout=remaining)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGTERM)
                    try:process.wait(timeout=15)
                    except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
                    raise TimeoutError('one-hour GPU proof deadline')
            if code:raise RuntimeError(f'proof phase {arguments} exited {code}')
        state['status']='passed'
    except Exception as error:
        failure=repr(error);state.update(status='failed',failure=failure)
    finally:
        state.update(finished_at=now(),elapsed_seconds=time.time()-started)
        atomic_write_json(ROOT/'workflow.json',state)
        try:sync()
        except Exception as error:
            atomic_write_json(ROOT/'sync-failure.json',{'error':repr(error),'at':now()})
        subprocess.run(['sudo','/usr/sbin/shutdown','-h','now'],check=False)
    if failure:raise RuntimeError(failure)

def launch(subnet):
    import boto3
    from botocore.exceptions import ClientError
    validate();a=read(APPROVAL)
    if a['hard_limit_usd']!=25 or a['six_run_pilot_authorized']:raise ValueError('scope mismatch')
    if 7200/3600*2.30+5>a['hard_limit_usd']:raise ValueError('maximum lifetime exceeds spending cap')
    ec2=boto3.client('ec2',region_name='us-east-1')
    identity=boto3.client('sts').get_caller_identity()
    if identity['Account']!='566629888938':raise ValueError('wrong AWS account')
    if (ROOT/'compute-ledger.json').exists():raise ValueError('reconcile existing ledger, never relaunch')
    active=ec2.describe_instances(Filters=[{'Name':'tag:Project','Values':['llm-tetris']},
        {'Name':'tag:Stage','Values':['6']},{'Name':'instance-state-name','Values':['pending','running','stopping','stopped']}])
    if any(r['Instances'] for r in active['Reservations']):raise ValueError('existing Stage6 worker; reconcile instead')
    if subnet not in ('subnet-07af2623089657057','subnet-011c567e0b89bdea6','subnet-01df3bd9aafb4de79'):
        raise ValueError('unapproved subnet')
    # Fixed wall-clock timer is armed before setup; shutdown stops, retaining evidence.
    userdata='''#!/bin/bash
set -eu
systemd-run --unit=feedback-proof-bootstrap-deadline --on-active=110m /usr/sbin/shutdown -h now
'''
    tags=[{'Key':k,'Value':v} for k,v in {'Project':'llm-tetris','Stage':'6',
         'RunId':RUN_ID,'ManagedBy':'llm-tetris'}.items()]
    request={'ImageId':'ami-0a4870b172edcb0f2','InstanceType':'g6e.2xlarge','MinCount':1,'MaxCount':1,
        'ClientToken':RUN_ID+'-20260903-'+subnet,'KeyName':'gpu-training','SubnetId':subnet,
        'SecurityGroupIds':['sg-0a3c367cb69c4ea87'],'IamInstanceProfile':{'Name':'LLMTetrisTelemetryProfile'},
        'InstanceInitiatedShutdownBehavior':'stop','MetadataOptions':{'HttpTokens':'required',
        'HttpEndpoint':'enabled','HttpPutResponseHopLimit':1},'UserData':userdata,
        'BlockDeviceMappings':[{'DeviceName':'/dev/sda1','Ebs':{'VolumeSize':100,'VolumeType':'gp3',
            'Encrypted':True,'DeleteOnTermination':True}}],
        'TagSpecifications':[{'ResourceType':'instance','Tags':tags+[{'Key':'Name','Value':RUN_ID},
            {'Key':'PilotDollarLimit','Value':'25'}]},{'ResourceType':'volume','Tags':tags}]}
    try:ec2.run_instances(**request,DryRun=True)
    except ClientError as error:
        if error.response['Error']['Code']!='DryRunOperation':raise
    request_path=ROOT/f'launch-request-{subnet}.json'
    if request_path.exists():
        if read(request_path)!=request:raise ValueError('idempotent request differs')
    else:write_new(request_path,request)
    try:response=ec2.run_instances(**request)
    except ClientError as error:
        write_new(ROOT/f'launch-failure-{time.time_ns()}.json',{'at':now(),'subnet':subnet,
            'error_code':error.response['Error']['Code'],'message':error.response['Error']['Message']})
        raise
    instance=response['Instances'][0];start=instance['LaunchTime'].timestamp()
    write_new(ROOT/'compute-ledger.json',{'run_id':RUN_ID,'instance_id':instance['InstanceId'],
        'instance_type':'g6e.2xlarge','region':'us-east-1','status':'launched','launch_epoch':start,
        'deadline_epoch':start+7200,'hard_limit_usd':25,'hourly_allowance_usd':2.30,
        'ancillary_reserve_usd':5,'compute_sessions':[{'start_epoch':start,'end_epoch':None}],
        'root_volume_id':None,'stage4_untouched':True,'operations_complete':False,
        'registration_sha256':file_sha256(REG),'approval_sha256':file_sha256(APPROVAL)})
    print(json.dumps({'instance_id':instance['InstanceId'],'deadline_epoch':start+7200}),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['launch','run','sync'])
    parser.add_argument('--subnet',default='subnet-07af2623089657057');args=parser.parse_args()
    if args.action=='launch':launch(args.subnet)
    elif args.action=='run':run()
    else:sync()
