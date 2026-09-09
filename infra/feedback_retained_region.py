#!/usr/bin/env python3
"""Prepare a free regional SSH key/security group only after GPU quota applies."""
from pathlib import Path
import argparse
import json
import subprocess
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from infra.feedback_session import SESSION, client, read
from tetris.rl import atomic_write_json


def prepare(region):
    assert region in ('us-east-2', 'us-west-2')
    quota = client('service-quotas', region).get_service_quota(ServiceCode='ec2', QuotaCode='L-DB2E81BA')['Quota']['Value']
    if quota < 8:
        raise ValueError('regional GPU quota not yet applied; no resources created')
    ec = client('ec2', region)
    approved = read(SESSION/'approval.json')
    assert client('sts').get_caller_identity()['Account'] == '566629888938'
    location = {'us-east-2':'US East (Ohio)', 'us-west-2':'US West (Oregon)'}[region]
    filters = {'instanceType':'g6e.2xlarge','location':location,'operatingSystem':'Linux',
               'tenancy':'Shared','preInstalledSw':'NA','capacitystatus':'Used'}
    prices = client('pricing').get_products(ServiceCode='AmazonEC2', Filters=[
        {'Type':'TERM_MATCH','Field':k,'Value':v} for k,v in filters.items()], MaxResults=100)
    products = [json.loads(p) for p in prices['PriceList']]
    rates = [float(d['pricePerUnit']['USD']) for p in products for t in p['terms'].get('OnDemand',{}).values()
             for d in t['priceDimensions'].values()]
    assert rates and max(rates) + .05 < approved['hourly_usd']
    original = read(SESSION/'original-ami-identity.json')
    images = ec.describe_images(Owners=[original['OwnerId']], Filters=[
        {'Name':'name','Values':[original['Name']]}, {'Name':'state','Values':['available']}])['Images']
    assert len(images) == 1
    zones = {o['Location'] for o in ec.describe_instance_type_offerings(LocationType='availability-zone',
        Filters=[{'Name':'instance-type','Values':['g6e.2xlarge']}])['InstanceTypeOfferings']}
    subnets = [s for s in ec.describe_subnets(Filters=[{'Name':'default-for-az','Values':['true']}])['Subnets']
               if s['AvailabilityZone'] in zones]
    assert subnets and len({s['VpcId'] for s in subnets}) == 1
    key_name = 'feedback-retained-v1'
    public_key = subprocess.run(['ssh-keygen','-y','-f',str(Path.home()/'.ssh/gpu-training.pem')],
                                check=True,capture_output=True,text=True).stdout.strip()
    keys = ec.describe_key_pairs(Filters=[{'Name':'key-name','Values':[key_name]}], IncludePublicKey=True)['KeyPairs']
    if keys:
        assert keys[0]['PublicKey'].split()[:2] == public_key.split()[:2]
    else:
        ec.import_key_pair(KeyName=key_name, PublicKeyMaterial=public_key.encode(),
                           TagSpecifications=[{'ResourceType':'key-pair','Tags':[{'Key':'Project','Value':'llm-tetris'},
                               {'Key':'Purpose','Value':'feedback-retained-v1'}]}])
    groups = ec.describe_security_groups(Filters=[{'Name':'group-name','Values':['feedback-retained-v1']},
        {'Name':'vpc-id','Values':[subnets[0]['VpcId']]}])['SecurityGroups']
    if groups:
        assert any(t['Key']=='Purpose' and t['Value']=='feedback-retained-v1' for t in groups[0].get('Tags',[]))
        group_id = groups[0]['GroupId']
    else:
        group_id = ec.create_security_group(GroupName='feedback-retained-v1', Description='Retained feedback pilot SSH and HTTPS',
            VpcId=subnets[0]['VpcId'], TagSpecifications=[{'ResourceType':'security-group','Tags':[
                {'Key':'Project','Value':'llm-tetris'},{'Key':'Purpose','Value':'feedback-retained-v1'}]}])['GroupId']
        # Persist owned resource IDs immediately so interrupted preparation can be cleaned up.
        atomic_write_json(SESSION/f'region-{region}-resources.json',
                          {'region':region,'security_group':group_id,'key_name':key_name,'at_epoch':time.time()})
    source = client('ec2').describe_security_groups(GroupIds=['sg-0a3c367cb69c4ea87'])['SecurityGroups'][0]
    wanted = [{'IpProtocol':'tcp','FromPort':22,'ToPort':22,'IpRanges':p['IpRanges']}
              for p in source['IpPermissions'] if p.get('FromPort')==22 and p.get('ToPort')==22]
    assert wanted and all(x['CidrIp'].endswith('/32') for p in wanted for x in p['IpRanges'])
    existing = ec.describe_security_groups(GroupIds=[group_id])['SecurityGroups'][0]
    if not existing['IpPermissions']:
        ec.authorize_security_group_ingress(GroupId=group_id,IpPermissions=wanted)
    egress = existing['IpPermissionsEgress']
    if any(p['IpProtocol']=='-1' for p in egress):
        ec.revoke_security_group_egress(GroupId=group_id,IpPermissions=egress)
        egress = []
    if not egress:
        ec.authorize_security_group_egress(GroupId=group_id,IpPermissions=[{
            'IpProtocol':'tcp','FromPort':443,'ToPort':443,'IpRanges':[{'CidrIp':'0.0.0.0/0'}]}])
    config = {'region':region,'ami':images[0]['ImageId'],'ami_name':images[0]['Name'],
              'subnets':[s['SubnetId'] for s in subnets],'security_group':group_id,'key_name':key_name,
              'quoted_hourly_usd':max(rates),'at_epoch':time.time()}
    atomic_write_json(SESSION/f'region-{region}.json',config)
    policy = read(SESSION/'execution-policy.json')
    policy.update(region=region,instance_type='g6e.2xlarge')
    atomic_write_json(SESSION/'execution-policy.json',policy)
    print(config)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('region',choices=['us-east-2','us-west-2'])
    prepare(parser.parse_args().region)
