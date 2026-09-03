from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

import pytest

from infra import scale10x_eval_resume as e


def test_commands_match_frozen_evaluation_phases():
    import ast
    source = ast.parse(Path('infra/scale10x_resume.py').read_text())
    calls = [node for node in ast.walk(source) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == 'execute'
             and isinstance(node.args[0], ast.Constant) and node.args[0].value in
             {'development', 'stage5', 'fresh_recovery', 'paired_analysis'}]
    namespace = {'sys': sys, 'c': e.c, 'PILOT_ROOT': e.PILOT_ROOT,
                 'adapter': str(e.PILOT_ROOT / 'adapter')}
    frozen = [(node.args[0].value, eval(compile(ast.Expression(node.args[1]), '<frozen>', 'eval'), namespace),
               node.args[2].value) for node in calls]
    assert e.commands() == frozen
    assert all('scripts/train_episode_rl.py' not in cmd for _, cmd, _ in e.commands())


def evidence():
    m = dict(status='completed', completed_updates=320, sample_count=106727,
             resume_used=True, resumed_from_update=164, output_adapter_sha256='weights')
    cpu = dict(status='passed', update=320, samples=106727, checks={'all': True})
    backup = dict(status='passed', encryption='AES256', adapter_directory_hash_recomputed_from_s3=True,
                  adapter_directory_sha256='weights')
    return m, cpu, backup


@pytest.mark.parametrize('target,key,value', [(0,'status','failed'), (0,'completed_updates',319),
    (0,'sample_count',0), (0,'resume_used',False), (0,'resumed_from_update',0),
    (1,'status','failed'), (1,'checks',{}), (1,'checks',{'x':False}),
    (2,'encryption','none'), (2,'adapter_directory_sha256','changed')])
def test_only_verified_completed_training_accepted(target, key, value):
    rows = evidence()
    e.validate_training(*rows)
    rows[target][key] = value
    with pytest.raises(ValueError):
        e.validate_training(*rows)


def test_only_zero_game_startup_can_retry(tmp_path):
    (tmp_path/'manifest.json').write_text(json.dumps({'status':'failed','suite':'development'}))
    for name in ('games.jsonl','states.jsonl','events.jsonl'):
        (tmp_path/name).touch()
    e.validate_empty_failure(tmp_path)
    (tmp_path/'games.jsonl').write_text('{}\n')
    with pytest.raises(ValueError, match='contains results'):
        e.validate_empty_failure(tmp_path)


def test_cache_alias_uses_exact_pinned_commit_no_newline_or_overwrite(tmp_path):
    with pytest.raises(ValueError, match='snapshot missing'):
        e.prepare_cache(tmp_path)
    snapshot=tmp_path/'snapshots'/e.REVISION
    snapshot.mkdir(parents=True);(snapshot/'config.json').write_text('{}')
    e.prepare_cache(tmp_path)
    assert (tmp_path/'refs/main').read_bytes()==e.REVISION.encode()
    e.prepare_cache(tmp_path)
    (tmp_path/'refs/main').write_text('different')
    with pytest.raises(ValueError, match='refusing to overwrite'):
        e.prepare_cache(tmp_path)


def test_archive_failure_upload_before_recoverable_move(monkeypatch, tmp_path):
    root, pilot, archive = tmp_path/'root', tmp_path/'pilot', tmp_path/'archive'
    root.mkdir(); pilot.mkdir(); (pilot/'stress-development').mkdir()
    (pilot/'stress-development/games.jsonl').touch()
    for path in [pilot/'manifest.json',pilot/'block-state.json',root/'resume-block-state-v1.json',
                 root/'resume-development-v1.log',root/'resume-shutdown-receipt-v1.json',root/'compute-ledger.json']:
        path.write_text('{}')
    for name, value in [('ROOT',root),('PILOT_ROOT',pilot),('ARCHIVE',archive)]:
        monkeypatch.setattr(e,name,value)
    def upload(cmd, **kwargs):
        assert (pilot/'stress-development/games.jsonl').exists()
        assert (archive/'archive-sha256.json').exists()
        assert 'AES256' in cmd
    monkeypatch.setattr(e.subprocess,'run',upload)
    e.archive_failure()
    assert (archive/'original-stress-development/games.jsonl').exists()
    assert not (pilot/'stress-development').exists()
    assert (pilot/'manifest.json').read_text()=='{}'
    with pytest.raises(FileExistsError):
        e.archive_failure()


@pytest.mark.parametrize('fail_sync',[False,True])
def test_eval_failure_keeps_training_incomplete_research_and_stops(monkeypatch,tmp_path,fail_sync):
    root,pilot=tmp_path/'root',tmp_path/'pilot';root.mkdir();pilot.mkdir()
    monkeypatch.setattr(e,'ROOT',root);monkeypatch.setattr(e,'PILOT_ROOT',pilot)
    monkeypatch.setattr(e,'BLOCK',root/'block.json');monkeypatch.setattr(e,'archive_failure',lambda:None)
    monkeypatch.setattr(e.c,'file_sha256',lambda _: 'hash')
    commands=[]
    def run(cmd,*args):
        commands.append(cmd)
        return 1 if 'scripts/eval_stress.py' in cmd or fail_sync else 0
    monkeypatch.setattr(e,'run_process',run)
    system=[];monkeypatch.setattr(e.subprocess,'run',lambda cmd,**kw:system.append(cmd))
    with pytest.raises(SystemExit):
        e.workflow({'deadline_epoch':time.time()+10000,'instance_id':'worker'},{})
    state=json.loads((root/'block.json').read_text())
    assert state['status']=='incomplete' and state['training_complete'] and not state['research_complete']
    assert not any('scripts/eval_closed_loop.py' in cmd for cmd in commands)
    assert system[-1]==['sudo','/usr/sbin/shutdown','-h','now']
    receipt=json.loads((root/'eval-resume-shutdown-receipt-v1.json').read_text())
    assert bool(receipt['sync_errors'])==fail_sync
