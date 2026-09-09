#!/usr/bin/env python3
"""Disposable GPU correctness proof, never a six-run pilot or model promotion.

Uses the registered trainer's actual rollout, loss and checkpoint functions.
All additions live outside its frozen source set. GPU work needs its own ledger.
"""
from __future__ import annotations
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tetris.rl import (atomic_write_json, directory_sha256, file_sha256,
    trajectory_advantages, discounted_reward_to_go, restore_game, EpisodeRewardWeights)
from tetris.recovery import load_start_bank, placement_failure
from scripts.stage6_feedback import schedule

ROOT = Path('experiments/stage6-feedback-v1/gpu-proof-v1')
REG = ROOT / 'registration.json'
APPROVAL = Path('experiments/stage6-feedback-v1/gpu-proof-approval-v1.json')
PILOT = Path('experiments/stage6-feedback-v1/registration.json')

def read(p):
    return json.loads(Path(p).read_text())

def now():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())

def write_new(p, value):
    if Path(p).exists():
        raise ValueError(f'refusing overwrite: {p}')
    atomic_write_json(Path(p), value)

def coefficients(case, method):
    if case == 'lone_illegal':
        return trajectory_advantages([[-10], [0, 0, -10]], method=method)[1][-1]
    if case == 'equal_illegal':
        return trajectory_advantages([[-10]] * 4, method=method)[0][-1]
    if case == 'delayed_positive':
        return trajectory_advantages([[0, 0, 10], [0, 0, -10]], method=method)[0][0]
    raise ValueError(case)

def prepare():
    from scripts.stage6_feedback import validate
    p = validate()
    a = read(APPROVAL)
    if a['hard_limit_usd'] != 25 or a['six_run_pilot_authorized']:
        raise ValueError('proof-only approval required')
    seeds = read(p['training_seed_file'])
    bank = load_start_bank(Path(p['recovery_start_file']), seeds)
    probes = []
    for state in bank:
        game = restore_game(state['seed'], state['action_prefix'], expected=state)
        blocked = [(rot, x) for rot in range(4) for x in range(10)
                   if placement_failure(game, (rot, x)) == 'blocked_at_top']
        if blocked:
            legal = game.legal_placements()[0]
            probes.append({'state': state, 'illegal_action': list(blocked[0]),
                           'legal_action': [legal['rot'], legal['x']], 'failure': 'blocked_at_top'})
        if len(probes) == 2:
            break
    if len(probes) != 2:
        raise ValueError('two training-only blocked-top probes required')
    paths = {**p['source_and_input_sha256'], str(PILOT): file_sha256(PILOT),
             str(APPROVAL): file_sha256(APPROVAL)}
    for name in ('infra/feedback_gpu_proof.py', 'infra/test_feedback_gpu_proof.py',
                 'infra/feedback_gpu_ops.py', 'infra/feedback-gpu-setup.sh',
                 'infra/feedback-gpu-needrestart.conf', 'tests/test_episode_proof.py',
                 'tests/test_episode_runtime.py',
                 'runs/sft-v1/closed_loop/manifest.json', 'runs/sft-v1/closed_loop/metrics.json'):
        paths[name] = file_sha256(Path(name))
    result = {'kind': 'feedback_gpu_correctness_only', 'registered_at': now(),
        'base_model': p['base_model'], 'base_model_revision': p['base_model_revision'],
        'initial_adapter': p['initial_adapter'], 'initial_adapter_sha256': p['initial_adapter_sha256'],
        'training_seed_file': p['training_seed_file'], 'recovery_start_file': p['recovery_start_file'],
        'source_and_input_sha256': paths, 'probes': probes,
        'training_seed': 6301, 'schedule': schedule(6301, seeds, bank, updates=4),
        'recipe': {**p['recipe'], 'updates': 4, 'save_every': 1, 'advantage_method': 'fixed_zero'},
        'optimizer': 'AdamW defaults; learning_rate=1e-6; clip_norm=1; cosine horizon 4, warmup 1',
        'directions': {'methods': ['active_group', 'fixed_zero'],
            'cases': ['lone_illegal', 'equal_illegal', 'delayed_positive'],
            'optimizer': 'fresh AdamW defaults at 1e-6, clip_norm=1; no scheduler for isolated single-step probe',
            'positive_reward_is_synthetic': True, 'zero_control_gradient_tolerance': 1e-6,
            'minimum_signed_logprob_change': 0},
        'token_absolute_tolerance': 0.0001, 'minimum_gpu_headroom': 0.15,
        'maximum_proof_seconds': 3600, 'final_test_access': False,
        'pilot_authorized': False, 'deployment_authorized': False,
        'note': 'Four-update disposable control and separate 2+2 process resume. No proof weights initialize any pilot.'}
    write_new(REG, result)
    print('proof registered', file_sha256(REG), flush=True)

def validate(require_session=False):
    r = read(REG)
    for name, digest in r['source_and_input_sha256'].items():
        if file_sha256(Path(name)) != digest:
            raise ValueError(f'proof registered input changed: {name}')
    if directory_sha256(Path(r['initial_adapter'])) != r['initial_adapter_sha256']:
        raise ValueError('original SFT changed')
    if require_session:
        ledger = read(ROOT / 'compute-ledger.json')
        if ledger['run_id'] != 'feedback-gpu-proof-v1' or ledger['hard_limit_usd'] != 25:
            raise ValueError('wrong proof ledger')
        if time.time() >= ledger['deadline_epoch']:
            raise ValueError('absolute worker deadline expired')
        state = read(ROOT / 'workflow.json')
        if time.time() - state['started_epoch'] >= r['maximum_proof_seconds']:
            raise ValueError('proof time exhausted')
    return r

def load_models(r):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from scripts.train_episode_rl import configure_execution, adapter_parameter_hash
    configure_execution()
    if not torch.cuda.is_available() or 'L40S' not in torch.cuda.get_device_name():
        raise ValueError('real L40S CUDA worker required')
    torch.manual_seed(r['training_seed']); torch.cuda.manual_seed_all(r['training_seed'])
    torch.cuda.reset_peak_memory_stats()
    tokenizer = AutoTokenizer.from_pretrained(r['base_model'], revision=r['base_model_revision'])
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    models = []
    for trainable in (True, False):
        base = AutoModelForCausalLM.from_pretrained(r['base_model'], revision=r['base_model_revision'],
                                                   dtype=torch.bfloat16).to('cuda')
        if base.config._commit_hash != r['base_model_revision']:
            raise ValueError('base revision mismatch')
        model = PeftModel.from_pretrained(base, r['initial_adapter'], is_trainable=trainable)
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout): module.p = 0.0
        if trainable:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        else:
            for param in model.parameters(): param.requires_grad_(False)
        model.eval(); models.append(model)
    if adapter_parameter_hash(models[0]) != adapter_parameter_hash(models[1]):
        raise ValueError('initial policy/reference differ')
    return tokenizer, *models

def gpu_memory():
    import torch
    total = torch.cuda.get_device_properties(0).total_memory
    peak = torch.cuda.max_memory_allocated()
    return {'name': torch.cuda.get_device_name(), 'total_bytes': total,
            'peak_allocated_bytes': peak, 'headroom_fraction': 1 - peak / total}

def directions():
    import torch
    from tetris.chat import build_generation_prompt
    from scripts.train_episode_rl import sequence_logprobs, action_loss_chunk, adapter_parameter_hash
    from scripts.check_episode_proof import independent_token_logprobs
    r = validate(True)
    tokenizer, policy, reference = load_models(r)
    parameters = [p for p in policy.parameters() if p.requires_grad]
    saved = [p.detach().clone() for p in parameters]
    initial = adapter_parameter_hash(policy); frozen = adapter_parameter_hash(reference)
    results = []
    for probe_index, probe in enumerate(r['probes']):
        prompt = build_generation_prompt(tokenizer, probe['state']['prompt'])
        rows = []
        for action in (probe['illegal_action'], probe['legal_action']):
            text = f'Action: rot={action[0]} x={action[1]}'
            rows.append({'prompt_ids': tokenizer(prompt, add_special_tokens=False)['input_ids'],
                'completion_ids': tokenizer(text, add_special_tokens=False)['input_ids'] + [tokenizer.eos_token_id],
                'raw_completion': text})
        for method in r['directions']['methods']:
            for case in r['directions']['cases']:
                with torch.no_grad():
                    for p, old in zip(parameters, saved): p.copy_(old)
                policy.train(); policy.zero_grad(set_to_none=True)
                row_index = int(case == 'delayed_positive'); row = dict(rows[row_index])
                with torch.no_grad():
                    before = independent_token_logprobs(policy, rows, tokenizer.pad_token_id, 1)
                    ref_tokens = independent_token_logprobs(reference, [row], tokenizer.pad_token_id, 1)[0]
                row['reference_token_logprobs'] = ref_tokens.cpu().tolist()
                row['advantage'] = coefficients(case, method)
                optimizer = torch.optim.AdamW(parameters, lr=1e-6)
                current = sequence_logprobs(policy, [row], tokenizer.pad_token_id, 1, return_tokens=True)
                alignment_error = float((current[0].detach() - before[row_index]).abs().max())
                loss, kl = action_loss_chunk(current, [row], .05, 1)
                loss.backward()
                norm = float(torch.nn.utils.clip_grad_norm_(parameters, 1, error_if_nonfinite=True))
                optimizer.step()
                with torch.no_grad():
                    after = independent_token_logprobs(policy, rows, tokenizer.pad_token_id, 1)
                deltas = [float(a.mean() - b.mean()) for a, b in zip(after, before)]
                advantage = row['advantage']
                passed = (norm <= 1e-6 if advantage == 0 else
                          math.isfinite(norm) and norm > 0 and deltas[row_index] * advantage > 0)
                result = {'probe_index': probe_index, 'method': method, 'case': case,
                    'advantage': advantage, 'gradient_norm_before_clip': norm, 'clipped': norm > 1,
                    'alignment_max_error': alignment_error, 'mean_logprob_before': [float(v.mean()) for v in before],
                    'mean_logprob_after': [float(v.mean()) for v in after],
                    'illegal_logprob_delta': deltas[0], 'legal_alternative_logprob_delta': deltas[1],
                    'passed': passed and alignment_error <= r['token_absolute_tolerance']}
                results.append(result); print(json.dumps(result), flush=True)
                del optimizer, current, loss, kl, before, after
    with torch.no_grad():
        for p, old in zip(parameters, saved): p.copy_(old)
    restored = adapter_parameter_hash(policy) == initial
    reference_unchanged = adapter_parameter_hash(reference) == frozen and all(
        not p.requires_grad and p.grad is None for p in reference.parameters())
    memory = gpu_memory()
    result = {'status': 'passed' if all(x['passed'] for x in results) and restored and reference_unchanged
              and memory['headroom_fraction'] >= r['minimum_gpu_headroom'] else 'not_passed',
        'generated_at': now(), 'registration_sha256': file_sha256(REG), 'probes': results,
        'policy_restored': restored, 'reference_unchanged': reference_unchanged, 'gpu': memory,
        'note': 'Controlled individual-completion updates, not evidence of aggregate illegal-action probability or gameplay improvement.'}
    write_new(ROOT / 'directions.json', result)
    if result['status'] != 'passed': raise ValueError('direction proof failed; no further GPU work')

def train(arm, stop_after, resume):
    import torch
    from transformers import get_cosine_schedule_with_warmup
    from scripts.train_episode_rl import (sample_group, flatten_steps, action_loss_chunk,
        sequence_logprobs, save_checkpoint, load_checkpoint, adapter_parameter_hash)
    r = validate(True)
    if read(ROOT / 'directions.json')['status'] != 'passed': raise ValueError('directions gate')
    destination = ROOT / arm
    if (arm, stop_after, resume) not in {('control', 4, False), ('resumed', 2, False), ('resumed', 4, True)}:
        raise ValueError('only registered disposable proof commands allowed')
    if destination.exists() and not resume: raise ValueError('proof arm already exists')
    tokenizer, policy, reference = load_models(r)
    reference_before = adapter_parameter_hash(reference)
    params = [p for p in policy.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=1e-6)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=1, num_training_steps=4)
    rng = random.Random(r['training_seed']); metrics = []; samples = begin = 0
    if resume:
        begin, samples = load_checkpoint(destination / 'checkpoint-2', model=policy, optimizer=optimizer,
                                         scheduler=scheduler, rng=rng, update_metrics=metrics)
        if begin != 2: raise ValueError('wrong resume checkpoint')
    bank = load_start_bank(Path(r['recovery_start_file']), read(r['training_seed_file']))
    by_hash = {s['state_hash']: s for s in bank}
    for update in range(begin + 1, stop_after + 1):
        validate_deadline = read(ROOT / 'workflow.json')['started_epoch'] + 3600
        if time.time() >= validate_deadline: raise ValueError('proof time exhausted')
        tick = time.time(); scheduled = r['schedule'][update-1]
        trajectories = sample_group(policy, reference, tokenizer, seed=scheduled['environment_seed'],
            group_size=4, horizon=128, temperature=1, reward_weights=EpisodeRewardWeights(),
            start=by_hash.get(scheduled['state_hash']))
        rewards = [[s['immediate_reward'] for s in t['steps']] for t in trajectories]
        advantages = trajectory_advantages(rewards, method='fixed_zero', reward_scale=10)
        for trajectory, values, rewards_row in zip(trajectories, advantages, rewards):
            for step, advantage, rtg in zip(trajectory['steps'], values, discounted_reward_to_go(rewards_row)):
                step.update(advantage=advantage, reward_to_go=rtg)
        write_new(destination / 'trajectory_batches' / f'update-{update:06d}.json',
            {'update': update, 'environment_seed': scheduled['environment_seed'], 'trajectories': trajectories})
        steps = flatten_steps(trajectories); policy.train(); optimizer.zero_grad(set_to_none=True)
        loss_sum = kl_sum = 0.0
        for offset in range(0, len(steps), 4):
            chunk = steps[offset:offset+4]
            token_rows = sequence_logprobs(policy, chunk, tokenizer.pad_token_id, 1, return_tokens=True)
            loss, kl = action_loss_chunk(token_rows, chunk, .05, len(steps))
            if not torch.isfinite(loss): raise ValueError('nonfinite loss')
            loss.backward(); loss_sum += float(loss.detach()); kl_sum += float(kl.detach())
        norm = float(torch.nn.utils.clip_grad_norm_(params, 1, error_if_nonfinite=True))
        optimizer.step(); scheduler.step(); samples += len(steps)
        metrics.append({'update': update, 'turns': len(steps), 'loss': loss_sum, 'kl': kl_sum,
                        'gradient_norm_before_clip': norm, 'seconds': time.time()-tick})
        save_checkpoint(destination / f'checkpoint-{update}', model=policy, optimizer=optimizer,
            scheduler=scheduler, update=update, samples=samples, rng=rng, update_metrics=metrics)
        print(json.dumps({'arm': arm, **metrics[-1]}), flush=True)
    reference_unchanged = adapter_parameter_hash(reference) == reference_before and all(
        not p.requires_grad and p.grad is None for p in reference.parameters())
    result = {'status': 'completed' if stop_after == 4 else 'paused', 'completed_updates': stop_after,
        'registration_sha256': file_sha256(REG), 'sample_count': samples, 'update_metrics': metrics,
        'reference_unchanged': reference_unchanged, 'reference_hash': reference_before,
        'adapter_parameter_hash': adapter_parameter_hash(policy), 'resume_used': resume,
        'gpu': gpu_memory(), 'completed_at': now()}
    write_new(destination / ('complete.json' if stop_after == 4 else 'paused.json'), result)
    if not reference_unchanged: raise ValueError('reference changed')

def nested_equal(a, b):
    import torch
    if isinstance(a, torch.Tensor): return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict): return isinstance(b, dict) and a.keys() == b.keys() and all(nested_equal(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)): return type(a) == type(b) and len(a) == len(b) and all(nested_equal(x,y) for x,y in zip(a,b))
    return a == b

def verify():
    import torch
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file
    from scripts.check_episode_proof import audit_batch, independent_token_logprobs
    from scripts.train_episode_rl import adapter_parameter_hash
    from tetris.chat import build_generation_prompt
    r = validate(True); manifests = [read(ROOT / arm / 'complete.json') for arm in ('control', 'resumed')]
    if any(m['status'] != 'completed' or m['completed_updates'] != 4 or not m['reference_unchanged'] for m in manifests):
        raise ValueError('incomplete proof training')
    left = ROOT / 'control'; right = ROOT / 'resumed'
    weights_equal = nested_equal(load_file(str(left/'checkpoint-4/adapter/adapter_model.safetensors')),
                                load_file(str(right/'checkpoint-4/adapter/adapter_model.safetensors')))
    states = [torch.load(d/'checkpoint-4/state.pt', map_location='cpu', weights_only=False) for d in (left, right)]
    state_equal = all(nested_equal(states[0][k], states[1][k]) for k in
                      ('optimizer','scheduler','python_rng','torch_rng','cuda_rng','samples','update'))
    del states
    seeds = read(r['training_seed_file']); bank = load_start_bank(Path(r['recovery_start_file']), seeds)
    batches = []; samples = 0
    for update in range(1,5):
        relative = Path('trajectory_batches')/f'update-{update:06d}.json'
        a,b = read(left/relative),read(right/relative)
        if a != b: raise ValueError(f'GPU trajectories differ at update {update}')
        samples += audit_batch(a,r['recipe'],seeds,bank); batches.append(a)
    tokenizer, policy, reference = load_models(r)
    max_error = 0.; tokens = 0
    for batch in batches:
        update = batch['update']
        source = Path(r['initial_adapter']) if update == 1 else left/f'checkpoint-{update-1}/adapter'
        set_peft_model_state_dict(policy,load_file(str(source/'adapter_model.safetensors')),adapter_name='default')
        for turn in range(max(len(t['steps']) for t in batch['trajectories'])):
            rows = [t['steps'][turn] for t in batch['trajectories'] if len(t['steps']) > turn]
            for row in rows:
                prompt = build_generation_prompt(tokenizer,row['serialized_prompt'])
                if (prompt != row['prompt'] or tokenizer(prompt,add_special_tokens=False)['input_ids'] != row['prompt_ids']
                    or tokenizer.decode(row['completion_ids'],skip_special_tokens=True) != row['raw_completion']):
                    raise ValueError('token text alignment failed')
            for model,key in ((policy,'policy_token_logprobs_at_sampling'),(reference,'reference_token_logprobs')):
                with torch.no_grad(): values=independent_token_logprobs(model,rows,tokenizer.pad_token_id,1)
                for row,value in zip(rows,values):
                    if not torch.isfinite(value).all():raise ValueError('nonfinite token probability')
                    max_error=max(max_error,float((value-torch.tensor(row[key],device=value.device)).abs().max()))
                    tokens+=len(value)
        print('independent token verification update',update,flush=True)
    directions_result=read(ROOT/'directions.json')
    headroom=min(gpu_memory()['headroom_fraction'],directions_result['gpu']['headroom_fraction'],
                 *(m['gpu']['headroom_fraction'] for m in manifests))
    checks={'directions':directions_result['status']=='passed','resume_trajectories_exact':True,
        'resume_weights_exact':weights_equal,'resume_optimizer_scheduler_rng_exact':state_equal,
        'all_trajectories_rewards_advantages_replayed':True,
        'token_alignment':tokens>0 and max_error<=r['token_absolute_tolerance'],
        'reference_unchanged':adapter_parameter_hash(reference)==manifests[0]['reference_hash']==manifests[1]['reference_hash'],
        'gpu_headroom':headroom>=r['minimum_gpu_headroom'],
        'original_sft_unchanged':directory_sha256(Path(r['initial_adapter']))==r['initial_adapter_sha256']}
    result={'status':'passed' if all(checks.values()) else 'not_passed','checks':checks,'generated_at':now(),
        'registration_sha256':file_sha256(REG),'sampled_decisions_per_arm':samples,'tokens_checked':tokens,
        'maximum_token_logprob_error':max_error,'minimum_headroom_fraction':headroom,
        'directions':directions_result,'training_manifests':manifests,
        'pilot_started':False,'gameplay_improvement_established':False,'final_test_access':False,
        'next':'Report correctness only; six-run pilot needs separate approval.'}
    write_new(ROOT/'gpu-proof.json',result)
    print(json.dumps({'status':result['status'],'checks':checks}),flush=True)
    if result['status']!='passed':raise ValueError('GPU proof not passed')

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','validate','directions','train','verify'])
    parser.add_argument('--arm',choices=['control','resumed']);parser.add_argument('--stop-after',type=int,default=4)
    parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    if args.action=='prepare':prepare()
    elif args.action=='validate':validate();print('proof inputs verified')
    elif args.action=='directions':directions()
    elif args.action=='train':train(args.arm,args.stop_after,args.resume)
    else:verify()
