"""Read-only GPU diagnosis of the retained direction-proof failure.

No optimizer, sampling, checkpoints, or changes to the registered proof.
"""
from pathlib import Path
import json
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    import torch
    from infra.feedback_gpu_proof import validate, load_models
    from scripts.train_episode_rl import sequence_logprobs, adapter_parameter_hash
    from scripts.check_episode_proof import independent_token_logprobs
    from scripts.eval_feedback import approved_session
    from tetris.chat import build_generation_prompt
    from tetris.rl import file_sha256, atomic_write_json

    root = Path('experiments/stage6-feedback-v1/session/alignment-diagnosis-v1')
    registration = json.loads((root / 'registration.json').read_text())
    assert file_sha256(Path(__file__)) == registration['script_sha256']
    approved_session(Path('experiments/stage6-feedback-v1/session/approval.json'))
    if (root / 'result.json').exists():
        raise ValueError('diagnosis result exists; preserve it')
    started = time.time()
    r = validate()
    tokenizer, policy, reference = load_models(r)
    initial = adapter_parameter_hash(policy)
    ref_initial = adapter_parameter_hash(reference)
    observations = []
    for index, probe in enumerate(r['probes']):
        prompt = build_generation_prompt(tokenizer, probe['state']['prompt'])
        rows = []
        for action in (probe['illegal_action'], probe['legal_action']):
            text = f'Action: rot={action[0]} x={action[1]}'
            rows.append({'prompt_ids': tokenizer(prompt, add_special_tokens=False)['input_ids'],
                         'completion_ids': tokenizer(text, add_special_tokens=False)['input_ids'] + [tokenizer.eos_token_id]})
        policy.train()
        with torch.no_grad():
            paired = independent_token_logprobs(policy, rows, tokenizer.pad_token_id, 1)
            paired_trainer = sequence_logprobs(policy, rows, tokenizer.pad_token_id, 1, return_tokens=True)
        for row_index, row in enumerate(rows):
            with torch.no_grad():
                solo = independent_token_logprobs(policy, [row], tokenizer.pad_token_id, 1)[0]
            current = sequence_logprobs(policy, [row], tokenizer.pad_token_id, 1, return_tokens=True)[0]
            observations.append({'probe_index': index, 'row_index': row_index,
                'single_same_input_error': float((current.detach() - solo).abs().max()),
                'paired_same_input_error': float((paired_trainer[row_index] - paired[row_index]).abs().max()),
                'batch_one_vs_two_error': float((solo - paired[row_index]).abs().max()),
                'original_proof_comparison_error': float((current.detach() - paired[row_index]).abs().max()),
                'prompt_length': len(row['prompt_ids']), 'completion_length': len(row['completion_ids'])})
            del current
    result = {'observations': observations, 'seconds': time.time() - started,
        'optimizer_steps': 0, 'policy_unchanged': adapter_parameter_hash(policy) == initial,
        'reference_unchanged': adapter_parameter_hash(reference) == ref_initial,
        'registration_sha256': file_sha256(root / 'registration.json'),
        'original_tolerance': r['token_absolute_tolerance'],
        'note': 'Diagnostic only; the original GPU proof remains failed. No tolerance is changed.'}
    atomic_write_json(root / 'result.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
