# Stage 6 RL research runbook

Stage 6 is implemented as a gated research ladder. The frozen SFT adapter remains the accepted policy unless replicated RL evidence clears both the Stage 5 non-inferiority gate and the `stress-v1` improvement rule. A null or negative result is a valid completed outcome.

## Current status — completed 10x negative result, 2026-09-03

The authorized 10x episode-RL iteration completed all 320 updates and 106,727
sampled decisions, followed by all frozen evaluations. Independent replay
reproduces the full failed promotion gate: development score improved only
0.1834% (required 3%), with paired mean-difference 95% CI [-14.010, 45.035]
spanning zero. Original recovery illegal deaths increased from 6/12 to 7/12;
fresh recovery remained 8/16. Stage 5 passed at 197.76 mean lines, with all
100 games surviving to the cap and no invalid actions. The original SFT remains
accepted; conditional replication is not applicable and final tests were not
accessed. This completes a bounded study, not a claim that RL cannot help.

All 378 artifacts across three scale runs passed independent AES256 S3 read-back,
including each final adapter. Worker `i-069539d2da9a2b0a2` is terminated and root
`vol-0bbe37041a8425e58` deleted; Stage 4 is unchanged. All five compute sessions
are closed. Estimated experiment cost: **$29.43**; tracked cumulative estimate:
**$44.96**, not an AWS invoice. No GPU work remains, the completed recurring
monitor was removed, and no new model is promoted.

Report: [10x outcome](runs/stage6-scale10x-v1/rl/report.md), with machine-readable
report, artifact audit, cleanup and closed ledger alongside it. Full execution
history and runtime-only approval: [10x plan](plan/original/stage-6-scale-10x.md).
The following sections retain earlier research outcomes without relabeling them.

## Completed recovery branch — 2026-09-02

The authorized Stage 6 recovery branch is complete. R3's independently
reproduced full gate is **not passed**: no qualifying improvement and remaining
illegal actions. Stage 5 passed at 197.87 mean lines with zero deaths/invalid
moves; fresh recovery had seven illegal deaths versus SFT's eight, but this
does not override the failed development criteria. The original SFT remains
accepted. Conditional replication and final tests are not applicable; final
tests were not accessed. Historical E3 and failed R2 outcomes remain unchanged.

All 171 retained artifacts across six recovery runs passed actual-byte AES256
S3 verification, including all final adapters. The exact Stage 6 instance
`i-04f0b80866f639597` is terminated and root `vol-0c8f944054d9c0a70` is deleted;
the cleanup checker passes with no Stage 6 instances or volumes remaining.
Stage 4 is unchanged. Final conservative compute/storage estimate: **$15.53**.
Report: `runs/stage6-recovery-v1/rl/report.md` and `report.json`; backup and
cleanup evidence are alongside them. No new model is promoted.
Stage 6 verification passes all four checks and all 75 tests. The recurring
monitor is paused following verified completion and final evidence upload.

### Execution history

**Live milestone (21:50:28 UTC): R3 development evaluation completed; Stage 5
is running.** Independent replay confirms one ordinary-game illegal action:
seed 30000002, decision 521 after 520 legal placements, `Action: rot=1 x=9`
for a J piece. The placement extends outside the board despite 34 legal
alternatives; the board was only five rows high. Nineteen other games reached
the full 2000-piece cap. Score per 100 pieces is 7884.717 versus SFT 7913.215
(-0.360%; paired 95% CI for the difference [-68.888, 12.035]). Original
recovery has seven illegal endings and two top-outs in 12 starts, versus
SFT's six illegal endings and eight total deaths. All seven recovery illegal
endings replay as top collisions despite legal alternatives. Fixed-state
parse and legality are still perfect. These development results fail the
unchanged promotion requirements; they are not a runtime or execution failure.
Stage 5, fresh recovery, the full independently reproduced R3 gate, backup
audit and cleanup still must finish. No replication or final-test access is
eligible. Evidence: `runs/rl-r3-episode-seed0/rl/candidate-development-local-check.json`.

**Live milestone (20:50:58 UTC): R3 training completed**, 32/32 updates,
9,948 sampled moves and 3,271.63 seconds (about 55 minutes). Independent local
validation reproduces every trajectory/reward/accounting check and confirms
the registered recipe and unchanged frozen reference. Final adapter SHA256:
`220a7e1a61cca99ca0c39ea3677a12b4c3b304cd22e5d79043ca1c66b9abaaa7`.
Checkpoint 32 retains adapter, optimizer, scheduler and complete RNG/history.
Development evaluation started at 20:51:04 UTC; Stage 5 and fresh recovery
follow automatically. Training completion is not a passed R3 research gate.
The timer and budgets are unchanged; no candidate is promoted yet.

The user authorized the recovery/data/episode recommendations end-to-end.
`plan/stage-6-recovery.md` and `runs/stage6-recovery-v1/rl/registration.json`
define a new independent branch, not a retroactive E3 pass. R0's saved audit
classifies all recovery illegal actions as top collisions despite legal
alternatives; KL 0.01 also made one out-of-bounds long-game move. R1 continues
frozen SFT on an 8192-row 50/50 recovery/ordinary mixture, then evaluates the
original suites and 16 newly held-out recovery starts. R2 proves GPU trajectory
replay/log-probability/resume correctness with 20 turns and recovery starts;
R3 independently tests episode rewards from original SFT. R1 is a comparator,
not silently substituted initialization. Replication remains conditional on
the original 3% gain and strict guardrails. See the new plan for exact rules.

Dataset `data/stage6-recovery-v1` is generated and hash-bound: 4096 recovery
training rows, 4096 ordinary rows, 512 validation rows, 129 replayable training
starts. Training uses the first 192 stress training seeds; fresh validation
uses the last 64, excluded from episode training. No old evaluation state
became a training example. Deployment and live status are recorded in the
compute ledger. Refresh its stopped-worker copy before resuming metadata cron.

R1 training completed at 16:25 UTC: 512/512 updates, 869.72 seconds, original
SFT unchanged. Final adapter SHA256 is
`0a40ef451396e6811e39397087efe8605125e6137c02ce4b4aa7e8848a82f245`;
its initial encrypted S3 upload finished at 16:25:41 UTC. All evaluations
finished at 18:17:42 UTC. An independent local gate exactly reproduces the
worker result except timestamp: **R1 not passed**. Original recovery illegal
deaths fell from 6 to 4 (target at most 3); fresh held-out illegal deaths
remained 8/16, while total deaths fell from 9 to 8. Normal development retained
zero deaths/invalid moves, with score down 0.229% (paired CI spans zero).
Stage 5 passed: 197.63 mean lines, 100/100 capped games, zero deaths/parse/illegal
actions. R1 is retained as a completed negative data-coverage experiment, not
a promoted model. Its result does not block independently initialized R2.
The existing ten-minute thread monitor remains active.

R2/R3 source was deployed between blocks after R1 stopped and its gate and
33 encrypted artifacts were independently verified. R2 was registered before
training (registration SHA256
`9c942afec1576e371461e3abdcaf38ee50998458ca16eb02e7fb9ead91fb2c06`).
The same worker restarted at 18:23:30 UTC with IP `44.204.99.12`; all 58
Stage 6 tests passed there. `llm-tetris-r2.service` launched at 18:25:26 UTC,
with metadata cron active and a one-hour shutdown timer. R2 **failed** at
18:29:34 UTC: its independent verifier found different resumed and uninterrupted
trajectories. Both four-update trainings completed, but this is not a passed
proof. Updates 1–2 match exactly; update 3 differs in policy probabilities,
and update 4 also differs in sampled actions. All 36 encrypted artifacts from
both attempts passed read-back verification. R3 remains unregistered and
unlaunched. A bounded read-only GPU diagnosis began after worker restart at
18:42:52 UTC; original registered source and failed evidence remain intact.

Diagnostics confirmed nondeterministic GPU gradients, not a checkpoint reload
defect: optimizer moments differed before the pause, and the reload restored
checkpoint weights exactly. Repeating the same backward pass differed by up
to 0.0003082 under default execution; deterministic repeats matched bitwise.
The diagnostic adapter stayed unchanged and no optimizer updates were applied.
The sole implementation retry uses new `rl-r2v2-episode-proof-*` runs, fixed
deterministic execution, completion-suffix logits and portable normalization.
All recipes, seeds, proof tolerances and budgets remain unchanged. Registration
SHA256 is `21674ccf7e322094951190829520bafe436cd40283c1bb8c5a771e050abff837`.
All 61 Stage 6 tests passed locally and on the worker. `llm-tetris-r2v2.service`
launched at 18:58:38 UTC on the same worker (IP `3.227.22.36`), with metadata
cron and a one-hour timer. The proof completed; its latest decision is below.

**Latest outcome (19:03:23 UTC):** R2 v2 finished and remains `not_passed` solely
because its 8626.25-second (2h 24m) pilot projection exceeds the registered
1.5-hour training cap. All nine correctness/memory checks passed: 301 samples,
5418 policy/reference tokens checked with zero error, bitwise-identical resumed
and uninterrupted trajectories and adapter tensors, positive delayed-credit
direction, unchanged reference, and 68.35% allocated GPU headroom. Independent
local replay and tensor comparison reproduce the result. AWS confirms the worker
is stopped; all 37 artifacts from the retry pair passed encrypted read-back.
The user subsequently approved a **12-hour** ceiling. The separate hash-bound
`runtime-amendment-v1.json` preserves the historical failed gate and revalidates
all nine correctness prerequisites. R3 is registered with 12-hour trainer and
overall ceilings, $20/pilot and $100 total unchanged. A nine-hour training
subprocess allowance reserves the remaining window for evaluations, analysis
and backups; the measured training estimate is only 2h 24m. All 75 Stage 6
tests passed locally and on the worker. R3 launched at **19:56:23 UTC** on the
same A10G worker, IP `44.200.55.54`; shutdown deadline is **2026-09-03 07:56:23 UTC**.
The five-minute metadata cron and existing ten-minute thread monitor are active.
Source, amendment and R3 registration passed actual-byte encrypted S3 read-back.
Live progress is tracked in the compute ledger. The monitor follows this
amendment and does not treat the historical R2 time gate as passed.
R2 has an uninterrupted four-update control plus the identical 2+2 resumed
run. Its independent checker replays rewards and states, re-forwards exact
policy/reference tokens, tests a throwaway positive delayed-credit update,
and checks GPU headroom and a conservative full-length R3 projection.
Registration freezes its numerical tolerances and source hashes before launch.
R3 registration requires all real R2 correctness evidence plus either the
original runtime gate or the explicit approved amendment, and a reproduced R1 result;
it never treats historical E3 as passed. Full commands and negative-result
closure requirements are in `plan/stage-6-recovery.md`.

### Historical E3 gate stop

E3 finished at 12:31:07 UTC. All three registered candidates trained and
completed strict development evaluation. Independent local selection exactly
reproduces the worker result (apart from its generation timestamp): **none
passes**, no KL is selected, and E4 must not launch. E0, E1 and E2 v2 passed.
E4–E7 remain unrun; the full Stage 6 research program is **not complete**.
See `runs/stage6-e3/rl/outcome.md`, `selection.json` and
`selection-local-check.json`. Historical progress entries below are retained.

| KL | Score / 100 pieces | Paired score change | Long deaths / 20 | Recovery illegal deaths / 12 |
|---|---:|---:|---:|---:|
| Frozen SFT | 7913.215 | baseline | 0 | 6 |
| 0.01 | 7914.926 | +0.0216% | 1 | 5 |
| 0.05 | 7913.575 | +0.00455% | 0 | 6 |
| 0.1 | 7913.065 | -0.00190% | 0 | 6 |

Every paired score confidence interval spans zero. E3's zero-illegal-recovery
gate is stricter than baseline non-regression: frozen SFT already has six
illegal recovery deaths. These failures alone do not prove RL introduced that
weakness. KL 0.01 separately regresses long-game survival. The registered gate
and original evidence remain unchanged; no final-test seeds were consumed.

At the historical E3 stop, the scheduled monitor was paused pending a user
decision. The user subsequently authorized the independent recovery branch
above; the monitor is now active. The old stop and its evidence remain intact.

## Local gate and frozen benchmark

```bash
source .venv-train/bin/activate
python scripts/verify_stages.py --stage 6
```

The committed benchmark is in `benchmarks/stress-v1/`:

- 256 training seeds
- 20 development seeds at a 2,000-piece cap
- 100 final-test seeds at a 5,000-piece cap
- 24 replayable recovery states and 24 board-quality probes, split evenly between development and final test with disjoint source seeds
- a 200-piece continuation horizon for recovery states
- frozen budgets, primary metric, thresholds, and promotion rules

Regeneration is intentionally guarded. `scripts/generate_stress_manifest.py` refuses to replace it unless `--force` is explicit.

## E0: frozen controls

Run the SFT control on development seeds before spending on RL:

```bash
python scripts/eval_stress.py \
  --suite development \
  --policies random,teacher,model \
  --adapter-dir runs/sft-v1/adapter \
  --policy-label sft \
  --data-dirs data/batch1 data/batch2 \
  --out-dir runs/stage6-e0/rl/stress-development
```

The original Stage 5 artifacts remain the frozen regression baseline. Re-run its model cohort only when evaluating a promoted checkpoint:

```bash
python scripts/eval_closed_loop.py \
  --policies model --modes strict \
  --adapter-dir runs/CANDIDATE/rl/adapter \
  --model-label CANDIDATE \
  --data-dirs data/batch1 data/batch2 \
  --out-dir runs/CANDIDATE/rl/stage5
```

## E1–E4: dense TRL GRPO

Create the GPU environment with `requirements-rl.txt`. Every invocation registers its question, hashes, seeds, reward, KL, limits, and projected cost before model loading. E1 rejects more than 50 updates. E2 rejects the main SFT adapter and requires `--initialization-kind weakened`; create that adapter reproducibly with `train_sft.py --max-train-rows ... --max-steps ...` or use a retained earlier checkpoint.

For newly trained weakened adapters, `train_sft.py` now seeds Python, NumPy,
and PyTorch before creating LoRA weights, rather than relying on Trainer's
later seed initialization. A local regression test checks repeatable RNG
state before loading data or creating the model. This does not retrain or
modify the frozen Stage 4 adapter. E2 is pre-registered in
`runs/rl-e2-seed0/rl/registration.json` and orchestrated by
`infra/rl-e2-block.sh`: 8,192-row / 512-step HF SFT with a pinned base revision,
then baseline development evaluation, then 512 dense GRPO updates only if the
baseline is measurably weaker and valid. The target is a mean reward gain of
at least 0.05 on all 24 fixed development states, with no greedy validity
regression. Paired differences and a bootstrap interval are reported. This
learning smoke does not authorize main-model promotion. The entire block,
including sync, is limited to five hours, about $5.25; final tests remain closed.

Example E1:

```bash
python scripts/train_rl.py \
  --experiment E1 \
  --question "Does dense GRPO fit, resume, and preserve the SFT reference on an A10G?" \
  --initialization-kind sft \
  --adapter-dir runs/sft-v1/adapter \
  --out-dir runs/rl-e1-seed0/rl \
  --max-updates 50 --states 512 --group-size 4 --batch-size 4 \
  --kl-beta 0.05 --instance-hourly-usd 1.01 --max-wall-clock-hours 2
```

Use separate registered runs for the E3 KL candidates `0.01`, `0.05`, and `0.1`; choose a coefficient before changing reward weights. Resume with `--resume` or `--resume PATH` so TRL restores model, optimizer, scheduler, and sample state.

The first live E1 block is prepared in `infra/rl-e1-block.sh` and must not start
until E0 passes. It registers 20 updates, 128 states, group/batch size 4,
accumulation 4, learning rate `1e-6`, KL `0.05`, and a one-hour cumulative limit.
It pauses after update 10, preserves that manifest, resumes checkpoint 10 with
the same scheduler horizon, and checks 20 completed updates / 320 completions
and at least 10% allocated-VRAM headroom. Supply the actual prior Stage 6 spend
as its first argument. A paused run is not a completed hardware gate.

Training now saves rollout observation counts alongside each checkpoint and
restores them on resume. After three measured updates it projects the remaining
run time and stops early if the registered time/cost limit would be exceeded.
Offline tests compare resumed versus uninterrupted completion tokens, exact
restored optimizer state, scheduler state, frozen reference and exact final
adapter parameters. They exposed a Transformers 5.16 PEFT resume bug: when
`ref/` exists, the library skips the default adapter stored at the checkpoint
root. The runner explicitly restores that policy adapter after constructing
the frozen SFT reference and before restoring the remaining Trainer state.
The real A10G E1 run completed all 20 updates on 2026-09-02 at 05:41:22 UTC,
resuming from update 10 with 160 then 320 completions and unchanged reference
hashes. Peak allocated memory was 3.94 GiB (82.1% headroom), median optimizer
step 3.154 seconds, and cumulative wall time 250.87 seconds. Its encrypted S3
adapter and receipt were verified. See `runs/rl-e1-seed0/rl/e1-gate.json` and
the conditional later-experiment forecast in
`runs/stage6-aws/rl/throughput-forecast.md`.

The dense reward is computed from a cloned `Game` transition. Illegal or malformed actions terminate with the registered penalty; assisted substitution is never used. TRL 0.29 copies the loaded pre-trained PEFT adapter into a frozen reference adapter when KL is active. The script hashes that copy before and after training and fails if it changes. See the [official GRPOTrainer documentation](https://huggingface.co/docs/trl/grpo_trainer).

## E5–E7: multi-turn proof and episode return

E5 enforces a 10–20 turn horizon:

```bash
python scripts/train_episode_rl.py \
  --experiment E5 \
  --question "Do exact saved tokens, log-probabilities, replay, credit assignment, and resume align?" \
  --adapter-dir runs/sft-v1/adapter \
  --out-dir runs/rl-e5-seed0/rl \
  --updates 2 --group-size 4 --horizon 20 --gamma 0.99 \
  --training-seed 0 --instance-hourly-usd 1.01 --max-wall-clock-hours 2
```

Each update writes one atomic `trajectory_batches/update-*.json` containing exact prompts and completion token IDs, policy/reference log-probabilities, decomposed rewards, state hashes, returns, advantages, and replay results. Checkpoints contain the adapter, optimizer, scheduler, Python RNG, Torch RNG, CUDA RNG, completed update, and sample count. Resume with `--resume runs/RUN/rl/checkpoint-N`; a repeated interrupted update replaces the same atomic batch file instead of duplicating samples.

E6 uses the same entry point with a registered longer horizon. E7 is three independent invocations with distinct `--training-seed` values. The custom trainer uses discounted reward-to-go and normalizes comparisons only across trajectories from the same starting seed and turn.

The episode trainer enables non-reentrant gradient checkpointing and computes
float32 log-softmax only at the saved completion positions. This avoids a
full-prompt probability tensor while keeping small KL differences out of bf16
rounding. Unit tests verify the selected positions, precision, and gradients;
real GPU memory and throughput remain E5 measurements, not assumed results.

For E5 resume proof, keep the registered total update count fixed and use
`--pause-after-update 1`, then resume `checkpoint-1` without that pause flag.
Pauses force a checkpoint even between scheduled saves. Checkpoints retain
committed update metrics as well as sample counts, and resume validates both.
The actual episode optimizer loop has an offline tiny-model test comparing
uninterrupted versus paused/resumed trajectories and exact final weights.
After three measured updates the trainer projects remaining time/cost and
stops at an update boundary when the registration would be exceeded. Final
checkpoint/export time is counted; an overrun is not marked completed. Abrupt
termination can still lose work since the last checkpoint, and the live E0
stress evaluator does not yet support mid-game resume. These improvements
were deployed in the tested source bundle before E1; real GPU episode
measurements still require E5.

## Promotion, confirmation, and final report

Evaluate each candidate greedily on development `stress-v1`. Only a candidate at least 3% above SFT with clean guardrails proceeds to final test. Final test is a one-time evaluation after selection.

```bash
python scripts/analyze_stage6.py \
  --suite test \
  --baseline sft=runs/stage6-e0/rl/stress-test \
  --candidate rl-seed0=runs/rl-seed0/rl/stress-test \
  --candidate rl-seed1=runs/rl-seed1/rl/stress-test \
  --candidate rl-seed2=runs/rl-seed2/rl/stress-test \
  --stage5-candidate rl-seed0=runs/rl-seed0/rl/stage5/metrics.json \
  --stage5-candidate rl-seed1=runs/rl-seed1/rl/stage5/metrics.json \
  --stage5-candidate rl-seed2=runs/rl-seed2/rl/stage5/metrics.json \
  --out-dir runs/stage6-final/rl
```

The report retains every paired difference, median, bootstrap 95% interval, guardrail, and unfavorable result. It selects `retain_rl` only when at least two of three training seeds improve, the combined paired interval is above zero, and every candidate clears Stage 5.

## AWS and artifact safety

### Active execution (2026-09-02 UTC)

The missing permissions were supplied and the SNS email subscription is confirmed.
The encrypted `g5.xlarge` worker `i-04f0b80866f639597` launched at 02:44:30 UTC.
Its root volume `vol-0c8f944054d9c0a70` is encrypted and delete-on-termination;
a termination dry run passed for this exact Stage 6 worker. Do not touch the
retained Stage 4 instance. Current EC2 compute is $1.006/hour; the ledger uses
$1.05/hour including a storage/IPv4 allowance. AWS billing remains authoritative.

E0 completed at 05:01:15 UTC using `infra/rl-e0-block.sh`. The Stage 5 portion
reproduced exactly all 100 original games, with 197.77 mean lines, zero deaths
and zero parse failures. The development suite also completed all three
controls, 24 fixed states and 12 recovery continuations per control. A local
review checked hashes, exact cohorts and encrypted S3 outputs/receipt;
`runs/stage6-e0/rl/e0-gate.json` records the passed control gate.

| Control | Mean lines / 2,000-piece game | Score / 100 pieces | Long-horizon deaths / 20 | Recovery deaths / 12 | Recovery illegal deaths / 12 |
|---|---:|---:|---:|---:|---:|
| Random | 0.05 | 1,659.300 | 20 | 12 | 0 |
| Teacher | 798.25 | 7,964.650 | 0 | 3 | 0 |
| SFT | 798.15 | 7,913.215 | 0 | 8 | 6 |

All controls had zero parse failures and illegal actions in the long-horizon
cohort and legal/parse rates of 100% on fixed one-action states. Recovery
deaths are retained evidence of harder-state weakness, not suppressed failures.
The control block synced artifacts and stopped the worker normally.

At 05:12 UTC, three E1 restart requests had failed with AWS
`InsufficientInstanceCapacity`. The worker's disk and hash-verified next-block
source bundle remained intact. Current launch IAM is restricted
to `subnet-094caed03deaaae73` / `us-east-1a`, so moving to another zone is not an
authorized workaround. The fifth restart request succeeded at 05:35:15 UTC,
without an IAM change or a replacement worker. Its current public IP is
`44.204.211.2`; SSH can use `HostKeyAlias=18.210.6.246` to verify the original
host key. Re-query the IP after any later restart.

The tested source bundle was deployed and all 29 tests passed on the worker.
E1 launched as `llm-tetris-e1.service` at 05:37:09 UTC with 20 updates and a
pause/resume at update 10, the one-hour cumulative training budget, and a
conservative $2.60 prior-spend allowance. It completed at 05:41:22 UTC, synced
its final adapter and stopped the worker normally. E0 and E1 are passed; do not
repeat them. E2 registration and its tested workflow are the next block. The
worker was confirmed stopped at 06:01 UTC; accrued conservative AWS cost was
about $2.69, while Cost Explorer still showed delayed estimated zero.

For E2, the original worker restarted at 06:10:04 UTC, IP `3.208.86.165`.
All 33 Stage 6 tests passed again on the worker, and `llm-tetris-e2.service`
launched at 06:11:37 UTC with $2.80 conservatively registered prior spend.
The five-hour shutdown timer expires at 11:10:39 UTC. The workflow is bounded
to 280 minutes with the remainder reserved for artifact sync and shutdown.
Its phase/status is in `runs/rl-e2-seed0/rl/block-state.json`; initial phase
is weakened SFT training. Do not start a duplicate block. Neither E2 baseline
eligibility nor learning is claimed until its corresponding gate passes.

E2 v1 stopped at 06:21:37 UTC, before any RL update. The 8,192-row / 512-step
SFT adapter parsed all outputs but failed baseline legality: 20/24 fixed
states (83.3%, floor 90%) and 94.725% mean long-game legality (floor 95%).
All 20 long games and all 12 recovery continuations ended on illegal actions.
Its weak adapter and complete baseline are retained in encrypted S3;
`runs/rl-e2-seed0/rl/outcome.json` records an inconclusive learning experiment,
not a negative result about GRPO. The worker stopped normally.

One bounded initialization repair is registered as E2 v2 in
`runs/rl-e2v2-seed0/rl/registration.json`, with
`infra/rl-e2v2-block.sh`. It trains on 65,536 rows for 4,096 SFT steps; the
baseline floors, dense RL configuration, seeds, reward and improvement rule
are exactly the same as v1. The component SFT limit is 90 minutes and the
whole block limit is six hours (at most $6.30 before incidental boot allowance,
still within $20/$100). The measured v1 SFT runtime predicts about 62.5 minutes
for this larger initialization. This is the sole revised initialization: if
it remains ineligible, do not sweep more recipes to chase the gate. Retain
both attempts and assess an honest early research stop.

E2 v2 restarted the same worker at 06:29:01 UTC, IP `18.204.43.234`, and
launched as `llm-tetris-e2v2.service` at 06:30:15 UTC after all 33 worker tests
passed. Its shutdown timer expires at 12:29:31 UTC. Registered prior spend is
$3.15 (conservative); the observed ledger estimate was about $3.00 at 06:30:35.
Inspect `runs/rl-e2v2-seed0/rl/block-state.json`; do not restart the old v1
service or duplicate v2. The new SFT metadata correctly names nested `rl/`
run directories; the original v1 metadata remains unchanged for provenance.

E2 v2 SFT completed all 4,096 steps at 07:33:43 UTC in 3,728.65 training
seconds. Baseline eligibility passed at 07:39:25 UTC and was reproduced
locally: mean fixed-state reward -0.50375 versus strong SFT 0.01417, fixed
parse/legality 100%, long-game parse 100% and mean legality 99.126%. Mean
lines were 36.25 and score/100 pieces 5,205.970; all 20 long games still ended
on illegal actions. This is an eligible weak learning control, not a main
policy candidate. Its actual adapter hash
`711989faf2330bfca98a837d0afd8024598d085e2a5003a4ca1f0096e1a7f05e`
matches the baseline and its retained S3 model is AES256 encrypted. The
frozen SFT hash was verified unchanged on the worker. Dense RL started its
CPU state-bank preparation at 07:39:28 UTC; no learning gain is claimed yet.

The dense run completed at 08:12:27 UTC: 512 updates, 8,192 completions,
1,978.48 seconds cumulative, 3.94 GiB peak allocated CUDA memory and identical
before/after reference hashes. Its retained adapter hash is
`802aca536dd55c7216d5d9f899f9d8f6ce7819eecd3d758a96b29385c9148ba0`.
Strict greedy development evaluation began at 08:12:28 UTC and was still
active at 08:19. Sampled training parse rate (99.988%) and legality (96.912%)
are temperature-1 diagnostics, not candidate greedy guardrail results. Do not
advance to E3 until the complete paired E2 learning gate passes.

E2 v2's learning gate passed at 08:21:24 UTC and was reproduced locally.
Mean paired fixed-state reward gain was 0.40167 (median 0; bootstrap 95% CI
[0.0025, 0.97542]), exceeding the registered 0.05 threshold. Fixed-state
parse/legality remained 100%; long-game legality rose from 99.126% to 99.324%.
Mean lines rose from 36.25 to 53.9 and score/100 pieces from 5,205.970 to
5,472.090. All 20 games still died (18 illegal actions, two top-outs), so this
does not establish strong-model improvement or authorize deployment. The
final adapter, evaluations and receipt were synced before normal shutdown.

E3 is now registered in `runs/stage6-e3/rl/registration.json` and orchestrated
by `infra/rl-e3-block.sh`. It compares KL 0.01/0.05/0.1 with identical 256-state
banks, 256 updates, learning rate 1e-6, group/batch four and accumulation four.
Every candidate starts from the exact frozen main SFT, not the E2 adapter.
All three development evaluations must complete. `scripts/select_e3_kl.py`
selects the smallest KL with ceiling parse/legality on fixed and long-game
states, zero long-game deaths, and no recovery parse/illegal-action failures.
It reports all paired score differences but does not select on score. The
block limit is six hours/$6.30; $20/$100 spending limits are unchanged. No
candidate passing means no E4 launch and no unregistered grid extension.

E3 restarted the original worker at 08:27:39 UTC, IP `13.218.190.150`, and
launched at 08:29:06 UTC after all 37 worker tests passed. Its first phase is
KL 0.01 state-bank preparation/training. The shutdown timer expires at
14:28:15 UTC, and registered prior spend is $5.30 (conservative). Inspect
`runs/stage6-e3/rl/block-state.json` and each `runs/rl-e3-kl*/rl/` directory;
do not duplicate the service or restart completed E2. The estimated accrued
Stage 6 spend was about $5.07 at 08:29:35 UTC.

At 08:45 UTC, E3's KL 0.01 candidate had reached update 200/256 with normal
checkpointing; no evaluation or KL selection had finished. While E3 runs,
`scripts/check_e4_pilot.py` and `infra/rl-e4-block.sh` are prepared locally,
not deployed over active E3 source. The local Stage 6 gate now has 44 passing
tests; the source running on AWS remains the independently tested 37-test E3
bundle. E4 has not been registered or launched.

At 09:49:37 UTC, E3 KL 0.01 completed development evaluation in 3,670.11
seconds. Its independently verified individual check is **not passed**:
one illegal-action death among 20 long games, and five illegal-action deaths
among 12 recovery starts (seven recovery deaths total). Fixed-state parse and
legality remain 100%, and long-game parse failures are zero. Mean lines are
788.55; score/100 pieces is 7,914.926 versus SFT 7,913.215, a +0.0216% change
(paired mean +1.7114, median -4.5, 95% bootstrap CI [-35.39, 38.7414]).
This score difference does not override validity failures. Replaying all
1,530 saved legal moves for seed 30,000,007 confirms its next parsed action
`rot=0 x=8` is illegal while the engine is not already terminal.

The adapter and complete evaluation were synced by 09:49:40 UTC. Evidence is
`runs/rl-e3-kl001-seed0/rl/candidate-local-check.json` and `sync-receipt.json`.
The unchanged registered block advanced to KL 0.05 training at 09:49:40 UTC.
KL 0.1 remains queued. Do not select a KL, launch E4, extend the grid or claim
E3 complete from this single candidate; both remaining candidates must finish.

At 11:10:18 UTC, KL 0.05 completed evaluation in 3,712.06 seconds. Its
long games have zero deaths, parse failures and illegal actions, and all 24
fixed states parse and act legally. Recovery still fails: six illegal-action
deaths and two top-outs among 12 starts. All six illegal-action trajectories
were independently replayed from their registered source states; the terminal
actions are parsed but illegal, not artifacts of an already-terminal engine.
The individual candidate is therefore **not passed**. Mean lines are 797.5;
score/100 pieces is 7,913.575, paired mean gain +0.36 (+0.00455%), median +10.3,
bootstrap 95% CI [-43.485, 42.58]. Every result is retained in
`runs/rl-e3-kl005-seed0/rl/candidate-local-check.json`.

KL 0.05's full sync completed at 11:10:21 UTC; the receipt was subsequently
synced and independently checked. KL 0.1 started CPU preparation at 11:10:21
UTC in the same registered block. Two candidates have failed, but E3 remains
open until the third completes and the full-grid selector runs. Do not launch
E4 or change the grid/guardrails on the basis of these results.

Interpretation clarification at 11:39 UTC: the original frozen SFT already
has six illegal-action recovery deaths and two top-outs. This detail was not
explicit in earlier progress summaries. The complete baseline cohort and all
six terminal illegal actions were independently replayed and verified in
`runs/stage6-e0/rl/recovery-baseline-audit.json`. Consequently E3's registered
zero-illegal-action recovery requirement is **stricter than baseline
non-regression**: it requires eliminating a pre-existing weakness, not merely
preserving the baseline. Do not relax or retroactively reinterpret the gate,
but do not claim the recovery failures alone prove RL introduced the weakness.
KL 0.01's new long-game illegal death remains a separate regression. E0's
control-reproduction gate remains passed. KL 0.1 finished training at 11:29:12
UTC and is still evaluating; its adapter has been verified and backed up.

After all E3 candidates complete, download and independently reproduce
`selection.json`, verify retained adapter hashes and encrypted sync receipts,
and close the E3 compute session in the ledger. If E3 passes, register E4:

```bash
python scripts/check_e4_pilot.py \
  --registration runs/rl-e4-seed0/rl/registration.json \
  --register-from-e3 runs/stage6-e3/rl/registration.json \
  --selection runs/stage6-e3/rl/selection.json \
  --prior-stage-spend-usd VERIFIED_CONSERVATIVE_PRIOR
```

Replace the prior-spend placeholder with a conservative ledger estimate that
also covers the next boot. The command refuses failed/incomplete E3 selection,
changed recipe, insufficient budget or overwriting an existing registration.
Deploy the tested E4 source and registration only between blocks, rerun the
worker gate, update/sync the ledger and re-arm a three-hour shutdown timer
before starting `llm-tetris-e4.service` with `bash infra/rl-e4-block.sh`.
Do not use a selected E3 adapter as initialization: E4 starts fresh from the
frozen main SFT, with selected KL, unchanged dense weights and 512 updates.

E4 is bounded to three hours including synchronization ($3.15 allowance),
with a 170-minute workflow deadline and ten-minute sync reserve. It evaluates
the complete development suite plus the original 100-seed/500-piece Stage 5
gate. At 09:00 UTC, E3's first long-game evaluation projected about 59 minutes
plus fixed/recovery probes. E4's not-yet-registered development component
therefore allows 75 minutes rather than 65, still inside the unchanged
three-hour block, 170-minute workflow and $20/$100 limits. No active E3
configuration changed. New Stage 5 metadata records the adapter hash and resolved base revision,
rejects changing weights or existing outputs, and leaves game rules, seeds,
decoding and metrics unchanged. Promotion requires the registered 3% score
gain, all E3 validity checks and Stage 5 non-inferiority. An honest dense null
result is recorded without scaling it; E5 trajectory correctness remains a
separate research question. E7 still requires an eligible pilot.

The deployed evaluator reuses the teacher scores already computed by each
evaluation path instead of making an identical second two-ply search. A
regression test checks exact actions, records, and diagnostics against the
original path. This does not alter completed E0 results.

The worker-local five-minute cron job in `infra/rl-monitor.cron` preserves live
metadata/logs in encrypted S3 without optimizer checkpoints. The ten-minute
scheduled task `complete-stage-6-rl-research` returns to this conversation to
inspect results, fix in-scope failures, and advance eligible experiments. It
must avoid duplicate jobs and stop itself only on genuine completion or when
new user authority is essential. Desktop-side scheduled checks require the
computer and app to remain running; the AWS experiment and cron run independently.

On later restarts, resolve the worker's new public IP and re-arm any shutdown
timer for the registered block duration. Do not assume the first-boot transient
timer survives a stop/start. Track wall-clock AWS spend from launch, not just
trainer-reported time. E3–E7 results are still pending; monitoring and passed
hardware checks are not evidence that Stage 6 research is complete.

### Launch preflight status (2026-09-02 UTC)

The first live AWS preflight used account `566629888938`, IAM user `gpu`.
The retained SFT adapter and Stage 5 artifacts are present in S3. The existing
Stage 4 worker is stopped; its unencrypted root volume was left untouched.
No Stage 6 worker was launched and no GPU experiment was run.

An encrypted `g5.xlarge` launch dry run was denied on `iam:PassRole` for the
existing `LLMTetrisTelemetryRole`. The principal also cannot inspect the AMI,
security group, instance profile, current pricing, or SNS subscriptions. The
monthly $200 project budget exists with notifications at 50%/100% actual and
80% forecast, but its SNS human endpoint has not been verified. This monthly
budget does not replace the stricter $20 pilot / $100 Stage 6 limits.

An account administrator can review and attach
`infra/rl-preflight-supplement-policy.json` to `gpu`. It supplements existing
permissions; it does not grant administrator access or permission to edit IAM,
and restricts role passing to the one existing role and EC2 service. It does not
grant CloudFormation access; after verification, direct EC2 CLI provisioning
can use the same encrypted root, tags, and instance-profile configuration as
the template below. Re-run the complete preflight after the permission change;
a successful dry run is still required. Verify lifecycle/cleanup permissions
and a confirmed human budget-alert subscription before launching long runs.

Evidence: `runs/stage6-aws-preflight/rl/manifest.json` and
`runs/stage6-aws-preflight/rl/aws-cleanup.json`. This is a blocked preflight,
not completed research or a negative RL result.

`infra/rl-instance.template.json` requires an encrypted root volume, delete-on-termination, IMDSv2, a scoped instance profile, and Stage 6 tags. Set the existing budget notification endpoint before launch. The training CLIs reject a wall-clock projection above the registered pilot limit and stop at the measured limit.

Sync retained metadata and final adapters, never optimizer checkpoints:

```bash
python scripts/sync_run_artifacts.py upload --run-id RUN_ID --include-adapter \
  --receipt runs/RUN_ID/rl/sync-receipt.json
```

Delete the CloudFormation stack immediately after synchronization, then write read-only cleanup evidence:

```bash
python scripts/check_rl_cleanup.py --region us-east-1 --out runs/stage6-final/rl/aws-cleanup.json
```

The cleanup check fails if a tagged instance remains pending, running, stopping, or stopped; if a tagged volume is unencrypted; or if an unattached tagged volume persists.

Pass `--cleanup-report` and `--sync-receipt` to `analyze_stage6.py` when writing the final report. The Stage 6 verifier remains `ready`, not `passed`, until both the scientific report and operational evidence are complete.
