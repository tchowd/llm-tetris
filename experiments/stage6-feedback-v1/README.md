# Phase 1A complete; Phase 1B pilot running

Current update, September 4 UTC: Phase 1A is implemented and validated. The
registered Phase 1B comparison is running sequentially on one retained L40S
worker in `us-east-2`, under the user's unchanged **$250 total hard limit**.
The original and revised seed-6201 runs, common SFT evaluation and GPU proof
are complete and fully audited. The first paired recovery result favors the
original method by 2.3 percentage points; this is interim evidence and not the
three-pair preregistered conclusion. The revised seed-6202 run is complete and
passed its training, recovery and ordinary-play audits. Its paired original
seed-6202 run completed training and difficult-state evaluation and is running
its ordinary-game guard. Both completed recovery pairs favor original RL, so
the registered 2-of-3 replication gate cannot pass. The retained runner will
finish this run and the other two registered runs, perform the
prospective analysis, back up the artifacts with AES256 and verified hashes,
and shut down. See [the execution handoff](session/HANDOFF.md).

The scientific registration remains frozen at SHA-256
`ca95de9c9849cfe0c2038b39a167d1fb8b062c60f0e6df065ea343901fc194e2`.
The final test remains sealed. Current local scientific and operations
validation passes 104 tests with no failures or skips; see
[`session/current-validation.json`](session/current-validation.json).

Earlier quota, capacity, proof-only and multi-worker notes below are retained
as historical execution evidence. They are superseded by the current retained
one-worker execution and do not describe the live state.

Update, September 3 at 23:25 UTC: the user approved the separate GPU correctness
check only, capped at $25. The new launch permission passes dry-run, but actual
L40S launches were rejected for capacity across all four eligible zones.
No instance or paid compute started. The isolated proof runner and 91 local
tests pass; GPU correctness remains untested. See [GPU-proof status and handoff](gpu-proof-v1/README.md).
This does not approve the six-run Phase 1B pilot described below.

The implementation, 128-case development recovery set, six paired-run
registration, evaluation/analysis tools and cost proposal are ready for review.
No AWS resource was provisioned and no paid compute started. GPU correctness
and performance are not yet established. The untouched final test stays sealed.

## Feedback change and evidence

- [Mathematical specification](feedback-spec.md): selectable fixed-zero baseline,
  A=discounted reward-to-go/10; original active-group method remains default.
- [Historical reproduction](historical-replay-check.json): update 6's lone
  survivor ends with reward -10, old advantage 0 and revised advantage -1.
  Every original coefficient reproduced exactly. This proves coefficient
  behavior, not improved model performance.
- [Local validation](local-validation.json): 86 tests passed, no failures/skips.
  Both arms have exact CPU uninterrupted/resume trajectory and weight equality;
  tests cover signs, delayed rewards, illegal actions, identical outcomes,
  loss weighting, token alignment, reference freezing and registration guards.
- Original source files are retained under `prechange/`; unrelated edits and
  completed-run artifacts were preserved.

## Registered pilot

[Protocol](protocol.md), [machine-readable registration](registration.json),
and [six command templates](training-commands.json).

Both methods independently load the exact original SFT for training seeds
6201/6202/6203, 32 updates each, matching starting-state schedules and all other
training settings. Greedy evaluation uses 128 recovery source games (cap 200)
and 20 ordinary games (cap 1,000), plus the original SFT as a common control.
Only final update 32 is eligible. Success requires a useful repeatable recovery
gain, paired uncertainty above zero, correct feedback, and gameplay guards.
Three seed pairs make this a diagnostic pilot, with an explicit inconclusive
outcome; it does not qualify a replacement model.

Registration SHA-256:
`ca95de9c9849cfe0c2038b39a167d1fb8b062c60f0e6df065ea343901fc194e2`.

## Spending proposal

[Runtime, cost and operational procedure](operations.md): one L40S worker,
approximately 12–17 hours expected, **$35–45 expected cost**, **$85 hard
incremental spending limit**, and **34-hour maximum worker session**.
This includes setup, bounded GPU validation, six training runs, seven greedy
evaluations, monitoring, backup and cleanup. A fresh GPU measurement must show
the complete workflow fits before all six runs launch.

Approval must explicitly cover this new budget and hard limit. Existing Stage 6
spending approvals are not reused. Once approved, create the operational annex,
arm the independent budget/deadline watchdog, pass the GPU proof, execute the
registered study, and report helps/hurts/inconclusive with cost and cleanup
evidence. The local pass is not a claim that the paid pilot is complete.
