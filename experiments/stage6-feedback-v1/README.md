# Phase 1A complete locally; Phase 1B awaiting budget approval

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
