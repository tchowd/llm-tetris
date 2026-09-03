# Launch-ready runtime and spending proposal — awaiting approval

No AWS resources or paid model work are authorized by this file. The requested
approval is for this six-run experiment only, including setup, proof,
evaluation, checkpoint backup, monitoring and cleanup. Historical budgets
and approvals are excluded.

Proposed worker: one On-Demand Linux g6e.2xlarge, L40S 48 GB, us-east-1.
AWS's public EC2 pricing feed was checked on 2026-09-03: $2.24208/hour for
this SKU. Use $2.30/hour for the session ledger including a 100 GB gp3 root
and public IPv4 allowance; recheck before launch. Price source:
https://b0.p.awsstatic.com/pricing/2.0/meteredUnitMaps/ec2/USD/current/ec2-ondemand-without-sec-sel/US%20East%20%28N.%20Virginia%29/Linux/index.json
Rate code HK3A8PU2TSC6EKP6.JRTCKXETXF.6YS6EN2CT7.

Local historical evidence: the completed L40S run took 32,799.63 seconds for
320 updates and 106,727 decisions; mean update compute was 101.19 seconds,
worst observed seconds/decision .63198. Six 32-update runs at that mixed-start
mean imply 5.40 training hours plus model load/export. At all-full trajectories,
98,304 decisions across six runs times the historical worst rate times 1.25
gives 21.57 hours, illustrating the uncertainty if recovery improves markedly.
Allow at most four hours per run; no runtime extension without new authority.

Historical greedy times: 2,812 seconds for 20 x 2,000-piece stress games plus
small probes, 174 seconds for 16 recovery starts at cap 200, 1,528 seconds for
100 x 500-piece Stage 5 games. New evaluation per adapter has 20 x 1,000
ordinary pieces and up to 128 x 200 recovery pieces. Budget about 45–60 minutes
per adapter, seven adapters including SFT, or 5–7 hours. Prefix generation is
local CPU and requires no GPU. Actual throughput depends on survival and CPU
teacher diagnostics; measure the first SFT evaluation before all training.

Expected total: roughly 12–17 worker hours, about $28–39 in the session ledger,
plus up to $5 in incremental backup/logging/transfer/storage allowance.
**Propose an expected budget of $35–45, with a hard incremental limit of $85
and a maximum 34-hour worker session.** At the ledger rate, 34 hours is $78.20;
the $5 ancillary allowance leaves $1.80 margin. These are estimates, not a bill
guarantee. No new allocation or restart may project above the approved limit.
AWS delayed billing is reconciled with a conservative live session ledger.

## Ordered launch procedure after explicit approval

1. Save a new approval annex containing the exact user authorization, approved
   USD cap, hourly rate, maximum session hours, registration SHA-256 and an
   absolute UTC shutdown deadline fixed at launch. The scientific registration
   stays immutable. Revalidate all pinned inputs and the accepted SFT hash.
2. Provision one separately tagged encrypted/delete-on-termination worker/root,
   with IMDSv2 and the existing scoped role. Use a unique launch client token;
   capacity failures do not justify duplicate workers or alternate instance
   types. Verify termination permission and account/region before launching.
   Do not restart or modify Stage 4 or any historical worker.
3. Transfer only registered source, dependencies, original SFT, train seed/bank
   files, new development states, and required Stage 5 metadata. Do not ship
   sealed stress states, new confirmation states, or previous RL adapters.
   Disable only job-disrupting maintenance for this scoped service, with a
   separate absolute shutdown timer that survives trainer failure.
4. Arm a one-minute watchdog independent of the training process: record actual
   launch/stop intervals, projected spend, running resource IDs and backup
   status; stop compute at the earlier dollar or absolute time boundary. Keep
   60 minutes for final sync/cleanup; never rely on AWS Budget alerts to stop
   the instance. Operator monitoring supplements, not replaces, the watchdog.
5. Run the local correctness suite on the worker. On disposable SFT copies,
   prove both estimator gradient directions and token alignment on CUDA, plus
   full-horizon four-update uninterrupted versus 2+2 resume for revised RL.
   Include terminal-illegal and delayed-positive probes, exact trajectory and
   adapter comparisons, frozen-reference equality and at least 15% GPU memory
   headroom. Probe optimizer states/adapters never initialize pilot runs. Use
   fresh training-only starts, preserve evidence, cap all proof work at one hour.
   Require a new GPU-proof artifact; historical GPU proofs do not pass this gate.
6. Evaluate SFT on the new development shards. Record measured runtime and
   conservative remaining-cost projections. Do not change cohorts or scientific
   thresholds in response to SFT results. Abort as incomplete if the complete
   six-run study plus evaluation and cleanup no longer fits.
7. Execute the six commands from `stage6_feedback.py commands` in their fixed
   order, appending the new approval file and session-derived trainer budget
   flags. Each command initializes from SFT. Save optimizer/RNG/scheduler every
   four updates (and on pause), atomically commit completed checkpoint evidence,
   retain at least the latest two complete checkpoints, and copy a complete
   checkpoint to a new encrypted S3 experiment prefix after each save. Do not
   use the historical uploader's optimizer-excluding default for resume backups.
8. After each run, greedily evaluate only update 32 using `eval_feedback.py`.
   Evaluation commits fixed 32-state shards and resumes only whole incomplete
   shards. No opportunistic intermediate evaluations or result-driven stopping.
   A trainer-budget stop or missing run yields an incomplete study, not success.
9. Run `analyze_feedback.py`, audit all trajectories/advantages/schedules and
   aggregate exact and threshold-zero counts. Verify complete encrypted S3
   read-back and hashes of all six final adapters, SFT identity, registrations,
   evaluations, report, logs and latest complete resume evidence before deleting
   local GPU storage. Download final evidence locally. Include unfavorable runs.
10. Terminate only the new recorded worker, verify root deletion and no tagged
    orphan volume/address, close every ledger interval and reconcile available
    billing. Retain backups and scientific artifacts; remove temporary optimizer
    backups after final evidence verification within the ancillary allowance.
    Produce a final scientific and operational report; retain original SFT.

The approval annex, provisioned worker identity, live watchdog, GPU proof and
session ledger are deliberately absent until approval. A CPU test pass does
not make those execution gates passed. If projected cost exceeds the cap, stop
and report incomplete; no reuse of historical spending headroom.
