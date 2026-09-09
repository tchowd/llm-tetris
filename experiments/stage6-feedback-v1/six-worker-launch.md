# Six workers, $250 total limit — approved, AWS prerequisites blocked

User authorization: “actually, let’s do it with six workers and increase the
budget to 250”. This supersedes the earlier one/three-worker execution choices
and $85 limit. The six scientific runs, original SFT initialization, training
settings, evaluation sets, final-checkpoint selection and statistical criteria
stay unchanged. The original scientific registration is preserved. The new
[approval annex](budget-approval-v2-six-workers.json) records concurrency and
budget separately; no historical spending allowance is reused.

One identical g6e.2xlarge L40S worker per registered run, six workers maximum.
Run the disposable correctness proof and common SFT evaluation on the first
worker before releasing the six optimizations. Every optimization initializes
independently from original SFT; no proof checkpoint is reused. Launch the
remaining five workers only after the gate passes, limiting idle cost. Both
methods retain identical hardware types but now use separate physical GPUs;
record hardware/software identities and deterministic-kernel checks on each.

Estimated completion: **4–6 hours after capacity is available**. AWS quota
approval and capacity waiting are additional and cannot be predicted here.
Six workers cost approximately $13.80/hour when all are running, at the
conservative $2.30 per-worker rate. Faster elapsed time does not mean less total
compute; setup, proof-gate waiting and duplicated caches can add overhead.

Hard incremental spending limit: **$250 including proof, setup, all six runs,
seven evaluations, storage, backup, monitoring and cleanup**. Set one global
deadline 16 hours from the first successful launch and never reset it. Every
worker gets that same absolute deadline and a fallback 16-hour boot timer.
Maximum 96 aggregate worker hours at $2.30 plus $15 ancillary reserve is
$235.80, leaving $14.20 margin. Stop earlier if measured or projected cost
exceeds the remaining allowance. Reserve the last hour for backup and cleanup.
The live ledger must aggregate all six resource IDs, including boot, idle,
failure and retry time; there is no $250 allowance per worker.

## Administrator actions needed

Live AWS checks on September 4, 2026 UTC found:

1. `gpu` can launch only the earlier proof/scale experiment tags. Pilot launch
   dry runs return `UnauthorizedOperation`. The new policy below adds only
   instance launch authorization for the six registered run IDs, L40S type,
   existing role, region, project tags and required IMDSv2.
2. The us-east-1 **Running On-Demand G and VT instances** quota is **8 vCPUs**.
   Six g6e.2xlarge workers need **48 vCPUs**. Requesting the increase returned
   `AccessDeniedException`; the current identity can read but cannot raise it.

Run these from an AWS administrator identity for account `566629888938`:

```bash
aws iam put-user-policy \
  --user-name gpu \
  --policy-name LaunchOnlySixStage6FeedbackPilotRuns \
  --policy-document file:///Users/turjchowder/Github/llm-tetris/infra/rl-feedback-six-worker-launch-policy.json

aws service-quotas request-service-quota-increase \
  --region us-east-1 \
  --service-code ec2 \
  --quota-code L-DB2E81BA \
  --desired-value 48
```

The second command requests an increase; AWS must approve it before all six
workers can run. Do not repurpose earlier experiment tags to bypass the launch
restriction. No identity policies were changed by the assistant. No new worker
exists and no paid compute has started; the existing Stage 4 worker stays stopped.

After the administrator applies the policy and AWS approves the quota, recheck
live permissions, quota and current instance inventory before launching. Reuse
the six saved requests/client tokens; reconcile uncertain launches before
retrying and never create a duplicate for a run. Preserve all capacity failures.
Then complete the approved GPU gate, six-run workflow, aggregate cost tracking,
encrypted evidence read-back and exact-resource cleanup. The earlier proof-only
launch helper must not be used to start or authorize the pilot.

Evidence: [six-worker preflight](six-worker-preflight.json),
[quota request denial](six-worker-quota-status.json), and the proposed policy
at `infra/rl-feedback-six-worker-launch-policy.json` in the repository.
