# GPU correctness proof: prepared, blocked on AWS capacity

Latest check: September 3, 2026, 23:25:33 UTC.

The user authorized step 1 only, with a $25 maximum. The two-hour worker lifetime
and one-hour GPU-proof bound remain unchanged. The six-run pilot, fresh SFT,
large RL and final-test access are not authorized.

The new inline IAM policy `LaunchOnlyStage6FeedbackGPUProofL40S` matches the
requested narrowly scoped permission. Complete launch dry-runs pass. Actual
`g6e.2xlarge` requests failed with `InsufficientInstanceCapacity` in us-east-1c,
us-east-1b, us-east-1d and us-east-1a, followed by another failed us-east-1c retry.
SDK retries occurred within these calls. AWS independently confirms no proof
instance exists. No compute session has started and no compute spend occurred.
Stage 4 remains stopped and untouched. This is not a failed GPU correctness test:
the GPU test has not run.

## Prepared evidence

- `registration.json`: immutable proof-only inputs and procedure, SHA-256
  `5f148409d52f0845d5a8c1b1e2ecdb9dc1b7d4dbfe71b0bdfab3f9ef48a952de`.
- `local-validation.json`: 91 passing tests (12 library warnings), no failures.
- `launch-request-*.json` and `launch-failure-*.json`: requests and capacity failures.
- `capacity-fallback-v1.json`: original permitted us-east-1a subnet fallback,
  without changing the frozen scientific inputs or original helper source.

Source archive: `/var/folders/d9/jwtpgh6n03j21g73kxwd__r40000gn/T/llm-tetris-feedback-proof-d6uv0w2o/source.tgz`.
SHA-256: `7b83cf7aca1c1dd4ca87331add155c20ad796e62400c3932d9e1646dac371eac`.
Contains 91 allowlisted files, including original SFT, but no sealed stress test
states. Preserve original pilot registration and all historical evidence.

## Resume when capacity is available

1. Revalidate with `.venv-train/bin/python infra/feedback_gpu_proof.py validate`.
   Inspect AWS and the proof ledger before any launch; reconcile lost responses
   instead of creating duplicate workers. Reuse the saved idempotent request.
2. `infra/feedback_gpu_ops.py launch` supports the three explicitly registered
   alternate-zone subnets. The saved us-east-1a request is a separately recorded
   operational fallback. Only one instance may exist. Do not change hardware.
3. Copy the verified archive and the newly created ledger to the worker, extract
   under `/home/ubuntu/llm-tetris`, and run `infra/feedback-gpu-setup.sh`. Verify
   its absolute shutdown timer, pinned offline model, CUDA compiler and tests.
   The setup script does not start optimization. Do not reset the launch deadline.
4. Start `infra/feedback_gpu_ops.py run` as the bounded `feedback-gpu-proof.service`
   with its working directory set to the repo, `.venv-rl/bin/python`, offline
   Hugging Face flags, `OMP_NUM_THREADS=1` and `TOKENIZERS_PARALLELISM=false`.
   This first checks actual AdamW update directions on disposable SFT copies,
   then runs four revised updates uninterrupted and a separate 2+2 process
   resume, followed by independent token/reward/weight/optimizer/RNG verification.
   The one-hour workflow deadline includes all these GPU phases.
5. Observe current service/logs. Failure must remain failure or incomplete;
   never tune gates after observing GPU outcomes. The workflow uploads evidence
   and stops the worker. Setup/preflight failures require operator stop handling.
6. Independently download and audit actual AES256 S3 bytes under
   `runs/feedback-gpu-proof-v1/`, close the ledger after confirmed stop, then
   terminate only the proof worker after verified backups. Verify exact root
   deletion and Stage 4 untouched, and report the observed result. No automatic
   pilot follows. No recurring monitor is currently scheduled for this proof.

Current public EC2 Linux L40S rate was rechecked at $2.24208/hour. The ledger
allowance is $2.30/hour plus $5 ancillary reserve. Two hours at that allowance
plus reserve is $9.60, below the user's $25 hard ceiling; AWS billing is authoritative.
