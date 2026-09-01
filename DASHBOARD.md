# Project operations dashboard

The dashboard is a local, read-only view of repository evidence and AWS read APIs. It never launches jobs, changes IAM, terminates instances, or rewrites run artifacts.

## Run it

```bash
python -m pip install -e .
cd web && npm ci && npm run build && cd ..
uvicorn server:app --reload
```

Open [http://127.0.0.1:8000/dashboard](http://127.0.0.1:8000/dashboard). The game remains at `/`.

For frontend hot reload, run `npm run dev` inside `web/` alongside FastAPI. Vite proxies `/api` to port 8000.

## Evidence commands

Write current-commit Stage 1 and 2 reports:

```bash
python scripts/verify_stages.py --stage 1 --stage 2
```

Write a durable Stage 3 validation report:

```bash
python scripts/validate_dataset.py \
  --data-dir data/batch1 \
  --report-json data/batch1/validation.json
```

Long-running generation, training, open-loop evaluation, and closed-loop evaluation commands append structured events to their run's `events.jsonl`. The dashboard tails these files locally; the same files can be shipped to CloudWatch.

Publish durable run evidence to the private artifact bucket, or retrieve only the small dashboard metadata files:

```bash
python scripts/sync_run_artifacts.py upload --run-id sft-v1 --include-adapter
python scripts/sync_run_artifacts.py download --run-id sft-v1
```

Checkpoint directories are always excluded. Adapter files are transferred only when `--include-adapter` is explicit.

## AWS setup

1. Copy `infra/dashboard.example.toml` to the gitignored `infra/dashboard.toml` and set the profile, regions, log group, and dashboard principal.
2. Attach a policy based on `infra/dashboard-readonly-policy.json` to the local dashboard principal.
3. Attach `infra/instance-telemetry-policy.json` through an EC2 instance role—do not copy static AWS keys to the instance.
4. On the instance, run `infra/bootstrap-cloudwatch.sh /home/ubuntu/llm-tetris` after installing the unified CloudWatch agent.
5. Tag project resources with `Project=llm-tetris`, `Stage=<n>`, `RunId=<id>`, and `ManagedBy=llm-tetris`.

AWS panels fail independently. For example, missing `billing:GetCredits` access produces a visible amber source error without hiding EC2 inventory or quotas.

## Routes

- `/dashboard` — command center and seven-stage stack
- `/dashboard/stages/:stage` — shared stage gate/evidence view
- `/dashboard/runs` — run history and lineage
- `/dashboard/dataset` — batch reconciliation and validation state
- `/dashboard/training` — Stage 4 progress and open-loop gates
- `/dashboard/eval` — Stage 5 policy comparison and artifact-backed replay browser
- `/dashboard/aws` — resources, jobs, telemetry, cost, credits, quotas, and IAM posture
- `/dashboard/issues` — derived red/amber/info queue

All `/api/dashboard/*` responses include `generated_at`, `partial`, `freshness`, and `errors` metadata.
