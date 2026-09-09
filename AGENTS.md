# Project guidance

## Model identity

- **Tetra** is the product/model-family name for this project's fine-tuned Tetris policy.
- Call the accepted historical model **Tetra-1.7B**. Its frozen artifact remains at
  `runs/sft-v1/adapter` and uses `Qwen/Qwen3-1.7B`.
- Call the Iteration 3 candidate **Tetra-0.6B**. It uses
  `Qwen/Qwen3-0.6B` and is not the accepted model until it passes the documented
  comparison gates.
- Use technical names in manifests as well as friendly names. “Tetra” never
  replaces the exact base-model ID, revision, run ID, or adapter hash.

## Planning and provenance

- `plan/original/` is the historical plan. Do not rewrite it to make later
  decisions appear original.
- `plan/iteration-3/` is the active plan for the Tetra-0.6B combined-data
  quality-transfer experiment. It is not a pure model-size ablation.
- Never rename, overwrite, or continue training `runs/sft-v1/adapter`. It is the
  frozen Tetra-1.7B comparison baseline and is referenced by historical hashes.
- Give every new training or evaluation run a new output directory. Commands for
  Tetra-0.6B must pass `--base-model Qwen/Qwen3-0.6B` explicitly.
- The primary Iteration 3 run is `tetra-qwen3-0.6b-combined-v1`. It starts from
  a fresh pinned Qwen3-0.6B base and uses all training rows from `data/batch1`,
  `data/batch2`, and `data/tetra17-recovery-v2`.
- Do not describe Tetra-0.6B as promoted, equivalent, or better until the
  open-loop, closed-loop, and stress comparison gates have actually passed.

## Experiment discipline

- Keep the engine, teacher, each source dataset's split, prompt serializer,
  chat template, decoding settings, and evaluation seeds unchanged. Preserve
  and hash the exact combined row composition and order.
- Never train on the 4,096 `split=eval` rows in
  `data/tetra17-recovery-v2`. Report that cohort separately from the original
  held-out evaluation rows.
- Preserve the imported dataset name, manifest, hashes, and provenance. Its
  72,000 on-policy correction rows came from Tetra-1.7B, not Tetra-0.6B.
- Do not attribute a Tetra-0.6B versus Tetra-1.7B performance difference to
  model size alone because their training corpora differ.
- Re-run the frozen Tetra-1.7B control with the same evaluation code,
  dependency environment, batching, and boards used for Tetra-0.6B. Historical
  headline numbers alone are not a matched control.
- Treat the 500-piece Stage 5 suite as an ordinary-play regression gate, not a
  sufficient quality benchmark; it is saturated. Include long ordinary games
  and a substantially sized recovery suite in every promotion decision.
- Separate illegal actions, legal top-outs, capped survival, score, and lines.
  Never report fewer illegal actions as improvement if they merely became
  legal top-outs.
- Never select the best checkpoint or best training seed after looking at
  confirmation results. Pre-register the eligible endpoint and replication
  seeds.
- Do not use an inspected development failure or any reserved confirmation
  state as training data. Generate analogous examples from training-only seeds.
- Pin and record the Hugging Face model revision before a full training run.
- A smoke adapter or partial-data pilot is diagnostic only and must never be
  promoted as the final Tetra-0.6B model.
- Preserve raw generated actions and per-game records so every failure can be
  replayed against the engine.
- Do not launch paid cloud training or modify AWS resources without an explicit
  user request to start the run.
