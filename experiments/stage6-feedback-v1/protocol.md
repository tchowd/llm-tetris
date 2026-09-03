# Feedback pilot registration, Phase 1B

Scientific design fixed before paid work. Registration JSON pins source,
input, adapter and evaluation hashes. Budget authority is a separate annex
referencing that registration hash. No historical spending approval applies.

Six independent runs: methods `active_group` and `fixed_zero`, each at training
seeds 6201, 6202, 6203. Every run loads the exact original SFT adapter
`7d753d616d3b9174f489e103f37193fccda28c46b1c7ac91e2f4e87efde01171`
on Qwen/Qwen3-1.7B revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`.
The frozen reference is a separate nontrainable copy of that same SFT.
No fresh SFT, previous RL initialization, model-size or prompt change.

Each run uses 32 updates, cosine horizon 32, one warmup update, four same-start
trajectories, horizon 128, gamma .99, temperature 1, top-p 1, top-k 0,
16 completion tokens, lr 1e-6, KL beta .05, batch size four, gradient clipping
at 1, and existing AdamW defaults. Reward remains score delta/100 minus
2 on legal top-out or 10 on illegal terminal action. No horizon bootstrap.
Only feedback estimator changes; its scale is fixed at 10. See feedback-spec.md.

Odd updates start empty; even updates use the unchanged training recovery bank
from the existing 192 training seeds. Python's dedicated scheduling RNG is
independent of generation's Torch RNG. Both paired runs therefore use identical
source states, and the registration stores the full 32-start schedule per seed.
Torch seeds are paired; divergent trajectories consume different numbers of
random draws, so later sampled actions are not assumed identical. Update count
is matched, not realized decisions or GPU seconds; report both. Execution order
is original/revised for seed 6201, revised/original for 6202, original/revised
for 6203. All six run regardless of intermediate performance if budget permits.

## Development evaluation and protected confirmation

128 recovery source games, seeds 81,000,000–81,000,127, one state per seed.
For each, use a noisy legal dense one-ply prefix (noise .4), and select the
first live state at height >=16 after >=8 decisions, with at most eight fixed
prefix-RNG attempts and 160 turns per attempt. Selection uses engine features
only, before any SFT or RL evaluation. Prefix RNG seed is source_seed*100+attempt.
No teacher continuation labels are added to training. Recovery cap: 200 legal
placements. Report illegal endings, legal top-outs, capped survivors, actual
legal placements, score and lines separately; a reduction in illegal endings
that merely substitutes top-outs is not recovery improvement.

Ordinary gameplay: 20 distinct source seeds 82,000,000–82,000,019, 1,000 legal
placements each. This is a pilot regression guard, not full Stage 5 or production
qualification. Existing Stage 5 and old 12/16 recovery results remain historical
references, not independent confirmation or the primary comparison.

Evaluate original SFT once and all six final-update adapters on identical cases,
strict greedy decoding, batch size 32, same pinned base and tokenizer. Fixed
32-state shards permit restart at shard boundaries without changing batching.
All metrics use continuation-only score and lines. Illegal attempts count as
failures, not legal survival. No action substitution or constrained decoding.

New source ranges [83,000,000,83,000,256) and [84,000,000,84,000,100) are
reserved for a future independent confirmation; do not generate or evaluate
them here. The existing final test remains sealed: no final-test state file is
opened, copied to the worker, evaluated, or used for selection. New ranges are
disjoint from Stage 3, Stage 5 and the existing stress seed partitions. Related
prefixes from a source game cannot cross partitions. Any follow-up needs its
own prospective registration; pilot advancement is not deployment approval.

## Selection, analysis and decisions

Only update 32 from each seed/arm is eligible. No best-checkpoint or best-seed
selection; intermediate checkpoints serve recovery only. Incomplete runs remain
incomplete, with no substituted seed, shortened endpoint, or complete-case win.

Primary endpoint: original-minus-revised absolute recovery illegal-ending rate,
averaged equally across the three seed pairs and 128 source games. Minimum
useful effect: 10 percentage points (about 13/128 cases per seed, not one case).
Retain each paired game difference and every per-seed mean.

Uncertainty: 10,000 crossed paired bootstrap replicates, NumPy RNG seed 73220;
resample training seed pairs and source games independently, preserving the
same sampled game indexes across all sampled pairs. Report 95% percentile
intervals and per-seed effects. Only three training seed pairs limits population
inference; this is not a powered confirmatory trial. Worst-case paired binary
SE across 128 independent games alone is 1/sqrt(128)=8.8 percentage points;
positive pairing may reduce it, seed variability may increase it. This pilot
can remain inconclusive even at the minimum useful effect.

Advance only if mean reduction >=10 points, the primary interval lower bound
is >0, at least two of three seed effects are positive, and all gates pass:

- Every revised terminal illegal action has negative advantage and neither
  exact nor effective zero (abs(A)<1e-10). All updates and gradients finite,
  frozen reference unchanged, replay/schedule/provenance checks passed.
- Each revised seed's recovery survival count is at least that of its paired
  original run and SFT; fewer illegal endings cannot mask lower survival.
- Each revised seed's ordinary mean score and lines are >=99% of both paired
  original and SFT. Both methods have zero ordinary illegal endings, top-outs
  and parse failures. Any control failure makes advancement unsafe.

Classify `helps` only when all advancement criteria pass; `hurts` when the
primary interval upper bound is below zero; otherwise `inconclusive`, stating
any observed regressions or failed safety gates explicitly. Secondary paired
intervals cover survival, top-outs, pieces, score and lines, revised minus
original; these are exploratory and unadjusted for multiplicity. Report SFT
levels and each arm's absolute results alongside the between-arm differences.

Training diagnostics are temperature-1 observations, never called greedy
performance: exact/effective zero counts with denominators, broken out by
empty/recovery starts, terminal illegal actions, terminal top-outs, single-active
steps and single-active illegal actions. Also retain reward variance, gradient
norm before clipping, clipping incidence, sampled k3 statistic, adapter L2 and
changed-tensor counts, sampled decisions and wall time. Compare coefficient
distributions; fixed-zero changes magnitude and regularization balance too.

No automatic larger run follows any outcome. For an inconclusive result,
identify whether uncertainty, short exposure, variance, or regression merits a
bounded follow-up; do not claim the old feedback issue caused prior performance.
