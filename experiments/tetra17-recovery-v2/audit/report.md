# Tetra-1.7B recovery data audit

## Technical summary

The frozen Tetra-1.7B SFT data is valid and internally consistent in the checks
performed, but its difficulty distribution does not match the recovery problem.
Of 1,005,785 rows, 86.57% are low difficulty, 8.92% moderate, 0.92% hard, and
3.60% critical. The previous recovery add-on then concentrated 99.63% of its
full-state examples in the critical band. This leaves the transition from
ordinary play into difficult recovery poorly represented.

The confirmed development behavior is also specific: 57/128 greedy recovery
games reached the 200-piece cap; 71/128 ended on parsed illegal actions. There
were no unparsable terminal actions or legal top-outs in this cohort. These
facts justify testing a progressive curriculum. They do not establish that
coverage imbalance caused the failures or that new SFT data will improve them.

## The missing middle is the main measured coverage gap

| Source | Low | Moderate | Hard | Critical | Full-state rows |
|---|---:|---:|---:|---:|---:|
| Original Stage 3 | 870,680 | 89,704 | 9,208 | 36,193 | 1,005,785 |
| Prior recovery add-on | 2 | 9 | 6 | 4,591 | 4,608 |

Difficulty bands are mutually exclusive: critical is maximum height 16+ or six
or more holes; hard is height 13–15 or three to five holes; moderate is height
8–12 or one to two holes; all remaining states are low.

All seven current and next pieces are closely balanced in the original data.
Rotations 2 and 3 have 120,218 and 87,992 rows, respectively. A deterministic
2,000-row teacher sample had 2,000 legal, teacher-optimal labels; 742 selected
placements touched a side boundary. This supports targeted difficulty work
without changing the teacher, piece representation, or base model.

## Identity, duplication, and leakage checks passed with one representation caveat

The audit found no missing required fields, malformed boards, unparsable stored
completions, completion/label mismatches, state-label conflicts, prompt-label
conflicts, or train/eval game/seed overlap. The original corpus contains 980,157
unique full states and 817,056 unique serialized prompts.

There are 104,245 prompt signatures that correspond to more than one full board
geometry. This is expected because the current input serializes aggregate
column features rather than board cells. None carried conflicting canonical
teacher labels in this corpus. It remains a plausible contributor to behavior
on unseen recovery boards, but the current task explicitly holds the input
representation fixed, so the experiment records this as a limitation rather
than changing it.

## Development failures are held out from generation

The failure profile reads four frozen evaluation shards and emits aggregates
only. It does not emit boards or action prefixes. All 128 starting states were
critical. Mean continuation length was 108.70 pieces, and every one of the 71
failures was a parsed illegal terminal placement. The new generation ranges
start at seed 30,000,000; the inspected development cohorts use 81,000,000 and
82,000,000 series seeds. Confirmation ranges 83,000,000 and 84,000,000 and the
final test remain sealed.

## Registered intervention and decision rule

The candidate dataset contains 240,000 training rows: 96,000 clean ordinary
rows, 48,000 progressive recovery rows, 24,000 safety rows, and 72,000 rows
from one frozen-model on-policy correction round. A separate 4,096-row
progressive evaluation split is held out from training.

The fresh SFT candidate starts from pinned `Qwen/Qwen3-1.7B` revision
`70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` with a newly initialized LoRA.
It passes only if recovery cap-outs rise from the matched control to at least
73/128, the paired gain is at least 10 percentage points with a bootstrap 95%
lower bound above zero, all 20 long ordinary games and all 100 Stage 5 games
survive, ordinary score and lines retain at least 99%, and open-loop parse,
legality, and exact-match gates pass. RL is skipped unless every SFT gate passes.

## Limitations and next check

The teacher sample is deterministic but sampled; full teacher replay validation
will run on the assembled dataset before training. The failure audit is
descriptive and the proposed coverage mechanism is a hypothesis. The decisive
evidence will be the matched greedy evaluation, with sampled on-policy training
behavior reported separately. The final test remains untouched during this
development decision.

Source: [`data-and-failures.json`](data-and-failures.json), generated from the
two original Stage 3 row files, the prior recovery dataset, and the frozen
development evaluation shards. As of 2026-09-08.
