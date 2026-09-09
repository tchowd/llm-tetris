# Stage 6 feedback pilot — inconclusive

## Decision

The revised `fixed_zero` feedback method did not pass the pre-registered promotion criteria. Keep **Tetra-1.7B** (`Qwen/Qwen3-1.7B`, frozen SFT adapter) as the baseline and do not scale this RL recipe. The final test set was not read (`final_test_access: false` throughout).

The primary metric is the paired reduction in greedy recovery illegal endings, calculated as original `active_group` minus revised `fixed_zero` across 128 fixed difficult states for each training seed. The three seed-pair effects were -2.34, -1.56, and +3.91 percentage points. Their mean is 0.00 points and the registered crossed-bootstrap 95% interval is -5.73 to +5.99 points. That interval includes both a material harm and a material benefit.

| Training seed | Original illegal endings | Revised illegal endings | Original minus revised |
| --- | ---: | ---: | ---: |
| 6201 | 72/128 | 75/128 | -3/128 (-2.34 pp) |
| 6202 | 70/128 | 72/128 | -2/128 (-1.56 pp) |
| 6203 | 74/128 | 69/128 | +5/128 (+3.91 pp) |

The replication gate required at least two improving seed pairs; only one improved. Revised recovery survival did not consistently meet the guardrail: it was 53 versus 56 for seed 6201 and 56 versus 58 for seed 6202, before improving from 54 to 59 for seed 6203. Ordinary gameplay was broadly stable, but seed 6203 had one illegal ending and 19/20 1,000-piece survivors under the revised method, so the ordinary guardrail did not pass either.

## What is confirmed

`active_group` centers and normalizes returns among active trajectories. When an illegal terminal trajectory is the lone survivor, or when all remaining returns are equal, its centered advantage is exactly zero. This is implementation behavior, not a performance conclusion.

`fixed_zero` uses discounted reward-to-go divided by the registered reward scale with an action-independent zero baseline. It does not force failures to be negative; their registered negative terminal reward makes them negative in these trajectories. In all three revised runs, terminal illegal actions had zero zero-advantages and every such action had a negative advantage. In the original runs, 34 of 165 sampled recovery terminal illegal actions were exactly zero, including all 30 lone-survivor illegal actions observed across the three seeds.

Both methods moved the policy by a similar small amount (relative adapter L2 movement 0.000398–0.000435), and every run verified the frozen reference policy hash before and after training. Those observations show the feedback change was applied and training occurred; they do not show that it improves gameplay.

## Evaluation distinction and limits

The advantage counts above come from sampled, temperature-1 training trajectories. The outcome table comes from separately generated greedy evaluations: 128 fixed recovery starts capped at 200 pieces and 20 ordinary games capped at 1,000 pieces. All evaluated runs had zero recovery top-outs and zero parse failures. The small three-seed sample makes the uncertainty interval descriptive rather than decisive, and secondary intervals in `report.json` are exploratory and unadjusted.

## Next step

Do not launch larger RL from this result. Retain the frozen SFT Tetra-1.7B baseline. If this mechanism is revisited, first diagnose why removing the zero-advantage pathology did not translate into stable recovery gains, then pre-register a new independent confirmation design before spending on a larger run.

Machine-readable paired analysis, per-run diagnostics, and all summary metrics are in [report.json](report.json). The experiment registration is [registration.json](registration.json), and the mathematical specification is [feedback-spec.md](feedback-spec.md).
