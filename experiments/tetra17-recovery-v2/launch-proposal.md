# Tetra-1.7B recovery-v2 launch proposal

## Execution

- Primary worker: one `g6e.2xlarge` in `us-east-1`, on-demand for reliability.
- The same worker runs generation, SFT, matched evaluation, and any conditional
  RL sequentially. This keeps one hard deadline and one cost ledger.
- Conditional RL starts only if the SFT gate passes; each run checkpoints every
  update.
- All artifacts sync to a new `tetra17-recovery-v2/` S3 prefix. Instances are
  terminated after verified read-back or any failure/budget stop.

The current us-east-1 on-demand rate is $2.24208 per g6e.2xlarge-hour. Recent
spot observations were $2.0601–$2.2291/hour, too small a discount to justify
interrupting this checkpointed but sequential workflow.

## Runtime and cost

| Work | Elapsed estimate | GPU instance-hours |
|---|---:|---:|
| Teacher curriculum + on-policy collection | 4–6 h | 4–6 |
| Fresh one-epoch SFT | 3–4 h | 3–4 |
| Matched control/candidate evaluation | 4–6 h | 4–6 |
| SFT analysis and cleanup | <1 h | <1 |
| Conditional three-seed RL + evaluation | 5–7 h | 5–7 |

Expected completion is 12–17 hours if the SFT gate fails and 17–24 hours if it
passes and conditional RL runs. The expected EC2 charge is about $30–$43 before
conditional RL and $42–$59 with it. A $75 hard experiment limit provides room
for EBS, S3, setup, checkpoint/read-back, and runtime variance. The launcher
must stop new work before projected accrued cost can exceed that limit.

No historical approval applies to this experiment. Provisioning requires a
new approval annex containing the registration hash, `$75` (or another amount
the user chooses) as the hard limit, and an absolute deadline.
