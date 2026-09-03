# Phase 1A: prospective feedback specification

Written before implementing the candidate, 2026-09-03. No compute approval.

Let G_it = sum_{k=t}^{T_i-1} gamma^(k-t) r_ik, with no bootstrap beyond
the existing rollout cap. The historical estimator subtracts the mean G at
the same relative turn over surviving trajectories and divides by population
standard deviation (epsilon 1e-8). For one survivor, or equal returns, this
is zero. Removing the denominator cannot fix mean subtraction.

Candidate `fixed_zero`: A_it = G_it / 10. Baseline b(s,t)=0; the positive
constant 10 is the already registered illegal-action penalty magnitude and
is fixed for this experiment, never estimated from the sampled group.
`active_group` remains the default and reproduces the old decimal50 formula.

For a trajectory score-function gradient, an action-independent baseline
satisfies E[b(s) grad log pi(a|s)] = b(s) grad sum_a pi(a|s) = 0.
Consequently zero is a valid (though potentially high-variance) baseline;
positive constant scaling preserves the unregularized reward objective's
direction. This removes self-comparison without fitting a critic or adding a
new reward. At an illegal terminal action G=-10, so A=-1 regardless of other
survivors. Equal positive outcomes receive positive credit; equal zero returns
still correctly yield zero. A legal top-out can have positive net reward if
score exceeds the death penalty. Earlier actions in an ultimately failed
trajectory can have positive return. No failure-label sign override is used.

With gamma=.99, rewards [0, 0, -10] give [-.9801, -.99, -1], while [0,0,10]
gives [.9801,.99,1]. These signs are a consequence of discounted rewards.
They are not a guarantee that every sampled action in a losing trajectory was
locally bad. Equal-reward trajectories may produce cancelling gradients in
expectation; a nonzero coefficient does not guarantee useful learning.

The unchanged loss is the mean over sampled decisions of each completion's
mean token loss: -A log pi + beta*(exp(log pi_ref-log pi) -
(log pi_ref-log pi) - 1). Each decision weighs 1/N, each of its L tokens
weighs 1/(N L); a trajectory with T decisions weighs T/N. Chunking must not
alter those weights. The action log probability is token-averaged, the random
decision count is the denominator, and there is no outer gamma^t factor.
Thus this implementation is a practical surrogate, not an unbiased estimator
of the exact initial-state discounted-return objective. Both arms keep these
limitations, AdamW defaults, gradient clipping, and KL identical.

Fixed scaling changes coefficient magnitude and its balance with beta=.05;
the pilot tests the complete specified estimator, not a pure baseline-only
causal effect. No post-hoc scale, learning-rate, or KL tuning is allowed.

At pi=pi_ref the k3 term has zero derivative. For a negative A, minimizing
the policy term lowers the sampled token's log probability; for positive A
it raises it. Away from reference the regularizer and other samples can oppose
that direction. A zero coefficient removes only the direct policy term;
earlier reward-to-go, shared parameters, weight decay and KL can still learn.
The k3 sample statistic is not an exact full-vocabulary KL measurement.

Confirmed: the old zero-coefficient behavior and loss weighting follow from
the implementation. Hypotheses: fixed-zero feedback improves recovery,
its variance is tolerable, and 32 updates produce measurable greedy changes.
Only the controlled pilot can address those hypotheses. CPU proofs are not
GPU training or checkpoint-recovery proof.

Original source copies are retained in `prechange/`; completed experiments,
their registrations, reports and model artifacts are not rewritten.
