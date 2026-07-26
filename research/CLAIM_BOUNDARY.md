# Claim boundary

## Current claim

For each supplied group, CliffQuant evaluates every positive finite IEEE binary16
stored scale and returns the scale with the lowest computed frozen
multi-environment diagonal reconstruction objective. Exact floating-point ties are
resolved by the lower FP16 bit pattern.

The implementation is tested against an independent pure-Python oracle that also
enumerates all 31,743 scales.

## Fixed scope

- One scale per supplied weight group.
- Round-to-nearest, ties-to-even, followed by integer clipping.
- Positive finite FP16 stored scales.
- Finite weights and finite, non-negative diagonal curvature values.
- Float64 proxy computation.
- No hidden environment normalization.

## Not claimed

- A faster search algorithm.
- Optimal model loss, perplexity, KL divergence, accuracy, or downstream quality.
- Full-model runtime or memory improvements.
- Robustness outside the supplied frozen environments.
- Optimality over other quantizers, zero-points, groupings, or scale dtypes.
- Novelty, state of the art, or a breakthrough.

Those questions require separate frozen experiments and evidence.
