# AutoPolicy 001 method

## Purpose

AutoPolicy 001 is a development measurement built from CliffQuant Experiment
001. It asks a narrower question than a mixed-bit model benchmark:

> Given a recorded W4, W8, and unquantized BF16 candidate for every frozen
> target matrix, which discrete assignment minimizes a recorded
> multi-environment reconstruction proxy under a logical target-payload budget?

This is not a preregistered confirmation experiment. Its role is to validate the
planner, artifact chain, byte model, and robust objective before any mixed-bit
checkpoint is exported.

## Frozen inputs

- Base model: `Qwen/Qwen3.5-0.8B-Base` at revision
  `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`.
- The 150 language-tower matrices and 128-weight grouping already frozen by
  Experiment 001.
- The four separately normalized calibration environments already frozen by
  Experiment 001: general, code, math, and multilingual.
- The validated CliffQuant W4 scale run and its exact stored FP16 scales.

The builder refuses model, calibration, module-inventory, or scale-run identity
drift.

## Candidate bank

Each matrix receives exactly three choices:

| Candidate | Codes | Stored scale | Distortion |
|---|---:|---|---|
| `w4_cliffquant` | signed INT4 | one FP16 scale per 128 weights | direct re-evaluation of the validated exact CliffQuant scales |
| `w8_absmax` | signed INT8 | one FP16 scale per 128 weights | sign-aware no-clipping AbsMax target, rounded to the nearest positive finite FP16 scale |
| `dense16` | unquantized BF16 source reference | none | zero reconstruction distortion by definition |

W8 uses round-to-nearest, ties-to-even, followed by clipping to `[-128, 127]`.
It is an AbsMax control, not an exact W8 scale optimizer.

For every candidate, the builder records:

- the sum of groupwise worst-environment normalized diagonal squared error; and
- the module-level squared-error total for each environment separately.

## Allocation granularity

The measured candidate bank contains all 150 target matrices. AutoPolicy exposes
two exact views over that table:

- matrix-level allocation for the additive `sum-groupwise-max` scalar
  objective; and
- a 24-transformer-block view for the four-environment robust-minimax
  objective, where all matrices in a selected block use the same candidate.

The block view sums the recorded matrix distortions and logical bytes without
changing them. It makes the exact multidimensional frontier tractable while
keeping the precision choice deployable at a clear architecture boundary. The
published AutoPolicy 001 frontier uses this block view. The full 150-matrix
robust search remains available as a fail-closed research path, but it is not
presented as the practical default.

## Robust allocation objective

For unit \(u\), candidate \(c\), environment \(e\), recorded distortion
\(d_{u,c,e}\), and logical byte cost \(m_{u,c}\), the robust planner solves

$$
\min_x \max_e \sum_{u,c} d_{u,c,e}x_{u,c}
$$

subject to

$$
\sum_c x_{u,c}=1\quad\forall u,\qquad
\sum_{u,c}m_{u,c}x_{u,c}\le B,\qquad
x_{u,c}\in\{0,1\}.
$$

For the published robust frontier, \(u\) is a transformer block. The
implementation keeps a componentwise byte/environment Pareto frontier. If the
exact search exceeds its configured state, transition, or Pareto-comparison
work limit, it aborts instead of silently returning an approximation. Objective
ties prefer the lower sum over environments, then fewer bytes, then
lexicographically smaller candidate IDs in sorted unit order.

## Logical byte model

For an integer candidate, the logical payload is:

$$
\left\lceil\frac{\text{weights}\times\text{bits}}{8}\right\rceil
+ 2\times\text{groups}.
$$

The unquantized BF16 source reference uses two bytes per target weight.

The model deliberately excludes biases, zero buffers, `g_idx`, packing padding,
non-target tensors, metadata, filesystem overhead, activations, KV cache, and
runtime workspace. Therefore the budget is called **logical target-payload
bytes**, not checkpoint size or runtime memory.

## Verification gates

- Every recomputed W4 group objective must match the validated stored objective
  within a strict float64 numerical tolerance.
- W8 vectorized results are tested against an independent scalar oracle.
- Candidate JSON and every W8 array archive are content-addressed.
- Candidate and plan JSON use strict duplicate-key rejection, finite-number
  validation, canonical encoding, and self-fingerprints.
- Plan verification reloads the exact candidate file and recomputes the exact
  allocation.
- Budgets below the all-W4 logical payload must fail closed.
- The all-W4 and all-BF16 endpoints must select their respective uniform
  policies.

## Claim boundary

AutoPolicy 001 can support only this claim:

> At the selected allocation granularity, the solver returns the exact optimum
> of the recorded discrete proxy table under the recorded logical-byte budget
> when its exact frontier completes.

It does not establish an optimum for NLL, accuracy, checkpoint size, runtime
memory, latency, energy, or hardware throughput. It does not produce a runnable
mixed-bit checkpoint. Those require a separate exporter and held-out
confirmation.
