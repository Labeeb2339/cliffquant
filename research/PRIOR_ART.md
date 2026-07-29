# Prior-art boundary

This is a bounded primary-source and public-code collision audit through
2026-07-30, not a proof of novelty.

The closest work I found is
[PiSO](https://arxiv.org/abs/2606.10890), which already gives an exact
breakpoint-based optimizer for round-to-nearest scales and applies it to
independent groupwise subproblems. That rules out broad claims that CliffQuant is
the first exact scale optimizer, the first diagonal-Hessian scale method, or the
first groupwise W4 scale search.

Other close references include:

- [MixQuant](https://arxiv.org/abs/2607.23047), which formulates per-module
  mixed-bit allocation as a multiple-choice knapsack problem using
  context-marginal distortion and one calibration for multiple budgets;
- [GAMMA](https://arxiv.org/abs/2605.18475), which uses teacher-forced
  reconstruction scoring and integer programming for exact-budget,
  quantizer-agnostic mixed precision with reusable any-budget scores;
- [NeUQI](https://arxiv.org/abs/2505.17595), which derives efficient scale and
  zero-point initialization for uniform low-bit quantization;
- [DASH-Q](https://arxiv.org/abs/2604.13806), which uses stable diagonal
  curvature and iterative weighted least squares;
- [ScaleSweep](https://arxiv.org/abs/2606.07618), which searches target-format
  block scales for NVFP4;
- [ScaleSearch](https://arxiv.org/abs/2605.12464), which searches representable
  microscaling values for NVFP4;
- [Optimal Quantization Using Scaled Codebook](https://research.nvidia.com/publication/2021-06_optimal-quantization-using-scaled-codebook),
  which globally optimizes a continuous scale and assignments for a fixed
  codebook under ordinary squared error;
- [GPTQ](https://arxiv.org/abs/2210.17323), which uses second-order calibration
  and sequential error compensation;
- [AWQ](https://arxiv.org/abs/2306.00978), which protects activation-salient
  channels through equivalent scaling;
- [SignRound / AutoRound](https://arxiv.org/abs/2309.05516), which learns
  rounding and clipping parameters;
- [OmniQuant](https://arxiv.org/abs/2308.13137), which learns clipping and
  equivalent transformations;
- [HQQ](https://github.com/dropbox/hqq), which optimizes groupwise scale and
  zero-point without calibration data;
- [MaCa](https://arxiv.org/abs/2602.07465), which separately normalizes sequence
  covariance estimates before averaging them into one calibration objective;
- [MixCal](https://arxiv.org/abs/2502.18424), which mixes generic and
  domain-specific calibration objectives through a weighted aggregate;
- [FairQuant](https://arxiv.org/abs/2602.23192), whose balanced reducer
  normalizes sensitive-group importance and takes a maximum for mixed-bit
  allocation and quantization-aware training;
- [C-PTQ](https://arxiv.org/abs/2607.21076), which mean-normalizes diagonal
  Fisher weights and searches channel-smoothing parameters against one
  aggregated calibration objective;
- [the worst-case PTQ benchmark](https://arxiv.org/abs/2303.13003), which makes
  distribution-shift reliability an explicit PTQ concern; and
- [the calibration-data study](https://arxiv.org/abs/2311.09755), which shows
  that PTQ quality can vary substantially with calibration data.

For AutoPolicy specifically, automatic budget-aware mixed precision is already
well established. The separate
[AutoPolicy prior-art boundary](AUTOPOLICY_PRIOR_ART.md) records the direct
allocation collisions and excludes first-MCKP, first-mixed-bit, or first
automatic-policy claims.

Generic worst-group optimization also predates this work; for example,
[Group DRO](https://arxiv.org/abs/1911.08731) optimizes worst-group loss under
distribution shifts. CliffQuant therefore does not claim minimax or
worst-group optimization itself as a new idea.

## Candidate distinction under test

CliffQuant investigates exact minimax scale selection for a fixed groupwise PTQ
proxy:

> For one symmetric integer weight group and a frozen collection of separately
> normalized calibration environments, select one deployment-compatible stored
> scale that minimizes the maximum diagonal reconstruction loss across those
> environments.

No source in this bounded audit combined all of the following:

- symmetric INT4 groupwise weights;
- one positive finite IEEE binary16 stored scale per group;
- separately normalized calibration environments retained as separate losses;
- a maximum-over-environments diagonal objective; and
- exact minimization over the permitted stored-scale set, with an exhaustive
  reference and a breakpoint-derived sufficient candidate set.

That absence is not proof that the combination is novel. The safe description
is "a candidate implementation distinction found in a bounded audit," not
"first," "novel," or "breakthrough."

## Claims excluded by this audit

- first exact PTQ scale optimizer;
- globally optimal quantization;
- first robust or multilingual calibration method;
- first target-dtype-aware scale search;
- guaranteed model quality;
- state of the art; or
- breakthrough.
