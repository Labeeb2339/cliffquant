# Prior-art boundary

This is a bounded collision audit, not a proof of novelty.

The closest work I found is
[PiSO](https://arxiv.org/abs/2606.10890), which already gives an exact
breakpoint-based optimizer for round-to-nearest scales and applies it to
independent groupwise subproblems. That rules out broad claims that CliffQuant is
the first exact scale optimizer, the first diagonal-Hessian scale method, or the
first groupwise W4 scale search.

Other close references include:

- [NeUQI](https://arxiv.org/abs/2505.17595), which derives efficient scale and
  zero-point initialization for uniform low-bit quantization;
- [DASH-Q](https://arxiv.org/abs/2604.13806), which uses stable diagonal
  curvature and iterative weighted least squares;
- [ScaleSweep](https://arxiv.org/abs/2606.07618), which searches target-format
  block scales for NVFP4;
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
- [the worst-case PTQ benchmark](https://arxiv.org/abs/2303.13003), which makes
  distribution-shift reliability an explicit PTQ concern; and
- [the calibration-data study](https://arxiv.org/abs/2311.09755), which shows
  that PTQ quality can vary substantially with calibration data.

## Candidate distinction under test

CliffQuant investigates exact minimax scale selection for a fixed groupwise PTQ
proxy:

> For one symmetric integer weight group and a frozen collection of separately
> normalized calibration environments, select one deployment-compatible stored
> scale that minimizes the maximum diagonal reconstruction loss across those
> environments.

No source in this bounded audit combined all of the following:

- symmetric INT4 groupwise weights;
- separately normalized calibration environments retained as separate losses;
- a maximum-over-environments diagonal objective; and
- exact minimization over the permitted stored-scale set.

That absence is not proof that the combination is novel. Until a wider
literature review and external review are complete, the safe description is
“candidate distinction” rather than “first,” “novel,” or “breakthrough.”

## Claims excluded by this audit

- first exact PTQ scale optimizer;
- globally optimal quantization;
- first robust or multilingual calibration method;
- first target-dtype-aware scale search;
- guaranteed model quality;
- state of the art; or
- breakthrough.
