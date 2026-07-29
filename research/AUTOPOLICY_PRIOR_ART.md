# AutoPolicy prior-art boundary

Automatic budget-aware mixed-precision quantization is established prior art.
AutoPolicy must not be presented as the first MCKP, dynamic-programming, or
mixed-bit allocator.

The closest direct collision found in the audit is
[MixQuant](https://arxiv.org/abs/2607.23047), posted on 25 July 2026. It already
uses per-module distortion, an MCKP surrogate, and one calibration process to
serve multiple budgets; its deployment allocator is greedy and is compared
against an ILP solver.

Other relevant primary sources include:

- [GAMMA](https://arxiv.org/abs/2605.18475), which uses a
  quantizer-agnostic teacher-forced reconstruction proxy, integer programming,
  exact budget constraints, and reusable scores for arbitrary budgets;
- [HAWQ-V2](https://arxiv.org/abs/1911.03852), which uses Hessian sensitivity
  and Pareto-based mixed precision;
- [HAWQ-V3](https://arxiv.org/abs/2011.10680), which uses hardware-aware
  integer programming;
- [LLM-MQ](https://neurips2023-enlsp.github.io/papers/paper_4.pdf), which uses
  first-order sensitivity and integer programming for low-bit LLM allocation;
- [SqueezeLLM](https://proceedings.mlr.press/v235/kim24f.html), which uses
  Fisher-based sensitivity;
- [CLADO](https://arxiv.org/abs/2307.05657) and
  [CoopQ](https://aclanthology.org/2026.findings-acl.373/), which model
  cross-layer interactions and expose the limits of an additive proxy;
- [ScaleBITS](https://arxiv.org/abs/2602.17698),
  [AMQ](https://aclanthology.org/2025.emnlp-main.1799/), and
  [LieQ](https://aclanthology.org/2026.findings-acl.771/), which automate
  mixed-precision selection; and
- [PiSO](https://arxiv.org/abs/2606.10890), which establishes exact
  breakpoint-based scale optimization as prior art.

The current implementation distinction under test is narrower:

- candidates scored across separately normalized environments;
- an exact transformer-block robust minimax allocation over the recorded
  environment vectors;
- fail-closed frontier limits;
- content-bound candidate and plan artifacts; and
- explicit logical-byte semantics.

That combination may be useful and may be distinctive, but this audit is not
proof of novelty.

In particular, AutoPolicy does not claim the first exact budget constraint, the
first reusable any-budget score table, or the first post-training global
allocator.
