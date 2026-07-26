# CliffQuant

Exact minimax scale selection for groupwise W4A16 quantization.

I built CliffQuant to test a narrow question: when one quantization scale has to
serve several calibration environments, is it better to optimize the worst
environment directly instead of averaging their sensitivity first?

![Worst-environment held-out proxy comparison](figures/proxy-policy-comparison.png)

## The result so far

The frozen Experiment 001 proxy gate passed on 4,096 real weight groups from all
150 quantized language-tower matrices in `Qwen3.5-0.8B-Base`.

| Policy | Held-out module-macro worst-environment weighted MSE | Relative to CliffQuant |
|---|---:|---:|
| CliffQuant minimax | `0.0002711874` | reference |
| Pooled-WMSE | `0.0002740030` | CliffQuant is `1.03%` lower |
| AbsMax | `0.0003770210` | CliffQuant is `28.1%` lower |

Against Pooled-WMSE, the paired module-stratified bootstrap effect was
`2.816e-6`, with a frozen 95% interval of `[2.255e-6, 3.408e-6]`. CliffQuant
won on 2,752 groups, tied on 381, and lost on 963. All four preregistered checks
passed against both baselines without changing a threshold after seeing the
outcome.

![Paired held-out evidence](figures/proxy-paired-evidence.png)

This is a positive reconstruction-proxy result. It is not yet a claim about
perplexity, downstream accuracy, general robustness, novelty, or state of the
art. The full packed-model and held-out NLL gates are still running.

## The objective

For weights \(w_i\), separately normalized environment diagonals \(d_{e,i}\),
and a deployment-compatible stored scale \(s\), CliffQuant minimizes

$$
\min_s \max_e \sum_i d_{e,i}\left(w_i-sq_i(s)\right)^2
$$

where \(q_i(s)\) uses round-to-nearest, ties-to-even, followed by clipping to
`[-8, 7]`. The allowed set contains every positive finite IEEE binary16 scale.
An exact objective tie is resolved by the lower FP16 bit pattern.

The production solver enumerates the analytically sufficient FP16 candidates
induced by code transitions, quadratic vertices, and environment intersections.
An independent reference enumerates all 31,743 positive finite FP16 values.

## Correctness before claims

- 10,000 deterministic exact-solver comparisons passed with zero mismatches.
- The solver certificate is bound to its Python, NumPy, OpenBLAS, protocol, and
  semantic source hashes.
- Calibration and held-out corpora replay exactly from content-addressed source
  pages and retokenize through a pinned tokenizer snapshot.
- The two phases contain 128 and 256 windows respectively, with zero record or
  token-window overlap.
- Activation diagonals are bound to the exact 488-tensor model snapshot.
- The proxy result is stored as checksummed raw group records, not copied from a
  chart or README.

The finalized proxy summary SHA256 is
`ef9cb2c5ab9ad582c42fb0f71dd3078c5838f7fbc9b467d53073e1e81c6eb05d`.

## Try the scale solver

```bash
python -m pip install -e .
```

```python
from cliffquant import QuantizerSpec, solve_breakpoint_exact

weights = [0.9, -0.4, 0.2, -1.1]
environment_diagonals = [
    [1.0, 0.8, 0.5, 1.2],
    [0.6, 1.4, 0.9, 0.7],
]

result = solve_breakpoint_exact(
    weights,
    environment_diagonals,
    QuantizerSpec(qmin=-8, qmax=7),
)

print(result.scale, result.objective, result.codes)
```

## Reproduce the checked surfaces

```bash
python -m pip install -e ".[dev,figures]"
python -m pytest
python -m ruff check .
python -m build
```

Regenerate the two figures from a finalized Experiment 001 artifact directory:

```bash
python scripts/generate_figures.py \
  --input-dir artifacts/experiment-001/proxy \
  --output-dir figures
```

The generator verifies every entry in `SHA256SUMS.json` before parsing a result
and writes its own output provenance to
[`figures/figure-manifest.json`](figures/figure-manifest.json).

## Research boundary

[PiSO](https://arxiv.org/abs/2606.10890) already establishes exact
breakpoint-based scale optimization as prior art. CliffQuant's candidate
distinction is the exact minimax objective over separately normalized
calibration environments, not exact scale optimization by itself. I have not
found the same combination in my bounded review, but absence from that review is
not proof of novelty.

Read the:

- [frozen Experiment 001 protocol](research/EXPERIMENT_001_PROTOCOL.md);
- [solver certification result](research/SOLVER_CERTIFICATION_001_RESULT.md);
- [prior-art boundary](research/PRIOR_ART.md); and
- [claim boundary](research/CLAIM_BOUNDARY.md).

## Full-model status

The AbsMax scale pass is complete for all 3,883,008 W4/G128 groups. The
CliffQuant minimax pass is in progress. A model will be published only after
both checkpoints pass independent GPTQ-v1 unpack checks, clean-environment load
and deterministic generation smokes, and the frozen held-out NLL regression
limits.

Apache-2.0.
