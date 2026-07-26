# CliffQuant

I am building CliffQuant as a correctness-first study of scale selection for
groupwise integer quantization.

Phase 1 deliberately uses a complete search. For each group, CliffQuant evaluates
all 31,743 positive finite IEEE binary16 scales under a frozen multi-environment
diagonal reconstruction proxy:

\[
J(s)=\max_e \sum_i d_{e,i}\left(w_i-sq_i(s)\right)^2
\]

where \(q_i(s)\) uses round-to-nearest, ties-to-even, followed by clipping to the
configured integer range. The smallest FP16 bit pattern wins an exact objective tie.

```python
from cliffquant import QuantizerSpec, solve_exact

weights = [0.9, -0.4, 0.2, -1.1]
curvature = [
    [1.0, 0.8, 0.5, 1.2],
    [0.6, 1.4, 0.9, 0.7],
]

result = solve_exact(weights, curvature, QuantizerSpec(qmin=-8, qmax=7))
print(result.scale, result.objective, result.codes)
```

## What is established

- The current solver searches the entire positive finite FP16 scale grid.
- The production solver is checked against a separate pure-Python exhaustive oracle.
- Inputs, rounding, clipping, scale storage, tie-breaking, and proxy evaluation are
  explicit and tested.

## What is not established

This phase does not claim faster scale search, full-model quantization, improved
language-model quality, novelty, or state-of-the-art results. See
[the claim boundary](research/CLAIM_BOUNDARY.md).

## Development

```bash
python -m pip install -e ".[dev]"
python -m pytest
python -m ruff check .
python -m build
```

Apache-2.0.
