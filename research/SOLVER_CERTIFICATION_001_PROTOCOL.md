# Solver certification 001 protocol

Status: frozen before running the 10,000-case campaign.

## Purpose

This campaign tests whether `solve_breakpoint_exact` returns the same result as
the exhaustive 31,743-scale production reference. It is a correctness gate, not
evidence that the quantization objective improves a model.

## Fixed campaign

- Seed: `20260726`.
- Cases: `10,000`, indexed from zero.
- Random generator: NumPy `SeedSequence([seed, case_index])` and `default_rng`.
- Environments: one through five.
- Group size: one through sixteen, with every fiftieth case using 128 weights.
- Quantizer ranges: fixed INT4-like ranges plus deterministic coverage of every
  supported signed bound.
- Problem families, allocated by case index: normal, log-uniform magnitude,
  exact round-to-nearest-even half ties, outlier mixtures, and sparse weights.
- Diagonals: lognormal nonnegative values with deterministic zeros.
- Deterministic all-zero weight and all-zero diagonal cases exercise exhaustive
  fallback and lower-bit tie breaking.

For every case, the campaign compares scale bits, widened scale value, integer
codes, every environment loss, and the minimax objective with exact equality.
Inputs and both outputs contribute to a per-case SHA256 fingerprint; ordered
case fingerprints contribute to one aggregate SHA256.

## Gate

All 10,000 comparisons must match. One mismatch fails the gate and is preserved
with its complete inputs and both outputs. Worker count and batch size may change
runtime but do not change case generation or ordered evidence.
