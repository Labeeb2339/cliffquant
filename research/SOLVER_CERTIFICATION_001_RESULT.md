# Solver certification 001 result

Date: 2026-07-26

Result: passed the frozen solver-correctness gate.

## Primary run

- Cases completed: 10,000
- Exact mismatches: 0
- Protocol gate: passed
- Seed: 20260726
- Workers / batch size: 16 / 32
- Elapsed wall time: 31.436 seconds
- Exhaustive fallbacks: 2,064
- Candidate-scale count: minimum 15, median 658.5, maximum 31,743
- Aggregate ordered-case fingerprint:
  `f1ee394153ea3e31add761e6e06793a3591620a22262bd5d21b500e84ef103bd`
- Raw certificate SHA256:
  `461fcf5b443496f0d6fbbdda3df296e7ffe9704eae15791034aeb61d7f2a6e75`

Each of the five frozen problem families contributed exactly 2,000 cases.
Certificate schema v2 also binds the result to hashes of the certification
protocol, certification engine, breakpoint solver, exhaustive solver, objective,
contracts, and FP16-grid implementation, plus sanitized runtime provenance.

## Scheduling replay

The complete campaign was rerun with seven workers and a batch size of 17.
It again returned 10,000 matches, zero mismatches, and the same aggregate
ordered-case fingerprint. Its elapsed wall time was 35.640 seconds and its raw
certificate SHA256 was:

`56716352dae1e1d9042d1823bb319596dc1caf0c476945104ab761c73bfed9f7`

The raw certificates are retained as local immutable-evidence candidates for a
future release asset. This result establishes agreement with the specified
exhaustive implementation across the frozen campaign. It does not establish
novelty, model quality, downstream accuracy, or a breakthrough.
