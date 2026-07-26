# Solver certification 001 result

Date: 2026-07-26

Result: passed the frozen solver-correctness gate.

## Primary run

- Cases completed: 10,000
- Exact mismatches: 0
- Protocol gate: passed
- Seed: 20260726
- Workers / batch size: 16 / 32
- Elapsed wall time: 32.046 seconds
- Exhaustive fallbacks: 2,064
- Candidate-scale count: minimum 15, median 658.5, maximum 31,743
- Aggregate ordered-case fingerprint:
  `f1ee394153ea3e31add761e6e06793a3591620a22262bd5d21b500e84ef103bd`
- Raw certificate SHA256:
  `eae9e53675f50b26be12ca05b71da5524c17cad91f04ecc7757b0b90c013d026`

Each of the five frozen problem families contributed exactly 2,000 cases.

## Scheduling replay

The complete campaign was rerun with seven workers and a batch size of 17.
It again returned 10,000 matches, zero mismatches, and the same aggregate
ordered-case fingerprint. Its elapsed wall time was 48.823 seconds and its raw
certificate SHA256 was:

`060f4e3773649d772e6280fe42bcd545fb1e8abe0598afcd46504198ddb78d93`

The raw certificates are retained as local immutable-evidence candidates for a
future release asset. This result establishes agreement with the specified
exhaustive implementation across the frozen campaign. It does not establish
novelty, model quality, downstream accuracy, or a breakthrough.
