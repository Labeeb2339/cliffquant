# Solver certification 001 result

Date: 2026-07-26

Result: passed the frozen solver-correctness gate.

## Experiment 001 runtime-bound run

- Cases completed: 10,000
- Exact mismatches: 0
- Protocol gate: passed
- Seed: 20260726
- Workers / batch size: 16 / 32
- Elapsed wall time: 22.242 seconds
- Exhaustive fallbacks: 2,064
- Candidate-scale count: minimum 15, median 658.5, maximum 31,743
- Aggregate ordered-case fingerprint:
  `f1ee394153ea3e31add761e6e06793a3591620a22262bd5d21b500e84ef103bd`
- Raw certificate SHA256:
  `a90b704c0d2e044c4a85c9dc45f604eedbfa5306c5f8719f149980b93d502cfd`
- Numerical runtime: Python 3.11.15, NumPy 2.2.6, OpenBLAS 0.3.29
  (`USE64BITINT`, Haswell target)

Each of the five frozen problem families contributed exactly 2,000 cases.
Certificate schema v2 also binds the result to hashes of the certification
protocol, certification engine, breakpoint solver, exhaustive solver, objective,
contracts, FP16-grid implementation, and provenance implementation. The
validator requires the current Python, NumPy, platform, and path-free NumPy/BLAS
build metadata to match the certificate. This is the certificate used by
Experiment 001.

## Earlier scheduling evidence

Before runtime/build binding was added, a primary run and a complete replay with
seven workers and batch size 17 both returned 10,000 matches, zero mismatches,
and the same aggregate ordered-case fingerprint. Their raw SHA256 values were:

`461fcf5b443496f0d6fbbdda3df296e7ffe9704eae15791034aeb61d7f2a6e75`
`56716352dae1e1d9042d1823bb319596dc1caf0c476945104ab761c73bfed9f7`

Those earlier certificates remain useful replay evidence but are not accepted
for Experiment 001 because they predate the final runtime schema. The raw
certificates are retained as local immutable-evidence candidates for a future
release asset. The runtime-bound result establishes agreement with the specified
exhaustive implementation across the frozen campaign. It does not establish
novelty, model quality, downstream accuracy, or a breakthrough.
