# Publication release gate

`scripts/build_publication_release.py` is the only supported path for creating
the small, reviewable Experiment 001 evidence bundle. It validates the live
model checkpoints before copying any evidence and writes to
`research/results/experiment-001` only after every gate passes.

The build requires explicit paths for:

- the live AbsMax and CliffQuant checkpoints;
- both structural verification reports;
- a clean-environment CliffQuant text-generation report;
- the raw held-out NLL result directory;
- the pinned local GPTQModel source tree used to load both checkpoints again;
- the finalized 4,096-group proxy directory;
- the 10,000-case solver certificate;
- all six frozen corpus manifest, token, and sidecar files;
- both completed scale-run manifests; and
- the three generated figures plus their source-bound figure manifest.

The NLL rerun defaults to `--device cuda:0 --batch-size 1`; both values are
recorded and checked. Run `python scripts/build_publication_release.py --help`
for the full command surface. The destination is immutable: an existing bundle,
broken link, symlink, or Windows reparse-point leaf is never overwritten.

## Fail-closed checks

Before construction, the builder:

1. re-hashes both live checkpoints through the strict structural report loader;
2. requires a clean-environment deterministic text smoke bound to the exact
   CliffQuant checkpoint identity;
3. recomputes every aggregate held-out NLL metric and both frozen regression
   gates from the raw per-token NLL arrays;
4. loads both checkpoints again in a fresh temporary result directory, reruns
   teacher-forced held-out NLL, and requires exact semantic reports and exact
   per-array dtype, shape, and canonical value hashes (the incidental NPZ ZIP
   container hash is deliberately ignored);
5. requires deterministic CUDA algorithms, deterministic cuDNN, disabled
   cuDNN benchmarking, disabled TF32, highest float32 matmul precision, and the
   fixed cuBLAS workspace contract before either model evaluation;
6. binds the NLL checkpoint identities to the structural identities and
   requires the exact quantized-module inventories to align, with at least one
   canonical `qweight` or `scales` tensor hash differing between policies;
7. requires distinct AbsMax and CliffQuant scale-run and safetensors payload
   identities;
8. fully loads both completed scale runs and binds them to the exact checkpoints,
   corpus replay, base snapshot, and solver certificate;
9. replays the frozen calibration and held-out corpus collection;
10. strictly parses all 4,096 canonical proxy group records, binds them to the
    frozen sample manifest and live checkpoints, and recomputes the complete
    module-macro summary, 10,000-replicate module-stratified bootstraps, and
    frozen gates;
11. regenerates all three figures with the current frozen generator in a fresh
    directory and requires exact manifests, logical-name-to-filename mappings,
    complete checksum-valid PNG pixel streams, and byte-identical images; and
12. copies only validated regular files into a sibling staging directory,
    repeats the full inventory and descriptor-hash pass after semantic reads,
    and atomically publishes without replacing any destination.

The held-out NLL writer uses the same stage-validate-no-replace pattern. Its raw
NPZ validator inspects the ZIP and NPY headers, exact member inventory, dtypes,
shapes, and bounded uncompressed sizes before NumPy is allowed to allocate or
decompress the arrays.

The curated bundle contains reports, raw NLL evidence, frozen corpus files,
proxy results, scale-run manifests, the solver certificate, and figures. It does
not duplicate the multi-gigabyte model checkpoints or per-module scale archives.

## Independent validation

Run the portable byte-inventory check:

```text
python scripts/validate_publication_release.py --inventory-only
```

This platform-independent mode rejects symlinks and reparse points, path
escapes, missing files, extra files, and size or hash drift. It is the check run
against the checked-in Experiment 001 bundle in CI.

Omit `--inventory-only` for the deeper semantic replay. That mode additionally
rejects non-canonical JSON/JSONL, status drift, gate drift, and identity
disagreement; recomputes the complete proxy summary; and reruns every other
semantic check that does not require the omitted live checkpoints. Its solver
certificate is intentionally bound to the recorded Windows, Python 3.11.15,
NumPy 2.2.6, and BLAS build, so recreate the frozen environment in
`research/PUBLICATION_VERIFICATION.md` before using it.

The portable check proves exact checked-in bytes. The deeper replay proves
internal consistency in the recorded runtime. Only the build command proves
that the evidence was bound to the supplied live checkpoints and independently
rerun at construction time. Passing any gate does not establish novelty, state
of the art, or a breakthrough claim.
