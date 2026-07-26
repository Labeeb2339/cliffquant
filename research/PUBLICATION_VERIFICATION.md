# Publication verification

This is the execution path for the packed-model and clean-load parts of the
Experiment 001 publication gate. It produces evidence; it does not declare the
proxy or NLL gates passed.

## Structural evidence

Run the independent unpack verifier outside the exported checkpoint directory:

```powershell
python scripts/verify_full_model_gptq.py structural `
  --base-model <pinned-qwen-snapshot> `
  --scale-run <completed-policy-scale-run> `
  --corpus-directory <frozen-corpus-directory> `
  --checkpoint <exported-gptq-checkpoint> `
  --report <evidence-directory>/structural.json
```

The report and `structural.sha256.json` are canonical JSON. The report binds the
result to:

- every checkpoint file and the `cliffquant_export.json` hash;
- the complete base-model snapshot descriptor;
- both offline-replayed corpus identities;
- the exact scale-run manifest and immutable policy tag; and
- the verifier source files and runtime.

Reports must be outside the checkpoint. An existing report is idempotently
accepted only when its bytes match; different evidence is never overwritten.

## Clean GPTQ load and smoke evidence

Use a new path for `--work-directory`; the runner rejects a path that already
exists.

```powershell
C:\path\to\cpython-3.11.15\python.exe `
  scripts/verify_in_clean_environment.py `
  --checkpoint <exported-gptq-checkpoint> `
  --output-directory <evidence-directory>/clean-load `
  --work-directory <new-disposable-directory> `
  --python C:\path\to\cpython-3.11.15\python.exe `
  --hashed-lock requirements/verification-cu128-win-py311-hashed.txt `
  --wheelhouse <external-wheel-only-directory> `
  --device cuda:0 `
  --image <optional-image>
```

The runner:

1. rejects a dirty CliffQuant source tree and makes a detached local clone of
   its exact commit and tree;
2. creates a new venv under an allowlisted environment, uses Python isolated
   mode, and disables user and pip configuration;
3. installs only wheels from the external wheelhouse, offline, with
   `--require-hashes`, `--only-binary`, and no dependency expansion;
4. requires the hashed lock to contain the same package versions as
   `requirements/verification-cu128.txt`, plus the pinned bootstrap tools;
5. fetches GPTQModel at commit
   `581bfd970b8b67372ed61b0ef449d88f5388d196`;
6. builds both projects from clean detached sources and installs non-editable
   wheels;
7. verifies imports resolve inside the new venv and requires a successful
   `pip check`;
8. records path-free commit, tree, lock, wheelhouse, built-wheel, package, and
   runtime identities;
9. loads the checkpoint through GPTQModel's `GPTQ_TORCH` backend; and
10. runs each greedy generation twice, rejecting differing token sequences or
   non-finite score tensors.

The external wheelhouse is not committed because it contains multi-gigabyte
CUDA wheels. Its path never enters the evidence: every wheel byte and the
committed hash lock are bound into `clean-build-source.json`.

The clean-build provenance is embedded in `clean-environment.json`, which is
validated and embedded in `text-smoke.json` and, when requested,
`image-smoke.json`. Each smoke report also embeds the same checkpoint identity
used by structural verification. Every JSON report has a canonical
`.sha256.json` sidecar.

Re-validate a saved runtime report and require the clean-environment binding:

```powershell
python scripts/verify_full_model_gptq.py report `
  --report <evidence-directory>/clean-load/text-smoke.json `
  --checkpoint <exported-gptq-checkpoint> `
  --require-clean-environment
```

Saved-report validation re-hashes the live checkpoint, requires the current
verifier source bytes, and validates the complete structural or generation
evidence. A sidecar detects accidental drift; the published Git commit and
reported hashes are the external evidence anchors.

## Hub release staging

The immutable exported checkpoint contains only the model payload described by
`cliffquant_export.json`. Keep that manifest and the saved verification reports
unchanged when preparing the Hugging Face repository.

Hub staging has one explicit opt-in envelope:

- `README.md`;
- `LICENSE`;
- `.gitattributes`;
- `assets/proxy-policy-comparison.png`; and
- `assets/heldout-nll-comparison.png`.

All four paths are required to be regular, non-symlink files inside the staged
checkpoint. Every other extra file is rejected. Default report validation
remains strict and rejects the envelope as unmanifested payload.

Validate an existing saved report against the Hub-staged directory with:

```powershell
python scripts/verify_full_model_gptq.py report `
  --report <evidence-directory>/clean-load/text-smoke.json `
  --checkpoint <hub-staged-checkpoint> `
  --require-clean-environment `
  --hub-release
```

The Hub result is labelled `model-payload-verified`. The verifier still hashes
and validates every model-payload byte against the immutable checkpoint
identity, but it validates publication-envelope paths and file types only.
README, license, `.gitattributes`, and graph bytes are deliberately not part of
the model-payload identity and are not claimed as byte-verified metadata.

The disposable work directory is intentionally retained after the run for
diagnosis. It is not publication evidence; only the canonical reports in the
output directory are.
