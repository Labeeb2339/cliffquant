# Experiment 001 protocol

Status: frozen before collecting Qwen activation statistics or comparing policies.

Addendum 001-A was frozen after implementing the first loader, but before any
real corpus or model collection. Addendum 001-B was then frozen after that
backend failed its usability diagnostic. Addendum 001-C freezes the paired
bootstrap arithmetic. All three addenda were frozen before accepting any real
model, policy, or outcome data.

## Question

Does keeping calibration environments separate during groupwise scale selection
improve the held-out worst-environment reconstruction proxy, and does that
improvement survive a complete W4A16 model smoke test?

This experiment is designed to test that question. It is not a novelty,
state-of-the-art, or downstream-accuracy claim.

## Fixed model and quantizer

- Base model: `Qwen/Qwen3.5-0.8B-Base`
- Base revision:
  `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`
- Quantized modules: the GPTQModel Qwen3.5 module definition at the experiment
  date, pinned to commit
  `581bfd970b8b67372ed61b0ef449d88f5388d196`.
- Quantized inventory: exactly 150 matrices under
  `model.language_model.layers`, totalling 497,025,024 weights and 3,883,008
  W4/G128 groups.
- Weight format: symmetric INT4, one scale for every 128 consecutive input
  features of each output channel.
- Integer range: `[-8, 7]`.
- Assignment: round-to-nearest, ties-to-even, then clipping.
- Stored scale: positive finite IEEE binary16.
- Activations: unquantized.
- Activation order: disabled.
- Unlisted modules, embeddings, normalization, convolution, vision components,
  biases, and the seven `mtp.layers.0` matrices remain at their base-model
  dtype.

## Calibration environments

All datasets are loaded at the pinned Hub revisions below. Sampling uses
`seed=2339`. Invalid or empty records are skipped deterministically before
sampling. Each environment contributes 32 non-overlapping 256-token windows.

| Environment | Dataset and subset | Split | Hub revision | Text fields |
| --- | --- | --- | --- | --- |
| general | `Salesforce/wikitext`, `wikitext-2-raw-v1` | train | `b08601e04326c79dfdd32d625aee71d232d685c3` | `text` |
| code | `google-research-datasets/mbpp`, `full` | train | `4bb6404fdc6cacfda99d4ac4205087b89d32030c` | task, solution, tests |
| math | `openai/gsm8k`, `main` | train | `740312add88f781978c0658806c59bc2815b9866` | question and answer |
| multilingual | `facebook/xnli`, `all_languages` | train | `b8dd5d7af51114dbda02c0e3f6133f332186418e` | premise and hypothesis in `ar`, `es`, `hi`, and `zh` |

For each quantized linear module and environment, the diagonal statistic is the
mean squared input activation over unmasked tokens. Each environment diagonal is
then divided by its own arithmetic mean for that module. A zero-mean diagonal is
rejected. This normalization gives each environment equal outer weight in the
minimax objective rather than allowing token count or global activation magnitude
to choose the winner.

The calibration manifest must record dataset revisions, selected row identifiers,
rendered-text SHA256 values, token counts, tokenizer files, model revision,
software versions, device, and every normalization constant.

## Addendum 001-A: rendering and deterministic sampling

The rendering rules in this addendum remain active. Its bounded streaming
sampling backend is superseded by Addendum 001-B below.

The tokenizer is called with `add_special_tokens=False`. Records use these exact
UTF-8 templates after trimming their component fields:

```text
general:       {text}\n\n
code:          Task:\n{task}\n\nSolution:\n{solution}\n\nTests:\n{tests}\n\n
math:          Question:\n{question}\n\nAnswer:\n{answer}\n\n
multilingual:  Language: {lang}\nPremise:\n{premise}\n\nHypothesis:\n{hypothesis}\n\n
```

Invalid records are filtered before sampling. For each pinned source split, the
stream loader fills a 10,000-valid-record buffer in source order, then uses
Python's `random.Random(2339)` to emit and replace uniformly selected buffer
slots until at most 10,000 records have been emitted or the stream ends. Original
source offsets are retained. The bounded records are then deduplicated by
rendered-text SHA256 and ordered by SHA256 of:

```text
seed, environment, phase, source identity, row identity, rendered-text SHA256
```

Tokens are concatenated in that order into non-overlapping 256-token windows.
An incomplete tail is discarded before moving to another split. Multilingual
windows are allocated equally to Arabic, Spanish, Hindi, and Chinese, then
interleaved in that order. The manifest records buffer size, record limit,
source offsets, text hashes, window hashes, and every segment used by a window.

This is a deterministic bounded-stream sample, not a claim of uniform sampling
over an entire dataset.

## Addendum 001-B: verified, resumable Dataset Viewer collection

On 2026-07-26, the pinned `datasets.load_dataset(..., streaming=True)` backend
did not return even a 100-row WikiText diagnostic after more than 20 minutes on
the experiment's Windows host. The diagnostic was terminated. No real corpus
rows, token manifests, activation statistics, policy comparisons, model
checkpoints, or outcome artifacts had been accepted or produced before this
amendment. The failure was a backend-usability result, not an experimental
result.

The replacement keeps Addendum 001-A's rendering, tokenization, filtering,
deduplication, and downstream ordering rules, but replaces its source sampler:

1. Before reading a cached or remote page, query the Hugging Face Hub dataset
   metadata endpoint and require its current repository SHA to equal the source's
   frozen 40-character revision exactly. Refuse the run on drift. Dataset Viewer
   pages are not accepted merely because an old local cache exists.
2. Obtain `num_rows_total` from a complete Dataset Viewer `/rows` response.
   Requests use fixed pages of 100 rows. Reject partial responses, missing or
   non-contiguous `row_idx` values, unexpected page lengths, changed totals, and
   any row whose `truncated_cells` list is non-empty.
3. Derive a source-specific integer seed from SHA256 of canonical JSON containing
   `seed=2339`, the environment, dataset, config, split, and revision. Shuffle
   the complete list of page indices once with `random.Random(derived_seed)`.
   Rows remain in ascending `row_idx` order within each selected page.
4. Preserve the Viewer `row_idx` as `__cliffquant_source_index__`. Apply the
   frozen environment renderer before accepting a row, so empty or invalid rows
   do not count toward the limit. Stop at exactly 10,000 valid records, or after
   all pages if the split contains fewer valid records.
5. Cache every response page beneath the caller's output directory. The cache
   identity includes dataset, revision, config, split, fixed page coordinates,
   and schema version. Each canonical response is SHA256-addressed; cache
   envelopes, filenames, and hashes are verified before reuse. Writes use an
   atomic same-directory replacement.
6. Use separate connection and socket-read timeouts with at most three retries
   after the first attempt. Emit page-level progress to standard error so a
   stalled collection is observable and a rerun visibly resumes from cache.

The phase manifest records the verified current SHA, total row count, page size,
derived page seed, SHA256 of the complete shuffled page order, every selection
page and canonical response hash, the accepted valid-record count, and the
content-addressed cache identity. Operational cache-hit status is deliberately
excluded from the hashed manifest so a resumed run produces the same scientific
provenance as its uninterrupted equivalent.

This is deterministic page sampling, not uniform row sampling. The protocol was
amended because the original backend was unusable before data collection, not
because an observed model or policy result was unfavorable.

## Held-out data

No held-out record may appear in calibration. The evaluator uses 64
non-overlapping 256-token windows per environment:

| Environment | Split |
| --- | --- |
| general | WikiText validation |
| code | MBPP validation and then test if more windows are required |
| math | GSM8K test |
| multilingual | XNLI validation, using the same four languages |

Held-out diagonals are collected independently and normalized with their own
per-module, per-environment means. Language-model evaluation reports
teacher-forced token NLL for the exact held-out token manifests. Prompt wrappers
are not changed between policies.

## Frozen policies

1. **AbsMax**: the smallest no-clipping scale,
   `max(max(w_pos)/7, max(abs(w_neg))/8)`, rounded to the nearest positive
   finite FP16 value with the lower bit pattern winning a distance tie.
2. **Pooled-WMSE**: exact scale selection after averaging the four normalized
   calibration diagonals into one diagonal.
3. **CliffQuant-minimax**: exact scale selection for the maximum of the four
   separate normalized calibration-environment losses.

All policies use the same weights, groups, integer range, stored-scale format,
packing path, and evaluator.

## Development sample

Before full-model fake quantization, policies are compared on a deterministic
stratified sample of 4,096 real weight groups. The sample spans every quantized
module family and all 24 language-model layers. The sample manifest is frozen by
module name, output row, group index, and SHA256.

## Primary metrics

- Per group: held-out maximum normalized diagonal-proxy loss.
- Aggregate: macro mean of each module's mean held-out maximum loss. This prevents
  large matrices from silently dominating the result.
- Paired effect: `baseline - CliffQuant`, reported with a deterministic
  module-stratified bootstrap 95% confidence interval.
- Full model: per-environment and macro token NLL, packed checkpoint size, peak
  device memory, load result, text-generation smoke result, and one image-text
  smoke result if the base processor supports it locally.

Raw losses, paired differences, failures, runtime samples, manifests, and hashes
are retained. Graphs are generated from those raw artifacts rather than manually
entered values.

## Addendum 001-C: module-stratified paired bootstrap

This addendum was frozen on 2026-07-26 before running the 4,096-group comparison
or accepting any real policy outcome.

For each baseline independently, the per-group paired effect is the baseline's
held-out maximum proxy loss minus CliffQuant-minimax's loss for the identical
weight group. Modules are fixed strata: the bootstrap does not resample modules.
Within every module, each replicate samples that module's paired effects with
replacement, retaining the module's original sampled-group count. The sampled
effects are averaged within each module, then those module means are
macro-averaged with equal module weight.

The interval uses exactly 10,000 replicates from NumPy PCG64. Modules are
processed in lexicographic module-name order and groups in frozen sample-manifest
order. The AbsMax comparison uses seed `2339`; the Pooled-WMSE comparison uses
seed `2340`. The 2.5th and 97.5th percentiles use NumPy's linear quantile method.
The reported paired point estimate is the same fixed-module macro mean before
resampling. Exact zero differences are ties and are excluded from both the win
and loss counts.

## Gates

### Solver correctness gate

- The accelerated exact solver must match the exhaustive 31,743-scale oracle in
  scale bits, codes, environment losses, and objective on the frozen adversarial
  suite and at least 10,000 deterministic randomized problems.
- Any mismatch blocks an exactness claim and blocks model packaging through that
  solver.

### Real-group proxy gate

CliffQuant-minimax must satisfy all of the following against both AbsMax and
Pooled-WMSE on the 4,096-group held-out sample:

- lower macro held-out maximum proxy loss;
- positive paired mean improvement;
- a module-stratified 95% bootstrap interval whose lower bound is greater than
  zero;
- wins on more groups than it loses, excluding exact ties.

Failure is reported as a negative result. The policies or thresholds are not
changed after observing the result.

### Publishable-model gate

A Hugging Face model is published only if:

- the proxy gate passes;
- every packed tensor round-trips through an independent unpack check;
- a clean environment loads the checkpoint through a standard supported GPTQ
  path;
- text generation is finite and deterministic at greedy settings;
- CliffQuant's held-out macro NLL is no worse than AbsMax by more than `0.01`;
- no environment NLL is worse than AbsMax by more than `0.02`.

The model card must still report every regression and must not translate a proxy
win into a downstream-accuracy or robustness claim.

## Confirmation boundary

Experiment 001 is development evidence on one small Qwen model and four chosen
text environments. A claim that the method generalizes requires a separately
frozen confirmation experiment on other model families, sizes, bit widths, and
unseen domains.
