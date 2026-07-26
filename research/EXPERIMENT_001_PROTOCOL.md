# Experiment 001 protocol

Status: frozen before collecting Qwen activation statistics or comparing policies.

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
  date.
- Weight format: symmetric INT4, one scale for every 128 consecutive input
  features of each output channel.
- Integer range: `[-8, 7]`.
- Assignment: round-to-nearest, ties-to-even, then clipping.
- Stored scale: positive finite IEEE binary16.
- Activations: unquantized.
- Activation order: disabled.
- Unlisted modules, embeddings, normalization, convolution, vision components,
  and biases remain at their base-model dtype.

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
