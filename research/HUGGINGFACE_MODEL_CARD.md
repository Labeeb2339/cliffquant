---
base_model: Qwen/Qwen3.5-0.8B-Base
base_model_relation: quantized
library_name: transformers
license: apache-2.0
model_name: Qwen3.5-0.8B-Base CliffQuant W4A16 G128
pipeline_tag: image-text-to-text
tags:
- qwen3_5
- multimodal
- gptq
- gptqmodel
- quantization
- 4-bit
- w4a16
- group-size-128
- model-compression
- cliffquant
---

# Qwen3.5-0.8B-Base CliffQuant W4A16 G128

This is a standard GPTQ-v1 W4A16 checkpoint of
[`Qwen/Qwen3.5-0.8B-Base`](https://huggingface.co/Qwen/Qwen3.5-0.8B-Base).
I quantized 150 language-tower matrices with symmetric 4-bit signed codes and
group size 128. The visual tower and other non-target tensors remain at their
original precision.

The only experimental change from the matched AbsMax control is the stored
group-scale policy. CliffQuant chooses the positive finite FP16 scale that
exactly minimizes the worst separately normalized calibration-environment
weighted reconstruction error.

[Code, protocol, and complete evidence](https://github.com/Labeeb2339/cliffquant)

![Held-out language-model loss](assets/heldout-nll-comparison.png)

## Measured result

Experiment 001 evaluates 64 windows and 16,320 target tokens per environment.
Negative deltas favor CliffQuant.

| Held-out environment | AbsMax NLL | CliffQuant NLL | CliffQuant - AbsMax |
|---|---:|---:|---:|
| Macro mean | `2.555326` | `2.500527` | `-0.054799` |
| General | `3.473953` | `3.410937` | `-0.063016` |
| Code | `1.513257` | `1.454046` | `-0.059211` |
| Math | `1.631195` | `1.613147` | `-0.018048` |
| Multilingual | `3.602898` | `3.523977` | `-0.078921` |

The frozen release gate allowed at most `+0.01` macro NLL regression and
`+0.02` in any environment. Both gates passed, and all four environments
improved.

The held-out reconstruction proxy also passed on 4,096 groups from all 150
target matrices:

| Scale policy | Module-macro worst-environment weighted MSE |
|---|---:|
| CliffQuant minimax | `0.0002711874` |
| Pooled-WMSE | `0.0002740030` |
| AbsMax | `0.0003770210` |

![Held-out reconstruction proxy](assets/proxy-policy-comparison.png)

## Quantization configuration

| Field | Value |
|---|---|
| Base model | `Qwen/Qwen3.5-0.8B-Base` |
| Base revision | `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68` |
| Format | GPTQ v1 |
| Target weights | 150 language-tower linear matrices |
| Weight bits | 4 |
| Activation precision | 16-bit model runtime |
| Group size | 128 |
| Symmetric | yes |
| Activation ordering | no (`desc_act=false`) |
| Stored scale | positive finite IEEE binary16 |
| Scale objective | exact four-environment minimax weighted MSE |
| GPTQModel revision | `581bfd970b8b67372ed61b0ef449d88f5388d196` |

CliffQuant changes scale selection, not the packed inference format. No custom
kernel is required.

## Load with GPTQModel

```bash
python -m pip install "gptqmodel==7.3.4"
```

```python
import os

os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

from gptqmodel import BACKEND, GPTQModel

model = GPTQModel.load(
    "labeebaryan/Qwen3.5-0.8B-Base-CliffQuant-W4A16-G128",
    device="cuda:0",
    backend=BACKEND.GPTQ_TORCH,
    trust_remote_code=False,
)

tokens = model.generate(
    "Exact minimax quantization",
    max_new_tokens=16,
)[0]
print(model.tokenizer.decode(tokens, skip_special_tokens=True))
```

The environment line keeps the portable `GPTQ_TORCH` path usable on Windows
without Triton; remove it if your TorchInductor/Triton setup is working. This
exact eager path was exercised against the released checkpoint.

This is a quantized **base** model, not an instruction-tuned assistant. The
upstream model card describes it as intended for fine-tuning, in-context
learning experiments, and research or development rather than direct
interaction.

## Verification

- `10,000 / 10,000` exact solver comparisons matched exhaustive FP16
  enumeration with zero mismatches.
- The full CliffQuant scale run completed all `3,883,008` W4/G128 groups.
- In 114 groups (`0.0029%`), a direct-candidate objective tie triggered
  conservative exhaustive-grid verification.
- Independent unpacking checked 150 module-buffer sets and `497,025,024`
  signed codes.
- All 338 non-target tensors remained byte-exact, with zero dense target
  weights left behind.
- Every target module's `qweight` and `scales` payload differs from the matched
  AbsMax checkpoint.
- A fresh empty-directory installation passed deterministic text and image-text
  generation twice.
- The fail-closed publication builder independently reloaded both checkpoints,
  reran held-out NLL, replayed the frozen corpora, recomputed the proxy and
  bootstrap gates, and regenerated the figures before release.

Important identities:

| Artifact | SHA256 |
|---|---|
| `model.safetensors` | `b88ea7316a858c2691c1e755a5eb73ebfc08e19b45273a481a262f2099893beb` |
| CliffQuant scale run | `1292f4102dad57b070071c00d92ba43cd096b64b6434b3b9a3b3b41b73db3cf5` |
| Solver certificate | `a90b704c0d2e044c4a85c9dc45f604eedbfa5306c5f8719f149980b93d502cfd` |
| Held-out raw NLL arrays | `fdb24aaa192df5a91189eabebeb2831056e06a84f9ff127fae67b70e72e78934` |
| Publication release manifest | `b2c5b71820320a08028d5ea2069d136020f0c9c5801901695748e2ae9ed9d447` |

The checkpoint itself was built from CliffQuant commit
`d176c4944962d0bcd43a8092059fbf5098653bc6`; later publication work does not
retroactively change that build provenance.

## Calibration and evaluation data

No fine-tuning or additional model training was performed. Pinned dataset
revisions were used only to construct deterministic calibration and held-out
text windows:

- general: WikiText-2 raw train / validation;
- code: MBPP train / validation and test;
- math: GSM8K train / test; and
- multilingual: XNLI train / validation.

Calibration contains 128 total 256-token windows. Held-out evaluation contains
256 disjoint 256-token windows, split equally across the four environments.
The repository records the exact revisions, row identities, rendered-text
hashes, token-window hashes, and zero-overlap checks.

## Limitations and claim boundary

This is one experiment on one base-model size, one quantization format, and one
frozen corpus construction. The positive proxy and NLL results do not establish
novelty, state of the art, downstream-task accuracy, latency gains, or broad
generalization.

Exact breakpoint-based scale optimization already has prior art. CliffQuant's
candidate distinction is the exact minimax objective over separately normalized
calibration environments, not exact scale optimization by itself. See the
[prior-art review](https://github.com/Labeeb2339/cliffquant/blob/main/research/PRIOR_ART.md)
and
[claim boundary](https://github.com/Labeeb2339/cliffquant/blob/main/research/CLAIM_BOUNDARY.md).

## License and attribution

The checkpoint and CliffQuant code are released under Apache-2.0. The base model
is also Apache-2.0; retain the upstream Qwen attribution and review its model
card before use.

If the method or evidence pipeline helps your work, citation metadata is
available in the repository's
[`CITATION.cff`](https://github.com/Labeeb2339/cliffquant/blob/main/CITATION.cff).
