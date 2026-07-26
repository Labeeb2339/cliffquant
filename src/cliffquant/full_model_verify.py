"""Independent structural and runtime verification for CliffQuant GPTQ exports."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .full_model_artifacts import (
    BASE_MODEL_ID,
    BASE_MODEL_REVISION,
    BITS,
    FROZEN_POLICIES,
    GROUP_SIZE,
    SafetensorsWeightSource,
    ScaleRun,
    load_scale_run,
    resolve_hf_snapshot_file,
    validate_corpus_replay_pair,
    verify_frozen_corpus_pair,
)
from .gptq_pack import (
    INT4_LOGICAL_ZERO,
    dequantize_gptq_int4,
    make_g_idx,
    pack_gptq_v1_qzeros,
    unpack_int4_qweight,
)
from .gptqmodel_export import (
    EXPORT_SCHEMA,
    fp16_values_from_bits,
    load_pinned_gptqmodel_contract,
    quantize_rows,
)
from .model_snapshot import validate_model_snapshot_descriptor
from .provenance import (
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
)
from .qwen35_modules import (
    QWEN35_08B_EXPECTED_QUANTIZED_MODULES,
    QWEN35_GPTQMODEL_TREE_REVISION,
)

VERIFICATION_SCHEMA = "cliffquant.gptq-verification.v1"


class TensorCheckpoint:
    """Strict local safetensors index with CPU tensor access."""

    def __init__(
        self,
        root: str | Path,
        *,
        allow_hf_blob_links: bool = False,
    ) -> None:
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(f"checkpoint directory does not exist: {self.root}")
        index_candidate = self.root / "model.safetensors.index.json"
        single_candidate = self.root / "model.safetensors"
        if index_candidate.is_file():
            index_path = self._safe_file(
                index_candidate.name,
                label="safetensors index",
                allow_hf_blob_links=allow_hf_blob_links,
            )
            data = json.loads(index_path.read_text(encoding="utf-8"))
            mapping = data.get("weight_map")
            if not isinstance(mapping, dict) or not mapping:
                raise ValueError(f"invalid safetensors weight map: {index_path}")
            if not all(
                isinstance(key, str) and isinstance(value, str) for key, value in mapping.items()
            ):
                raise TypeError("safetensors weight map must map strings to strings")
            self.index_path: Path | None = index_path
            self._weight_map = dict(mapping)
        elif single_candidate.is_file():
            single_path = self._safe_file(
                single_candidate.name,
                label="safetensors checkpoint",
                allow_hf_blob_links=allow_hf_blob_links,
            )
            try:
                from safetensors import safe_open
            except ImportError as exc:
                raise RuntimeError("safetensors is required for checkpoint verification") from exc
            with safe_open(single_path, framework="pt", device="cpu") as handle:
                self._weight_map = {key: single_path.name for key in handle}
            self.index_path = None
        else:
            raise FileNotFoundError(f"checkpoint has no standard model safetensors: {self.root}")
        self.keys = frozenset(self._weight_map)
        self._shards: dict[str, Path] = {}
        for name in sorted(set(self._weight_map.values())):
            candidate = self._safe_file(
                name,
                label="safetensors shard",
                allow_hf_blob_links=allow_hf_blob_links,
            )
            self._shards[name] = candidate

        try:
            from safetensors import safe_open
        except ImportError as exc:
            raise RuntimeError("safetensors is required for checkpoint verification") from exc
        actual_locations: dict[str, list[str]] = {}
        shard_keys: dict[str, frozenset[str]] = {}
        for name, path in self._shards.items():
            with safe_open(path, framework="pt", device="cpu") as handle:
                keys = frozenset(str(key) for key in handle.keys())  # noqa: SIM118
            shard_keys[name] = keys
            for key in keys:
                actual_locations.setdefault(key, []).append(name)
        duplicated = sorted(key for key, names in actual_locations.items() if len(names) != 1)
        if duplicated:
            raise ValueError(f"safetensors keys occur in multiple shards: {duplicated}")
        actual_keys = frozenset(actual_locations)
        if actual_keys != self.keys:
            missing = sorted(self.keys - actual_keys)
            extra = sorted(actual_keys - self.keys)
            raise ValueError(
                f"safetensors index coverage mismatch; missing={missing}, extra={extra}"
            )
        misplaced = sorted(
            key for key, shard_name in self._weight_map.items() if key not in shard_keys[shard_name]
        )
        if misplaced:
            raise ValueError(f"safetensors index maps keys to the wrong shard: {misplaced}")

    def _safe_file(
        self,
        relative: str,
        *,
        label: str,
        allow_hf_blob_links: bool,
    ) -> Path:
        if allow_hf_blob_links:
            return resolve_hf_snapshot_file(self.root, relative, label=label)
        candidate = (self.root / relative).resolve()
        if self.root != candidate and self.root not in candidate.parents:
            raise ValueError(f"{label} escapes checkpoint directory: {relative!r}")
        if not candidate.is_file():
            raise FileNotFoundError(f"missing {label}: {candidate}")
        return candidate

    @property
    def file_hashes(self) -> list[dict[str, Any]]:
        files = [
            {
                "file": path.relative_to(self.root).as_posix(),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for path in sorted(self._shards.values())
        ]
        if self.index_path is not None:
            files.append(
                {
                    "file": self.index_path.name,
                    "sha256": sha256_file(self.index_path),
                    "size_bytes": self.index_path.stat().st_size,
                }
            )
        return sorted(files, key=lambda entry: entry["file"])

    def read(self, key: str) -> Any:
        try:
            from safetensors import safe_open
        except ImportError as exc:
            raise RuntimeError("safetensors is required for checkpoint verification") from exc
        try:
            path = self._shards[self._weight_map[key]]
        except KeyError as exc:
            raise KeyError(f"tensor is absent from checkpoint: {key}") from exc
        with safe_open(path, framework="pt", device="cpu") as handle:
            return handle.get_tensor(key).detach().cpu().contiguous()


def tensor_sha256(tensor: Any) -> str:
    """Hash any torch tensor, including bfloat16, without numeric conversion."""

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for tensor hashing") from exc
    value = tensor.detach().cpu().contiguous()
    raw_view = value.view(torch.uint8)
    raw = raw_view.numpy().tobytes(order="C")
    header = canonical_json_bytes(
        {
            "dtype": str(value.dtype),
            "shape": [int(dimension) for dimension in value.shape],
        }
    )
    return sha256_bytes(header + raw)


def _numpy(tensor: Any, *, float64: bool = False) -> NDArray[np.generic]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for checkpoint verification") from exc
    value = tensor.detach().cpu()
    if float64:
        value = value.to(dtype=torch.float64)
    return np.ascontiguousarray(value.numpy())


def _verify_export_manifest(checkpoint: Path) -> dict[str, Any]:
    path = checkpoint / "cliffquant_export.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict) or metadata.get("schema") != EXPORT_SCHEMA:
        raise ValueError("checkpoint lacks a supported CliffQuant export manifest")
    policy = metadata.get("policy")
    if policy not in FROZEN_POLICIES:
        raise ValueError("export manifest is missing an immutable policy tag")
    if metadata.get("base_model", {}).get("id") != BASE_MODEL_ID:
        raise ValueError("export manifest base-model id drift")
    if metadata.get("base_model", {}).get("revision") != BASE_MODEL_REVISION:
        raise ValueError("export manifest base-model revision drift")
    base_source = validate_model_snapshot_descriptor(
        metadata.get("base_model", {}).get("source"),
        model_id=BASE_MODEL_ID,
        revision=BASE_MODEL_REVISION,
    )
    if metadata.get("gptqmodel", {}).get("revision") != QWEN35_GPTQMODEL_TREE_REVISION:
        raise ValueError("export manifest GPTQModel revision drift")
    gptq = metadata.get("gptq")
    if not isinstance(gptq, dict) or {
        "bits": gptq.get("bits"),
        "format": gptq.get("format"),
        "group_size": gptq.get("group_size"),
        "method": gptq.get("method"),
        "sym": gptq.get("sym"),
        "zero": gptq.get("zero"),
    } != {
        "bits": BITS,
        "format": "gptq",
        "group_size": GROUP_SIZE,
        "method": "gptq",
        "sym": True,
        "zero": INT4_LOGICAL_ZERO,
    }:
        raise ValueError("export manifest GPTQ contract drift")
    scale_run = metadata.get("scale_run")
    if not isinstance(scale_run, dict) or not isinstance(scale_run.get("manifest_sha256"), str):
        raise ValueError("export manifest scale-run identity is missing")
    corpus_replay = validate_corpus_replay_pair(metadata.get("corpus_replay"))

    descriptors = metadata.get("files")
    if not isinstance(descriptors, list):
        raise ValueError("export manifest files must be a list")
    described_files: set[str] = set()
    for descriptor in descriptors:
        if not isinstance(descriptor, dict) or not isinstance(descriptor.get("file"), str):
            raise ValueError("invalid export file descriptor")
        if descriptor["file"] in described_files:
            raise ValueError(f"duplicate export file descriptor: {descriptor['file']}")
        described_files.add(descriptor["file"])
        candidate = (checkpoint / descriptor["file"]).resolve()
        if checkpoint != candidate and checkpoint not in candidate.parents:
            raise ValueError("export manifest file escapes checkpoint")
        if not candidate.is_file():
            raise FileNotFoundError(f"export file is missing: {descriptor['file']}")
        if sha256_file(candidate) != descriptor.get("sha256"):
            raise ValueError(f"export file hash mismatch: {candidate}")
        if candidate.stat().st_size != descriptor.get("size_bytes"):
            raise ValueError(f"export file size mismatch: {descriptor['file']}")
    actual_files = {
        candidate.relative_to(checkpoint).as_posix()
        for candidate in checkpoint.rglob("*")
        if candidate.is_file() and candidate.name != path.name
    }
    if described_files != actual_files:
        missing = sorted(actual_files - described_files)
        extra = sorted(described_files - actual_files)
        raise ValueError(f"export file coverage mismatch; missing={missing}, extra={extra}")

    quantize_config_path = checkpoint / "quantize_config.json"
    quantize_config = json.loads(quantize_config_path.read_text(encoding="utf-8"))
    if not isinstance(quantize_config, dict):
        raise ValueError("quantize_config must contain a JSON object")
    method = str(quantize_config.get("quant_method", quantize_config.get("method", ""))).lower()
    format_name = str(
        quantize_config.get("format", quantize_config.get("checkpoint_format", ""))
    ).lower()
    if method != "gptq" or format_name != "gptq":
        raise ValueError("quantize_config is not standard GPTQ-v1")
    meta = quantize_config.get("meta")
    embedded = meta.get("cliffquant") if isinstance(meta, dict) else None
    if not isinstance(embedded, dict):
        raise ValueError("quantize_config has no embedded CliffQuant identity")
    if (
        embedded.get("policy") != policy
        or embedded.get("scale_run_sha256") != scale_run["manifest_sha256"]
        or embedded.get("base_snapshot_sha256") != base_source["fingerprint_sha256"]
        or embedded.get("corpus_replay_sha256") != corpus_replay["sha256"]
        or embedded.get("base_model") != BASE_MODEL_ID
        or embedded.get("base_revision") != BASE_MODEL_REVISION
        or embedded.get("gptqmodel_revision") != QWEN35_GPTQMODEL_TREE_REVISION
    ):
        raise ValueError("quantize_config embedded CliffQuant identity drift")
    return metadata


def _verify_auxiliary_files(base: Path, output: Path) -> list[dict[str, Any]]:
    names = (
        "chat_template.json",
        "chat_template.jinja",
        "merges.txt",
        "preprocessor_config.json",
        "processor_config.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "video_preprocessor_config.json",
        "vocab.json",
    )
    evidence: list[dict[str, Any]] = []
    for name in names:
        base_file = base / name
        if not base_file.is_file():
            continue
        output_file = output / name
        if not output_file.is_file():
            raise ValueError(f"GPTQModel writer did not preserve auxiliary file: {name}")
        base_sha256 = sha256_file(base_file)
        output_sha256 = sha256_file(output_file)
        if output_sha256 != base_sha256:
            raise ValueError(f"GPTQModel writer changed preserved auxiliary file: {name}")
        evidence.append(
            {
                "base_sha256": base_sha256,
                "file": name,
                "output_sha256": output_sha256,
            }
        )
    if not any(entry["file"].startswith("tokenizer") for entry in evidence):
        raise ValueError("no tokenizer artifacts were preserved")
    if not any("processor" in entry["file"] for entry in evidence):
        raise ValueError("no processor artifacts were preserved")
    return evidence


def _verify_module_buffers(
    *,
    base: TensorCheckpoint,
    output: TensorCheckpoint,
    scale_run: ScaleRun,
    row_chunk_size: int,
    dequant_blocks_per_module: int,
) -> list[dict[str, Any]]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for checkpoint verification") from exc
    evidence: list[dict[str, Any]] = []
    for module in scale_run.modules:
        prefix = module.name
        dense_key = f"{prefix}.weight"
        if dense_key in output.keys:
            raise ValueError(f"target dense weight survived GPTQ export: {dense_key}")
        required = {
            "g_idx": f"{prefix}.g_idx",
            "qweight": f"{prefix}.qweight",
            "qzeros": f"{prefix}.qzeros",
            "scales": f"{prefix}.scales",
        }
        if any(key not in output.keys for key in required.values()):
            missing = sorted(key for key in required.values() if key not in output.keys)
            raise ValueError(f"incomplete GPTQ buffer set for {prefix}: {missing}")
        qweight_tensor = output.read(required["qweight"])
        qzeros_tensor = output.read(required["qzeros"])
        scales_tensor = output.read(required["scales"])
        g_idx_tensor = output.read(required["g_idx"])
        if qweight_tensor.dtype != torch.int32 or qzeros_tensor.dtype != torch.int32:
            raise TypeError(f"GPTQ packed buffers must be int32 for {prefix}")
        if g_idx_tensor.dtype != torch.int32:
            raise TypeError(f"GPTQ g_idx must be int32 for {prefix}")
        if scales_tensor.dtype != torch.float16:
            raise TypeError(f"GPTQ scales must be FP16 for {prefix}")

        qweight = _numpy(qweight_tensor)
        qzeros = _numpy(qzeros_tensor)
        scales = _numpy(scales_tensor)
        g_idx = _numpy(g_idx_tensor)
        expected_bits = scale_run.artifacts[prefix].export_scale_bits
        expected_scales = fp16_values_from_bits(expected_bits)
        np.testing.assert_array_equal(scales.view(np.uint16), expected_scales.T.view(np.uint16))
        np.testing.assert_array_equal(g_idx, make_g_idx(module.in_features, GROUP_SIZE))
        np.testing.assert_array_equal(
            qzeros,
            pack_gptq_v1_qzeros(
                num_groups=module.groups_per_row,
                out_features=module.out_features,
                logical_zero=INT4_LOGICAL_ZERO,
            ),
        )

        dense = base.read(dense_key)
        for start in range(0, module.out_features, row_chunk_size):
            stop = min(start + row_chunk_size, module.out_features)
            dense_rows = _numpy(dense[start:stop], float64=True)
            expected_codes = quantize_rows(dense_rows, expected_scales[start:stop])
            actual_codes = unpack_int4_qweight(qweight[:, start:stop])
            np.testing.assert_array_equal(
                actual_codes,
                expected_codes,
                err_msg=f"signed-code parity failed for {prefix} rows {start}:{stop}",
            )

        block_count = min(dequant_blocks_per_module, module.out_features // 8)
        if block_count <= 0:
            raise ValueError(f"module is too narrow for GPTQ dequant verification: {prefix}")
        seed = int(sha256_bytes(prefix.encode())[:16], 16)
        generator = np.random.default_rng(seed)
        block_indices = np.sort(
            generator.choice(module.out_features // 8, size=block_count, replace=False)
        )
        dequant_samples = 0
        for block_index in block_indices:
            start = int(block_index) * 8
            stop = start + 8
            reconstructed = dequantize_gptq_int4(
                qweight[:, start:stop],
                qzeros[:, start // 8 : stop // 8],
                scales[:, start:stop],
                g_idx,
            )
            codes = unpack_int4_qweight(qweight[:, start:stop]).astype(np.float32)
            expected = codes * scales[g_idx].T.astype(np.float32)
            np.testing.assert_array_equal(reconstructed, expected)
            dequant_samples += int(reconstructed.size)
        evidence.append(
            {
                "dequant_values_checked": dequant_samples,
                "g_idx_sha256": tensor_sha256(g_idx_tensor),
                "module": prefix,
                "qweight_sha256": tensor_sha256(qweight_tensor),
                "qzeros_sha256": tensor_sha256(qzeros_tensor),
                "scales_sha256": tensor_sha256(scales_tensor),
                "signed_codes_checked": module.in_features * module.out_features,
            }
        )
    return evidence


def verify_full_model_structure(
    *,
    base_model_dir: str | Path,
    scale_run_dir: str | Path,
    corpus_directory: str | Path,
    checkpoint_dir: str | Path,
    report_path: str | Path,
    row_chunk_size: int = 32,
    dequant_blocks_per_module: int = 2,
) -> dict[str, Any]:
    """Verify every target and preserved tensor in a written GPTQ checkpoint."""

    if type(row_chunk_size) is not int or row_chunk_size <= 0:
        raise ValueError("row_chunk_size must be a positive integer")
    if type(dequant_blocks_per_module) is not int or dequant_blocks_per_module <= 0:
        raise ValueError("dequant_blocks_per_module must be a positive integer")
    base_root = Path(base_model_dir).resolve()
    if base_root.name != BASE_MODEL_REVISION:
        raise ValueError("base_model_dir is not the pinned Hugging Face snapshot")
    # Re-hash pinned shards and verify the scale run before comparing tensors.
    base_source = SafetensorsWeightSource(base_root)
    scale_run = load_scale_run(scale_run_dir, strict_inventory=True)
    corpus_replay = verify_frozen_corpus_pair(corpus_directory)
    scale_corpus_replay = validate_corpus_replay_pair(scale_run.metadata.get("corpus_replay"))
    if corpus_replay != scale_corpus_replay:
        raise ValueError("scale run and replayed corpus bundles differ")
    expected_fingerprint = sha256_bytes(canonical_json_bytes(dict(base_source.provenance)))
    if scale_run.metadata.get("base_fingerprint") != expected_fingerprint:
        raise ValueError("scale run and base checkpoint fingerprints differ")

    checkpoint_root = Path(checkpoint_dir).resolve()
    export_manifest = _verify_export_manifest(checkpoint_root)
    if export_manifest["base_model"]["source"] != dict(base_source.provenance):
        raise ValueError("export manifest and live base snapshot bytes differ")
    if export_manifest.get("scale_run", {}).get("manifest_sha256") != scale_run.manifest_sha256:
        raise ValueError("export manifest and scale-run hash differ")
    if export_manifest.get("policy") != scale_run.policy:
        raise ValueError("export manifest and scale-run policy differ")
    if validate_corpus_replay_pair(export_manifest.get("corpus_replay")) != corpus_replay:
        raise ValueError("export manifest and replayed corpus bundles differ")
    base = TensorCheckpoint(base_root, allow_hf_blob_links=True)
    output = TensorCheckpoint(checkpoint_root)

    target_weights = {module.weight_key for module in scale_run.modules}
    quant_buffers = {
        f"{module.name}.{suffix}"
        for module in scale_run.modules
        for suffix in ("g_idx", "qweight", "qzeros", "scales")
    }
    expected_output_keys = (base.keys - target_weights) | quant_buffers
    if output.keys != expected_output_keys:
        missing = sorted(expected_output_keys - output.keys)
        extra = sorted(output.keys - expected_output_keys)
        raise ValueError(f"checkpoint tensor-key mismatch; missing={missing}, extra={extra}")

    preserved: list[dict[str, Any]] = []
    for key in sorted(base.keys - target_weights):
        before = base.read(key)
        after = output.read(key)
        if before.dtype != after.dtype or tuple(before.shape) != tuple(after.shape):
            raise ValueError(f"preserved tensor metadata changed: {key}")
        if not bool(__import__("torch").equal(before, after)):
            raise ValueError(f"preserved non-target tensor changed: {key}")
        preserved.append({"key": key, "sha256": tensor_sha256(before)})

    modules = _verify_module_buffers(
        base=base,
        output=output,
        scale_run=scale_run,
        row_chunk_size=row_chunk_size,
        dequant_blocks_per_module=dequant_blocks_per_module,
    )
    if len(modules) != QWEN35_08B_EXPECTED_QUANTIZED_MODULES:
        raise RuntimeError("structural verifier did not check exactly 150 modules")
    report = {
        "auxiliary_files": _verify_auxiliary_files(base_root, checkpoint_root),
        "base_checkpoint_files": base.file_hashes,
        "checkpoint_files": output.file_hashes,
        "corpus_replay": corpus_replay,
        "export_manifest_sha256": sha256_file(checkpoint_root / "cliffquant_export.json"),
        "modules": modules,
        "policy": scale_run.policy,
        "preserved_non_target_tensor_count": len(preserved),
        "preserved_non_target_tensors": preserved,
        "scale_run_sha256": scale_run.manifest_sha256,
        "schema": VERIFICATION_SCHEMA,
        "status": "pass",
        "summary": {
            "dense_target_weights_present": 0,
            "module_buffer_sets": len(modules),
            "non_target_tensors_exact": len(preserved),
            "signed_codes_checked": sum(item["signed_codes_checked"] for item in modules),
        },
    }
    destination = Path(report_path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(canonical_json_bytes(report))
    hashes = {
        "report": {
            "file": destination.name,
            "sha256": sha256_file(destination),
            "size_bytes": destination.stat().st_size,
        }
    }
    destination.with_name(f"{destination.stem}.sha256.json").write_bytes(
        canonical_json_bytes(hashes)
    )
    return report


def load_fresh_gptq_torch(
    *,
    checkpoint_dir: str | Path,
    gptqmodel_source: str | Path,
    device: str,
) -> tuple[Any, Any]:
    """Load a written checkpoint only through pinned GPTQ_TORCH."""

    contract = load_pinned_gptqmodel_contract(gptqmodel_source)
    try:
        wrapper = contract.GPTQModel.load(
            str(Path(checkpoint_dir).resolve()),
            device=device,
            backend=contract.BACKEND.GPTQ_TORCH,
            trust_remote_code=False,
            local_files_only=True,
        )
    except Exception as exc:
        raise RuntimeError("fresh GPTQ_TORCH checkpoint load failed") from exc
    quantized = [
        (name, module)
        for name, module in wrapper.model.named_modules()
        if isinstance(module, contract.BaseQuantLinear)
    ]
    if len(quantized) != QWEN35_08B_EXPECTED_QUANTIZED_MODULES:
        raise RuntimeError(
            f"fresh GPTQ_TORCH load found {len(quantized)} quantized modules, expected 150"
        )
    return contract, wrapper


def _move_inputs(inputs: Mapping[str, Any], device: str) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in inputs.items():
        to = getattr(value, "to", None)
        moved[key] = to(device) if callable(to) else value
    return moved


def _generate_once(
    wrapper: Any,
    inputs: Mapping[str, Any],
    *,
    max_new_tokens: int,
) -> tuple[Any, list[Any]]:
    generator = getattr(wrapper, "generate", None)
    if not callable(generator):
        generator = getattr(wrapper.model, "generate", None)
    if not callable(generator):
        raise RuntimeError("loaded GPTQModel wrapper exposes no generate method")
    output = generator(
        **inputs,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        output_scores=True,
        return_dict_in_generate=True,
        use_cache=True,
    )
    sequences = getattr(output, "sequences", None)
    scores = list(getattr(output, "scores", ()))
    if sequences is None:
        raise RuntimeError("generation returned no token sequences")
    return sequences, scores


def _check_generation(
    first: tuple[Any, list[Any]],
    second: tuple[Any, list[Any]],
) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for runtime verification") from exc
    first_sequences, first_scores = first
    second_sequences, second_scores = second
    if not torch.equal(first_sequences, second_sequences):
        raise ValueError("greedy generation is not deterministic across repeated calls")
    if len(first_scores) != len(second_scores):
        raise ValueError("repeated greedy generations returned different score counts")
    if not first_scores:
        raise ValueError("generation returned no score tensors")
    for score in [*first_scores, *second_scores]:
        if not bool(torch.all(torch.isfinite(score)).item()):
            raise ValueError("generation produced non-finite logits")
    return {
        "generated_tokens": int(first_sequences.shape[-1]),
        "score_steps": len(first_scores),
        "sequence_sha256": tensor_sha256(first_sequences),
        "scores_sha256": [tensor_sha256(score) for score in first_scores],
    }


def run_text_generation_smoke(
    *,
    checkpoint_dir: str | Path,
    gptqmodel_source: str | Path,
    prompt: str,
    device: str = "cuda:0",
    max_new_tokens: int = 16,
) -> dict[str, Any]:
    """Fresh-load GPTQ_TORCH and check finite deterministic greedy text output."""

    if not prompt:
        raise ValueError("text smoke prompt must be non-empty")
    if type(max_new_tokens) is not int or max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be a positive integer")
    _contract, wrapper = load_fresh_gptq_torch(
        checkpoint_dir=checkpoint_dir,
        gptqmodel_source=gptqmodel_source,
        device=device,
    )
    tokenizer = getattr(wrapper, "tokenizer", None)
    if tokenizer is None:
        tokenizer = getattr(wrapper, "processor", None)
    tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    if not callable(tokenizer):
        raise RuntimeError("fresh GPTQModel load did not preserve a callable tokenizer")
    inputs = _move_inputs(tokenizer(prompt, return_tensors="pt"), device)
    first = _generate_once(wrapper, inputs, max_new_tokens=max_new_tokens)
    second = _generate_once(wrapper, inputs, max_new_tokens=max_new_tokens)
    evidence = _check_generation(first, second)
    decode = getattr(tokenizer, "decode", None)
    decoded = (
        decode(first[0][0].detach().cpu(), skip_special_tokens=True) if callable(decode) else None
    )
    return {
        **evidence,
        "backend": "GPTQ_TORCH",
        "decoded_text": decoded,
        "device": device,
        "kind": "text",
        "status": "pass",
    }


def run_image_generation_smoke(
    *,
    checkpoint_dir: str | Path,
    gptqmodel_source: str | Path,
    image_path: str | Path,
    prompt: str,
    device: str = "cuda:0",
    max_new_tokens: int = 16,
) -> dict[str, Any]:
    """Separately check one finite deterministic image-text generation."""

    if not prompt:
        raise ValueError("image smoke prompt must be non-empty")
    path = Path(image_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"image smoke input does not exist: {path}")
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required for the image-text smoke") from exc
    _contract, wrapper = load_fresh_gptq_torch(
        checkpoint_dir=checkpoint_dir,
        gptqmodel_source=gptqmodel_source,
        device=device,
    )
    processor = getattr(wrapper, "processor", None)
    if not callable(processor):
        raise RuntimeError("fresh GPTQModel load did not preserve a callable processor")
    with Image.open(path) as image:
        inputs = processor(text=prompt, images=image.convert("RGB"), return_tensors="pt")
    moved = _move_inputs(inputs, device)
    first = _generate_once(wrapper, moved, max_new_tokens=max_new_tokens)
    second = _generate_once(wrapper, moved, max_new_tokens=max_new_tokens)
    return {
        **_check_generation(first, second),
        "backend": "GPTQ_TORCH",
        "device": device,
        "image_sha256": sha256_file(path),
        "kind": "image-text",
        "status": "pass",
    }
