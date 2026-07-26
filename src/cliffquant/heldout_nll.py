"""Paired held-out token-NLL evaluation for CliffQuant and AbsMax GPTQ models."""

from __future__ import annotations

import gc
import json
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .experiment001_io import load_corpus_artifact
from .full_model_artifacts import (
    BASE_MODEL_ID,
    BASE_MODEL_REVISION,
    FROZEN_ENVIRONMENTS,
    validate_corpus_replay_pair,
    verify_frozen_corpus_pair,
)
from .full_model_verify import load_fresh_gptq_torch
from .model_snapshot import validate_model_snapshot_descriptor
from .provenance import (
    canonical_array_sha256,
    canonical_json_bytes,
    runtime_metadata,
    sha256_bytes,
    sha256_file,
)
from .qwen35_modules import QWEN35_GPTQMODEL_TREE_REVISION

HELDOUT_NLL_SCHEMA = "cliffquant.heldout-nll-comparison.v1"
EXPECTED_WINDOWS = 64
EXPECTED_WINDOW_TOKENS = 256
MACRO_REGRESSION_LIMIT = 0.01
ENVIRONMENT_REGRESSION_LIMIT = 0.02


def load_heldout_tokens(
    manifest_path: str | Path,
) -> tuple[dict[str, Any], dict[str, tuple[NDArray[np.int64], NDArray[np.uint8]]]]:
    """Validate the exact frozen held-out token manifest and arrays."""

    path = Path(manifest_path).resolve()
    corpus = load_corpus_artifact(path.parent, phase="heldout")
    if corpus.manifest_path != path:
        raise ValueError("NLL gate requires the exact heldout.manifest.json artifact")

    tokens: dict[str, tuple[NDArray[np.int64], NDArray[np.uint8]]] = {}
    with np.load(corpus.archive_path, allow_pickle=False) as arrays:
        expected_keys = {
            f"{environment}__{suffix}"
            for environment in FROZEN_ENVIRONMENTS
            for suffix in ("attention_mask", "input_ids")
        }
        if set(arrays.files) != expected_keys:
            raise ValueError("held-out token archive array coverage mismatch")
        for environment in FROZEN_ENVIRONMENTS:
            input_ids = np.ascontiguousarray(
                arrays[f"{environment}__input_ids"],
                dtype=np.int64,
            )
            attention_mask = np.ascontiguousarray(
                arrays[f"{environment}__attention_mask"],
                dtype=np.uint8,
            )
            expected_shape = (EXPECTED_WINDOWS, EXPECTED_WINDOW_TOKENS)
            if input_ids.shape != expected_shape or attention_mask.shape != expected_shape:
                raise ValueError(f"held-out token shape mismatch for {environment}")
            if not np.all(attention_mask == 1):
                raise ValueError(f"held-out attention mask drift for {environment}")
            tokens[environment] = (input_ids, attention_mask)
    manifest = {
        "_validated": {
            "archive_file": corpus.archive_path.name,
            "archive_sha256": corpus.archive_sha256,
            "manifest_file": corpus.manifest_path.name,
            "manifest_sha256": corpus.manifest_sha256,
            "sidecar_sha256": corpus.sidecar_sha256,
        }
    }
    return manifest, tokens


def _checkpoint_fingerprint(root: Path) -> dict[str, Any]:
    files = [
        {
            "file": path.relative_to(root).as_posix(),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]
    model_files = [
        descriptor
        for descriptor in files
        if descriptor["file"].endswith(".safetensors")
        or descriptor["file"].endswith(".safetensors.index.json")
    ]
    if not any(descriptor["file"].endswith(".safetensors") for descriptor in model_files):
        raise ValueError(f"checkpoint has no safetensors model payload: {root}")
    return {
        "files": files,
        "model_files": model_files,
        "model_sha256": sha256_bytes(canonical_json_bytes(model_files)),
        "sha256": sha256_bytes(canonical_json_bytes(files)),
    }


def _standard_gptq_config(root: Path) -> dict[str, Any]:
    quant_path = root / "quantize_config.json"
    config_path = root / "config.json"
    export_path = root / "cliffquant_export.json"
    payload: dict[str, Any] | None = None
    source_path: Path | None = None
    source_label: str | None = None
    if quant_path.is_file():
        value = json.loads(quant_path.read_text(encoding="utf-8"))
        if isinstance(value, dict):
            payload = value
            source_path = quant_path
            source_label = quant_path.name
    elif config_path.is_file():
        value = json.loads(config_path.read_text(encoding="utf-8"))
        nested = value.get("quantization_config") if isinstance(value, dict) else None
        if isinstance(nested, dict):
            payload = nested
            source_path = config_path
            source_label = f"{config_path.name}:quantization_config"
    elif export_path.is_file():
        value = json.loads(export_path.read_text(encoding="utf-8"))
        nested = value.get("gptq") if isinstance(value, dict) else None
        if isinstance(nested, dict):
            payload = nested
            source_path = export_path
            source_label = export_path.name
    if payload is None or source_path is None or source_label is None:
        raise ValueError(f"checkpoint has no readable standard GPTQ configuration: {root}")
    method = str(payload.get("quant_method", payload.get("method", ""))).lower()
    format_name = str(payload.get("format", payload.get("checkpoint_format", ""))).lower()
    if method != "gptq" or format_name != "gptq":
        raise ValueError(f"checkpoint is not standard method=gptq, format=gptq: {root}")
    return {
        "cliffquant": (
            payload.get("meta", {}).get("cliffquant")
            if isinstance(payload.get("meta"), dict)
            else None
        ),
        "config_sha256": sha256_file(source_path),
        "format": format_name,
        "method": method,
        "source": source_label,
    }


def _export_identity(
    root: Path,
    *,
    expected_policy: str,
) -> dict[str, Any]:
    export_path = root / "cliffquant_export.json"
    if not export_path.is_file():
        raise ValueError(f"{expected_policy} checkpoint is missing export provenance")
    export = json.loads(export_path.read_text(encoding="utf-8"))
    if not isinstance(export, dict) or export.get("schema") != "cliffquant.gptq-export.v1":
        raise ValueError(f"{expected_policy} checkpoint has invalid export provenance")
    if export.get("policy") != expected_policy:
        raise ValueError(
            f"checkpoint policy mismatch: expected {expected_policy}, "
            f"found {export.get('policy')!r}"
        )
    if export.get("base_model", {}).get("id") != BASE_MODEL_ID:
        raise ValueError(f"{expected_policy} checkpoint base-model id drift")
    if export.get("base_model", {}).get("revision") != BASE_MODEL_REVISION:
        raise ValueError(f"{expected_policy} checkpoint base-model revision drift")
    model_source = validate_model_snapshot_descriptor(
        export.get("base_model", {}).get("source"),
        model_id=BASE_MODEL_ID,
        revision=BASE_MODEL_REVISION,
    )
    if export.get("gptqmodel", {}).get("revision") != QWEN35_GPTQMODEL_TREE_REVISION:
        raise ValueError(f"{expected_policy} checkpoint GPTQModel revision drift")
    gptq = export.get("gptq")
    if not isinstance(gptq, dict) or {
        "bits": gptq.get("bits"),
        "format": gptq.get("format"),
        "group_size": gptq.get("group_size"),
        "method": gptq.get("method"),
        "sym": gptq.get("sym"),
        "zero": gptq.get("zero"),
    } != {
        "bits": 4,
        "format": "gptq",
        "group_size": 128,
        "method": "gptq",
        "sym": True,
        "zero": 8,
    }:
        raise ValueError(f"{expected_policy} checkpoint GPTQ contract drift")
    quantization = _standard_gptq_config(root)
    corpus_replay = validate_corpus_replay_pair(export.get("corpus_replay"))
    scale_run = export.get("scale_run")
    if (
        not isinstance(scale_run, dict)
        or not isinstance(scale_run.get("manifest_sha256"), str)
        or re.fullmatch(r"[0-9a-f]{64}", scale_run["manifest_sha256"]) is None
    ):
        raise ValueError(f"{expected_policy} checkpoint has no scale-run fingerprint")
    embedded = quantization.get("cliffquant")
    if not isinstance(embedded, dict):
        raise ValueError(f"{expected_policy} checkpoint has no embedded policy identity")
    if (
        embedded.get("policy") != expected_policy
        or embedded.get("scale_run_sha256") != scale_run["manifest_sha256"]
        or embedded.get("base_snapshot_sha256") != model_source["fingerprint_sha256"]
        or embedded.get("corpus_replay_sha256") != corpus_replay["sha256"]
        or embedded.get("base_model") != BASE_MODEL_ID
        or embedded.get("base_revision") != BASE_MODEL_REVISION
        or embedded.get("gptqmodel_revision") != QWEN35_GPTQMODEL_TREE_REVISION
    ):
        raise ValueError(f"{expected_policy} checkpoint embedded policy identity drift")
    descriptors = export.get("files")
    if not isinstance(descriptors, list):
        raise ValueError("export file descriptors must be a list")
    described_files: set[str] = set()
    for descriptor in descriptors:
        if not isinstance(descriptor, dict) or not isinstance(descriptor.get("file"), str):
            raise ValueError("invalid export file descriptor")
        if descriptor["file"] in described_files:
            raise ValueError(f"duplicate export file descriptor: {descriptor['file']}")
        described_files.add(descriptor["file"])
        candidate = (root / descriptor["file"]).resolve()
        if candidate != root and root not in candidate.parents:
            raise ValueError("export file descriptor escapes checkpoint")
        if not candidate.is_file():
            raise FileNotFoundError(f"export file is missing: {descriptor['file']}")
        if sha256_file(candidate) != descriptor.get("sha256"):
            raise ValueError(f"export file hash mismatch: {descriptor['file']}")
        if candidate.stat().st_size != descriptor.get("size_bytes"):
            raise ValueError(f"export file size mismatch: {descriptor['file']}")
    actual_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != export_path.name
    }
    if described_files != actual_files:
        missing = sorted(actual_files - described_files)
        extra = sorted(described_files - actual_files)
        raise ValueError(f"export file coverage mismatch; missing={missing}, extra={extra}")
    return {
        "export_manifest_sha256": sha256_file(export_path),
        "corpus_replay": corpus_replay,
        "model_source": model_source,
        "policy": expected_policy,
        "quantization": quantization,
        "scale_run_sha256": scale_run["manifest_sha256"],
    }


def _evaluate_checkpoint(
    *,
    checkpoint: Path,
    gptqmodel_source: Path,
    tokens: Mapping[str, tuple[NDArray[np.int64], NDArray[np.uint8]]],
    device: str,
    batch_size: int,
) -> tuple[dict[str, Any], dict[str, NDArray[np.generic]]]:
    try:
        import torch
        import torch.nn.functional as functional
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for held-out NLL evaluation") from exc

    _contract, wrapper = load_fresh_gptq_torch(
        checkpoint_dir=checkpoint,
        gptqmodel_source=gptqmodel_source,
        device=device,
    )
    model = wrapper.model
    model.eval()
    arrays: dict[str, NDArray[np.generic]] = {}
    metrics: dict[str, Any] = {}
    with torch.inference_mode():
        for environment in FROZEN_ENVIRONMENTS:
            input_ids, attention_mask = tokens[environment]
            token_nll = np.zeros(
                (input_ids.shape[0], input_ids.shape[1] - 1),
                dtype=np.float64,
            )
            target_mask = attention_mask[:, 1:].astype(np.uint8, copy=True)
            for start in range(0, input_ids.shape[0], batch_size):
                stop = min(start + batch_size, input_ids.shape[0])
                ids = torch.as_tensor(input_ids[start:stop], dtype=torch.long, device=device)
                mask = torch.as_tensor(
                    attention_mask[start:stop],
                    dtype=torch.long,
                    device=device,
                )
                outputs = model(
                    input_ids=ids,
                    attention_mask=mask,
                    use_cache=False,
                    return_dict=True,
                )
                logits = getattr(outputs, "logits", None)
                if logits is None or logits.ndim != 3:
                    raise RuntimeError(f"model returned invalid logits for {environment}")
                shifted_logits = logits[:, :-1, :].to(dtype=torch.float32)
                targets = ids[:, 1:]
                losses = functional.cross_entropy(
                    shifted_logits.reshape(-1, shifted_logits.shape[-1]),
                    targets.reshape(-1),
                    reduction="none",
                ).reshape(targets.shape)
                if not bool(torch.all(torch.isfinite(losses)).item()):
                    raise ValueError(f"non-finite held-out token NLL for {environment}")
                token_nll[start:stop] = losses.to(dtype=torch.float64, device="cpu").numpy()

            masked_nll = token_nll * target_mask.astype(np.float64)
            window_sums = np.sum(masked_nll, axis=1, dtype=np.float64)
            window_counts = np.sum(target_mask, axis=1, dtype=np.int64)
            if np.any(window_counts <= 0):
                raise ValueError(f"held-out window has no target tokens for {environment}")
            total_nll = float(np.sum(window_sums, dtype=np.float64))
            total_tokens = int(np.sum(window_counts, dtype=np.int64))
            metrics[environment] = {
                "mean_nll": total_nll / total_tokens,
                "target_tokens": total_tokens,
                "total_nll": total_nll,
                "windows": int(input_ids.shape[0]),
            }
            arrays[f"{environment}__target_mask"] = target_mask
            arrays[f"{environment}__token_nll"] = token_nll
            arrays[f"{environment}__window_nll_sum"] = window_sums
            arrays[f"{environment}__window_target_tokens"] = window_counts
    metrics["macro_mean_nll"] = float(
        np.mean([metrics[environment]["mean_nll"] for environment in FROZEN_ENVIRONMENTS])
    )

    del model
    del wrapper
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics, arrays


def compare_heldout_nll(
    *,
    cliffquant_checkpoint: str | Path,
    absmax_checkpoint: str | Path,
    heldout_manifest: str | Path,
    gptqmodel_source: str | Path,
    output_dir: str | Path,
    device: str = "cuda:0",
    batch_size: int = 1,
) -> dict[str, Any]:
    """Evaluate both checkpoints on identical tokens and apply frozen gates."""

    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    destination = Path(output_dir).resolve()
    report_path = destination / "heldout-nll.json"
    raw_path = destination / "heldout-nll.raw.npz"
    hashes_path = destination / "heldout-nll.sha256.json"
    if any(path.exists() for path in (report_path, raw_path, hashes_path)):
        raise FileExistsError("refusing to overwrite an existing held-out NLL result")
    destination.mkdir(parents=True, exist_ok=True)

    heldout_path = Path(heldout_manifest).resolve()
    corpus_replay = verify_frozen_corpus_pair(heldout_path.parent)
    manifest, tokens = load_heldout_tokens(heldout_path)
    heldout_identity = manifest["_validated"]
    replayed_heldout = corpus_replay["heldout"]
    if (
        heldout_identity["archive_sha256"] != replayed_heldout["token_archive"]["sha256"]
        or heldout_identity["manifest_sha256"] != replayed_heldout["manifest"]["sha256"]
        or heldout_identity["sidecar_sha256"] != replayed_heldout["sidecar"]["sha256"]
    ):
        raise ValueError("held-out token artifact disagrees with offline corpus replay")
    checkpoints = {
        "absmax": Path(absmax_checkpoint).resolve(),
        "cliffquant": Path(cliffquant_checkpoint).resolve(),
    }
    expected_policies = {
        "absmax": "absmax",
        "cliffquant": "cliffquant_minimax",
    }
    checkpoint_identities: dict[str, Any] = {}
    checkpoint_fingerprints: dict[str, Any] = {}
    for policy, checkpoint in checkpoints.items():
        if not checkpoint.is_dir():
            raise FileNotFoundError(f"{policy} checkpoint does not exist: {checkpoint}")
        checkpoint_identities[policy] = _export_identity(
            checkpoint,
            expected_policy=expected_policies[policy],
        )
        checkpoint_fingerprints[policy] = _checkpoint_fingerprint(checkpoint)
        if checkpoint_identities[policy]["corpus_replay"] != corpus_replay:
            raise ValueError(f"{policy} checkpoint corpus replay identity drift")
    if (
        checkpoint_fingerprints["absmax"]["sha256"]
        == checkpoint_fingerprints["cliffquant"]["sha256"]
    ):
        raise ValueError("AbsMax and CliffQuant checkpoints must have distinct fingerprints")
    if (
        checkpoint_fingerprints["absmax"]["model_sha256"]
        == checkpoint_fingerprints["cliffquant"]["model_sha256"]
    ):
        raise ValueError("AbsMax and CliffQuant model payloads must be distinct")
    if (
        checkpoint_identities["absmax"]["scale_run_sha256"]
        == checkpoint_identities["cliffquant"]["scale_run_sha256"]
    ):
        raise ValueError("AbsMax and CliffQuant scale runs must be distinct")
    if (
        checkpoint_identities["absmax"]["model_source"]
        != checkpoint_identities["cliffquant"]["model_source"]
    ):
        raise ValueError("AbsMax and CliffQuant checkpoints use different base model bytes")

    all_arrays: dict[str, NDArray[np.generic]] = {}
    policy_metrics: dict[str, Any] = {}
    # Evaluate baseline first so the gate direction remains explicit.
    for policy in ("absmax", "cliffquant"):
        metrics, arrays = _evaluate_checkpoint(
            checkpoint=checkpoints[policy],
            gptqmodel_source=Path(gptqmodel_source).resolve(),
            tokens=tokens,
            device=device,
            batch_size=batch_size,
        )
        policy_metrics[policy] = metrics
        for name, value in arrays.items():
            all_arrays[f"{policy}__{name}"] = value

    environment_deltas = {
        environment: (
            policy_metrics["cliffquant"][environment]["mean_nll"]
            - policy_metrics["absmax"][environment]["mean_nll"]
        )
        for environment in FROZEN_ENVIRONMENTS
    }
    macro_delta = (
        policy_metrics["cliffquant"]["macro_mean_nll"] - policy_metrics["absmax"]["macro_mean_nll"]
    )
    environment_gate = all(
        delta <= ENVIRONMENT_REGRESSION_LIMIT for delta in environment_deltas.values()
    )
    macro_gate = macro_delta <= MACRO_REGRESSION_LIMIT

    temporary = raw_path.with_name(f".{raw_path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **all_arrays)
    os.replace(temporary, raw_path)
    arrays_metadata = {
        name: {
            "dtype": value.dtype.str,
            "sha256": canonical_array_sha256(value),
            "shape": list(value.shape),
        }
        for name, value in sorted(all_arrays.items())
    }
    report = {
        "checkpoints": {
            policy: {
                "identity": checkpoint_identities[policy],
                "role": policy,
                **checkpoint_fingerprints[policy],
            }
            for policy in checkpoints
        },
        "comparison": {
            "environment_delta_cliffquant_minus_absmax": environment_deltas,
            "macro_delta_cliffquant_minus_absmax": macro_delta,
        },
        "corpus_replay": corpus_replay,
        "gates": {
            "environment": {
                "limit": ENVIRONMENT_REGRESSION_LIMIT,
                "pass": environment_gate,
            },
            "macro": {
                "limit": MACRO_REGRESSION_LIMIT,
                "pass": macro_gate,
            },
            "publishable_model_nll_gate": environment_gate and macro_gate,
        },
        "heldout": manifest["_validated"],
        "policies": policy_metrics,
        "raw_evidence": {
            "arrays": arrays_metadata,
            "file": raw_path.name,
            "sha256": sha256_file(raw_path),
            "size_bytes": raw_path.stat().st_size,
        },
        "runtime": {
            **runtime_metadata(device=device),
            "batch_size": batch_size,
            "teacher_forcing": True,
        },
        "schema": HELDOUT_NLL_SCHEMA,
        "status": "pass" if environment_gate and macro_gate else "fail",
    }
    report_path.write_bytes(canonical_json_bytes(report))
    hashes = {
        "inputs": {
            "corpus_replay_pair_sha256": corpus_replay["sha256"],
            "heldout_manifest_sha256": manifest["_validated"]["manifest_sha256"],
            "heldout_tokens_sha256": manifest["_validated"]["archive_sha256"],
        },
        "raw": {
            "file": raw_path.name,
            "sha256": sha256_file(raw_path),
            "size_bytes": raw_path.stat().st_size,
        },
        "report": {
            "file": report_path.name,
            "sha256": sha256_file(report_path),
            "size_bytes": report_path.stat().st_size,
        },
    }
    hashes_path.write_bytes(canonical_json_bytes(hashes))
    return report
