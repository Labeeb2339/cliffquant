"""Paired held-out token-NLL evaluation for CliffQuant and AbsMax GPTQ models."""

from __future__ import annotations

import ctypes
import errno
import gc
import json
import math
import os
import re
import shutil
import stat
import sys
import tempfile
import zipfile
from collections.abc import Mapping
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .experiment001_io import load_corpus_artifact
from .full_model_artifacts import (
    FROZEN_ENVIRONMENTS,
    validate_corpus_replay_pair,
    verify_frozen_corpus_pair,
)
from .full_model_verify import (
    describe_exported_checkpoint,
    load_fresh_gptq_torch,
    validate_exported_checkpoint_identity,
)
from .provenance import (
    canonical_array_sha256,
    canonical_json_bytes,
    runtime_metadata,
    sha256_bytes,
    sha256_file,
)

HELDOUT_NLL_SCHEMA = "cliffquant.heldout-nll-comparison.v1"
HELDOUT_NLL_SIDECAR_SCHEMA = "cliffquant.heldout-nll-sidecar.v1"
HELDOUT_NLL_EVALUATOR_SCHEMA = "cliffquant.heldout-nll-evaluator.v1"
EXPECTED_WINDOWS = 64
EXPECTED_WINDOW_TOKENS = 256
MACRO_REGRESSION_LIMIT = 0.01
ENVIRONMENT_REGRESSION_LIMIT = 0.02
DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG = ":4096:8"

_RESULT_FILES = frozenset(
    {
        "heldout-nll.json",
        "heldout-nll.raw.npz",
        "heldout-nll.sha256.json",
    }
)
_POLICY_ROLES = ("absmax", "cliffquant")
_EXPECTED_POLICY_NAMES = {
    "absmax": "absmax",
    "cliffquant": "cliffquant_minimax",
}
_NLL_RUNTIME_PACKAGE_VERSIONS = {
    "accelerate": "1.14.0",
    "datasets": "5.0.0",
    "gptqmodel": "7.3.4",
    "huggingface-hub": "1.24.0",
    "numpy": "2.2.6",
    "tokenizers": "0.22.2",
    "torch": "2.11.0+cu128",
    "transformers": "5.14.1",
}
_NLL_EVALUATOR_SOURCE_FILES = (
    "corpus.py",
    "corpus_replay.py",
    "dataset_viewer.py",
    "experiment001_io.py",
    "full_model_artifacts.py",
    "full_model_verify.py",
    "gptqmodel_export.py",
    "heldout_nll.py",
    "model_snapshot.py",
    "provenance.py",
    "qwen35_modules.py",
)
_RAW_SUFFIX_SPECS: dict[str, tuple[str, tuple[int, ...]]] = {
    "target_mask": ("|u1", (EXPECTED_WINDOWS, EXPECTED_WINDOW_TOKENS - 1)),
    "token_nll": ("<f8", (EXPECTED_WINDOWS, EXPECTED_WINDOW_TOKENS - 1)),
    "window_nll_sum": ("<f8", (EXPECTED_WINDOWS,)),
    "window_target_tokens": ("<i8", (EXPECTED_WINDOWS,)),
}
_MAX_RAW_NPY_HEADER_BYTES = 4_096
_MAX_RAW_ZIP_FIXED_OVERHEAD_BYTES = 65_536
_MAX_RAW_ZIP_ENTRY_OVERHEAD_BYTES = 1_024


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


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _model_payload_sha256(identity: Mapping[str, Any]) -> str:
    model_files = [
        descriptor
        for descriptor in identity["checkpoint_files"]
        if descriptor["file"].endswith(".safetensors")
    ]
    if not model_files:
        raise ValueError("checkpoint identity has no safetensors model payload")
    return sha256_bytes(canonical_json_bytes(model_files))


def _checkpoint_result_identity(checkpoint: Path, *, role: str) -> dict[str, Any]:
    identity = describe_exported_checkpoint(checkpoint)
    expected_policy = _EXPECTED_POLICY_NAMES[role]
    if identity["policy"] != expected_policy:
        raise ValueError(
            f"checkpoint policy mismatch: expected {expected_policy}, found {identity['policy']!r}"
        )
    return {
        "identity": identity,
        "model_payload_sha256": _model_payload_sha256(identity),
        "role": role,
    }


def _nll_evaluator_identity() -> dict[str, Any]:
    source_root = Path(__file__).resolve().parent
    files = []
    for name in _NLL_EVALUATOR_SOURCE_FILES:
        path = source_root / name
        if not path.is_file():
            raise FileNotFoundError(f"held-out NLL evaluator source file is missing: {name}")
        files.append(
            {
                "file": name,
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    payload = {
        "files": files,
        "schema": HELDOUT_NLL_EVALUATOR_SCHEMA,
    }
    return {
        **payload,
        "sha256": sha256_bytes(canonical_json_bytes(payload)),
    }


def _validate_current_nll_evaluator(value: Any) -> dict[str, Any]:
    evaluator = _require_exact_fields(
        value,
        {"files", "schema", "sha256"},
        label="held-out NLL evaluator",
    )
    if evaluator["schema"] != HELDOUT_NLL_EVALUATOR_SCHEMA:
        raise ValueError("held-out NLL evaluator schema drift")
    files = evaluator["files"]
    if not isinstance(files, list) or len(files) != len(_NLL_EVALUATOR_SOURCE_FILES):
        raise ValueError("held-out NLL evaluator source coverage drift")
    names = []
    for descriptor in files:
        validated = _validate_result_descriptor(
            descriptor,
            expected_file=descriptor.get("file") if isinstance(descriptor, Mapping) else "",
            label="held-out NLL evaluator source",
        )
        names.append(validated["file"])
    if tuple(names) != _NLL_EVALUATOR_SOURCE_FILES:
        raise ValueError("held-out NLL evaluator source order drift")
    payload = {"files": files, "schema": evaluator["schema"]}
    if evaluator["sha256"] != sha256_bytes(canonical_json_bytes(payload)):
        raise ValueError("held-out NLL evaluator aggregate drift")
    current = _nll_evaluator_identity()
    if dict(evaluator) != current:
        raise ValueError("held-out NLL evaluator differs from current source bytes")
    return current


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
        cache_dequantized_weights=True,
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


def _configure_deterministic_evaluation(device: str) -> None:
    """Configure the exact deterministic contract recorded by CUDA evaluations."""

    if not device.startswith("cuda:"):
        return
    prior_workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for held-out NLL evaluation") from exc
    if torch.cuda.is_initialized() and prior_workspace != DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG:
        raise RuntimeError(
            "CUDA was initialized before the deterministic cuBLAS workspace contract"
        )
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)


def _compare_heldout_nll_into_directory(
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
    _configure_deterministic_evaluation(device)
    evaluation_runtime = {
        **runtime_metadata(
            device=device,
            package_names=tuple(_NLL_RUNTIME_PACKAGE_VERSIONS),
        ),
        "batch_size": batch_size,
        "dequantized_weight_cache": True,
        "teacher_forcing": True,
    }
    _validate_runtime(evaluation_runtime)
    output_candidate = Path(output_dir)
    _reject_symlink_path_components(output_candidate, label="held-out NLL output path")
    destination = output_candidate.resolve()
    report_path = destination / "heldout-nll.json"
    raw_path = destination / "heldout-nll.raw.npz"
    hashes_path = destination / "heldout-nll.sha256.json"
    if destination.exists():
        if destination.is_symlink() or not destination.is_dir():
            raise ValueError("held-out NLL output must be a regular directory")
        if any(destination.iterdir()):
            raise FileExistsError("refusing to write into a non-empty held-out NLL directory")
    else:
        destination.mkdir(parents=True)

    heldout_candidate = Path(heldout_manifest)
    _reject_symlink_path_components(
        heldout_candidate,
        label="held-out NLL manifest path",
    )
    heldout_path = heldout_candidate.resolve()
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
    evaluator_identity = _nll_evaluator_identity()
    checkpoint_candidates = {
        "absmax": Path(absmax_checkpoint),
        "cliffquant": Path(cliffquant_checkpoint),
    }
    for policy, candidate in checkpoint_candidates.items():
        _reject_symlink_path_components(candidate, label=f"{policy} checkpoint path")
    checkpoints = {
        policy: candidate.resolve() for policy, candidate in checkpoint_candidates.items()
    }
    gptqmodel_candidate = Path(gptqmodel_source)
    _reject_symlink_path_components(gptqmodel_candidate, label="GPTQModel source path")
    gptqmodel_root = gptqmodel_candidate.resolve()
    checkpoint_results: dict[str, Any] = {}
    for policy, checkpoint in checkpoints.items():
        if not checkpoint.is_dir():
            raise FileNotFoundError(f"{policy} checkpoint does not exist: {checkpoint}")
        checkpoint_results[policy] = _checkpoint_result_identity(
            checkpoint,
            role=policy,
        )
        if checkpoint_results[policy]["identity"]["corpus_replay"] != corpus_replay:
            raise ValueError(f"{policy} checkpoint corpus replay identity drift")
    if (
        checkpoint_results["absmax"]["identity"]["sha256"]
        == checkpoint_results["cliffquant"]["identity"]["sha256"]
    ):
        raise ValueError("AbsMax and CliffQuant checkpoints must have distinct fingerprints")
    if (
        checkpoint_results["absmax"]["model_payload_sha256"]
        == checkpoint_results["cliffquant"]["model_payload_sha256"]
    ):
        raise ValueError("AbsMax and CliffQuant model payloads must be distinct")
    if (
        checkpoint_results["absmax"]["identity"]["scale_run"]["manifest_sha256"]
        == checkpoint_results["cliffquant"]["identity"]["scale_run"]["manifest_sha256"]
    ):
        raise ValueError("AbsMax and CliffQuant scale runs must be distinct")
    if (
        checkpoint_results["absmax"]["identity"]["base_model"]
        != checkpoint_results["cliffquant"]["identity"]["base_model"]
    ):
        raise ValueError("AbsMax and CliffQuant checkpoints use different base model bytes")

    all_arrays: dict[str, NDArray[np.generic]] = {}
    policy_metrics: dict[str, Any] = {}
    # Evaluate baseline first so the gate direction remains explicit.
    for policy in ("absmax", "cliffquant"):
        metrics, arrays = _evaluate_checkpoint(
            checkpoint=checkpoints[policy],
            gptqmodel_source=gptqmodel_root,
            tokens=tokens,
            device=device,
            batch_size=batch_size,
        )
        policy_metrics[policy] = metrics
        for name, value in arrays.items():
            all_arrays[f"{policy}__{name}"] = value

    for policy, checkpoint in checkpoints.items():
        if _checkpoint_result_identity(checkpoint, role=policy) != checkpoint_results[policy]:
            raise ValueError(f"{policy} checkpoint changed during held-out NLL evaluation")
    if _nll_evaluator_identity() != evaluator_identity:
        raise ValueError("held-out NLL evaluator source changed during evaluation")

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
        "checkpoints": checkpoint_results,
        "comparison": {
            "environment_delta_cliffquant_minus_absmax": environment_deltas,
            "macro_delta_cliffquant_minus_absmax": macro_delta,
        },
        "corpus_replay": corpus_replay,
        "evaluator": evaluator_identity,
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
        "runtime": evaluation_runtime,
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
        "schema": HELDOUT_NLL_SIDECAR_SCHEMA,
    }
    hashes_path.write_bytes(canonical_json_bytes(hashes))
    return report


def _atomic_rename_directory_no_replace(source: Path, destination: Path) -> None:
    """Atomically publish a directory while refusing to replace any destination."""

    if os.path.lexists(destination):
        raise FileExistsError(f"held-out NLL destination already exists: {destination}")
    if os.name == "nt":
        # Windows rename is atomic and fails when the destination already exists.
        os.rename(source, destination)
        return

    source_bytes = os.fsencode(source)
    destination_bytes = os.fsencode(destination)
    libc = ctypes.CDLL(None, use_errno=True)
    at_fdcwd = -100
    if sys.platform.startswith("linux"):
        try:
            rename_no_replace = libc.renameat2
        except AttributeError as exc:
            raise RuntimeError(
                "this Linux runtime lacks atomic no-replace directory publication"
            ) from exc
        rename_no_replace.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        rename_no_replace.restype = ctypes.c_int
        result = rename_no_replace(
            at_fdcwd,
            source_bytes,
            at_fdcwd,
            destination_bytes,
            1,  # RENAME_NOREPLACE
        )
    elif sys.platform == "darwin":
        try:
            rename_no_replace = libc.renameatx_np
        except AttributeError as exc:
            raise RuntimeError(
                "this macOS runtime lacks atomic no-replace directory publication"
            ) from exc
        rename_no_replace.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        rename_no_replace.restype = ctypes.c_int
        result = rename_no_replace(
            at_fdcwd,
            source_bytes,
            at_fdcwd,
            destination_bytes,
            4,  # RENAME_EXCL
        )
    else:
        raise RuntimeError(
            "atomic no-replace directory publication is unsupported on this platform"
        )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(
            error_number,
            os.strerror(error_number),
            destination,
        )
    raise OSError(error_number, os.strerror(error_number), destination)


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
    """Build, validate, and atomically publish paired held-out NLL evidence."""

    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    output_candidate = Path(output_dir)
    _reject_symlink_path_components(output_candidate, label="held-out NLL output path")
    destination = output_candidate.resolve()
    if os.path.lexists(destination):
        raise FileExistsError("held-out NLL output directory must be absent")
    destination.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_path_components(destination.parent, label="held-out NLL output parent")
    if not destination.parent.is_dir():
        raise ValueError("held-out NLL output parent must be a regular directory")

    stage = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.staging-",
            dir=destination.parent,
        )
    )
    try:
        report = _compare_heldout_nll_into_directory(
            cliffquant_checkpoint=cliffquant_checkpoint,
            absmax_checkpoint=absmax_checkpoint,
            heldout_manifest=heldout_manifest,
            gptqmodel_source=gptqmodel_source,
            output_dir=stage,
            device=device,
            batch_size=batch_size,
        )
        validated = validate_heldout_nll_result(stage)
        if validated != report:
            raise ValueError("staged held-out NLL evidence differs after validation")
        _atomic_rename_directory_no_replace(stage, destination)
        return report
    finally:
        if os.path.lexists(stage):
            if _is_symlink_or_reparse_point(stage):
                if stat.S_ISDIR(stage.lstat().st_mode):
                    os.rmdir(stage)
                else:
                    stage.unlink()
            else:
                shutil.rmtree(stage)


def _read_canonical_json_object(path: Path, *, label: str) -> dict[str, Any]:
    payload = path.read_bytes()
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    if payload != canonical_json_bytes(value):
        raise ValueError(f"{label} is not canonical JSON")
    return value


def _require_exact_fields(value: Any, expected: set[str], *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError(f"{label} fields drift")
    return value


def _validate_result_descriptor(
    value: Any,
    *,
    expected_file: str,
    label: str,
) -> dict[str, Any]:
    descriptor = _require_exact_fields(
        value,
        {"file", "sha256", "size_bytes"},
        label=label,
    )
    if (
        descriptor["file"] != expected_file
        or not _is_sha256(descriptor["sha256"])
        or type(descriptor["size_bytes"]) is not int
        or descriptor["size_bytes"] <= 0
    ):
        raise ValueError(f"{label} identity drift")
    return dict(descriptor)


def _reject_absolute_strings(value: Any, *, label: str = "held-out NLL evidence") -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            _reject_absolute_strings(key, label=label)
            _reject_absolute_strings(nested, label=label)
        return
    if isinstance(value, list):
        for nested in value:
            _reject_absolute_strings(nested, label=label)
        return
    if isinstance(value, str):
        windows_path = PureWindowsPath(value)
        if (
            Path(value).is_absolute()
            or PurePosixPath(value).is_absolute()
            or windows_path.is_absolute()
            or bool(windows_path.drive)
            or bool(windows_path.root)
            or value.casefold().startswith("file:")
        ):
            raise ValueError(f"{label} contains an absolute path")


def _is_symlink_or_reparse_point(path: Path) -> bool:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return False
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    file_attributes = getattr(metadata, "st_file_attributes", 0)
    return path.is_symlink() or bool(reparse_flag and file_attributes & reparse_flag)


def _reject_symlink_path_components(path: Path, *, label: str) -> None:
    absolute = Path(os.path.abspath(path))
    for component in (absolute, *absolute.parents):
        if _is_symlink_or_reparse_point(component):
            raise ValueError(f"{label} must not contain symlink or reparse-point components")


def _validate_runtime(value: Any) -> None:
    common_fields = {
        "batch_size",
        "dequantized_weight_cache",
        "device",
        "implementation",
        "machine",
        "numpy",
        "numpy_build",
        "packages",
        "platform",
        "python",
        "python_executable",
        "teacher_forcing",
    }
    if not isinstance(value, Mapping) or not isinstance(value.get("device"), str):
        raise ValueError("held-out NLL runtime fields drift")
    device = value["device"]
    cuda_match = re.fullmatch(r"cuda:(0|[1-9][0-9]*)", device)
    if device != "cpu" and cuda_match is None:
        raise ValueError("held-out NLL runtime device drift")
    expected_fields = common_fields | ({"accelerator"} if cuda_match is not None else set())
    runtime = _require_exact_fields(value, expected_fields, label="held-out NLL runtime")
    if (
        type(runtime["batch_size"]) is not int
        or runtime["batch_size"] <= 0
        or runtime["dequantized_weight_cache"] is not True
        or runtime["teacher_forcing"] is not True
    ):
        raise ValueError("held-out NLL runtime evaluation contract drift")
    for field in ("implementation", "numpy", "platform", "python", "python_executable"):
        if not isinstance(runtime[field], str) or not runtime[field]:
            raise ValueError(f"held-out NLL runtime {field} drift")
    if not isinstance(runtime["machine"], str):
        raise ValueError("held-out NLL runtime machine drift")
    executable = Path(runtime["python_executable"])
    if executable.is_absolute() or executable.name != runtime["python_executable"]:
        raise ValueError("held-out NLL runtime executable path drift")

    packages = _require_exact_fields(
        runtime["packages"],
        set(_NLL_RUNTIME_PACKAGE_VERSIONS),
        label="held-out NLL runtime packages",
    )
    if dict(packages) != _NLL_RUNTIME_PACKAGE_VERSIONS:
        raise ValueError("held-out NLL runtime package inventory drift")
    if packages["numpy"] != runtime["numpy"]:
        raise ValueError("held-out NLL runtime NumPy identity drift")

    numpy_build = _require_exact_fields(
        runtime["numpy_build"],
        {"blas", "host", "lapack", "simd"},
        label="held-out NLL NumPy build",
    )
    dependency_fields = {"found", "name", "openblas configuration", "version"}
    _require_exact_fields(
        numpy_build["blas"],
        dependency_fields,
        label="held-out NLL NumPy BLAS build",
    )
    _require_exact_fields(
        numpy_build["lapack"],
        dependency_fields,
        label="held-out NLL NumPy LAPACK build",
    )
    if not isinstance(numpy_build["host"], Mapping) or not isinstance(numpy_build["simd"], Mapping):
        raise ValueError("held-out NLL NumPy build metadata drift")

    if cuda_match is None:
        return
    accelerator = runtime["accelerator"]
    base_fields = {
        "available",
        "cuda_runtime",
        "deterministic_algorithms",
        "deterministic_algorithms_warn_only",
    }
    if not isinstance(accelerator, Mapping) or accelerator.get("available") is not True:
        raise ValueError("held-out NLL accelerator fields drift")
    available_fields = {
        "capability",
        "cublas_workspace_config",
        "cudnn_allow_tf32",
        "cudnn_benchmark",
        "cudnn_deterministic",
        "cudnn_version",
        "device_index",
        "float32_matmul_precision",
        "matmul_allow_tf32",
        "name",
        "total_memory_bytes",
    }
    expected_accelerator_fields = base_fields | available_fields
    _require_exact_fields(
        accelerator,
        expected_accelerator_fields,
        label="held-out NLL accelerator",
    )
    if (
        type(accelerator["deterministic_algorithms"]) is not bool
        or type(accelerator["deterministic_algorithms_warn_only"]) is not bool
        or (
            accelerator["cuda_runtime"] is not None
            and not isinstance(accelerator["cuda_runtime"], str)
        )
    ):
        raise ValueError("held-out NLL accelerator identity drift")
    capability = accelerator["capability"]
    if (
        not isinstance(capability, list)
        or len(capability) != 2
        or not all(type(item) is int and item >= 0 for item in capability)
        or type(accelerator["device_index"]) is not int
        or accelerator["device_index"] != int(cuda_match.group(1))
        or not isinstance(accelerator["float32_matmul_precision"], str)
        or not accelerator["float32_matmul_precision"]
        or not isinstance(accelerator["name"], str)
        or not accelerator["name"]
        or type(accelerator["total_memory_bytes"]) is not int
        or accelerator["total_memory_bytes"] <= 0
    ):
        raise ValueError("held-out NLL accelerator device identity drift")
    for field in (
        "cudnn_allow_tf32",
        "cudnn_benchmark",
        "cudnn_deterministic",
        "matmul_allow_tf32",
    ):
        if type(accelerator[field]) is not bool:
            raise ValueError(f"held-out NLL accelerator {field} drift")
    if (
        accelerator["cublas_workspace_config"] is not None
        and not isinstance(accelerator["cublas_workspace_config"], str)
    ) or (type(accelerator["cudnn_version"]) is not int or accelerator["cudnn_version"] <= 0):
        raise ValueError("held-out NLL accelerator library identity drift")
    if not isinstance(accelerator["cuda_runtime"], str) or not accelerator["cuda_runtime"]:
        raise ValueError("held-out NLL CUDA runtime identity drift")
    deterministic_contract = {
        "cublas_workspace_config": DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG,
        "cudnn_allow_tf32": False,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "deterministic_algorithms": True,
        "deterministic_algorithms_warn_only": False,
        "float32_matmul_precision": "highest",
        "matmul_allow_tf32": False,
    }
    if any(accelerator[field] != expected for field, expected in deterministic_contract.items()):
        raise ValueError("held-out NLL deterministic CUDA contract drift")


def _validate_checkpoint_results(value: Any, corpus_replay: Mapping[str, Any]) -> None:
    checkpoints = _require_exact_fields(
        value,
        set(_POLICY_ROLES),
        label="held-out NLL checkpoints",
    )
    identities: dict[str, dict[str, Any]] = {}
    model_payload_sha256: dict[str, str] = {}
    for role in _POLICY_ROLES:
        entry = _require_exact_fields(
            checkpoints[role],
            {"identity", "model_payload_sha256", "role"},
            label=f"{role} checkpoint result",
        )
        if entry["role"] != role:
            raise ValueError(f"{role} checkpoint role drift")
        identity = validate_exported_checkpoint_identity(entry["identity"])
        if identity["policy"] != _EXPECTED_POLICY_NAMES[role]:
            raise ValueError(f"{role} checkpoint policy drift")
        if identity["corpus_replay"] != corpus_replay:
            raise ValueError(f"{role} checkpoint corpus replay identity drift")
        expected_model_sha256 = _model_payload_sha256(identity)
        if entry["model_payload_sha256"] != expected_model_sha256:
            raise ValueError(f"{role} checkpoint model payload fingerprint drift")
        identities[role] = identity
        model_payload_sha256[role] = expected_model_sha256

    if identities["absmax"]["sha256"] == identities["cliffquant"]["sha256"]:
        raise ValueError("AbsMax and CliffQuant checkpoint fingerprints are identical")
    if model_payload_sha256["absmax"] == model_payload_sha256["cliffquant"]:
        raise ValueError("AbsMax and CliffQuant model payloads are identical")
    if (
        identities["absmax"]["scale_run"]["manifest_sha256"]
        == identities["cliffquant"]["scale_run"]["manifest_sha256"]
    ):
        raise ValueError("AbsMax and CliffQuant scale-run fingerprints are identical")
    if identities["absmax"]["base_model"] != identities["cliffquant"]["base_model"]:
        raise ValueError("AbsMax and CliffQuant base-model identities differ")
    if identities["absmax"]["gptq"] != identities["cliffquant"]["gptq"]:
        raise ValueError("AbsMax and CliffQuant GPTQ contracts differ")
    if identities["absmax"]["gptqmodel"] != identities["cliffquant"]["gptqmodel"]:
        raise ValueError("AbsMax and CliffQuant GPTQModel identities differ")


def _validate_heldout_identity(
    value: Any,
    *,
    corpus_replay: Mapping[str, Any],
) -> None:
    heldout = _require_exact_fields(
        value,
        {
            "archive_file",
            "archive_sha256",
            "manifest_file",
            "manifest_sha256",
            "sidecar_sha256",
        },
        label="held-out token identity",
    )
    replay = corpus_replay["heldout"]
    expected = {
        "archive_file": replay["token_archive"]["file"],
        "archive_sha256": replay["token_archive"]["sha256"],
        "manifest_file": replay["manifest"]["file"],
        "manifest_sha256": replay["manifest"]["sha256"],
        "sidecar_sha256": replay["sidecar"]["sha256"],
    }
    if dict(heldout) != expected:
        raise ValueError("held-out token identity differs from corpus replay")


def _expected_raw_keys() -> set[str]:
    return {
        f"{policy}__{environment}__{suffix}"
        for policy in _POLICY_ROLES
        for environment in FROZEN_ENVIRONMENTS
        for suffix in _RAW_SUFFIX_SPECS
    }


def _preflight_raw_npz(raw_path: Path, expected_keys: set[str]) -> None:
    expected_entries = {f"{name}.npy": name for name in expected_keys}
    expected_payload_bytes = {
        f"{name}.npy": int(np.prod(_RAW_SUFFIX_SPECS[name.rsplit("__", 1)[1]][1]))
        * np.dtype(_RAW_SUFFIX_SPECS[name.rsplit("__", 1)[1]][0]).itemsize
        for name in expected_keys
    }
    maximum_archive_bytes = (
        sum(
            payload_bytes + _MAX_RAW_NPY_HEADER_BYTES
            for payload_bytes in expected_payload_bytes.values()
        )
        + _MAX_RAW_ZIP_FIXED_OVERHEAD_BYTES
        + len(expected_entries) * _MAX_RAW_ZIP_ENTRY_OVERHEAD_BYTES
    )
    if raw_path.stat().st_size > maximum_archive_bytes:
        raise ValueError("held-out NLL raw NPZ exceeds the conservative size limit")

    try:
        archive = zipfile.ZipFile(raw_path, mode="r")
    except (OSError, zipfile.BadZipFile) as exc:
        raise ValueError("held-out NLL raw evidence is not a readable NPZ archive") from exc
    with archive:
        infos = archive.infolist()
        filenames = [info.filename for info in infos]
        if len(filenames) != len(expected_entries) or set(filenames) != set(expected_entries):
            raise ValueError(
                "held-out NLL raw NPZ entry inventory has duplicate, extra, or missing entries"
            )
        for info in infos:
            payload_bytes = expected_payload_bytes[info.filename]
            if (
                info.is_dir()
                or info.flag_bits & 0x1
                or info.compress_type != zipfile.ZIP_DEFLATED
                or info.file_size <= payload_bytes
                or info.file_size > payload_bytes + _MAX_RAW_NPY_HEADER_BYTES
            ):
                raise ValueError(f"held-out NLL raw NPZ entry metadata drift: {info.filename}")
            try:
                with archive.open(info, mode="r") as member:
                    version = np.lib.format.read_magic(member)
                    if version == (1, 0):
                        shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(
                            member,
                            max_header_size=_MAX_RAW_NPY_HEADER_BYTES,
                        )
                    elif version == (2, 0):
                        shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(
                            member,
                            max_header_size=_MAX_RAW_NPY_HEADER_BYTES,
                        )
                    else:
                        raise ValueError("unsupported NPY format version")
                    header_and_magic_bytes = member.tell()
            except (EOFError, OSError, ValueError, zipfile.BadZipFile) as exc:
                raise ValueError(
                    f"held-out NLL raw NPZ entry header is unsafe: {info.filename}"
                ) from exc
            array_name = expected_entries[info.filename]
            suffix = array_name.rsplit("__", 1)[1]
            expected_dtype, expected_shape = _RAW_SUFFIX_SPECS[suffix]
            if (
                dtype.hasobject
                or dtype.str != expected_dtype
                or tuple(shape) != expected_shape
                or fortran_order is not False
                or info.file_size != header_and_magic_bytes + payload_bytes
            ):
                raise ValueError(f"held-out NLL raw NPZ array header drift: {array_name}")


def _load_and_validate_raw_arrays(
    raw_path: Path,
    raw_evidence: Mapping[str, Any],
) -> dict[str, NDArray[np.generic]]:
    expected_keys = _expected_raw_keys()
    metadata = _require_exact_fields(
        raw_evidence["arrays"],
        expected_keys,
        label="held-out NLL raw array metadata",
    )
    _preflight_raw_npz(raw_path, expected_keys)
    loaded: dict[str, NDArray[np.generic]] = {}
    try:
        archive = np.load(raw_path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError("held-out NLL raw evidence is not a readable NPZ archive") from exc
    with archive:
        if len(archive.files) != len(expected_keys) or set(archive.files) != expected_keys:
            raise ValueError("held-out NLL raw array coverage drift")
        for name in sorted(expected_keys):
            try:
                value = archive[name]
            except ValueError as exc:
                raise ValueError(f"held-out NLL raw array is unsafe or unreadable: {name}") from exc
            suffix = name.rsplit("__", 1)[1]
            expected_dtype, expected_shape = _RAW_SUFFIX_SPECS[suffix]
            descriptor = _require_exact_fields(
                metadata[name],
                {"dtype", "sha256", "shape"},
                label=f"held-out NLL raw array descriptor {name}",
            )
            if (
                value.dtype.str != expected_dtype
                or value.shape != expected_shape
                or descriptor["dtype"] != expected_dtype
                or descriptor["shape"] != list(expected_shape)
                or not _is_sha256(descriptor["sha256"])
                or descriptor["sha256"] != canonical_array_sha256(value)
            ):
                raise ValueError(f"held-out NLL raw array identity drift: {name}")
            loaded[name] = np.ascontiguousarray(value).copy()
    return loaded


def _require_report_float(value: Any, expected: float, *, label: str) -> None:
    if type(value) is not float or not math.isfinite(value) or value.hex() != expected.hex():
        raise ValueError(f"{label} drift")


def _recompute_policy_metrics(
    arrays: Mapping[str, NDArray[np.generic]],
) -> dict[str, dict[str, Any]]:
    metrics: dict[str, dict[str, Any]] = {}
    reference_masks: dict[str, NDArray[np.uint8]] = {}
    for policy in _POLICY_ROLES:
        policy_metrics: dict[str, Any] = {}
        for environment in FROZEN_ENVIRONMENTS:
            prefix = f"{policy}__{environment}__"
            target_mask = arrays[f"{prefix}target_mask"]
            token_nll = arrays[f"{prefix}token_nll"]
            window_sums = arrays[f"{prefix}window_nll_sum"]
            window_counts = arrays[f"{prefix}window_target_tokens"]
            if not np.all(target_mask == 1):
                raise ValueError(f"{policy}/{environment} target mask drift")
            if not np.all(np.isfinite(token_nll)) or np.any(token_nll < 0.0):
                raise ValueError(f"{policy}/{environment} token NLL is non-finite or negative")
            expected_window_sums = np.sum(
                token_nll * target_mask.astype(np.float64),
                axis=1,
                dtype=np.float64,
            )
            expected_window_counts = np.sum(target_mask, axis=1, dtype=np.int64)
            if not np.all(np.isfinite(window_sums)):
                raise ValueError(f"{policy}/{environment} window NLL totals are non-finite")
            if not np.array_equal(window_sums, expected_window_sums):
                raise ValueError(f"{policy}/{environment} window NLL totals drift")
            if not np.array_equal(window_counts, expected_window_counts):
                raise ValueError(f"{policy}/{environment} window token counts drift")
            if np.any(window_counts <= 0):
                raise ValueError(f"{policy}/{environment} contains an empty held-out window")
            if policy == "absmax":
                reference_masks[environment] = target_mask
            elif not np.array_equal(target_mask, reference_masks[environment]):
                raise ValueError(f"{environment} target masks differ between policies")

            total_nll = float(np.sum(window_sums, dtype=np.float64))
            total_tokens = int(np.sum(window_counts, dtype=np.int64))
            if not math.isfinite(total_nll):
                raise ValueError(f"{policy}/{environment} total NLL is non-finite")
            policy_metrics[environment] = {
                "mean_nll": total_nll / total_tokens,
                "target_tokens": total_tokens,
                "total_nll": total_nll,
                "windows": EXPECTED_WINDOWS,
            }
        policy_metrics["macro_mean_nll"] = float(
            np.mean(
                [policy_metrics[environment]["mean_nll"] for environment in FROZEN_ENVIRONMENTS]
            )
        )
        metrics[policy] = policy_metrics
    return metrics


def _validate_saved_metrics_and_gates(
    report: Mapping[str, Any],
    recomputed: Mapping[str, Mapping[str, Any]],
) -> None:
    policies = _require_exact_fields(
        report["policies"],
        set(_POLICY_ROLES),
        label="held-out NLL policies",
    )
    metric_fields = {"mean_nll", "target_tokens", "total_nll", "windows"}
    for policy in _POLICY_ROLES:
        saved_policy = _require_exact_fields(
            policies[policy],
            {*FROZEN_ENVIRONMENTS, "macro_mean_nll"},
            label=f"{policy} NLL metrics",
        )
        for environment in FROZEN_ENVIRONMENTS:
            saved_metric = _require_exact_fields(
                saved_policy[environment],
                metric_fields,
                label=f"{policy}/{environment} NLL metric",
            )
            expected_metric = recomputed[policy][environment]
            _require_report_float(
                saved_metric["mean_nll"],
                expected_metric["mean_nll"],
                label=f"{policy}/{environment} mean NLL",
            )
            _require_report_float(
                saved_metric["total_nll"],
                expected_metric["total_nll"],
                label=f"{policy}/{environment} total NLL",
            )
            if (
                type(saved_metric["target_tokens"]) is not int
                or saved_metric["target_tokens"] != expected_metric["target_tokens"]
                or type(saved_metric["windows"]) is not int
                or saved_metric["windows"] != EXPECTED_WINDOWS
            ):
                raise ValueError(f"{policy}/{environment} NLL count drift")
        _require_report_float(
            saved_policy["macro_mean_nll"],
            recomputed[policy]["macro_mean_nll"],
            label=f"{policy} macro mean NLL",
        )

    environment_deltas = {
        environment: (
            recomputed["cliffquant"][environment]["mean_nll"]
            - recomputed["absmax"][environment]["mean_nll"]
        )
        for environment in FROZEN_ENVIRONMENTS
    }
    macro_delta = (
        recomputed["cliffquant"]["macro_mean_nll"] - recomputed["absmax"]["macro_mean_nll"]
    )
    comparison = _require_exact_fields(
        report["comparison"],
        {
            "environment_delta_cliffquant_minus_absmax",
            "macro_delta_cliffquant_minus_absmax",
        },
        label="held-out NLL comparison",
    )
    saved_deltas = _require_exact_fields(
        comparison["environment_delta_cliffquant_minus_absmax"],
        set(FROZEN_ENVIRONMENTS),
        label="held-out NLL environment deltas",
    )
    for environment, expected_delta in environment_deltas.items():
        _require_report_float(
            saved_deltas[environment],
            expected_delta,
            label=f"{environment} NLL delta",
        )
    _require_report_float(
        comparison["macro_delta_cliffquant_minus_absmax"],
        macro_delta,
        label="macro NLL delta",
    )

    environment_pass = all(
        delta <= ENVIRONMENT_REGRESSION_LIMIT for delta in environment_deltas.values()
    )
    macro_pass = macro_delta <= MACRO_REGRESSION_LIMIT
    gates = _require_exact_fields(
        report["gates"],
        {"environment", "macro", "publishable_model_nll_gate"},
        label="held-out NLL gates",
    )
    environment_gate = _require_exact_fields(
        gates["environment"],
        {"limit", "pass"},
        label="held-out NLL environment gate",
    )
    macro_gate = _require_exact_fields(
        gates["macro"],
        {"limit", "pass"},
        label="held-out NLL macro gate",
    )
    if (
        type(environment_gate["limit"]) is not float
        or type(environment_gate["pass"]) is not bool
        or type(macro_gate["limit"]) is not float
        or type(macro_gate["pass"]) is not bool
        or type(gates["publishable_model_nll_gate"]) is not bool
    ):
        raise ValueError("held-out NLL gate value types drift")
    expected_gates = {
        "environment": {
            "limit": ENVIRONMENT_REGRESSION_LIMIT,
            "pass": environment_pass,
        },
        "macro": {
            "limit": MACRO_REGRESSION_LIMIT,
            "pass": macro_pass,
        },
        "publishable_model_nll_gate": environment_pass and macro_pass,
    }
    if gates != expected_gates:
        raise ValueError("held-out NLL frozen gates drift")
    expected_status = "pass" if environment_pass and macro_pass else "fail"
    if report["status"] != expected_status:
        raise ValueError("held-out NLL status disagrees with frozen gates")


def _validate_live_nll_bindings(
    report: Mapping[str, Any],
    arrays: Mapping[str, NDArray[np.generic]],
    *,
    heldout_manifest: str | Path,
    absmax_checkpoint: str | Path,
    cliffquant_checkpoint: str | Path,
) -> None:
    heldout_candidate = Path(heldout_manifest)
    _reject_symlink_path_components(
        heldout_candidate,
        label="live held-out manifest path",
    )
    heldout_path = heldout_candidate.resolve(strict=True)
    if heldout_path.name != "heldout.manifest.json":
        raise ValueError("live held-out manifest filename drift")
    live_corpus_replay = verify_frozen_corpus_pair(heldout_path.parent)
    if live_corpus_replay != report["corpus_replay"]:
        raise ValueError("live frozen corpus replay differs from held-out NLL report")
    live_manifest, live_tokens = load_heldout_tokens(heldout_path)
    if live_manifest["_validated"] != report["heldout"]:
        raise ValueError("live held-out token identity differs from NLL report")
    for role in _POLICY_ROLES:
        for environment in FROZEN_ENVIRONMENTS:
            expected_mask = np.ascontiguousarray(
                live_tokens[environment][1][:, 1:],
                dtype=np.uint8,
            )
            if not np.array_equal(
                arrays[f"{role}__{environment}__target_mask"],
                expected_mask,
            ):
                raise ValueError(f"{role}/{environment} mask differs from live held-out corpus")

    checkpoint_values = {
        "absmax": Path(absmax_checkpoint),
        "cliffquant": Path(cliffquant_checkpoint),
    }
    resolved_checkpoints: dict[str, Path] = {}
    for role, candidate in checkpoint_values.items():
        _reject_symlink_path_components(candidate, label=f"live {role} checkpoint path")
        root = candidate.resolve(strict=True)
        if not root.is_dir():
            raise ValueError(f"live {role} checkpoint must be a directory")
        live_identity = _checkpoint_result_identity(root, role=role)
        if live_identity != report["checkpoints"][role]:
            raise ValueError(f"live {role} checkpoint differs from held-out NLL report")
        resolved_checkpoints[role] = root
    if resolved_checkpoints["absmax"] == resolved_checkpoints["cliffquant"]:
        raise ValueError("live AbsMax and CliffQuant checkpoint paths must be distinct")


def validate_heldout_nll_result(
    result_dir: str | Path,
    *,
    heldout_manifest: str | Path | None = None,
    absmax_checkpoint: str | Path | None = None,
    cliffquant_checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    """Validate and recompute a saved held-out NLL result from its raw arrays."""

    candidate = Path(result_dir)
    _reject_symlink_path_components(candidate, label="held-out NLL result path")
    try:
        root_mode = candidate.lstat().st_mode
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"held-out NLL result directory is missing: {candidate}") from exc
    if _is_symlink_or_reparse_point(candidate) or not stat.S_ISDIR(root_mode):
        raise ValueError(
            "held-out NLL result must be a regular non-symlink directory "
            "(reparse points are forbidden)"
        )
    root = candidate.resolve(strict=True)
    children = list(root.iterdir())
    if {child.name for child in children} != _RESULT_FILES:
        missing = sorted(_RESULT_FILES - {child.name for child in children})
        extra = sorted({child.name for child in children} - _RESULT_FILES)
        raise ValueError(
            f"held-out NLL result file coverage mismatch; missing={missing}, extra={extra}"
        )
    paths: dict[str, Path] = {}
    for child in children:
        mode = child.lstat().st_mode
        if _is_symlink_or_reparse_point(child) or not stat.S_ISREG(mode):
            raise ValueError(
                "held-out NLL result files must be regular non-symlink files "
                f"(reparse points are forbidden): {child.name}"
            )
        resolved = child.resolve(strict=True)
        if resolved.parent != root:
            raise ValueError(f"held-out NLL result file escapes directory: {child.name}")
        paths[child.name] = resolved

    report_path = paths["heldout-nll.json"]
    raw_path = paths["heldout-nll.raw.npz"]
    sidecar_path = paths["heldout-nll.sha256.json"]
    report = _read_canonical_json_object(report_path, label="held-out NLL report")
    sidecar = _read_canonical_json_object(sidecar_path, label="held-out NLL sidecar")
    _reject_absolute_strings(report)
    _reject_absolute_strings(sidecar)

    expected_report_fields = {
        "checkpoints",
        "comparison",
        "corpus_replay",
        "evaluator",
        "gates",
        "heldout",
        "policies",
        "raw_evidence",
        "runtime",
        "schema",
        "status",
    }
    _require_exact_fields(report, expected_report_fields, label="held-out NLL report")
    if report["schema"] != HELDOUT_NLL_SCHEMA or report["status"] not in {"pass", "fail"}:
        raise ValueError("held-out NLL report schema or status drift")

    _require_exact_fields(
        sidecar,
        {"inputs", "raw", "report", "schema"},
        label="held-out NLL sidecar",
    )
    if sidecar["schema"] != HELDOUT_NLL_SIDECAR_SCHEMA:
        raise ValueError("held-out NLL sidecar schema drift")
    report_descriptor = _validate_result_descriptor(
        sidecar["report"],
        expected_file=report_path.name,
        label="held-out NLL report descriptor",
    )
    raw_descriptor = _validate_result_descriptor(
        sidecar["raw"],
        expected_file=raw_path.name,
        label="held-out NLL raw descriptor",
    )
    for path, descriptor, label in (
        (report_path, report_descriptor, "report"),
        (raw_path, raw_descriptor, "raw evidence"),
    ):
        if (
            path.stat().st_size != descriptor["size_bytes"]
            or sha256_file(path) != descriptor["sha256"]
        ):
            raise ValueError(f"held-out NLL {label} live hash or size mismatch")

    inputs = _require_exact_fields(
        sidecar["inputs"],
        {
            "corpus_replay_pair_sha256",
            "heldout_manifest_sha256",
            "heldout_tokens_sha256",
        },
        label="held-out NLL sidecar inputs",
    )
    if not all(_is_sha256(value) for value in inputs.values()):
        raise ValueError("held-out NLL sidecar input hash drift")

    corpus_replay = validate_corpus_replay_pair(report["corpus_replay"])
    _validate_current_nll_evaluator(report["evaluator"])
    _validate_heldout_identity(report["heldout"], corpus_replay=corpus_replay)
    expected_inputs = {
        "corpus_replay_pair_sha256": corpus_replay["sha256"],
        "heldout_manifest_sha256": report["heldout"]["manifest_sha256"],
        "heldout_tokens_sha256": report["heldout"]["archive_sha256"],
    }
    if dict(inputs) != expected_inputs:
        raise ValueError("held-out NLL sidecar input binding drift")
    _validate_checkpoint_results(report["checkpoints"], corpus_replay)
    _validate_runtime(report["runtime"])

    raw_evidence = _require_exact_fields(
        report["raw_evidence"],
        {"arrays", "file", "sha256", "size_bytes"},
        label="held-out NLL raw evidence",
    )
    if {
        "file": raw_evidence["file"],
        "sha256": raw_evidence["sha256"],
        "size_bytes": raw_evidence["size_bytes"],
    } != raw_descriptor:
        raise ValueError("held-out NLL raw evidence differs from sidecar")
    arrays = _load_and_validate_raw_arrays(raw_path, raw_evidence)
    recomputed = _recompute_policy_metrics(arrays)
    _validate_saved_metrics_and_gates(report, recomputed)
    live_values = (heldout_manifest, absmax_checkpoint, cliffquant_checkpoint)
    if any(value is not None for value in live_values):
        if not all(value is not None for value in live_values):
            raise ValueError(
                "live held-out NLL validation requires the manifest and both checkpoints"
            )
        _validate_live_nll_bindings(
            report,
            arrays,
            heldout_manifest=heldout_manifest,
            absmax_checkpoint=absmax_checkpoint,
            cliffquant_checkpoint=cliffquant_checkpoint,
        )
    return json.loads(canonical_json_bytes(report))
