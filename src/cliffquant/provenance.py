"""Canonical hashing and runtime provenance helpers for experiments."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import sys
from collections.abc import Mapping
from pathlib import Path, PureWindowsPath
from typing import Any

import numpy as np


def numpy_build_metadata() -> dict[str, Any]:
    """Return path-free NumPy/BLAS build facts that can affect numerical evidence."""

    config = getattr(np.__config__, "CONFIG", {})
    dependencies = config.get("Build Dependencies", {})

    def dependency(name: str) -> dict[str, Any]:
        raw = dependencies.get(name, {})
        return {key: raw.get(key) for key in ("found", "name", "openblas configuration", "version")}

    return {
        "blas": dependency("blas"),
        "host": dict(config.get("Machine Information", {}).get("host", {})),
        "lapack": dependency("lapack"),
        "simd": dict(config.get("SIMD Extensions", {})),
    }


def sha256_bytes(data: bytes) -> str:
    """Return a lowercase SHA256 digest."""

    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    """Hash a file without loading it all into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    """Encode strict, stable UTF-8 JSON suitable for content hashing."""

    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode()


def canonical_array_sha256(array: np.ndarray) -> str:
    """Hash an array in a platform-independent little-endian representation."""

    contiguous = np.ascontiguousarray(array)
    dtype = contiguous.dtype.newbyteorder("<")
    canonical = contiguous.astype(dtype, copy=False)
    header = canonical_json_bytes({"dtype": canonical.dtype.str, "shape": list(canonical.shape)})
    return sha256_bytes(header + canonical.tobytes(order="C"))


def runtime_metadata(
    *,
    device: str,
    package_names: tuple[str, ...] = (
        "numpy",
        "torch",
        "transformers",
        "datasets",
        "huggingface-hub",
        "tokenizers",
        "accelerate",
    ),
) -> dict[str, Any]:
    """Capture software and device facts without importing optional packages."""

    packages: dict[str, str | None] = {}
    for name in package_names:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None

    metadata: dict[str, Any] = {
        "device": device,
        "implementation": platform.python_implementation(),
        "machine": platform.machine(),
        "numpy": np.__version__,
        "numpy_build": numpy_build_metadata(),
        "packages": packages,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "python_executable": Path(sys.executable).name,
    }
    if device.startswith("cuda"):
        try:
            import torch
        except ImportError:
            metadata["accelerator"] = {"available": False}
        else:
            available = bool(torch.cuda.is_available())
            accelerator: dict[str, Any] = {
                "available": available,
                "cuda_runtime": torch.version.cuda,
                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                "deterministic_algorithms_warn_only": (
                    torch.is_deterministic_algorithms_warn_only_enabled()
                ),
            }
            if available:
                resolved = torch.device(device)
                index = resolved.index
                if index is None:
                    index = torch.cuda.current_device()
                properties = torch.cuda.get_device_properties(index)
                accelerator.update(
                    {
                        "capability": list(torch.cuda.get_device_capability(index)),
                        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
                        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
                        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
                        "cudnn_version": torch.backends.cudnn.version(),
                        "device_index": index,
                        "float32_matmul_precision": torch.get_float32_matmul_precision(),
                        "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
                        "name": properties.name,
                        "total_memory_bytes": properties.total_memory,
                    }
                )
            metadata["accelerator"] = accelerator
    return metadata


def tokenizer_metadata(
    tokenizer: Any,
    *,
    model_revision: str,
    explicit_files: Mapping[str, str | Path] | None = None,
    logical_name_or_path: str | None = None,
) -> dict[str, Any]:
    """Fingerprint a tokenizer and the local tokenizer files used by a run."""

    candidates: dict[str, Path] = {}
    if explicit_files is not None:
        candidates.update({str(name): Path(path) for name, path in explicit_files.items()})
    else:
        vocab_names = getattr(tokenizer, "vocab_files_names", {}) or {}
        for logical_name in sorted(vocab_names):
            path_value = getattr(tokenizer, logical_name, None)
            if isinstance(path_value, str) and Path(path_value).is_file():
                candidates[logical_name] = Path(path_value)

        source = getattr(tokenizer, "name_or_path", None)
        if isinstance(source, str) and Path(source).is_dir():
            allowed_names = {
                "added_tokens.json",
                "chat_template.jinja",
                "merges.txt",
                "sentencepiece.bpe.model",
                "special_tokens_map.json",
                "tokenizer.json",
                "tokenizer.model",
                "tokenizer_config.json",
                "vocab.json",
                "vocab.txt",
            }
            for path in sorted(Path(source).iterdir()):
                if path.is_file() and path.name in allowed_names:
                    candidates.setdefault(path.name, path)

    files = {
        name: {
            "path": name,
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for name, path in sorted(candidates.items())
    }
    special_ids = {
        name: getattr(tokenizer, name, None)
        for name in (
            "bos_token_id",
            "eos_token_id",
            "pad_token_id",
            "unk_token_id",
        )
    }
    raw_name_or_path = str(getattr(tokenizer, "name_or_path", ""))
    is_local_absolute_path = (
        Path(raw_name_or_path).is_absolute() or PureWindowsPath(raw_name_or_path).is_absolute()
    )
    public_name_or_path = (
        logical_name_or_path
        if logical_name_or_path is not None
        else (
            raw_name_or_path
            if not is_local_absolute_path
            else PureWindowsPath(raw_name_or_path).name
        )
    )
    return {
        "class": type(tokenizer).__name__,
        "files": files,
        "model_revision": model_revision,
        "name_or_path": public_name_or_path,
        "special_token_ids": special_ids,
        "vocab_size": getattr(tokenizer, "vocab_size", None),
    }
