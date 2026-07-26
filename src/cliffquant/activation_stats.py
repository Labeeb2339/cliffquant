"""Forward-hook collection of normalized input-square diagonals."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .provenance import (
    canonical_array_sha256,
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
)
from .qwen35_modules import (
    QWEN35_08B_EXPECTED_QUANTIZED_MODULES,
    QWEN35_GPTQMODEL_TREE_REVISION,
    QuantizedModule,
    discover_qwen35_quantized_modules,
)


@dataclass(frozen=True, slots=True)
class EnvironmentDiagonal:
    """Raw and normalized statistics for one module/environment pair."""

    mean_input_square: np.ndarray
    normalized: np.ndarray
    normalization_constant: float
    token_rows: int


@dataclass(frozen=True, slots=True)
class ActivationStatistics:
    """Final statistics plus the frozen module inventory."""

    modules: tuple[QuantizedModule, ...]
    environments: tuple[str, ...]
    values: Mapping[tuple[str, str], EnvironmentDiagonal]


@dataclass(slots=True)
class _Accumulator:
    sum_squares: np.ndarray
    token_rows: int = 0


def _as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    detached = getattr(value, "detach", None)
    if callable(detached):
        value = detached()
    cpu = getattr(value, "cpu", None)
    if callable(cpu):
        value = cpu()
    numpy_method = getattr(value, "numpy", None)
    if callable(numpy_method):
        return np.asarray(numpy_method())
    return np.asarray(value)


def _sum_squares(
    inputs: Any,
    attention_mask: Any,
    *,
    expected_features: int,
) -> tuple[np.ndarray, int]:
    try:
        import torch
    except ImportError:
        torch = None
    if torch is not None and isinstance(inputs, torch.Tensor):
        if inputs.ndim < 2:
            raise ValueError("hook input must have at least a token and feature dimension")
        if inputs.shape[-1] != expected_features:
            raise ValueError(
                f"hook input has {inputs.shape[-1]} features; expected {expected_features}"
            )
        rows = inputs.detach().reshape(-1, expected_features)
        mask = torch.as_tensor(attention_mask, device=rows.device).reshape(-1)
        if mask.numel() != rows.shape[0]:
            raise ValueError(
                f"attention mask has {mask.numel()} token rows; input has {rows.shape[0]}"
            )
        selected = mask != 0
        count = int(torch.count_nonzero(selected).item())
        if count == 0:
            raise ValueError("batch contains no unmasked token rows")
        selected_rows = rows[selected].to(dtype=torch.float32)
        if not bool(torch.all(torch.isfinite(selected_rows)).item()):
            raise ValueError("hook input contains non-finite unmasked activations")
        # Reduce on-device, then transfer only one feature vector to the CPU.
        sums = selected_rows.square().sum(dim=0, dtype=torch.float32)
        return sums.to(dtype=torch.float64, device="cpu").numpy(), count

    array = _as_numpy(inputs)
    if array.ndim < 2:
        raise ValueError("hook input must have at least a token and feature dimension")
    if array.shape[-1] != expected_features:
        raise ValueError(f"hook input has {array.shape[-1]} features; expected {expected_features}")
    if array.dtype.kind not in {"f", "i", "u"}:
        raise TypeError("hook input must be a real numeric tensor")
    rows = np.asarray(array, dtype=np.float64).reshape(-1, expected_features)
    mask = _as_numpy(attention_mask)
    if mask.dtype.kind not in {"b", "i", "u", "f"}:
        raise TypeError("attention mask must be numeric or boolean")
    flattened_mask = mask.reshape(-1)
    if flattened_mask.size != rows.shape[0]:
        raise ValueError(
            f"attention mask has {flattened_mask.size} token rows; input has {rows.shape[0]}"
        )
    selected = flattened_mask != 0
    count = int(np.count_nonzero(selected))
    if count == 0:
        raise ValueError("batch contains no unmasked token rows")
    selected_rows = rows[selected]
    if not np.all(np.isfinite(selected_rows)):
        raise ValueError("hook input contains non-finite unmasked activations")
    return np.sum(np.square(selected_rows), axis=0, dtype=np.float64), count


def _first_input(args: Any) -> Any:
    if isinstance(args, tuple) and args:
        return args[0]
    if isinstance(args, list) and args:
        return args[0]
    raise ValueError("quantized linear module received no positional input tensor")


class ActivationDiagonalCollector:
    """Register hooks and collect token-masked input-square statistics."""

    def __init__(self, modules: Sequence[QuantizedModule]) -> None:
        if not modules:
            raise ValueError("at least one quantized module is required")
        self.modules = tuple(modules)
        self._module_by_name = {item.name: item for item in self.modules}
        if len(self._module_by_name) != len(self.modules):
            raise ValueError("quantized module names must be unique")
        self._accumulators: dict[tuple[str, str], _Accumulator] = {}
        self._current_environment: str | None = None
        self._current_attention_mask: Any = None
        self._handles: list[Any] = []
        self._closed = False
        for item in self.modules:
            register = getattr(item.module, "register_forward_pre_hook", None)
            if not callable(register):
                self.close()
                raise TypeError(f"module {item.name} does not support forward pre-hooks")
            self._handles.append(register(self._make_hook(item)))

    def _make_hook(self, item: QuantizedModule) -> Any:
        def hook(_module: Any, args: Any) -> None:
            if self._current_environment is None or self._current_attention_mask is None:
                raise RuntimeError(
                    f"{item.name} ran outside collector.capture(environment, attention_mask)"
                )
            sums, count = _sum_squares(
                _first_input(args),
                self._current_attention_mask,
                expected_features=item.in_features,
            )
            key = (self._current_environment, item.name)
            accumulator = self._accumulators.get(key)
            if accumulator is None:
                self._accumulators[key] = _Accumulator(sums, count)
            else:
                accumulator.sum_squares += sums
                accumulator.token_rows += count

        return hook

    @contextmanager
    def capture(self, environment: str, attention_mask: Any) -> Iterable[None]:
        """Set the exact environment/mask applied to the next forward pass."""

        if self._closed:
            raise RuntimeError("collector is closed")
        if not environment:
            raise ValueError("environment must be non-empty")
        if self._current_environment is not None:
            raise RuntimeError("collector.capture contexts cannot be nested")
        self._current_environment = environment
        self._current_attention_mask = attention_mask
        try:
            yield
        finally:
            self._current_environment = None
            self._current_attention_mask = None

    def finalize(self, environments: Sequence[str]) -> ActivationStatistics:
        """Normalize every module/environment diagonal by its arithmetic mean."""

        if not environments or len(set(environments)) != len(environments):
            raise ValueError("environments must be non-empty and unique")
        values: dict[tuple[str, str], EnvironmentDiagonal] = {}
        for environment in environments:
            for item in self.modules:
                key = (environment, item.name)
                accumulator = self._accumulators.get(key)
                if accumulator is None or accumulator.token_rows == 0:
                    raise ValueError(f"missing activation rows for {environment}/{item.name}")
                raw = accumulator.sum_squares / accumulator.token_rows
                if not np.all(np.isfinite(raw)):
                    raise ValueError(f"non-finite diagonal for {environment}/{item.name}")
                normalization = float(np.mean(raw, dtype=np.float64))
                if not np.isfinite(normalization) or normalization <= 0.0:
                    raise ValueError(f"zero or invalid diagonal mean for {environment}/{item.name}")
                normalized = raw / normalization
                raw.setflags(write=False)
                normalized.setflags(write=False)
                values[key] = EnvironmentDiagonal(
                    mean_input_square=raw,
                    normalized=normalized,
                    normalization_constant=normalization,
                    token_rows=accumulator.token_rows,
                )
        return ActivationStatistics(
            modules=self.modules,
            environments=tuple(environments),
            values=values,
        )

    def close(self) -> None:
        """Remove all registered hooks."""

        if self._closed:
            return
        for handle in self._handles:
            remove = getattr(handle, "remove", None)
            if callable(remove):
                remove()
        self._handles.clear()
        self._closed = True

    def __enter__(self) -> ActivationDiagonalCollector:
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()


def save_activation_statistics(
    output_dir: str | Path,
    *,
    statistics: ActivationStatistics,
    phase: str,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Save arrays, canonical metadata, and independently checkable hashes."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    npz_path = destination / f"{phase}.diagonals.npz"
    arrays: dict[str, np.ndarray] = {}
    entries: list[dict[str, Any]] = []

    for module_index, module in enumerate(statistics.modules):
        for environment_index, environment in enumerate(statistics.environments):
            value = statistics.values[(environment, module.name)]
            raw_key = f"m{module_index:03d}__e{environment_index:02d}__raw"
            normalized_key = f"m{module_index:03d}__e{environment_index:02d}__normalized"
            arrays[raw_key] = np.asarray(value.mean_input_square, dtype=np.float64)
            arrays[normalized_key] = np.asarray(value.normalized, dtype=np.float64)
            entries.append(
                {
                    "environment": environment,
                    "module": module.name,
                    "normalization_constant": value.normalization_constant,
                    "normalized": {
                        "array_key": normalized_key,
                        "dtype": "<f8",
                        "sha256": canonical_array_sha256(arrays[normalized_key]),
                        "shape": list(arrays[normalized_key].shape),
                    },
                    "raw": {
                        "array_key": raw_key,
                        "dtype": "<f8",
                        "sha256": canonical_array_sha256(arrays[raw_key]),
                        "shape": list(arrays[raw_key].shape),
                    },
                    "token_rows": value.token_rows,
                }
            )
    np.savez_compressed(npz_path, **arrays)

    metadata = {
        "arrays": entries,
        "environments": list(statistics.environments),
        "modules": [
            {
                "branch": module.branch,
                "in_features": module.in_features,
                "layer": module.layer,
                "name": module.name,
                "out_features": module.out_features,
            }
            for module in statistics.modules
        ],
        "module_tree": {
            "implementation": "GPTQModel qwen3_5.py",
            "revision": QWEN35_GPTQMODEL_TREE_REVISION,
        },
        "phase": phase,
        "provenance": dict(provenance),
        "schema": "cliffquant.activation-diagonals.v1",
        "statistics_archive": {
            "file": npz_path.name,
            "sha256": sha256_file(npz_path),
            "size_bytes": npz_path.stat().st_size,
        },
    }
    metadata_path = destination / f"{phase}.diagonals.json"
    metadata_bytes = canonical_json_bytes(metadata)
    metadata_path.write_bytes(metadata_bytes)
    hashes = {
        "metadata": {
            "file": metadata_path.name,
            "sha256": sha256_bytes(metadata_bytes),
            "size_bytes": len(metadata_bytes),
        },
        "statistics": metadata["statistics_archive"],
    }
    hashes_path = destination / f"{phase}.diagonals.sha256.json"
    hashes_path.write_bytes(canonical_json_bytes(hashes))
    return hashes


def collect_model_windows(
    model: Any,
    *,
    token_arrays: Mapping[str, tuple[np.ndarray, np.ndarray]],
    batch_size: int,
    device: str,
) -> ActivationStatistics:
    """Run exact token windows through a model and collect all four environments."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for real model collection") from exc

    modules = discover_qwen35_quantized_modules(model)
    if len(modules) != QWEN35_08B_EXPECTED_QUANTIZED_MODULES:
        raise ValueError(
            "Qwen3.5-0.8B module inventory drift: "
            f"expected {QWEN35_08B_EXPECTED_QUANTIZED_MODULES}, found {len(modules)} "
            f"for GPTQModel {QWEN35_GPTQMODEL_TREE_REVISION}"
        )
    environments = tuple(token_arrays)
    model.eval()
    with ActivationDiagonalCollector(modules) as collector, torch.inference_mode():
        for environment, (input_ids, attention_mask) in token_arrays.items():
            if input_ids.shape != attention_mask.shape or input_ids.ndim != 2:
                raise ValueError(f"invalid token arrays for {environment}")
            for start in range(0, input_ids.shape[0], batch_size):
                stop = min(start + batch_size, input_ids.shape[0])
                ids = torch.as_tensor(input_ids[start:stop], dtype=torch.long, device=device)
                mask = torch.as_tensor(
                    attention_mask[start:stop],
                    dtype=torch.long,
                    device=device,
                )
                with collector.capture(environment, mask):
                    model(input_ids=ids, attention_mask=mask, use_cache=False)
        return collector.finalize(environments)
