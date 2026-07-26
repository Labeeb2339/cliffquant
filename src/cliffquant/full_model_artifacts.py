"""Validated artifacts for the frozen Qwen3.5 CliffQuant full-model run.

This module deliberately contains no model-loading or GPTQModel imports.  It
provides strict, dependency-light readers for the calibration diagonals, pinned
base safetensors, and scale-search artifacts shared by the solve, export, and
verification phases.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from .model_snapshot import (
    describe_model_snapshot,
    validate_model_snapshot_descriptor,
)
from .provenance import (
    canonical_array_sha256,
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
)
from .qwen35_modules import (
    QWEN35_08B_EXPECTED_QUANTIZED_MODULES,
    QWEN35_GPTQMODEL_TREE_REVISION,
)

BASE_MODEL_ID = "Qwen/Qwen3.5-0.8B-Base"
BASE_MODEL_REVISION = "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"
FROZEN_ENVIRONMENTS = ("general", "code", "math", "multilingual")
GROUP_SIZE = 128
BITS = 4
QMIN = -8
QMAX = 7
EXPECTED_QUANTIZED_WEIGHTS = 497_025_024
EXPECTED_QUANTIZED_GROUPS = 3_883_008
POLICY_CLIFFQUANT = "cliffquant_minimax"
POLICY_ABSMAX = "absmax"
FROZEN_POLICIES = (POLICY_CLIFFQUANT, POLICY_ABSMAX)
SCALE_RUN_SCHEMA = "cliffquant.full-model-scales.v2"
SCALE_MODULE_SCHEMA = "cliffquant.module-scales.v2"
SCALE_CHUNK_SCHEMA = "cliffquant.scale-chunk.v2"
CORPUS_REPLAY_PAIR_SCHEMA = "cliffquant.corpus-replay-pair.v1"

_TARGET_RE = re.compile(
    r"^model\.language_model\.layers\.(?P<layer>\d+)\."
    r"(?P<branch>self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|"
    r"linear_attn\.(?:in_proj_qkv|in_proj_z|out_proj)|"
    r"mlp\.(?:gate_proj|up_proj|down_proj))$"
)


class WeightSource(Protocol):
    """Minimal row-addressable dense-weight source used by the solver."""

    @property
    def keys(self) -> frozenset[str]: ...

    @property
    def provenance(self) -> Mapping[str, Any]: ...

    def shape(self, key: str) -> tuple[int, ...]: ...

    def read_rows(self, key: str, start: int, stop: int) -> NDArray[np.generic]: ...

    def read_tensor(self, key: str) -> NDArray[np.generic]: ...


@dataclass(frozen=True, slots=True)
class FrozenModuleSpec:
    """One matrix in the exact frozen 150-module inventory."""

    name: str
    layer: int
    branch: str
    in_features: int
    out_features: int

    @property
    def weight_key(self) -> str:
        return f"{self.name}.weight"

    @property
    def groups_per_row(self) -> int:
        return self.in_features // GROUP_SIZE


@dataclass(frozen=True, slots=True)
class CalibrationDiagonals:
    """Fully validated normalized diagonals for every frozen module."""

    metadata_path: Path
    archive_path: Path
    metadata_sha256: str
    archive_sha256: str
    modules: tuple[FrozenModuleSpec, ...]
    environments: tuple[str, ...]
    normalized: Mapping[str, NDArray[np.float64]]
    source: Mapping[str, Any] | None = None
    corpus_replay: Mapping[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class ScaleModuleArtifact:
    """Validated module-level scale arrays and their metadata."""

    module: FrozenModuleSpec
    metadata_path: Path
    archive_path: Path
    metadata_sha256: str
    archive_sha256: str
    scale_bits: NDArray[np.uint16]
    export_scale_bits: NDArray[np.uint16]
    objectives: NDArray[np.float64]
    metadata: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class ScaleRun:
    """Validated complete 150-module scale run."""

    root: Path
    manifest_path: Path
    manifest_sha256: str
    policy: str
    modules: tuple[FrozenModuleSpec, ...]
    artifacts: Mapping[str, ScaleModuleArtifact]
    metadata: Mapping[str, Any]


def _load_json(path: Path) -> dict[str, Any]:
    def strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key in {path}: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=strict_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON number in {path}: {token}")
            ),
        )
    except FileNotFoundError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must contain an object: {path}")
    return value


def _safe_child(root: Path, relative: str, *, label: str) -> Path:
    candidate = (root / relative).resolve()
    resolved_root = root.resolve()
    if candidate != resolved_root and resolved_root not in candidate.parents:
        raise ValueError(f"{label} escapes artifact directory: {relative!r}")
    return candidate


def resolve_hf_snapshot_file(root: Path, relative: str, *, label: str) -> Path:
    """Resolve one lexical snapshot child while allowing HF blob symlinks.

    Hugging Face snapshots normally contain symlinks into the model cache's
    sibling ``blobs`` directory.  Artifact readers must still reject absolute
    paths, ``..`` traversal, and links anywhere else.
    """

    raw = Path(relative)
    if raw.is_absolute() or raw.drive or any(part == ".." for part in raw.parts):
        raise ValueError(f"{label} escapes pinned snapshot: {relative!r}")
    candidate = Path(os.path.abspath(root / raw))
    resolved_root = root.resolve()
    if candidate != resolved_root and resolved_root not in candidate.parents:
        raise ValueError(f"{label} escapes pinned snapshot: {relative!r}")
    if not candidate.is_file():
        raise FileNotFoundError(f"missing {label}: {candidate}")

    resolved = candidate.resolve()
    if resolved != candidate and resolved_root not in resolved.parents:
        blob_root = (resolved_root.parents[1] / "blobs").resolve()
        if resolved != blob_root and blob_root not in resolved.parents:
            raise ValueError(f"{label} symlink escapes the Hugging Face blob cache")
    return candidate


def _require_sha(path: Path, expected: Any, *, label: str) -> str:
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError(f"{label} must be a lowercase SHA256")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{label} hash mismatch: expected {expected}, got {actual}")
    return actual


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _validate_replay_file(
    value: Any,
    *,
    expected_file: str,
    label: str,
) -> None:
    if not isinstance(value, Mapping) or set(value) != {"file", "sha256", "size_bytes"}:
        raise ValueError(f"{label} descriptor fields drift")
    if value["file"] != expected_file or not _is_sha256(value["sha256"]):
        raise ValueError(f"{label} descriptor identity drift")
    if type(value["size_bytes"]) is not int or value["size_bytes"] <= 0:
        raise ValueError(f"{label} descriptor size drift")


def _validate_corpus_replay(value: Any, *, phase: str) -> None:
    expected_fields = {
        "accepted_source_rows",
        "cache",
        "manifest",
        "phase",
        "schema",
        "sidecar",
        "token_archive",
        "tokenizer_sha256",
        "used_rendered_records",
        "window_count",
    }
    if not isinstance(value, Mapping) or set(value) != expected_fields:
        raise ValueError(f"{phase} corpus replay fields drift")
    if value["schema"] != "cliffquant.corpus-replay.v1" or value["phase"] != phase:
        raise ValueError(f"{phase} corpus replay identity drift")
    for field in ("accepted_source_rows", "used_rendered_records"):
        if type(value[field]) is not int or value[field] <= 0:
            raise ValueError(f"{phase} corpus replay {field} drift")
    expected_windows = 4 * (32 if phase == "calibration" else 64)
    if type(value["window_count"]) is not int or value["window_count"] != expected_windows:
        raise ValueError(f"{phase} corpus replay window count drift")
    cache = value["cache"]
    if (
        not isinstance(cache, Mapping)
        or set(cache) != {"page_count", "pages_sha256"}
        or type(cache["page_count"]) is not int
        or cache["page_count"] <= 0
        or not _is_sha256(cache["pages_sha256"])
    ):
        raise ValueError(f"{phase} corpus replay cache identity drift")
    _validate_replay_file(
        value["manifest"],
        expected_file=f"{phase}.manifest.json",
        label=f"{phase} replay manifest",
    )
    _validate_replay_file(
        value["sidecar"],
        expected_file=f"{phase}.sha256.json",
        label=f"{phase} replay sidecar",
    )
    _validate_replay_file(
        value["token_archive"],
        expected_file=f"{phase}.tokens.npz",
        label=f"{phase} replay token archive",
    )
    if not _is_sha256(value["tokenizer_sha256"]):
        raise ValueError(f"{phase} corpus replay tokenizer identity drift")


def validate_corpus_replay_pair(value: Any) -> dict[str, Any]:
    """Validate and return a canonical, path-free two-phase replay identity."""

    if not isinstance(value, Mapping) or set(value) != {
        "calibration",
        "heldout",
        "schema",
        "sha256",
    }:
        raise ValueError("corpus replay pair fields drift")
    if value["schema"] != CORPUS_REPLAY_PAIR_SCHEMA:
        raise ValueError("corpus replay pair schema drift")
    _validate_corpus_replay(value["calibration"], phase="calibration")
    _validate_corpus_replay(value["heldout"], phase="heldout")
    if value["calibration"]["tokenizer_sha256"] != value["heldout"]["tokenizer_sha256"]:
        raise ValueError("calibration and heldout replay tokenizer identities differ")
    identity = {
        "calibration": value["calibration"],
        "heldout": value["heldout"],
    }
    expected_sha256 = sha256_bytes(canonical_json_bytes(identity))
    if value["sha256"] != expected_sha256:
        raise ValueError("corpus replay pair fingerprint mismatch")
    return json.loads(canonical_json_bytes(dict(value)))


def verify_frozen_corpus_pair(
    corpus_directory: str | Path,
    *,
    tokenizer: Any | None = None,
) -> dict[str, Any]:
    """Replay both adjacent corpus bundles with one pinned tokenizer load."""

    from .corpus_replay import (
        load_pinned_tokenizer,
        verify_corpus_replay,
    )

    root = Path(corpus_directory).resolve()
    pinned_tokenizer = load_pinned_tokenizer() if tokenizer is None else tokenizer
    identity = {
        "calibration": verify_corpus_replay(
            root,
            phase="calibration",
            tokenizer=pinned_tokenizer,
        ),
        "heldout": verify_corpus_replay(
            root,
            phase="heldout",
            tokenizer=pinned_tokenizer,
        ),
    }
    pair = {
        **identity,
        "schema": CORPUS_REPLAY_PAIR_SCHEMA,
        "sha256": sha256_bytes(canonical_json_bytes(identity)),
    }
    return validate_corpus_replay_pair(pair)


def _parse_module(entry: Mapping[str, Any]) -> FrozenModuleSpec:
    try:
        name = entry["name"]
        branch = entry["branch"]
        layer = entry["layer"]
        in_features = entry["in_features"]
        out_features = entry["out_features"]
    except KeyError as exc:
        raise ValueError(f"module entry is missing {exc.args[0]!r}") from exc
    if not isinstance(name, str) or not isinstance(branch, str):
        raise TypeError("module name and branch must be strings")
    match = _TARGET_RE.fullmatch(name)
    if match is None:
        raise ValueError(f"module is outside the frozen Qwen3.5 inventory: {name}")
    if type(layer) is not int or int(match.group("layer")) != layer:
        raise ValueError(f"module layer metadata disagrees with its name: {name}")
    if branch != match.group("branch"):
        raise ValueError(f"module branch metadata disagrees with its name: {name}")
    if type(in_features) is not int or type(out_features) is not int:
        raise TypeError(f"module dimensions must be integers: {name}")
    if in_features <= 0 or out_features <= 0:
        raise ValueError(f"module dimensions must be positive: {name}")
    if in_features % GROUP_SIZE:
        raise ValueError(f"module input width is not divisible by {GROUP_SIZE}: {name}")
    if out_features % 8:
        raise ValueError(f"module output width is not GPTQ INT4-packable: {name}")
    return FrozenModuleSpec(
        name=name,
        layer=layer,
        branch=branch,
        in_features=in_features,
        out_features=out_features,
    )


def validate_frozen_inventory(
    modules: tuple[FrozenModuleSpec, ...],
    *,
    strict: bool = True,
) -> None:
    """Reject duplicates, out-of-tree modules, and frozen-inventory drift."""

    names = [module.name for module in modules]
    if len(names) != len(set(names)):
        raise ValueError("frozen module inventory contains duplicate names")
    if strict and len(modules) != QWEN35_08B_EXPECTED_QUANTIZED_MODULES:
        raise ValueError(
            "Qwen3.5 module inventory drift: "
            f"expected {QWEN35_08B_EXPECTED_QUANTIZED_MODULES}, found {len(modules)}"
        )
    if strict:
        weights = sum(module.in_features * module.out_features for module in modules)
        groups = sum(module.out_features * module.groups_per_row for module in modules)
        if weights != EXPECTED_QUANTIZED_WEIGHTS:
            raise ValueError(
                f"frozen weight total drift: expected {EXPECTED_QUANTIZED_WEIGHTS}, got {weights}"
            )
        if groups != EXPECTED_QUANTIZED_GROUPS:
            raise ValueError(
                f"frozen group total drift: expected {EXPECTED_QUANTIZED_GROUPS}, got {groups}"
            )


def load_calibration_diagonals(
    metadata_path: str | Path,
    *,
    strict_inventory: bool = True,
    corpus_replay: Mapping[str, Any] | None = None,
) -> CalibrationDiagonals:
    """Load and independently validate a saved calibration-diagonal bundle."""

    # Experiment 001's shared loader binds the diagonals to the exact corpus
    # manifest and verifies strict JSON, all sidecars, archive coverage, raw
    # arithmetic, token counts, inventory totals, and pinned provenance.
    from .experiment001_io import load_diagonal_artifact

    artifact = load_diagonal_artifact(
        metadata_path,
        expected_phase="calibration",
        require_frozen_inventory=strict_inventory,
    )
    modules = tuple(
        FrozenModuleSpec(
            name=module.name,
            layer=module.layer,
            branch=module.branch,
            in_features=module.in_features,
            out_features=module.out_features,
        )
        for module in artifact.modules
    )
    replay = validate_corpus_replay_pair(corpus_replay) if corpus_replay is not None else None
    if strict_inventory and replay is None:
        raise ValueError("strict full-model calibration requires offline corpus replay")
    if replay is not None:
        source = artifact.source_descriptor()
        corpus = source.get("corpus")
        calibration_replay = replay["calibration"]
        if (
            not isinstance(corpus, Mapping)
            or corpus.get("phase") != "calibration"
            or corpus.get("manifest", {}).get("sha256") != calibration_replay["manifest"]["sha256"]
            or corpus.get("archive", {}).get("sha256")
            != calibration_replay["token_archive"]["sha256"]
            or corpus.get("sidecar_sha256") != calibration_replay["sidecar"]["sha256"]
        ):
            raise ValueError("calibration diagonals disagree with offline corpus replay")
    return CalibrationDiagonals(
        metadata_path=artifact.metadata_path,
        archive_path=artifact.archive_path,
        metadata_sha256=artifact.metadata_sha256,
        archive_sha256=artifact.archive_sha256,
        modules=modules,
        environments=artifact.environments,
        normalized=artifact.normalized,
        source=artifact.source_descriptor(),
        corpus_replay=replay,
    )


class SafetensorsWeightSource:
    """Read tensor rows from a locally cached, exactly pinned HF snapshot."""

    def __init__(
        self,
        snapshot_dir: str | Path,
        *,
        expected_revision: str = BASE_MODEL_REVISION,
        require_snapshot_name: bool = True,
    ) -> None:
        self.root = Path(snapshot_dir).resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(f"base model snapshot does not exist: {self.root}")
        if require_snapshot_name and self.root.name != expected_revision:
            raise ValueError(
                "base model directory must be the pinned Hugging Face snapshot directory "
                f"named {expected_revision}"
            )
        model_source = describe_model_snapshot(
            self.root,
            model_id=BASE_MODEL_ID,
            revision=expected_revision,
        )

        index_candidate = self.root / "model.safetensors.index.json"
        single_candidate = self.root / "model.safetensors"
        if index_candidate.is_file():
            index_path = resolve_hf_snapshot_file(
                self.root,
                index_candidate.name,
                label="safetensors index",
            )
            index = _load_json(index_path)
            raw_map = index.get("weight_map")
            if not isinstance(raw_map, dict) or not raw_map:
                raise ValueError("safetensors index has no weight_map")
            if not all(
                isinstance(key, str) and isinstance(value, str) for key, value in raw_map.items()
            ):
                raise TypeError("safetensors weight_map must map strings to strings")
            weight_map = dict(raw_map)
        elif single_candidate.is_file():
            single_path = resolve_hf_snapshot_file(
                self.root,
                single_candidate.name,
                label="safetensors checkpoint",
            )
            try:
                from safetensors import safe_open
            except ImportError as exc:
                raise RuntimeError(
                    "safetensors is required to inspect the pinned base checkpoint"
                ) from exc
            with safe_open(single_path, framework="pt", device="cpu") as handle:
                weight_map = {key: single_path.name for key in handle}
        else:
            raise FileNotFoundError("base snapshot has no model safetensors checkpoint")

        shard_paths: dict[str, Path] = {}
        for shard_name in sorted(set(weight_map.values())):
            shard = resolve_hf_snapshot_file(
                self.root,
                shard_name,
                label="safetensors shard",
            )
            shard_paths[shard_name] = shard

        self._weight_map = weight_map
        self._shard_paths = shard_paths
        self._keys = frozenset(weight_map)
        self._provenance = model_source

    @property
    def keys(self) -> frozenset[str]:
        return self._keys

    @property
    def provenance(self) -> Mapping[str, Any]:
        return self._provenance

    def _path_for(self, key: str) -> Path:
        try:
            return self._shard_paths[self._weight_map[key]]
        except KeyError as exc:
            raise KeyError(f"tensor is absent from pinned base checkpoint: {key}") from exc

    def _open(self, key: str) -> Any:
        try:
            from safetensors import safe_open
        except ImportError as exc:
            raise RuntimeError("safetensors is required to read model weights") from exc
        return safe_open(self._path_for(key), framework="pt", device="cpu")

    def shape(self, key: str) -> tuple[int, ...]:
        with self._open(key) as handle:
            tensor_slice = handle.get_slice(key)
            return tuple(int(value) for value in tensor_slice.get_shape())

    def read_rows(self, key: str, start: int, stop: int) -> NDArray[np.generic]:
        if type(start) is not int or type(stop) is not int or start < 0 or stop <= start:
            raise ValueError("row bounds must be increasing nonnegative integers")
        with self._open(key) as handle:
            tensor_slice = handle.get_slice(key)
            shape = tuple(int(value) for value in tensor_slice.get_shape())
            if len(shape) != 2 or stop > shape[0]:
                raise ValueError(f"invalid row slice {start}:{stop} for {key} with shape {shape}")
            value = tensor_slice[start:stop]
            detached = getattr(value, "detach", None)
            if callable(detached):
                import torch

                value = detached().to(dtype=torch.float64).cpu().numpy()
            return np.ascontiguousarray(np.asarray(value))

    def read_tensor(self, key: str) -> NDArray[np.generic]:
        with self._open(key) as handle:
            value = handle.get_tensor(key)
            detached = getattr(value, "detach", None)
            if callable(detached):
                import torch

                value = detached().to(dtype=torch.float64).cpu().numpy()
            return np.ascontiguousarray(np.asarray(value))


class MappingWeightSource:
    """Small in-memory source used by dependency-free unit tests."""

    def __init__(self, tensors: Mapping[str, NDArray[np.generic]]) -> None:
        self._tensors = {key: np.ascontiguousarray(value) for key, value in tensors.items()}

    @property
    def keys(self) -> frozenset[str]:
        return frozenset(self._tensors)

    @property
    def provenance(self) -> Mapping[str, Any]:
        return {"kind": "in-memory-test-source"}

    def shape(self, key: str) -> tuple[int, ...]:
        return tuple(int(value) for value in self._tensors[key].shape)

    def read_rows(self, key: str, start: int, stop: int) -> NDArray[np.generic]:
        return np.ascontiguousarray(self._tensors[key][start:stop])

    def read_tensor(self, key: str) -> NDArray[np.generic]:
        return self._tensors[key].copy()


def _load_scale_module(
    root: Path,
    descriptor: Mapping[str, Any],
    expected_module: FrozenModuleSpec,
    *,
    expected_policy: str,
    expected_base_fingerprint: str,
    expected_calibration: Mapping[str, Any],
    expected_corpus_replay_sha256: str | None,
    expected_engine: Mapping[str, Any],
) -> ScaleModuleArtifact:
    metadata_name = descriptor.get("metadata_file")
    if not isinstance(metadata_name, str):
        raise TypeError("scale module metadata_file must be a string")
    metadata_path = _safe_child(root, metadata_name, label="scale module metadata")
    metadata_sha = _require_sha(
        metadata_path,
        descriptor.get("metadata_sha256"),
        label=f"{expected_module.name} scale metadata",
    )
    metadata = _load_json(metadata_path)
    if metadata.get("schema") != SCALE_MODULE_SCHEMA:
        raise ValueError(f"unsupported scale module schema for {expected_module.name}")
    policy = metadata.get("policy")
    if (
        policy not in FROZEN_POLICIES
        or policy != descriptor.get("policy")
        or policy != expected_policy
    ):
        raise ValueError(f"scale module policy mismatch for {expected_module.name}")
    if metadata.get("module") != expected_module.name:
        raise ValueError(f"scale module identity mismatch for {expected_module.name}")
    if metadata.get("base_fingerprint") != expected_base_fingerprint:
        raise ValueError(f"scale module base fingerprint mismatch for {expected_module.name}")
    if metadata.get("calibration") != dict(expected_calibration):
        raise ValueError(f"scale module calibration mismatch for {expected_module.name}")
    if metadata.get("corpus_replay_sha256") != expected_corpus_replay_sha256:
        raise ValueError(f"scale module corpus replay mismatch for {expected_module.name}")
    if metadata.get("engine") != dict(expected_engine):
        raise ValueError(f"scale module engine identity mismatch for {expected_module.name}")
    archive_name = metadata.get("archive_file")
    if not isinstance(archive_name, str):
        raise TypeError("scale module archive_file must be a string")
    archive_path = _safe_child(metadata_path.parent, archive_name, label="scale module archive")
    archive_sha = _require_sha(
        archive_path,
        metadata.get("archive_sha256"),
        label=f"{expected_module.name} scale archive",
    )
    if descriptor.get("archive_sha256") != archive_sha:
        raise ValueError(f"scale run/module archive hash disagreement for {expected_module.name}")

    with np.load(archive_path, allow_pickle=False) as arrays:
        required = {"scale_bits", "export_scale_bits", "objectives"}
        if set(arrays.files) != required:
            raise ValueError(f"unexpected arrays in {expected_module.name} scale archive")
        scale_bits = np.asarray(arrays["scale_bits"])
        export_bits = np.asarray(arrays["export_scale_bits"])
        objectives = np.asarray(arrays["objectives"])
        expected_shape = (expected_module.out_features, expected_module.groups_per_row)
        if scale_bits.shape != expected_shape or export_bits.shape != expected_shape:
            raise ValueError(f"scale-bit shape mismatch for {expected_module.name}")
        if objectives.shape != expected_shape:
            raise ValueError(f"objective shape mismatch for {expected_module.name}")
        if scale_bits.dtype != np.dtype("<u2") and scale_bits.dtype != np.dtype("uint16"):
            raise TypeError(f"scale_bits must be uint16 for {expected_module.name}")
        if export_bits.dtype != np.dtype("<u2") and export_bits.dtype != np.dtype("uint16"):
            raise TypeError(f"export_scale_bits must be uint16 for {expected_module.name}")
        if objectives.dtype.kind != "f" or not np.all(np.isfinite(objectives)):
            raise ValueError(f"objectives must be finite floating point for {expected_module.name}")
        if np.any(objectives < 0):
            raise ValueError(f"objectives must be nonnegative for {expected_module.name}")
        arrays_meta = metadata.get("arrays")
        if not isinstance(arrays_meta, dict):
            raise ValueError(f"missing scale array metadata for {expected_module.name}")
        values = {
            "scale_bits": np.ascontiguousarray(scale_bits, dtype=np.uint16),
            "export_scale_bits": np.ascontiguousarray(export_bits, dtype=np.uint16),
            "objectives": np.ascontiguousarray(objectives, dtype=np.float64),
        }
        for name, value in values.items():
            entry = arrays_meta.get(name)
            if (
                not isinstance(entry, dict)
                or entry.get("dtype") != value.dtype.str
                or entry.get("shape") != list(value.shape)
                or entry.get("sha256") != canonical_array_sha256(value)
            ):
                raise ValueError(f"{name} hash mismatch for {expected_module.name}")
        if np.any(values["scale_bits"] == 0) or np.any(values["scale_bits"] > 0x7BFF):
            raise ValueError(f"invalid positive-finite FP16 scale bits for {expected_module.name}")
        if np.any(values["export_scale_bits"] == 0) or np.any(values["export_scale_bits"] > 0x7BFF):
            raise ValueError(f"invalid export FP16 scale bits for {expected_module.name}")
        changed_for_export = values["scale_bits"] != values["export_scale_bits"]
        if np.any(
            changed_for_export
            & (
                (values["scale_bits"] != 0x0001)
                | (values["export_scale_bits"] != 0x3C00)
                | (values["objectives"] != 0.0)
            )
        ):
            raise ValueError(
                f"only exact all-zero tie optima may use canonical export scales for "
                f"{expected_module.name}"
            )
        for value in values.values():
            value.setflags(write=False)

    return ScaleModuleArtifact(
        module=expected_module,
        metadata_path=metadata_path,
        archive_path=archive_path,
        metadata_sha256=metadata_sha,
        archive_sha256=archive_sha,
        scale_bits=values["scale_bits"],
        export_scale_bits=values["export_scale_bits"],
        objectives=values["objectives"],
        metadata=metadata,
    )


def load_scale_run(
    root_or_manifest: str | Path,
    *,
    strict_inventory: bool = True,
) -> ScaleRun:
    """Load and validate a completed full-model scale-search run."""

    supplied = Path(root_or_manifest).resolve()
    manifest_path = supplied / "scale-run.json" if supplied.is_dir() else supplied
    root = manifest_path.parent
    metadata = _load_json(manifest_path)
    if metadata.get("schema") != SCALE_RUN_SCHEMA:
        raise ValueError("unsupported or incomplete full-model scale run")
    if metadata.get("status") != "complete":
        raise ValueError("full-model scale run is not complete")
    policy = metadata.get("policy")
    if policy not in FROZEN_POLICIES:
        raise ValueError("full-model scale run has no valid immutable policy tag")
    if metadata.get("base_model", {}).get("id") != BASE_MODEL_ID:
        raise ValueError("scale run base-model id drift")
    if metadata.get("base_model", {}).get("revision") != BASE_MODEL_REVISION:
        raise ValueError("scale run base-model revision drift")
    if metadata.get("gptqmodel_revision") != QWEN35_GPTQMODEL_TREE_REVISION:
        raise ValueError("scale run GPTQModel revision drift")
    base_fingerprint = metadata.get("base_fingerprint")
    if not isinstance(base_fingerprint, str) or not re.fullmatch(r"[0-9a-f]{64}", base_fingerprint):
        raise ValueError("scale run base fingerprint is invalid")
    base_model = metadata.get("base_model")
    source_descriptor = base_model.get("source") if isinstance(base_model, Mapping) else None
    if strict_inventory:
        source_descriptor = validate_model_snapshot_descriptor(
            source_descriptor,
            model_id=BASE_MODEL_ID,
            revision=BASE_MODEL_REVISION,
        )
    elif not isinstance(source_descriptor, Mapping):
        raise ValueError("scale run base source descriptor is missing")
    if base_fingerprint != sha256_bytes(canonical_json_bytes(dict(source_descriptor))):
        raise ValueError("scale run base fingerprint disagrees with its source descriptor")
    calibration = metadata.get("calibration")
    if not isinstance(calibration, dict):
        raise ValueError("scale run calibration descriptor is missing")
    expected_calibration = {
        "archive_sha256": calibration.get("archive_sha256"),
        "metadata_sha256": calibration.get("metadata_sha256"),
    }
    if not all(
        isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value)
        for value in expected_calibration.values()
    ):
        raise ValueError("scale run calibration hashes are invalid")
    raw_corpus_replay = metadata.get("corpus_replay")
    if raw_corpus_replay is None:
        if strict_inventory:
            raise ValueError("strict full-model scale run has no offline corpus replay")
        corpus_replay = None
        corpus_replay_sha256 = None
    else:
        corpus_replay = validate_corpus_replay_pair(raw_corpus_replay)
        corpus_replay_sha256 = corpus_replay["sha256"]
    engine = metadata.get("engine")
    if not isinstance(engine, dict):
        raise ValueError("scale run engine identity is missing")
    quantizer = metadata.get("quantizer")
    expected_selection = (
        "solve_breakpoint_exact"
        if policy == POLICY_CLIFFQUANT
        else "absmax_signed_range_nearest_fp16"
    )
    if not isinstance(quantizer, dict) or {
        "bits": quantizer.get("bits"),
        "group_size": quantizer.get("group_size"),
        "qmax": quantizer.get("qmax"),
        "qmin": quantizer.get("qmin"),
        "selection": quantizer.get("selection"),
    } != {
        "bits": BITS,
        "group_size": GROUP_SIZE,
        "qmax": QMAX,
        "qmin": QMIN,
        "selection": expected_selection,
    }:
        raise ValueError("scale run quantizer or policy selection drift")

    raw_modules = metadata.get("modules")
    if not isinstance(raw_modules, list):
        raise TypeError("scale run modules must be a list")
    modules: list[FrozenModuleSpec] = []
    descriptors: dict[str, Mapping[str, Any]] = {}
    for entry in raw_modules:
        if not isinstance(entry, dict):
            raise TypeError("scale run module descriptors must be objects")
        module_entry = entry.get("module")
        if not isinstance(module_entry, dict):
            raise ValueError("scale run descriptor is missing module metadata")
        module = _parse_module(module_entry)
        modules.append(module)
        descriptors[module.name] = entry
    module_tuple = tuple(modules)
    validate_frozen_inventory(module_tuple, strict=strict_inventory)
    if len(descriptors) != len(module_tuple):
        raise ValueError("scale run contains duplicate module descriptors")

    artifacts = {
        module.name: _load_scale_module(
            root,
            descriptors[module.name],
            module,
            expected_policy=policy,
            expected_base_fingerprint=base_fingerprint,
            expected_calibration=expected_calibration,
            expected_corpus_replay_sha256=corpus_replay_sha256,
            expected_engine=engine,
        )
        for module in module_tuple
    }
    return ScaleRun(
        root=root,
        manifest_path=manifest_path,
        manifest_sha256=sha256_file(manifest_path),
        policy=policy,
        modules=module_tuple,
        artifacts=artifacts,
        metadata=metadata,
    )


def iter_safetensor_files(root: str | Path) -> Iterator[Path]:
    """Yield output safetensor files in stable order."""

    directory = Path(root)
    for path in sorted(directory.glob("*.safetensors")):
        if path.is_file():
            yield path
