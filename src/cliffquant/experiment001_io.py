"""Strict inputs and deterministic real-group sampling for Experiment 001."""

from __future__ import annotations

import hashlib
import json
import math
import random
import struct
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from cliffquant.corpus import (
    CALIBRATION_SOURCES,
    CALIBRATION_WINDOWS,
    HELDOUT_SOURCES,
    HELDOUT_WINDOWS,
    MAX_VALID_RECORDS,
    MODEL_ID,
    SEED,
    WINDOW_TOKENS,
    collection_contract_is_supported,
    expected_tokenizer_metadata,
)
from cliffquant.dataset_viewer import CACHE_SCHEMA_VERSION, VIEWER_PAGE_SIZE
from cliffquant.model_snapshot import (
    describe_model_snapshot,
    validate_model_snapshot_descriptor,
)
from cliffquant.provenance import (
    canonical_array_sha256,
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
)
from cliffquant.qwen35_modules import (
    QWEN35_08B_EXPECTED_QUANTIZED_MODULES,
    QWEN35_GPTQMODEL_TREE_REVISION,
)

EXPERIMENT_001_MODEL_REVISION = "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"
EXPERIMENT_001_MODEL_ID = "Qwen/Qwen3.5-0.8B-Base"
EXPERIMENT_001_ENVIRONMENTS = ("general", "code", "math", "multilingual")
EXPERIMENT_001_GROUP_SIZE = 128
EXPERIMENT_001_SAMPLE_SIZE = 4096
EXPERIMENT_001_SEED = 2339
EXPERIMENT_001_EXPECTED_WEIGHTS = 497_025_024
EXPERIMENT_001_EXPECTED_GROUPS = 3_883_008

_BRANCH_ORDER = {
    "self_attn.q_proj": 0,
    "self_attn.k_proj": 1,
    "self_attn.v_proj": 2,
    "self_attn.o_proj": 3,
    "linear_attn.in_proj_qkv": 0,
    "linear_attn.in_proj_z": 1,
    "linear_attn.out_proj": 2,
    "mlp.gate_proj": 4,
    "mlp.up_proj": 5,
    "mlp.down_proj": 6,
}

_SELF_ATTENTION_BRANCHES = {
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
}
_LINEAR_ATTENTION_BRANCHES = {
    "in_proj_qkv",
    "in_proj_z",
    "out_proj",
}
_MLP_BRANCHES = {"gate_proj", "up_proj", "down_proj"}

_SAFETENSORS_DTYPES: dict[str, tuple[str, int]] = {
    "BOOL": ("u1", 1),
    "I8": ("i1", 1),
    "U8": ("u1", 1),
    "I16": ("<i2", 2),
    "U16": ("<u2", 2),
    "I32": ("<i4", 4),
    "U32": ("<u4", 4),
    "I64": ("<i8", 8),
    "U64": ("<u8", 8),
    "F16": ("<f2", 2),
    "BF16": ("<u2", 2),
    "F32": ("<f4", 4),
    "F64": ("<f8", 8),
}


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_strict_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON number: {token}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read strict JSON from {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _required_string(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _required_int(value: Any, *, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _required_sha256(value: Any, *, name: str) -> str:
    digest = _required_string(value, name=name)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{name} must be a lowercase SHA256 digest")
    return digest


def _safe_relative_path(value: Any, *, name: str) -> str:
    text = _required_string(value, name=name)
    path = Path(text)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{name} must be a safe relative path")
    return text


_RUNTIME_PACKAGE_NAMES = {
    "accelerate",
    "datasets",
    "huggingface-hub",
    "numpy",
    "tokenizers",
    "torch",
    "transformers",
}
_RUNTIME_COMMON_FIELDS = {
    "device",
    "implementation",
    "machine",
    "numpy",
    "numpy_build",
    "packages",
    "platform",
    "python",
    "python_executable",
}
_ACCELERATOR_FIELDS = {
    "available",
    "capability",
    "cublas_workspace_config",
    "cuda_runtime",
    "cudnn_allow_tf32",
    "cudnn_benchmark",
    "cudnn_deterministic",
    "cudnn_version",
    "deterministic_algorithms",
    "deterministic_algorithms_warn_only",
    "device_index",
    "float32_matmul_precision",
    "matmul_allow_tf32",
    "name",
    "total_memory_bytes",
}


def _validate_numpy_build(value: Any, *, label: str) -> None:
    if not isinstance(value, dict) or set(value) != {"blas", "host", "lapack", "simd"}:
        raise ValueError(f"{label} NumPy build metadata is incomplete")
    for dependency in ("blas", "lapack"):
        descriptor = value[dependency]
        if not isinstance(descriptor, dict) or set(descriptor) != {
            "found",
            "name",
            "openblas configuration",
            "version",
        }:
            raise ValueError(f"{label} {dependency} build metadata is incomplete")
        if type(descriptor["found"]) is not bool:
            raise ValueError(f"{label} {dependency} found flag must be boolean")
        for key in ("name", "openblas configuration", "version"):
            if descriptor[key] is not None and not isinstance(descriptor[key], str):
                raise ValueError(f"{label} {dependency} {key} must be text or null")
    host = value["host"]
    if not isinstance(host, dict) or not {
        "cpu",
        "endian",
        "family",
        "system",
    } <= set(host):
        raise ValueError(f"{label} NumPy host metadata is incomplete")
    if not all(isinstance(item, str) and item for item in host.values()):
        raise ValueError(f"{label} NumPy host metadata must contain non-empty strings")
    simd = value["simd"]
    if (
        not isinstance(simd, dict)
        or not {"baseline", "found"} <= set(simd)
        or not set(simd) <= {"baseline", "found", "not found"}
    ):
        raise ValueError(f"{label} NumPy SIMD metadata is incomplete")
    for key in simd:
        if not isinstance(simd[key], list) or not all(
            isinstance(item, str) and item for item in simd[key]
        ):
            raise ValueError(f"{label} NumPy SIMD entries must be strings")


def _validate_accelerator(value: Any, *, deterministic: bool, label: str) -> None:
    if not isinstance(value, dict) or set(value) != _ACCELERATOR_FIELDS:
        raise ValueError(f"{label} accelerator metadata is incomplete")
    if value["available"] is not True:
        raise ValueError(f"{label} requires an available CUDA accelerator")
    if (
        not isinstance(value["name"], str)
        or not value["name"]
        or not isinstance(value["cuda_runtime"], str)
        or not value["cuda_runtime"]
        or type(value["device_index"]) is not int
        or type(value["total_memory_bytes"]) is not int
        or value["total_memory_bytes"] <= 0
        or not isinstance(value["capability"], list)
        or len(value["capability"]) != 2
        or any(type(item) is not int or item < 0 for item in value["capability"])
    ):
        raise ValueError(f"{label} accelerator identity is invalid")
    if value["cudnn_version"] is not None and (
        type(value["cudnn_version"]) is not int or value["cudnn_version"] <= 0
    ):
        raise ValueError(f"{label} cuDNN version is invalid")
    if deterministic and (
        value["cublas_workspace_config"] != ":4096:8"
        or value["deterministic_algorithms"] is not True
        or value["deterministic_algorithms_warn_only"] is not False
        or value["cudnn_benchmark"] is not False
        or value["cudnn_deterministic"] is not True
        or value["cudnn_allow_tf32"] is not False
        or value["matmul_allow_tf32"] is not False
        or value["float32_matmul_precision"] != "highest"
    ):
        raise ValueError(f"{label} deterministic CUDA settings drift")


def _validate_runtime(
    value: Any,
    *,
    label: str,
    require_deterministic_cuda: bool,
) -> None:
    extra = {"model_parameter_dtypes", "seed"} if require_deterministic_cuda else set()
    allowed = _RUNTIME_COMMON_FIELDS | extra
    if isinstance(value, dict) and "accelerator" in value:
        allowed = allowed | {"accelerator"}
    if not isinstance(value, dict) or set(value) != allowed:
        raise ValueError(f"{label} runtime metadata fields are incomplete")
    for key in ("device", "implementation", "machine", "numpy", "platform", "python"):
        if not isinstance(value[key], str) or not value[key]:
            raise ValueError(f"{label} runtime {key} must be non-empty text")
    executable = value["python_executable"]
    if not isinstance(executable, str) or not executable or Path(executable).name != executable:
        raise ValueError(f"{label} Python executable must be a basename")
    packages = value["packages"]
    if not isinstance(packages, dict) or set(packages) != _RUNTIME_PACKAGE_NAMES:
        raise ValueError(f"{label} runtime package inventory drift")
    if any(version is not None and not isinstance(version, str) for version in packages.values()):
        raise ValueError(f"{label} runtime package versions must be text or null")
    _validate_numpy_build(value["numpy_build"], label=label)
    accelerator = value.get("accelerator")
    if value["device"].startswith("cuda"):
        _validate_accelerator(
            accelerator,
            deterministic=require_deterministic_cuda,
            label=label,
        )
    elif accelerator is not None:
        raise ValueError(f"{label} has accelerator metadata for a non-CUDA device")
    if require_deterministic_cuda:
        if not value["device"].startswith("cuda"):
            raise ValueError(f"{label} must record a CUDA device")
        if value["seed"] != EXPERIMENT_001_SEED:
            raise ValueError(f"{label} seed drift")
        if value["model_parameter_dtypes"] != ["torch.bfloat16"]:
            raise ValueError(f"{label} model parameter dtype inventory drift")


def _parse_target_name(name: str) -> tuple[int, str] | None:
    prefix = "model.language_model.layers."
    suffix = ".weight"
    candidate = name[: -len(suffix)] if name.endswith(suffix) else name
    if not candidate.startswith(prefix):
        return None
    remainder = candidate[len(prefix) :]
    layer_text, separator, branch = remainder.partition(".")
    if not separator or not layer_text.isdigit():
        return None
    pieces = branch.split(".")
    if len(pieces) != 2:
        return None
    family, projection = pieces
    if family == "self_attn" and projection in _SELF_ATTENTION_BRANCHES:
        return int(layer_text), branch
    if family == "linear_attn" and projection in _LINEAR_ATTENTION_BRANCHES:
        return int(layer_text), branch
    if family == "mlp" and projection in _MLP_BRANCHES:
        return int(layer_text), branch
    return None


def _inventory_sort_key(module: ModuleInventoryEntry) -> tuple[int, int, str]:
    return module.layer, _BRANCH_ORDER[module.branch], module.name


@dataclass(frozen=True, slots=True)
class ModuleInventoryEntry:
    """One matrix in the frozen GPTQModel Qwen3.5 inventory."""

    name: str
    layer: int
    branch: str
    in_features: int
    out_features: int

    @property
    def groups_per_row(self) -> int:
        return self.in_features // EXPERIMENT_001_GROUP_SIZE

    @property
    def group_count(self) -> int:
        return self.out_features * self.groups_per_row

    @property
    def weight_count(self) -> int:
        return self.out_features * self.in_features

    @property
    def tensor_name(self) -> str:
        return f"{self.name}.weight"


@dataclass(frozen=True, slots=True)
class CorpusArtifact:
    """Validated corpus manifest, exact token archive, and overlap fingerprints."""

    phase: str
    manifest_path: Path
    archive_path: Path
    manifest_sha256: str
    archive_sha256: str
    sidecar_sha256: str
    rendered_sha256: frozenset[str]
    window_sha256: frozenset[str]
    tokenizer: dict[str, Any]

    def source_descriptor(self) -> dict[str, Any]:
        return {
            "archive": {
                "file": self.archive_path.name,
                "sha256": self.archive_sha256,
                "size_bytes": self.archive_path.stat().st_size,
            },
            "manifest": {
                "file": self.manifest_path.name,
                "sha256": self.manifest_sha256,
                "size_bytes": self.manifest_path.stat().st_size,
            },
            "phase": self.phase,
            "sidecar_sha256": self.sidecar_sha256,
        }


@dataclass(frozen=True, slots=True)
class DiagonalArtifact:
    """Validated normalized diagonals plus their corpus trust chain."""

    phase: str
    environments: tuple[str, ...]
    modules: tuple[ModuleInventoryEntry, ...]
    normalized: dict[str, NDArray[np.float64]]
    metadata_path: Path
    archive_path: Path
    metadata_sha256: str
    archive_sha256: str
    sidecar_sha256: str
    corpus: CorpusArtifact
    model_source: dict[str, Any]

    def source_descriptor(self) -> dict[str, Any]:
        return {
            "archive": {
                "file": self.archive_path.name,
                "sha256": self.archive_sha256,
                "size_bytes": self.archive_path.stat().st_size,
            },
            "corpus": self.corpus.source_descriptor(),
            "metadata": {
                "file": self.metadata_path.name,
                "sha256": self.metadata_sha256,
                "size_bytes": self.metadata_path.stat().st_size,
            },
            "model": self.model_source,
            "phase": self.phase,
            "sidecar_sha256": self.sidecar_sha256,
        }


def _parse_inventory(metadata: dict[str, Any]) -> tuple[ModuleInventoryEntry, ...]:
    raw_modules = metadata.get("modules")
    if not isinstance(raw_modules, list) or not raw_modules:
        raise ValueError("diagonal metadata modules must be a non-empty list")
    modules: list[ModuleInventoryEntry] = []
    for index, raw in enumerate(raw_modules):
        if not isinstance(raw, dict):
            raise ValueError(f"modules[{index}] must be an object")
        name = _required_string(raw.get("name"), name=f"modules[{index}].name")
        parsed = _parse_target_name(name)
        if parsed is None:
            raise ValueError(f"modules[{index}] is outside the frozen target tree: {name}")
        parsed_layer, parsed_branch = parsed
        layer = _required_int(raw.get("layer"), name=f"modules[{index}].layer")
        branch = _required_string(raw.get("branch"), name=f"modules[{index}].branch")
        in_features = _required_int(
            raw.get("in_features"),
            name=f"modules[{index}].in_features",
            minimum=1,
        )
        out_features = _required_int(
            raw.get("out_features"),
            name=f"modules[{index}].out_features",
            minimum=1,
        )
        if (layer, branch) != (parsed_layer, parsed_branch):
            raise ValueError(f"module metadata disagrees with its name for {name}")
        if in_features % EXPERIMENT_001_GROUP_SIZE:
            raise ValueError(f"{name} input width is not divisible by 128")
        modules.append(
            ModuleInventoryEntry(
                name=name,
                layer=layer,
                branch=branch,
                in_features=in_features,
                out_features=out_features,
            )
        )
    if len({module.name for module in modules}) != len(modules):
        raise ValueError("diagonal metadata contains duplicate modules")
    if tuple(modules) != tuple(sorted(modules, key=_inventory_sort_key)):
        raise ValueError("diagonal metadata modules are not in frozen inventory order")
    return tuple(modules)


def _validate_frozen_inventory(modules: Sequence[ModuleInventoryEntry]) -> None:
    if len(modules) != QWEN35_08B_EXPECTED_QUANTIZED_MODULES:
        raise ValueError(
            "inventory drift: expected "
            f"{QWEN35_08B_EXPECTED_QUANTIZED_MODULES} modules, found {len(modules)}"
        )
    weight_count = sum(module.weight_count for module in modules)
    group_count = sum(module.group_count for module in modules)
    if weight_count != EXPERIMENT_001_EXPECTED_WEIGHTS:
        raise ValueError(
            f"inventory drift: expected {EXPERIMENT_001_EXPECTED_WEIGHTS} weights, "
            f"found {weight_count}"
        )
    if group_count != EXPERIMENT_001_EXPECTED_GROUPS:
        raise ValueError(
            f"inventory drift: expected {EXPERIMENT_001_EXPECTED_GROUPS} groups, "
            f"found {group_count}"
        )
    if {module.layer for module in modules} != set(range(24)):
        raise ValueError("inventory drift: target modules do not span layers 0 through 23")


def _sidecar_path(metadata_path: Path) -> Path:
    suffix = ".json"
    if not metadata_path.name.endswith(suffix):
        raise ValueError("diagonal metadata path must end in .json")
    return metadata_path.with_name(f"{metadata_path.name[: -len(suffix)]}.sha256.json")


def _expected_source_table(phase: str) -> dict[str, tuple[Any, ...]]:
    if phase == "calibration":
        return CALIBRATION_SOURCES
    if phase == "heldout":
        return HELDOUT_SOURCES
    raise ValueError("phase must be calibration or heldout")


def _source_manifest_value(source: Any) -> dict[str, str]:
    return {
        "dataset": source.dataset,
        "revision": source.revision,
        "split": source.split,
        "subset": source.subset,
    }


def _source_audit_identity(source: Any) -> dict[str, str]:
    return {
        "config": source.subset,
        "dataset": source.dataset,
        "revision": source.revision,
        "split": source.split,
    }


def _validate_loader_audit(
    audit: Any,
    *,
    source: Any,
    environment: str,
    label: str,
) -> None:
    if not isinstance(audit, dict):
        raise ValueError(f"{label} must be an object")
    expected_keys = {
        "accepted_valid_records",
        "backend",
        "cache_schema_version",
        "derived_page_seed",
        "max_valid_records",
        "metadata_page",
        "page_count",
        "page_order_sha256",
        "page_size",
        "selection_pages",
        "source",
        "total_rows",
        "verified_current_sha",
    }
    if set(audit) != expected_keys:
        raise ValueError(f"{label} fields do not match Addendum 001-B")
    if audit["backend"] != "hugging-face-dataset-viewer/rows":
        raise ValueError(f"{label} backend drift")
    if audit["cache_schema_version"] != CACHE_SCHEMA_VERSION:
        raise ValueError(f"{label} cache schema drift")
    if audit["max_valid_records"] != MAX_VALID_RECORDS:
        raise ValueError(f"{label} record limit drift")
    if audit["page_size"] != VIEWER_PAGE_SIZE:
        raise ValueError(f"{label} page size drift")
    if audit["source"] != _source_audit_identity(source):
        raise ValueError(f"{label} source identity drift")
    if audit["verified_current_sha"] != source.revision:
        raise ValueError(f"{label} verified revision drift")

    total_rows = _required_int(audit["total_rows"], name=f"{label}.total_rows", minimum=1)
    expected_page_count = (total_rows + VIEWER_PAGE_SIZE - 1) // VIEWER_PAGE_SIZE
    if audit["page_count"] != expected_page_count:
        raise ValueError(f"{label} page count mismatch")
    accepted = _required_int(
        audit["accepted_valid_records"],
        name=f"{label}.accepted_valid_records",
        minimum=1,
    )
    if accepted > MAX_VALID_RECORDS:
        raise ValueError(f"{label} accepted too many valid records")

    seed_material = {
        "environment": environment,
        "seed": EXPERIMENT_001_SEED,
        "source": _source_audit_identity(source),
    }
    expected_derived_seed = int(sha256_bytes(canonical_json_bytes(seed_material))[:16], 16)
    if audit["derived_page_seed"] != expected_derived_seed:
        raise ValueError(f"{label} derived page seed mismatch")
    page_order = list(range(expected_page_count))
    random.Random(expected_derived_seed).shuffle(page_order)
    if audit["page_order_sha256"] != sha256_bytes(canonical_json_bytes(page_order)):
        raise ValueError(f"{label} page-order hash mismatch")

    metadata_page = audit["metadata_page"]
    if not isinstance(metadata_page, dict) or set(metadata_page) != {
        "cache_file",
        "page_index",
        "response_sha256",
    }:
        raise ValueError(f"{label} metadata page is invalid")
    metadata_cache_file = _safe_relative_path(
        metadata_page["cache_file"],
        name=f"{label}.metadata_page.cache_file",
    )
    if metadata_page["page_index"] != 0:
        raise ValueError(f"{label} metadata page must be page zero")
    metadata_response_hash = _required_sha256(
        metadata_page["response_sha256"],
        name=f"{label}.metadata_page.response_sha256",
    )
    if Path(metadata_cache_file).name != (f"page-00000000-{metadata_response_hash}.json"):
        raise ValueError(f"{label} metadata cache filename/hash mismatch")

    selection_pages = audit["selection_pages"]
    if not isinstance(selection_pages, list) or not selection_pages:
        raise ValueError(f"{label} has no selection pages")
    if len(selection_pages) > expected_page_count:
        raise ValueError(f"{label} has too many selection pages")
    accepted_sum = 0
    selected_indices: list[int] = []
    for index, page in enumerate(selection_pages):
        page_label = f"{label}.selection_pages[{index}]"
        if not isinstance(page, dict) or set(page) != {
            "accepted_valid_records",
            "cache_file",
            "page_index",
            "response_sha256",
        }:
            raise ValueError(f"{page_label} is invalid")
        page_index = _required_int(
            page["page_index"],
            name=f"{page_label}.page_index",
        )
        if page_index >= expected_page_count:
            raise ValueError(f"{page_label} index is outside the source")
        selected_indices.append(page_index)
        page_accepted = _required_int(
            page["accepted_valid_records"],
            name=f"{page_label}.accepted_valid_records",
        )
        if page_accepted > VIEWER_PAGE_SIZE:
            raise ValueError(f"{page_label} accepted more rows than one page contains")
        accepted_sum += page_accepted
        cache_file = _safe_relative_path(
            page["cache_file"],
            name=f"{page_label}.cache_file",
        )
        response_hash = _required_sha256(
            page["response_sha256"],
            name=f"{page_label}.response_sha256",
        )
        if Path(cache_file).name != f"page-{page_index:08d}-{response_hash}.json":
            raise ValueError(f"{page_label} cache filename/hash mismatch")
    if selected_indices != page_order[: len(selection_pages)]:
        raise ValueError(f"{label} selection pages are not the seeded prefix")
    if accepted_sum != accepted:
        raise ValueError(f"{label} accepted-record sum mismatch")
    if accepted < MAX_VALID_RECORDS and len(selection_pages) != expected_page_count:
        raise ValueError(f"{label} stopped before exhausting a short source")


def _validate_source_loading(
    source_loading: Any,
    *,
    phase: str,
    used_source_keys: set[str],
) -> None:
    if not isinstance(source_loading, dict) or set(source_loading) != {
        "cache_directory",
        "contract",
        "max_records_per_split",
        "method",
        "page_size",
        "sources",
    }:
        raise ValueError("source_loading fields do not match Addendum 001-B")
    if source_loading["cache_directory"] != "dataset-viewer-cache":
        raise ValueError("source_loading cache directory drift")
    if not collection_contract_is_supported(source_loading["contract"]):
        raise ValueError("source_loading collection contract drift")
    if source_loading["max_records_per_split"] != MAX_VALID_RECORDS:
        raise ValueError("source_loading record limit drift")
    if source_loading["method"] != (
        "Hub-SHA-verified Dataset Viewer /rows; deterministic "
        "100-row page shuffle; content-addressed local cache"
    ):
        raise ValueError("source_loading method drift")
    if source_loading["page_size"] != VIEWER_PAGE_SIZE:
        raise ValueError("source_loading page size drift")
    audits = source_loading["sources"]
    if not isinstance(audits, dict):
        raise ValueError("source_loading.sources must be an object")

    expected_by_key: dict[str, tuple[str, Any]] = {}
    for environment, sources in _expected_source_table(phase).items():
        for source in sources:
            expected_by_key[source.key] = (environment, source)
    if not used_source_keys <= set(audits) <= set(expected_by_key):
        raise ValueError("source_loading audit set disagrees with used pinned sources")
    for key, audit in audits.items():
        environment, source = expected_by_key[key]
        _validate_loader_audit(
            audit,
            source=source,
            environment=environment,
            label=f"source_loading.sources[{key}]",
        )


def _validate_record(
    record: Any,
    *,
    expected_sources: Sequence[Any],
    label: str,
) -> tuple[str, str]:
    if not isinstance(record, dict) or set(record) != {
        "dataset",
        "rendered_sha256",
        "revision",
        "row_id",
        "source_index",
        "split",
        "subset",
        "token_count",
        "variant",
    }:
        raise ValueError(f"{label} fields are invalid")
    identity = {
        "dataset": record["dataset"],
        "revision": record["revision"],
        "split": record["split"],
        "subset": record["subset"],
    }
    matching = [source for source in expected_sources if _source_manifest_value(source) == identity]
    if len(matching) != 1:
        raise ValueError(f"{label} source is outside the frozen dataset table")
    digest = _required_sha256(record["rendered_sha256"], name=f"{label}.rendered_sha256")
    _required_string(record["row_id"], name=f"{label}.row_id")
    _required_int(record["source_index"], name=f"{label}.source_index")
    _required_int(record["token_count"], name=f"{label}.token_count", minimum=1)
    variant = record["variant"]
    if variant is not None and not isinstance(variant, str):
        raise ValueError(f"{label}.variant must be null or a string")
    return digest, matching[0].key


def _validate_window(
    window: Any,
    *,
    environment: str,
    index: int,
    window_tokens: int,
    record_hashes: set[str],
    input_ids: NDArray[np.int64],
) -> str:
    label = f"environments.{environment}.windows[{index}]"
    if not isinstance(window, dict) or set(window) != {
        "index",
        "language",
        "segments",
        "token_count",
        "token_sha256",
    }:
        raise ValueError(f"{label} fields are invalid")
    if window["index"] != index or window["token_count"] != window_tokens:
        raise ValueError(f"{label} index or token count drift")
    language = window["language"]
    if environment == "multilingual":
        expected_language = ("ar", "es", "hi", "zh")[index % 4]
        if language != expected_language:
            raise ValueError(f"{label} multilingual interleave drift")
    elif language is not None:
        raise ValueError(f"{label} unexpectedly has a language label")
    token_hash = _required_sha256(window["token_sha256"], name=f"{label}.token_sha256")
    expected_hash = sha256_bytes(canonical_json_bytes(input_ids.tolist()))
    if token_hash != expected_hash:
        raise ValueError(f"{label} token-array hash mismatch")

    segments = window["segments"]
    if not isinstance(segments, list) or not segments:
        raise ValueError(f"{label} has no source segments")
    cursor = 0
    for segment_index, segment in enumerate(segments):
        segment_label = f"{label}.segments[{segment_index}]"
        if not isinstance(segment, dict) or set(segment) != {
            "record_token_end",
            "record_token_start",
            "rendered_sha256",
            "row_id",
            "window_token_end",
            "window_token_start",
        }:
            raise ValueError(f"{segment_label} fields are invalid")
        rendered_hash = _required_sha256(
            segment["rendered_sha256"],
            name=f"{segment_label}.rendered_sha256",
        )
        if rendered_hash not in record_hashes:
            raise ValueError(f"{segment_label} references an unknown rendered record")
        _required_string(segment["row_id"], name=f"{segment_label}.row_id")
        record_start = _required_int(
            segment["record_token_start"],
            name=f"{segment_label}.record_token_start",
        )
        record_end = _required_int(
            segment["record_token_end"],
            name=f"{segment_label}.record_token_end",
            minimum=1,
        )
        window_start = _required_int(
            segment["window_token_start"],
            name=f"{segment_label}.window_token_start",
        )
        window_end = _required_int(
            segment["window_token_end"],
            name=f"{segment_label}.window_token_end",
            minimum=1,
        )
        if (
            window_start != cursor
            or window_end <= window_start
            or window_end > window_tokens
            or record_end <= record_start
            or record_end - record_start != window_end - window_start
        ):
            raise ValueError(f"{segment_label} token ranges are inconsistent")
        cursor = window_end
    if cursor != window_tokens:
        raise ValueError(f"{label} segments do not cover the token window")
    return token_hash


def load_corpus_artifact(
    directory: str | Path,
    *,
    phase: str,
) -> CorpusArtifact:
    """Verify the exact corpus manifest, token archive, and Addendum 001-B audit."""

    root = Path(directory).resolve()
    manifest_path = root / f"{phase}.manifest.json"
    sidecar_path = root / f"{phase}.sha256.json"
    manifest = _load_strict_json(manifest_path)
    expected_top_keys = {
        "dry_run",
        "environments",
        "model",
        "phase",
        "rendering",
        "runtime",
        "sampling",
        "schema",
        "source_loading",
        "token_archive",
        "tokenizer",
        "window_count_per_environment",
        "window_tokens",
    }
    if set(manifest) != expected_top_keys:
        raise ValueError("corpus manifest fields do not match schema v1")
    if manifest["schema"] != "cliffquant.corpus-manifest.v1":
        raise ValueError("unsupported corpus manifest schema")
    if manifest["phase"] != phase:
        raise ValueError("corpus manifest phase mismatch")
    if manifest["dry_run"] is not False:
        raise ValueError("dry-run corpus artifacts cannot enter Experiment 001")
    if MODEL_ID != EXPERIMENT_001_MODEL_ID or manifest["model"] != {
        "id": EXPERIMENT_001_MODEL_ID,
        "revision": EXPERIMENT_001_MODEL_REVISION,
    }:
        raise ValueError("corpus manifest model id or revision drift")
    expected_count = CALIBRATION_WINDOWS if phase == "calibration" else HELDOUT_WINDOWS
    if manifest["window_count_per_environment"] != expected_count:
        raise ValueError("corpus window count drift")
    if manifest["window_tokens"] != WINDOW_TOKENS or WINDOW_TOKENS != 256:
        raise ValueError("corpus window size drift")
    expected_sampling = {
        "deduplication": "first seeded-rank occurrence of each rendered SHA256",
        "non_overlapping": True,
        "ordering": "SHA256(seed, environment, phase, source, row ID, rendered SHA256)",
        "seed": SEED,
        "split_boundary_policy": "discard incomplete tail before the next split",
    }
    if SEED != EXPERIMENT_001_SEED or manifest["sampling"] != expected_sampling:
        raise ValueError("corpus sampling metadata drift")
    _validate_runtime(
        manifest["runtime"],
        label="corpus",
        require_deterministic_cuda=False,
    )
    if manifest["tokenizer"] != expected_tokenizer_metadata():
        raise ValueError("corpus tokenizer identity drift")
    if not isinstance(manifest["rendering"], dict) or set(manifest["rendering"]) != set(
        EXPERIMENT_001_ENVIRONMENTS
    ):
        raise ValueError("corpus rendering metadata is incomplete")

    archive_descriptor = manifest["token_archive"]
    if not isinstance(archive_descriptor, dict) or set(archive_descriptor) != {
        "file",
        "sha256",
        "size_bytes",
    }:
        raise ValueError("corpus token archive descriptor is invalid")
    archive_name = _required_string(
        archive_descriptor["file"],
        name="token_archive.file",
    )
    if Path(archive_name).name != archive_name or archive_name != f"{phase}.tokens.npz":
        raise ValueError("corpus token archive filename drift")
    archive_path = root / archive_name
    if not archive_path.is_file() or archive_path.stat().st_size != _required_int(
        archive_descriptor["size_bytes"], name="token_archive.size_bytes"
    ):
        raise ValueError("corpus token archive size mismatch")
    archive_hash = sha256_file(archive_path)
    if archive_hash != _required_sha256(
        archive_descriptor["sha256"],
        name="token_archive.sha256",
    ):
        raise ValueError("corpus token archive hash mismatch")

    manifest_hash = sha256_file(manifest_path)
    if manifest_path.read_bytes() != canonical_json_bytes(manifest):
        raise ValueError("corpus manifest is not canonical JSON")
    sidecar = _load_strict_json(sidecar_path)
    if set(sidecar) != {"manifest", "tokens"}:
        raise ValueError("corpus hash sidecar fields are invalid")
    manifest_descriptor = sidecar["manifest"]
    if not isinstance(manifest_descriptor, dict) or manifest_descriptor != {
        "file": manifest_path.name,
        "sha256": manifest_hash,
        "size_bytes": manifest_path.stat().st_size,
    }:
        raise ValueError("corpus manifest sidecar hash mismatch")
    if sidecar["tokens"] != archive_descriptor:
        raise ValueError("corpus token sidecar disagrees with manifest")

    environments = manifest["environments"]
    if not isinstance(environments, dict) or set(environments) != set(EXPERIMENT_001_ENVIRONMENTS):
        raise ValueError("corpus environment order drift")
    expected_sources = _expected_source_table(phase)
    rendered_hashes: set[str] = set()
    window_hashes: set[str] = set()
    used_source_keys: set[str] = set()
    expected_array_keys = {
        f"{environment}__{kind}"
        for environment in EXPERIMENT_001_ENVIRONMENTS
        for kind in ("attention_mask", "input_ids")
    }
    try:
        arrays_context = np.load(archive_path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError("cannot open corpus token archive") from exc
    with arrays_context as arrays:
        if set(arrays.files) != expected_array_keys:
            raise ValueError("corpus token archive array set drift")
        for environment in EXPERIMENT_001_ENVIRONMENTS:
            environment_value = environments[environment]
            if not isinstance(environment_value, dict) or set(environment_value) != {
                "datasets",
                "records",
                "windows",
            }:
                raise ValueError(f"invalid corpus environment entry: {environment}")
            expected_dataset_table = [
                _source_manifest_value(source) for source in expected_sources[environment]
            ]
            if environment_value["datasets"] != expected_dataset_table:
                raise ValueError(f"{environment} dataset revision table drift")
            records = environment_value["records"]
            if not isinstance(records, list) or not records:
                raise ValueError(f"{environment} has no rendered records")
            environment_record_hashes: set[str] = set()
            for record_index, record in enumerate(records):
                digest, source_key = _validate_record(
                    record,
                    expected_sources=expected_sources[environment],
                    label=f"environments.{environment}.records[{record_index}]",
                )
                if digest in environment_record_hashes:
                    raise ValueError("corpus manifest contains a duplicate rendered record")
                rendered_hashes.add(digest)
                environment_record_hashes.add(digest)
                used_source_keys.add(source_key)

            ids = np.asarray(arrays[f"{environment}__input_ids"])
            mask = np.asarray(arrays[f"{environment}__attention_mask"])
            expected_shape = (expected_count, WINDOW_TOKENS)
            if ids.dtype.str != "<i8" or ids.shape != expected_shape or np.any(ids < 0):
                raise ValueError(f"{environment} input_ids array drift")
            if mask.dtype.str != "|u1" or mask.shape != expected_shape or not np.all(mask == 1):
                raise ValueError(f"{environment} attention_mask array drift")
            windows = environment_value["windows"]
            if not isinstance(windows, list) or len(windows) != expected_count:
                raise ValueError(f"{environment} window manifest count drift")
            for window_index, window in enumerate(windows):
                token_hash = _validate_window(
                    window,
                    environment=environment,
                    index=window_index,
                    window_tokens=WINDOW_TOKENS,
                    record_hashes=environment_record_hashes,
                    input_ids=ids[window_index],
                )
                window_hashes.add(token_hash)

    _validate_source_loading(
        manifest["source_loading"],
        phase=phase,
        used_source_keys=used_source_keys,
    )
    return CorpusArtifact(
        phase=phase,
        manifest_path=manifest_path,
        archive_path=archive_path,
        manifest_sha256=manifest_hash,
        archive_sha256=archive_hash,
        sidecar_sha256=sha256_file(sidecar_path),
        rendered_sha256=frozenset(rendered_hashes),
        window_sha256=frozenset(window_hashes),
        tokenizer=dict(manifest["tokenizer"]),
    )


def _validated_array(
    arrays: Any,
    descriptor: Any,
    *,
    expected_features: int,
    label: str,
) -> NDArray[np.float64]:
    if not isinstance(descriptor, dict):
        raise ValueError(f"{label} descriptor must be an object")
    key = _required_string(descriptor.get("array_key"), name=f"{label}.array_key")
    if descriptor.get("dtype") != "<f8":
        raise ValueError(f"{label} dtype must be <f8")
    if descriptor.get("shape") != [expected_features]:
        raise ValueError(f"{label} shape disagrees with module width")
    if key not in arrays.files:
        raise ValueError(f"{label} array key is missing from archive: {key}")
    value = np.asarray(arrays[key])
    if value.dtype.str != "<f8" or value.shape != (expected_features,):
        raise ValueError(f"{label} archive array has unexpected dtype or shape")
    if not np.all(np.isfinite(value)) or np.any(value < 0.0):
        raise ValueError(f"{label} must contain finite non-negative values")
    expected_hash = _required_string(descriptor.get("sha256"), name=f"{label}.sha256")
    if canonical_array_sha256(value) != expected_hash:
        raise ValueError(f"{label} array hash mismatch")
    result = np.ascontiguousarray(value, dtype=np.float64)
    result.setflags(write=False)
    return result


def load_diagonal_artifact(
    metadata_path: str | Path,
    *,
    expected_phase: str,
    require_frozen_inventory: bool = True,
) -> DiagonalArtifact:
    """Load one activation-statistics artifact after checking every recorded hash."""

    path = Path(metadata_path).resolve()
    metadata = _load_strict_json(path)
    if path.name != f"{expected_phase}.diagonals.json":
        raise ValueError("diagonal metadata filename drift")
    if set(metadata) != {
        "arrays",
        "environments",
        "module_tree",
        "modules",
        "phase",
        "provenance",
        "schema",
        "statistics_archive",
    }:
        raise ValueError("diagonal metadata fields do not match schema v1")
    if metadata.get("schema") != "cliffquant.activation-diagonals.v1":
        raise ValueError("unsupported diagonal metadata schema")
    if metadata.get("phase") != expected_phase:
        raise ValueError(f"expected {expected_phase} diagonals, found {metadata.get('phase')!r}")
    environments_raw = metadata.get("environments")
    if not isinstance(environments_raw, list) or not all(
        isinstance(environment, str) for environment in environments_raw
    ):
        raise ValueError("diagonal environments must be a string list")
    environments = tuple(environments_raw)
    if environments != EXPERIMENT_001_ENVIRONMENTS:
        raise ValueError(
            f"environment drift: expected {EXPERIMENT_001_ENVIRONMENTS}, found {environments}"
        )

    module_tree = metadata.get("module_tree")
    if not isinstance(module_tree, dict) or module_tree != {
        "implementation": "GPTQModel qwen3_5.py",
        "revision": QWEN35_GPTQMODEL_TREE_REVISION,
    }:
        raise ValueError("missing module_tree provenance")

    corpus = load_corpus_artifact(path.parent, phase=expected_phase)
    provenance = metadata.get("provenance")
    if not isinstance(provenance, dict) or set(provenance) != {
        "corpus_manifest_sha256",
        "gptqmodel_module_tree_revision",
        "model",
        "runtime",
        "tokenizer",
    }:
        raise ValueError("diagonal collection provenance is incomplete")
    model_source = validate_model_snapshot_descriptor(
        provenance["model"],
        model_id=EXPERIMENT_001_MODEL_ID,
        revision=EXPERIMENT_001_MODEL_REVISION,
    )
    if provenance["gptqmodel_module_tree_revision"] != QWEN35_GPTQMODEL_TREE_REVISION:
        raise ValueError("diagonal provenance GPTQModel tree revision drift")
    if provenance["corpus_manifest_sha256"] != corpus.manifest_sha256:
        raise ValueError("diagonal provenance corpus manifest hash mismatch")
    _validate_runtime(
        provenance["runtime"],
        label="diagonal",
        require_deterministic_cuda=True,
    )
    if provenance["tokenizer"] != expected_tokenizer_metadata():
        raise ValueError("diagonal tokenizer identity drift")
    if provenance["tokenizer"] != corpus.tokenizer:
        raise ValueError("diagonal tokenizer disagrees with its corpus artifact")

    modules = _parse_inventory(metadata)
    if require_frozen_inventory:
        _validate_frozen_inventory(modules)

    archive_descriptor = metadata.get("statistics_archive")
    if not isinstance(archive_descriptor, dict):
        raise ValueError("missing statistics_archive descriptor")
    archive_name = _required_string(
        archive_descriptor.get("file"),
        name="statistics_archive.file",
    )
    if Path(archive_name).name != archive_name or archive_name != f"{expected_phase}.diagonals.npz":
        raise ValueError("statistics archive filename drift")
    archive_path = path.with_name(archive_name)
    expected_archive_size = _required_int(
        archive_descriptor.get("size_bytes"),
        name="statistics_archive.size_bytes",
    )
    if not archive_path.is_file() or archive_path.stat().st_size != expected_archive_size:
        raise ValueError("statistics archive size mismatch")
    archive_hash = sha256_file(archive_path)
    if archive_hash != archive_descriptor.get("sha256"):
        raise ValueError("statistics archive hash mismatch")

    sidecar_path = _sidecar_path(path)
    sidecar = _load_strict_json(sidecar_path)
    if set(sidecar) != {"metadata", "statistics"}:
        raise ValueError("diagonal hash sidecar fields are invalid")
    metadata_descriptor = sidecar.get("metadata")
    statistics_descriptor = sidecar.get("statistics")
    if not isinstance(metadata_descriptor, dict) or not isinstance(statistics_descriptor, dict):
        raise ValueError("diagonal hash sidecar is incomplete")
    metadata_hash = sha256_file(path)
    if (
        metadata_descriptor.get("file") != path.name
        or metadata_descriptor.get("sha256") != metadata_hash
        or metadata_descriptor.get("size_bytes") != path.stat().st_size
    ):
        raise ValueError("diagonal metadata sidecar hash mismatch")
    if statistics_descriptor != archive_descriptor:
        raise ValueError("diagonal archive sidecar disagrees with metadata")

    raw_entries = metadata.get("arrays")
    if not isinstance(raw_entries, list):
        raise ValueError("diagonal arrays must be a list")
    module_by_name = {module.name: module for module in modules}
    expected_pairs = {
        (environment, module.name) for module in modules for environment in environments
    }
    observed_pairs: set[tuple[str, str]] = set()
    normalized_by_module: dict[str, NDArray[np.float64]] = {
        module.name: np.empty(
            (len(environments), module.in_features),
            dtype=np.float64,
        )
        for module in modules
    }
    referenced_keys: set[str] = set()

    try:
        arrays_context = np.load(archive_path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError("cannot open statistics archive") from exc
    with arrays_context as arrays:
        for entry_index, entry in enumerate(raw_entries):
            if not isinstance(entry, dict):
                raise ValueError(f"arrays[{entry_index}] must be an object")
            environment = _required_string(
                entry.get("environment"),
                name=f"arrays[{entry_index}].environment",
            )
            module_name = _required_string(
                entry.get("module"),
                name=f"arrays[{entry_index}].module",
            )
            pair = (environment, module_name)
            if pair not in expected_pairs:
                raise ValueError(f"unexpected diagonal pair: {environment}/{module_name}")
            if pair in observed_pairs:
                raise ValueError(f"duplicate diagonal pair: {environment}/{module_name}")
            observed_pairs.add(pair)
            module = module_by_name[module_name]
            normalization = entry.get("normalization_constant")
            if isinstance(normalization, bool) or not isinstance(normalization, (int, float)):
                raise ValueError(f"invalid normalization constant for {environment}/{module_name}")
            normalization = float(normalization)
            if not np.isfinite(normalization) or normalization <= 0.0:
                raise ValueError(f"invalid normalization constant for {environment}/{module_name}")
            token_rows = _required_int(
                entry.get("token_rows"),
                name=f"token_rows for {environment}/{module_name}",
                minimum=1,
            )
            expected_token_rows = (
                CALIBRATION_WINDOWS if expected_phase == "calibration" else HELDOUT_WINDOWS
            ) * WINDOW_TOKENS
            if token_rows != expected_token_rows:
                raise ValueError(f"token row count drift for {environment}/{module_name}")

            raw = _validated_array(
                arrays,
                entry.get("raw"),
                expected_features=module.in_features,
                label=f"raw {environment}/{module_name}",
            )
            normalized = _validated_array(
                arrays,
                entry.get("normalized"),
                expected_features=module.in_features,
                label=f"normalized {environment}/{module_name}",
            )
            raw_key = entry["raw"]["array_key"]
            normalized_key = entry["normalized"]["array_key"]
            if raw_key in referenced_keys or normalized_key in referenced_keys:
                raise ValueError("an archive array key is referenced more than once")
            referenced_keys.update((raw_key, normalized_key))
            if not np.allclose(
                raw / normalization,
                normalized,
                rtol=1e-12,
                atol=1e-12,
            ):
                raise ValueError(f"normalized diagonal arithmetic mismatch for {pair}")
            if not np.isclose(
                np.mean(normalized, dtype=np.float64),
                1.0,
                rtol=1e-12,
                atol=1e-12,
            ):
                raise ValueError(f"normalized diagonal does not have mean one for {pair}")
            environment_index = environments.index(environment)
            normalized_by_module[module_name][environment_index] = normalized
        if observed_pairs != expected_pairs:
            missing = sorted(expected_pairs - observed_pairs)
            raise ValueError(f"missing diagonal module/environment pairs: {missing[:3]}")
        if set(arrays.files) != referenced_keys:
            raise ValueError("statistics archive contains unreferenced arrays")

    for value in normalized_by_module.values():
        value.setflags(write=False)
    return DiagonalArtifact(
        phase=expected_phase,
        environments=environments,
        modules=modules,
        normalized=normalized_by_module,
        metadata_path=path,
        archive_path=archive_path,
        metadata_sha256=metadata_hash,
        archive_sha256=archive_hash,
        sidecar_sha256=sha256_file(sidecar_path),
        corpus=corpus,
        model_source=model_source,
    )


def validate_diagonal_pair(
    calibration: DiagonalArtifact,
    heldout: DiagonalArtifact,
    *,
    require_frozen_inventory: bool = True,
) -> None:
    """Reject any inventory or environment drift between both experiment phases."""

    if calibration.phase != "calibration" or heldout.phase != "heldout":
        raise ValueError("expected calibration and heldout artifacts")
    if calibration.environments != heldout.environments:
        raise ValueError("calibration and heldout environment order differs")
    if calibration.modules != heldout.modules:
        raise ValueError("calibration and heldout module inventories differ")
    if calibration.model_source != heldout.model_source:
        raise ValueError("calibration and heldout model-byte provenance differs")
    record_overlap = calibration.corpus.rendered_sha256 & heldout.corpus.rendered_sha256
    window_overlap = calibration.corpus.window_sha256 & heldout.corpus.window_sha256
    if record_overlap or window_overlap:
        details: list[str] = []
        if record_overlap:
            details.append(f"{len(record_overlap)} rendered-text hashes")
        if window_overlap:
            details.append(f"{len(window_overlap)} token-window hashes")
        raise ValueError("calibration/held-out overlap detected: " + ", ".join(details))
    if require_frozen_inventory:
        _validate_frozen_inventory(calibration.modules)


@dataclass(frozen=True, slots=True)
class _TensorLocation:
    shard: Path
    dtype: str
    shape: tuple[int, ...]
    absolute_data_start: int
    absolute_data_end: int


def _read_safetensors_header(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as handle:
            prefix = handle.read(8)
            if len(prefix) != 8:
                raise ValueError("truncated safetensors length prefix")
            header_length = struct.unpack("<Q", prefix)[0]
            if header_length == 0 or header_length > path.stat().st_size - 8:
                raise ValueError("invalid safetensors header length")
            header = handle.read(header_length)
            if len(header) != header_length:
                raise ValueError("truncated safetensors header")
    except OSError as exc:
        raise ValueError(f"cannot read safetensors header from {path}") from exc
    try:
        value = json.loads(
            header.decode("utf-8").rstrip(" "),
            object_pairs_hook=_strict_object,
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid safetensors header in {path}") from exc
    if not isinstance(value, dict):
        raise ValueError("safetensors header must be an object")
    value["_cliffquant_data_start"] = 8 + header_length
    return value


class GroupWeightSource(Protocol):
    """Minimal source used by deterministic sample materialization."""

    modules: tuple[ModuleInventoryEntry, ...]

    def read_groups(
        self,
        coordinates: Sequence[GroupCoordinate],
    ) -> dict[str, NDArray[np.float64]]:
        """Return one finite group for every coordinate, keyed by group id."""


class SafetensorWeightStore:
    """Read only selected Qwen weight groups directly from safetensors shards."""

    def __init__(
        self,
        model_dir: str | Path,
        *,
        revision: str = EXPERIMENT_001_MODEL_REVISION,
        require_revision_directory: bool = True,
        hash_shards: bool = True,
    ) -> None:
        root = Path(model_dir).resolve()
        if not root.is_dir():
            raise ValueError(f"model directory does not exist: {root}")
        if revision != EXPERIMENT_001_MODEL_REVISION:
            raise ValueError("Experiment 001 model revision is frozen")
        if require_revision_directory and root.name != revision:
            raise ValueError(
                "model directory must be the pinned Hugging Face snapshot directory "
                f"named {revision}"
            )
        model_source = (
            describe_model_snapshot(
                root,
                model_id=EXPERIMENT_001_MODEL_ID,
                revision=revision,
            )
            if hash_shards
            else None
        )
        index_path = root / "model.safetensors.index.json"
        index = _load_strict_json(index_path)
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("model safetensors index has no weight_map")
        if not all(
            isinstance(key, str) and isinstance(value, str) for key, value in weight_map.items()
        ):
            raise ValueError("model weight_map must map string tensor names to shard basenames")

        target_names = sorted(
            name
            for name in weight_map
            if name.endswith(".weight") and _parse_target_name(name) is not None
        )
        if len(target_names) != QWEN35_08B_EXPECTED_QUANTIZED_MODULES:
            raise ValueError(
                "model inventory drift: expected "
                f"{QWEN35_08B_EXPECTED_QUANTIZED_MODULES} target tensors, "
                f"found {len(target_names)}"
            )

        shard_names = sorted({weight_map[name] for name in target_names})
        locations: dict[str, _TensorLocation] = {}
        shard_descriptors: dict[str, dict[str, Any]] = {}
        for shard_name in shard_names:
            if Path(shard_name).name != shard_name:
                raise ValueError("safetensors shard must be a basename")
            shard = root / shard_name
            if not shard.is_file():
                raise ValueError(f"missing safetensors shard: {shard_name}")
            header = _read_safetensors_header(shard)
            data_start = _required_int(
                header.pop("_cliffquant_data_start"),
                name="safetensors data start",
                minimum=8,
            )
            file_size = shard.stat().st_size
            for tensor_name in target_names:
                if weight_map[tensor_name] != shard_name:
                    continue
                descriptor = header.get(tensor_name)
                if not isinstance(descriptor, dict):
                    raise ValueError(f"{tensor_name} is missing from {shard_name}")
                dtype = _required_string(
                    descriptor.get("dtype"),
                    name=f"{tensor_name}.dtype",
                )
                if dtype not in _SAFETENSORS_DTYPES:
                    raise ValueError(f"unsupported target tensor dtype {dtype} for {tensor_name}")
                shape_raw = descriptor.get("shape")
                if (
                    not isinstance(shape_raw, list)
                    or len(shape_raw) != 2
                    or any(type(item) is not int or item <= 0 for item in shape_raw)
                ):
                    raise ValueError(f"{tensor_name} must have a positive rank-2 shape")
                shape = tuple(shape_raw)
                offsets = descriptor.get("data_offsets")
                if (
                    not isinstance(offsets, list)
                    or len(offsets) != 2
                    or any(type(item) is not int for item in offsets)
                ):
                    raise ValueError(f"invalid data offsets for {tensor_name}")
                relative_start, relative_end = offsets
                item_size = _SAFETENSORS_DTYPES[dtype][1]
                expected_bytes = math.prod(shape) * item_size
                if (
                    relative_start < 0
                    or relative_end - relative_start != expected_bytes
                    or data_start + relative_end > file_size
                ):
                    raise ValueError(f"invalid tensor extent for {tensor_name}")
                locations[tensor_name] = _TensorLocation(
                    shard=shard,
                    dtype=dtype,
                    shape=shape,
                    absolute_data_start=data_start + relative_start,
                    absolute_data_end=data_start + relative_end,
                )
            shard_descriptors[shard_name] = {
                "sha256": sha256_file(shard) if hash_shards else None,
                "size_bytes": file_size,
            }

        modules: list[ModuleInventoryEntry] = []
        for tensor_name in target_names:
            parsed = _parse_target_name(tensor_name)
            if parsed is None:  # pragma: no cover - target_names was filtered above
                raise RuntimeError("target-name filter changed during inventory construction")
            layer, branch = parsed
            out_features, in_features = locations[tensor_name].shape
            modules.append(
                ModuleInventoryEntry(
                    name=tensor_name.removesuffix(".weight"),
                    layer=layer,
                    branch=branch,
                    in_features=in_features,
                    out_features=out_features,
                )
            )
        modules.sort(key=_inventory_sort_key)
        _validate_frozen_inventory(modules)
        self.root = root
        self.revision = revision
        self.modules = tuple(modules)
        self._locations = locations
        self._module_by_name = {module.name: module for module in modules}
        self.source = (
            model_source
            if model_source is not None
            else {
                "index": {
                    "file": index_path.name,
                    "sha256": sha256_file(index_path),
                    "size_bytes": index_path.stat().st_size,
                },
                "revision": revision,
                "shards": shard_descriptors,
            }
        )

    @staticmethod
    def _decode_group(raw: bytes, dtype: str) -> NDArray[np.float64]:
        numpy_dtype, item_size = _SAFETENSORS_DTYPES[dtype]
        if len(raw) != EXPERIMENT_001_GROUP_SIZE * item_size:
            raise ValueError("short read from safetensors weight group")
        encoded = np.frombuffer(raw, dtype=numpy_dtype, count=EXPERIMENT_001_GROUP_SIZE)
        if dtype == "BF16":
            expanded = encoded.astype(np.uint32) << np.uint32(16)
            values = expanded.view("<f4")
        else:
            values = encoded
        decoded = np.ascontiguousarray(values, dtype=np.float64)
        if not np.all(np.isfinite(decoded)):
            raise ValueError("weight group contains non-finite values")
        decoded.setflags(write=False)
        return decoded

    def read_groups(
        self,
        coordinates: Sequence[GroupCoordinate],
    ) -> dict[str, NDArray[np.float64]]:
        """Read selected groups while opening each shard only once."""

        by_shard: dict[Path, list[GroupCoordinate]] = {}
        for coordinate in coordinates:
            module = self._module_by_name.get(coordinate.module)
            if module is None:
                raise ValueError(f"sample references an unknown module: {coordinate.module}")
            if not 0 <= coordinate.row < module.out_features:
                raise ValueError(f"sample row is outside {coordinate.module}")
            if not 0 <= coordinate.group < module.groups_per_row:
                raise ValueError(f"sample group is outside {coordinate.module}")
            location = self._locations[module.tensor_name]
            by_shard.setdefault(location.shard, []).append(coordinate)

        result: dict[str, NDArray[np.float64]] = {}
        for shard, shard_coordinates in sorted(by_shard.items(), key=lambda item: item[0].name):
            with shard.open("rb") as handle:
                for coordinate in sorted(shard_coordinates, key=lambda item: item.sample_index):
                    module = self._module_by_name[coordinate.module]
                    location = self._locations[module.tensor_name]
                    _, item_size = _SAFETENSORS_DTYPES[location.dtype]
                    flat_start = (
                        coordinate.row * module.in_features
                        + coordinate.group * EXPERIMENT_001_GROUP_SIZE
                    )
                    byte_start = location.absolute_data_start + flat_start * item_size
                    byte_end = byte_start + EXPERIMENT_001_GROUP_SIZE * item_size
                    if byte_end > location.absolute_data_end:
                        raise ValueError(f"sample extent exceeds tensor {module.tensor_name}")
                    handle.seek(byte_start)
                    result[coordinate.group_id] = self._decode_group(
                        handle.read(EXPERIMENT_001_GROUP_SIZE * item_size),
                        location.dtype,
                    )
        if len(result) != len(coordinates):
            raise ValueError("sample contained duplicate group identities")
        return result


@dataclass(frozen=True, slots=True)
class GroupCoordinate:
    """One deterministic output-row/input-group coordinate."""

    sample_index: int
    module: str
    layer: int
    branch: str
    row: int
    group: int
    input_start: int
    input_stop: int
    group_id: str

    def identity(self) -> dict[str, Any]:
        return {
            "branch": self.branch,
            "group": self.group,
            "group_id": self.group_id,
            "input_start": self.input_start,
            "input_stop": self.input_stop,
            "layer": self.layer,
            "module": self.module,
            "row": self.row,
            "sample_index": self.sample_index,
        }


@dataclass(frozen=True, slots=True)
class SampledGroup:
    """One materialized group and its canonical numerical fingerprint."""

    coordinate: GroupCoordinate
    weights: NDArray[np.float64]
    weight_sha256: str


def stratified_module_quotas(
    modules: Sequence[ModuleInventoryEntry],
    *,
    sample_size: int = EXPERIMENT_001_SAMPLE_SIZE,
) -> dict[str, int]:
    """Allocate an exact sample with one guaranteed group from every module."""

    if not modules:
        raise ValueError("at least one module is required")
    if sample_size < len(modules):
        raise ValueError("sample_size cannot cover every module")
    total_groups = sum(module.group_count for module in modules)
    if sample_size > total_groups:
        raise ValueError("sample_size exceeds the available groups")
    quotas = {module.name: 1 for module in modules}
    remaining = sample_size - len(modules)
    if remaining == 0:
        return quotas
    capacities = {module.name: module.group_count - 1 for module in modules}
    total_capacity = sum(capacities.values())
    allocated = 0
    remainders: list[tuple[int, str]] = []
    for module in modules:
        numerator = remaining * capacities[module.name]
        share, remainder = divmod(numerator, total_capacity)
        quotas[module.name] += share
        allocated += share
        remainders.append((remainder, module.name))
    for _, name in sorted(remainders, key=lambda item: (-item[0], item[1]))[
        : remaining - allocated
    ]:
        quotas[name] += 1
    if sum(quotas.values()) != sample_size:
        raise RuntimeError("stratified quota arithmetic did not preserve sample size")
    if any(quotas[module.name] > module.group_count for module in modules):
        raise RuntimeError("stratified quota exceeds a module's capacity")
    return quotas


def _module_positions(module: ModuleInventoryEntry, quota: int, seed: int) -> list[int]:
    count = module.group_count
    digest = hashlib.sha256(f"{seed},{module.name}".encode()).digest()
    start = int.from_bytes(digest[:8], "big") % count
    stride = int.from_bytes(digest[8:16], "big") % count
    if stride == 0:
        stride = 1
    while math.gcd(stride, count) != 1:
        stride += 1
        if stride == count:
            stride = 1
    return sorted((start + index * stride) % count for index in range(quota))


def deterministic_group_coordinates(
    modules: Sequence[ModuleInventoryEntry],
    *,
    sample_size: int = EXPERIMENT_001_SAMPLE_SIZE,
    seed: int = EXPERIMENT_001_SEED,
) -> tuple[GroupCoordinate, ...]:
    """Create the frozen module-stratified sample without reading model weights."""

    if type(seed) is not int:
        raise TypeError("seed must be an integer")
    ordered = tuple(sorted(modules, key=_inventory_sort_key))
    if tuple(modules) != ordered:
        raise ValueError("modules must be in frozen inventory order")
    quotas = stratified_module_quotas(ordered, sample_size=sample_size)
    pending: list[tuple[ModuleInventoryEntry, int]] = []
    for module in ordered:
        pending.extend(
            (module, flat_index)
            for flat_index in _module_positions(module, quotas[module.name], seed)
        )

    coordinates: list[GroupCoordinate] = []
    for sample_index, (module, flat_index) in enumerate(pending):
        row, group = divmod(flat_index, module.groups_per_row)
        identity_without_id = {
            "branch": module.branch,
            "group": group,
            "input_start": group * EXPERIMENT_001_GROUP_SIZE,
            "input_stop": (group + 1) * EXPERIMENT_001_GROUP_SIZE,
            "layer": module.layer,
            "module": module.name,
            "row": row,
            "sample_index": sample_index,
        }
        group_id = hashlib.sha256(canonical_json_bytes(identity_without_id)).hexdigest()
        coordinates.append(
            GroupCoordinate(
                sample_index=sample_index,
                module=module.name,
                layer=module.layer,
                branch=module.branch,
                row=row,
                group=group,
                input_start=group * EXPERIMENT_001_GROUP_SIZE,
                input_stop=(group + 1) * EXPERIMENT_001_GROUP_SIZE,
                group_id=group_id,
            )
        )
    if len(coordinates) != sample_size:
        raise RuntimeError("deterministic sample has the wrong size")
    if len({coordinate.group_id for coordinate in coordinates}) != sample_size:
        raise RuntimeError("deterministic sample contains duplicate coordinates")
    return tuple(coordinates)


def materialize_sample(
    source: GroupWeightSource,
    coordinates: Sequence[GroupCoordinate],
) -> tuple[SampledGroup, ...]:
    """Read and fingerprint every selected group."""

    values = source.read_groups(coordinates)
    materialized: list[SampledGroup] = []
    for coordinate in coordinates:
        weights = np.asarray(values[coordinate.group_id], dtype=np.float64)
        if weights.shape != (EXPERIMENT_001_GROUP_SIZE,) or not np.all(np.isfinite(weights)):
            raise ValueError(f"invalid weight group for {coordinate.group_id}")
        weights = np.ascontiguousarray(weights)
        weights.setflags(write=False)
        weight_hash = canonical_array_sha256(weights.astype("<f8"))
        materialized.append(
            SampledGroup(
                coordinate=coordinate,
                weights=weights,
                weight_sha256=weight_hash,
            )
        )
    return tuple(materialized)


def sample_manifest(
    groups: Sequence[SampledGroup],
    *,
    seed: int = EXPERIMENT_001_SEED,
) -> dict[str, Any]:
    """Return the canonical public manifest for a materialized sample."""

    return {
        "entries": [
            {
                **group.coordinate.identity(),
                "weight_sha256": group.weight_sha256,
            }
            for group in groups
        ],
        "group_size": EXPERIMENT_001_GROUP_SIZE,
        "sample_size": len(groups),
        "sampling": {
            "allocation": "one-per-module-then-capacity-proportional-largest-remainder",
            "positions": "sha256-derived-coprime-modular-traversal",
            "seed": seed,
        },
        "schema": "cliffquant.experiment-001.sample.v1",
        "weight_sha256_encoding": "canonical-array-float64-little-endian",
    }
