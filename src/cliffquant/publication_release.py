"""Fail-closed construction and offline validation of Experiment 001 evidence.

The release bundle deliberately excludes model checkpoints and per-module scale
archives.  Those large inputs are validated live before construction; their
path-free identities and the small scale-run manifests are retained in the
curated bundle.
"""

from __future__ import annotations

import errno
import importlib.util
import json
import math
import os
import re
import shutil
import stat
import struct
import sys
import tempfile
import zlib
from collections.abc import Mapping, Sequence
from datetime import datetime
from functools import cache
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import ModuleType
from typing import Any

import numpy as np

from .experiment001_eval import aggregate_group_results
from .experiment001_io import (
    EXPERIMENT_001_ENVIRONMENTS,
    EXPERIMENT_001_GROUP_SIZE,
    EXPERIMENT_001_SAMPLE_SIZE,
    EXPERIMENT_001_SEED,
)
from .fp16_grid import bits_to_scale
from .full_model_artifacts import (
    BASE_MODEL_ID,
    BASE_MODEL_REVISION,
    FROZEN_ENVIRONMENTS,
    POLICY_ABSMAX,
    POLICY_CLIFFQUANT,
    SCALE_RUN_SCHEMA,
    _parse_module,
    load_scale_run,
    validate_corpus_replay_pair,
    validate_frozen_inventory,
)
from .full_model_verify import (
    RUNTIME_VERIFICATION_SCHEMA,
    VERIFICATION_SCHEMA,
    _validate_runtime_report,
    _validate_structural_report,
    describe_exported_checkpoint,
    load_verification_report,
    validate_exported_checkpoint_identity,
)
from .heldout_nll import (
    DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG,
    compare_heldout_nll,
    validate_heldout_nll_result,
)
from .model_snapshot import validate_model_snapshot_descriptor
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
from .solver_certificate import validate_solver_certificate

RELEASE_SCHEMA = "cliffquant.publication-release.v1"
RELEASE_SIDECAR_SCHEMA = "cliffquant.publication-release-sidecar.v1"
FIGURE_MANIFEST_SCHEMA = "cliffquant.figures.v2"
EXPERIMENT_NAME = "experiment-001"

_RELEASE_MANIFEST = "release-manifest.json"
_RELEASE_SIDECAR = "release-manifest.sha256.json"
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_REPORT_LOCATIONS = {
    "absmax": PurePosixPath("verification/absmax-structural/report.json"),
    "cliffquant": PurePosixPath("verification/cliffquant-structural/report.json"),
    "cliffquant_text": PurePosixPath("verification/cliffquant-clean-text/report.json"),
}
_FIGURE_LOCATIONS = {
    "heldout_nll_comparison": PurePosixPath("figures/heldout-nll-comparison.png"),
    "paired_evidence": PurePosixPath("figures/proxy-paired-evidence.png"),
    "policy_worst_environment": PurePosixPath("figures/proxy-policy-comparison.png"),
}
_COLLECTION_FILES = (
    "calibration.manifest.json",
    "calibration.sha256.json",
    "calibration.tokens.npz",
    "heldout.manifest.json",
    "heldout.sha256.json",
    "heldout.tokens.npz",
)
_PROXY_FILES = (
    "SHA256SUMS.json",
    "groups.jsonl",
    "run.inputs.json",
    "sample.manifest.json",
    "summary.json",
)
_NLL_FILES = (
    "heldout-nll.json",
    "heldout-nll.raw.npz",
    "heldout-nll.sha256.json",
)
_STATIC_PAYLOAD_PATHS = frozenset(
    {
        *{f"corpus/{name}" for name in _COLLECTION_FILES},
        *{str(path) for path in _FIGURE_LOCATIONS.values()},
        "figures/figure-manifest.json",
        *{f"nll/{name}" for name in _NLL_FILES},
        *{f"proxy/{name}" for name in _PROXY_FILES},
        "scales/absmax/scale-run.json",
        "scales/cliffquant/scale-run.json",
        "solver/certificate.json",
        *{str(path) for path in _REPORT_LOCATIONS.values()},
        *{str(path.with_name(f"{path.stem}.sha256.json")) for path in _REPORT_LOCATIONS.values()},
    }
)


def _scale_source_binding_sha256(source: Mapping[str, Any]) -> str:
    """Return the scale-run binding for the complete snapshot descriptor."""

    return sha256_bytes(canonical_json_bytes(dict(source)))


def _strict_json_object(payload: bytes, *, label: str) -> dict[str, Any]:
    def object_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"{label} contains duplicate JSON key {key!r}")
            value[key] = item
        return value

    try:
        value = json.loads(
            payload,
            object_pairs_hook=object_hook,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"{label} contains non-finite number {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return value


def _canonical_json_file(path: Path, *, label: str) -> dict[str, Any]:
    payload = path.read_bytes()
    value = _strict_json_object(payload, label=label)
    if payload != canonical_json_bytes(value):
        raise ValueError(f"{label} is not canonical JSON")
    return value


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256_PATTERN.fullmatch(value) is not None


def _is_link_or_reparse(path: Path, mode: int) -> bool:
    attributes = getattr(path.lstat(), "st_file_attributes", 0)
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return stat.S_ISLNK(mode) or bool(attributes & reparse_flag)


def _reject_symlink_components(path: Path, *, label: str) -> None:
    absolute = Path(os.path.abspath(path))
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            break
        if _is_link_or_reparse(current, mode):
            raise ValueError(f"{label} traverses a symlink or reparse point: {current}")


def _regular_file(path: str | Path, *, label: str) -> Path:
    candidate = Path(path)
    _reject_symlink_components(candidate, label=label)
    try:
        mode = candidate.lstat().st_mode
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"{label} is missing: {candidate}") from exc
    if _is_link_or_reparse(candidate, mode) or not stat.S_ISREG(mode):
        raise ValueError(f"{label} must be a regular non-symlink file: {candidate}")
    return candidate.resolve(strict=True)


def _regular_directory(path: str | Path, *, label: str) -> Path:
    candidate = Path(path)
    _reject_symlink_components(candidate, label=label)
    try:
        mode = candidate.lstat().st_mode
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"{label} is missing: {candidate}") from exc
    if _is_link_or_reparse(candidate, mode) or not stat.S_ISDIR(mode):
        raise ValueError(f"{label} must be a regular non-symlink directory: {candidate}")
    return candidate.resolve(strict=True)


def _safe_release_path(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty relative POSIX path")
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    if (
        posix.is_absolute()
        or windows.is_absolute()
        or windows.drive
        or "\\" in value
        or any(part in {"", ".", ".."} for part in posix.parts)
        or str(posix) != value
    ):
        raise ValueError(f"{label} is unsafe: {value!r}")
    return value


def _release_child(root: Path, relative: str, *, label: str) -> Path:
    safe = _safe_release_path(relative, label=label)
    candidate = root.joinpath(*PurePosixPath(safe).parts)
    resolved = candidate.resolve(strict=False)
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"{label} escapes the release root: {relative!r}")
    return candidate


def _descriptor(path: Path, *, relative: str) -> dict[str, Any]:
    return {
        "path": relative,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _validate_descriptor(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256", "size_bytes"}:
        raise ValueError(f"{label} descriptor fields drift")
    path = _safe_release_path(value["path"], label=f"{label} path")
    if (
        not _is_sha256(value["sha256"])
        or type(value["size_bytes"]) is not int
        or value["size_bytes"] <= 0
    ):
        raise ValueError(f"{label} descriptor values drift")
    return {
        "path": path,
        "sha256": value["sha256"],
        "size_bytes": value["size_bytes"],
    }


def _verification_sidecar_payload(report_payload: bytes) -> bytes:
    sidecar = {
        "report": {
            "file": "report.json",
            "sha256": sha256_bytes(report_payload),
            "size_bytes": len(report_payload),
        },
        "schema": "cliffquant.verification-report-sidecar.v1",
    }
    return canonical_json_bytes(sidecar)


def _load_bundled_report(root: Path, relative: PurePosixPath) -> dict[str, Any]:
    report_path = _release_child(root, str(relative), label="verification report")
    sidecar_relative = relative.with_name(f"{relative.stem}.sha256.json")
    sidecar_path = _release_child(
        root,
        str(sidecar_relative),
        label="verification report sidecar",
    )
    report_payload = report_path.read_bytes()
    report = _strict_json_object(report_payload, label=str(relative))
    if report_payload != canonical_json_bytes(report):
        raise ValueError(f"verification report is not canonical JSON: {relative}")
    sidecar_payload = sidecar_path.read_bytes()
    if sidecar_payload != _verification_sidecar_payload(report_payload):
        raise ValueError(f"verification report sidecar mismatch: {sidecar_relative}")
    return report


@cache
def _load_figure_helpers() -> ModuleType:
    repository_root = Path(__file__).resolve().parents[2]
    script = repository_root / "scripts" / "generate_figures.py"
    if not script.is_file():
        raise RuntimeError("publication release requires scripts/generate_figures.py")
    spec = importlib.util.spec_from_file_location(
        "_cliffquant_publication_generate_figures",
        script,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the frozen figure validator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _png_dimensions(path: Path) -> tuple[int, int]:
    if path.stat().st_size > 32 * 1024 * 1024:
        raise ValueError(f"figure PNG exceeds the publication resource limit: {path.name}")
    payload = path.read_bytes()
    if len(payload) < 45 or payload[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"figure is not a complete PNG: {path.name}")

    offset = 8
    ihdr: bytes | None = None
    idat = bytearray()
    saw_idat = False
    ended_idat = False
    saw_iend = False
    while offset < len(payload):
        if offset + 12 > len(payload):
            raise ValueError(f"figure contains a truncated PNG chunk: {path.name}")
        length = struct.unpack(">I", payload[offset : offset + 4])[0]
        chunk_type = payload[offset + 4 : offset + 8]
        chunk_end = offset + 12 + length
        if chunk_end > len(payload):
            raise ValueError(f"figure contains a truncated PNG chunk: {path.name}")
        data = payload[offset + 8 : offset + 8 + length]
        expected_crc = struct.unpack(">I", payload[offset + 8 + length : chunk_end])[0]
        actual_crc = zlib.crc32(data, zlib.crc32(chunk_type)) & 0xFFFFFFFF
        if actual_crc != expected_crc:
            raise ValueError(f"figure contains a PNG checksum mismatch: {path.name}")

        if ihdr is None:
            if chunk_type != b"IHDR" or length != 13:
                raise ValueError(f"figure does not begin with a canonical IHDR: {path.name}")
            ihdr = data
        elif chunk_type == b"IHDR":
            raise ValueError(f"figure contains duplicate PNG IHDR chunks: {path.name}")

        if chunk_type == b"IDAT":
            if ended_idat:
                raise ValueError(f"figure has non-consecutive PNG image data: {path.name}")
            saw_idat = True
            idat.extend(data)
        elif saw_idat and chunk_type != b"IEND":
            ended_idat = True

        if chunk_type == b"IEND":
            if length != 0 or saw_iend or chunk_end != len(payload):
                raise ValueError(f"figure has an invalid or non-terminal PNG IEND: {path.name}")
            saw_iend = True
        offset = chunk_end

    if ihdr is None or not saw_idat or not saw_iend:
        raise ValueError(f"figure is missing required PNG chunks: {path.name}")
    width, height, bit_depth, color_type, compression, filtering, interlace = struct.unpack(
        ">IIBBBBB",
        ihdr,
    )
    if (
        width <= 0
        or height <= 0
        or width > 4096
        or height > 4096
        or (bit_depth, color_type, compression, filtering, interlace) != (8, 6, 0, 0, 0)
    ):
        raise ValueError(f"figure PNG encoding differs from the frozen RGBA format: {path.name}")

    expected_size = height * (1 + 4 * width)
    if expected_size > 64 * 1024 * 1024 or len(idat) > 32 * 1024 * 1024:
        raise ValueError(f"figure PNG exceeds the publication resource limit: {path.name}")
    decompressor = zlib.decompressobj()
    decoded = decompressor.decompress(bytes(idat), expected_size + 1)
    if len(decoded) <= expected_size:
        decoded += decompressor.flush(expected_size + 1 - len(decoded))
    if (
        len(decoded) != expected_size
        or not decompressor.eof
        or decompressor.unused_data
        or decompressor.unconsumed_tail
    ):
        raise ValueError(f"figure PNG pixel stream is truncated or oversized: {path.name}")
    row_size = 1 + 4 * width
    if any(decoded[row * row_size] not in range(5) for row in range(height)):
        raise ValueError(f"figure PNG contains an invalid row filter: {path.name}")
    return width, height


def _model_payload_sha256(identity: Mapping[str, Any]) -> str:
    files = [
        descriptor
        for descriptor in identity["checkpoint_files"]
        if descriptor["file"].endswith(".safetensors")
    ]
    if not files:
        raise ValueError("checkpoint identity contains no safetensors model payload")
    return sha256_bytes(canonical_json_bytes(files))


def _report_source(path: str | Path, *, label: str) -> tuple[Path, Path]:
    report = _regular_file(path, label=label)
    sidecar = _regular_file(
        report.with_name(f"{report.stem}.sha256.json"),
        label=f"{label} sidecar",
    )
    if sidecar.parent != report.parent:
        raise ValueError(f"{label} sidecar escapes its report directory")
    return report, sidecar


def _finite_number(value: Any, *, label: str, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        raise ValueError(f"{label} must be finite")
    return result


def _utc_timestamp(value: Any, *, label: str) -> datetime:
    if (
        not isinstance(value, str)
        or re.fullmatch(
            r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z",
            value,
        )
        is None
    ):
        raise ValueError(f"{label} must be a canonical UTC timestamp")
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{label} is not a real UTC timestamp") from exc


def _validate_proxy_policy_record(
    value: Any,
    *,
    policy: str,
    environments: tuple[str, ...],
) -> None:
    fields = {
        "calibration_loss_by_environment",
        "codes",
        "codes_sha256",
        "heldout_loss_by_environment",
        "heldout_max_loss",
        "runtime_seconds",
        "scale",
        "scale_bits",
        "selection",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError(f"proxy group {policy} policy fields drift")
    scale_bits = value["scale_bits"]
    if type(scale_bits) is not int:
        raise ValueError(f"proxy group {policy} scale_bits must be an integer")
    try:
        scale = bits_to_scale(scale_bits)
    except (IndexError, TypeError, ValueError) as exc:
        raise ValueError(f"proxy group {policy} scale_bits is invalid") from exc
    if value["scale"] != scale:
        raise ValueError(f"proxy group {policy} scale differs from scale_bits")

    codes = value["codes"]
    if (
        not isinstance(codes, list)
        or len(codes) != EXPERIMENT_001_GROUP_SIZE
        or any(type(code) is not int or not -8 <= code <= 7 for code in codes)
        or value["codes_sha256"] != canonical_array_sha256(np.asarray(codes, dtype="<i2"))
    ):
        raise ValueError(f"proxy group {policy} code payload drift")

    losses: dict[str, dict[str, float]] = {}
    for field in ("calibration_loss_by_environment", "heldout_loss_by_environment"):
        raw = value[field]
        if not isinstance(raw, Mapping) or set(raw) != set(environments):
            raise ValueError(f"proxy group {policy} {field} environment order drift")
        losses[field] = {
            environment: _finite_number(
                raw[environment],
                label=f"proxy group {policy} {field}.{environment}",
                nonnegative=True,
            )
            for environment in environments
        }
    heldout_max = _finite_number(
        value["heldout_max_loss"],
        label=f"proxy group {policy} held-out maximum",
        nonnegative=True,
    )
    if heldout_max != max(losses["heldout_loss_by_environment"].values()):
        raise ValueError(f"proxy group {policy} held-out maximum is inconsistent")
    _finite_number(
        value["runtime_seconds"],
        label=f"proxy group {policy} runtime",
        nonnegative=True,
    )

    selection = value["selection"]
    expected_method = {
        "absmax": "range-derived-absmax-nearest-positive-fp16",
        "cliffquant_minimax": "exact-breakpoint-four-environment-minimax",
        "pooled_wmse": "exact-breakpoint-pooled-wmse",
    }[policy]
    if not isinstance(selection, Mapping) or selection.get("method") != expected_method:
        raise ValueError(f"proxy group {policy} selection method drift")
    if policy == "absmax":
        if dict(selection) != {
            "method": expected_method,
            "objective": "none",
            "rounding_tie": "lower-fp16-bit-pattern",
        }:
            raise ValueError("proxy group AbsMax selection provenance drift")
        return
    if set(selection) != {
        "method",
        "objective",
        "solver_provenance",
        "solver_returned_objective",
    } or not isinstance(selection["solver_provenance"], Mapping):
        raise ValueError(f"proxy group {policy} exact-solver provenance drift")
    expected_objective = (
        float(
            np.mean(
                tuple(losses["calibration_loss_by_environment"].values()),
                dtype=np.float64,
            )
        )
        if policy == "pooled_wmse"
        else max(losses["calibration_loss_by_environment"].values())
    )
    if selection["objective"] != expected_objective:
        raise ValueError(f"proxy group {policy} selection objective is inconsistent")
    _finite_number(
        selection["solver_returned_objective"],
        label=f"proxy group {policy} solver objective",
        nonnegative=True,
    )


def _validate_proxy_group(
    value: Any,
    *,
    expected_entry: Mapping[str, Any],
    input_fingerprint: str,
    environments: tuple[str, ...],
) -> dict[str, Any]:
    fields = {
        "identity",
        "input_fingerprint",
        "paired_baseline_minus_cliff",
        "policies",
        "runtime_seconds",
        "schema",
        "weight_sha256",
    }
    if (
        not isinstance(value, dict)
        or set(value) != fields
        or value["schema"] != "cliffquant.experiment-001.group-result.v1"
        or value["input_fingerprint"] != input_fingerprint
    ):
        raise ValueError("proxy group top-level contract drift")
    identity_fields = {
        "branch",
        "group",
        "group_id",
        "input_start",
        "input_stop",
        "layer",
        "module",
        "row",
        "sample_index",
    }
    identity = value["identity"]
    if (
        not isinstance(identity, Mapping)
        or set(identity) != identity_fields
        or dict(identity) != {field: expected_entry[field] for field in identity_fields}
        or value["weight_sha256"] != expected_entry["weight_sha256"]
    ):
        raise ValueError("proxy group identity differs from the frozen sample manifest")
    _finite_number(value["runtime_seconds"], label="proxy group runtime", nonnegative=True)

    policies = value["policies"]
    policy_names = ("absmax", "cliffquant_minimax", "pooled_wmse")
    if not isinstance(policies, Mapping) or set(policies) != set(policy_names):
        raise ValueError("proxy group policy inventory drift")
    for policy in policy_names:
        _validate_proxy_policy_record(
            policies[policy],
            policy=policy,
            environments=environments,
        )
    paired = value["paired_baseline_minus_cliff"]
    if not isinstance(paired, Mapping) or set(paired) != {"absmax", "pooled_wmse"}:
        raise ValueError("proxy group paired-effect inventory drift")
    cliff_loss = policies["cliffquant_minimax"]["heldout_max_loss"]
    for baseline in ("absmax", "pooled_wmse"):
        if paired[baseline] != policies[baseline]["heldout_max_loss"] - cliff_loss:
            raise ValueError("proxy group paired effect is inconsistent")
    return value


def _recompute_proxy_summary(
    root: Path,
    summary: Mapping[str, Any],
    *,
    require_checkpoints: bool,
) -> None:
    manifest_path = root / "sample.manifest.json"
    run_inputs_path = root / "run.inputs.json"
    groups_path = root / "groups.jsonl"
    manifest = _canonical_json_file(manifest_path, label="proxy sample manifest")
    run_inputs = _canonical_json_file(run_inputs_path, label="proxy run inputs")
    if set(manifest) != {
        "entries",
        "group_size",
        "sample_size",
        "sampling",
        "schema",
        "source_fingerprint",
        "weight_sha256_encoding",
    } or manifest != {
        **manifest,
        "group_size": EXPERIMENT_001_GROUP_SIZE,
        "sample_size": EXPERIMENT_001_SAMPLE_SIZE,
        "sampling": {
            "allocation": "one-per-module-then-capacity-proportional-largest-remainder",
            "positions": "sha256-derived-coprime-modular-traversal",
            "seed": EXPERIMENT_001_SEED,
        },
        "schema": "cliffquant.experiment-001.sample.v1",
        "weight_sha256_encoding": "canonical-array-float64-little-endian",
    }:
        raise ValueError("proxy sample manifest contract drift")
    entries = manifest["entries"]
    if not isinstance(entries, list) or len(entries) != EXPERIMENT_001_SAMPLE_SIZE:
        raise ValueError("proxy sample manifest inventory drift")

    if (
        not isinstance(run_inputs, Mapping)
        or run_inputs.get("schema") != "cliffquant.experiment-001.run-inputs.v1"
        or run_inputs.get("sample_size") != EXPERIMENT_001_SAMPLE_SIZE
        or run_inputs.get("seed") != EXPERIMENT_001_SEED
        or run_inputs.get("environments") != list(EXPERIMENT_001_ENVIRONMENTS)
    ):
        raise ValueError("proxy run-input contract drift")
    sample_descriptor = run_inputs.get("sample_manifest")
    expected_sample_descriptor = {
        "file": "sample.manifest.json",
        "sha256": sha256_file(manifest_path),
        "size_bytes": manifest_path.stat().st_size,
    }
    if sample_descriptor != expected_sample_descriptor:
        raise ValueError("proxy run inputs do not bind the sample manifest")
    source = {
        key: value
        for key, value in run_inputs.items()
        if key not in {"input_fingerprint", "sample_manifest", "schema"}
    }
    source_fingerprint = sha256_bytes(canonical_json_bytes(source))
    if manifest["source_fingerprint"] != source_fingerprint:
        raise ValueError("proxy sample manifest source fingerprint is not reproducible")
    input_fingerprint = sha256_bytes(
        canonical_json_bytes(
            {
                "sample_manifest_sha256": expected_sample_descriptor["sha256"],
                "source_fingerprint": source_fingerprint,
            }
        )
    )
    if run_inputs.get("input_fingerprint") != input_fingerprint:
        raise ValueError("proxy run-input fingerprint is not reproducible")

    if groups_path.stat().st_size > 128 * 1024 * 1024:
        raise ValueError("proxy groups JSONL exceeds the publication resource limit")
    results: list[dict[str, Any]] = []
    canonical_lines: list[bytes] = []
    with groups_path.open("rb") as handle:
        for sample_index, line in enumerate(handle):
            if sample_index >= EXPERIMENT_001_SAMPLE_SIZE:
                raise ValueError("proxy groups JSONL contains extra records")
            if len(line) > 128 * 1024 or not line.endswith(b"\n"):
                raise ValueError("proxy groups JSONL contains a non-canonical record")
            record = _strict_json_object(line, label=f"proxy group {sample_index}")
            if line != canonical_json_bytes(record):
                raise ValueError(f"proxy group {sample_index} is not canonical JSON")
            entry = entries[sample_index]
            if (
                not isinstance(entry, Mapping)
                or set(entry)
                != {
                    "branch",
                    "group",
                    "group_id",
                    "input_start",
                    "input_stop",
                    "layer",
                    "module",
                    "row",
                    "sample_index",
                    "weight_sha256",
                }
                or entry["sample_index"] != sample_index
                or not _is_sha256(entry["group_id"])
                or not _is_sha256(entry["weight_sha256"])
                or not isinstance(entry["branch"], str)
                or not isinstance(entry["module"], str)
                or entry["module"]
                != (f"model.language_model.layers.{entry['layer']}.{entry['branch']}")
                or any(
                    type(entry[field]) is not int or entry[field] < 0
                    for field in (
                        "group",
                        "input_start",
                        "input_stop",
                        "layer",
                        "row",
                    )
                )
                or entry["layer"] >= 24
                or entry["input_start"] != entry["group"] * EXPERIMENT_001_GROUP_SIZE
                or entry["input_stop"] != entry["input_start"] + EXPERIMENT_001_GROUP_SIZE
            ):
                raise ValueError("proxy sample manifest entry contract drift")
            identity_without_id = {
                field: entry[field]
                for field in (
                    "branch",
                    "group",
                    "input_start",
                    "input_stop",
                    "layer",
                    "module",
                    "row",
                    "sample_index",
                )
            }
            if entry["group_id"] != sha256_bytes(canonical_json_bytes(identity_without_id)):
                raise ValueError("proxy sample manifest group identity hash drift")
            results.append(
                _validate_proxy_group(
                    record,
                    expected_entry=entry,
                    input_fingerprint=input_fingerprint,
                    environments=EXPERIMENT_001_ENVIRONMENTS,
                )
            )
            canonical_lines.append(line)
    if len(results) != EXPERIMENT_001_SAMPLE_SIZE:
        raise ValueError("proxy groups JSONL inventory drift")
    if (
        len({result["identity"]["group_id"] for result in results}) != EXPERIMENT_001_SAMPLE_SIZE
        or len({result["identity"]["module"] for result in results})
        != QWEN35_08B_EXPECTED_QUANTIZED_MODULES
    ):
        raise ValueError("proxy groups contain duplicate coordinates or incomplete modules")

    if require_checkpoints:
        checkpoint_root = root / "checkpoints"
        expected_names: set[str] = set()
        for record, line in zip(results, canonical_lines, strict=True):
            identity = record["identity"]
            filename = f"{identity['sample_index']:04d}-{identity['group_id'][:16]}.json"
            expected_names.add(filename)
            checkpoint = _regular_file(
                checkpoint_root / filename,
                label="proxy group checkpoint",
            )
            if checkpoint.parent != checkpoint_root or checkpoint.read_bytes() != line:
                raise ValueError("proxy group JSONL differs from its live checkpoint")
        if {child.name for child in checkpoint_root.iterdir()} != expected_names:
            raise ValueError("proxy checkpoint filename inventory drift")

    aggregate = aggregate_group_results(
        results,
        environments=EXPERIMENT_001_ENVIRONMENTS,
    )
    aggregate_fields = set(aggregate)
    if (
        not isinstance(summary, Mapping)
        or set(summary) != aggregate_fields | {"input_fingerprint", "runtime"}
        or summary.get("input_fingerprint") != input_fingerprint
        or {field: summary[field] for field in aggregate_fields} != aggregate
    ):
        raise ValueError("proxy summary differs from recomputed frozen aggregation")
    runtime = summary["runtime"]
    if (
        not isinstance(runtime, Mapping)
        or set(runtime) != {"group_runtime_seconds_sum", "host", "wall_seconds", "workers"}
        or type(runtime["workers"]) is not int
        or runtime["workers"] <= 0
        or not isinstance(runtime["host"], Mapping)
        or runtime["group_runtime_seconds_sum"]
        != float(
            np.sum(
                [result["runtime_seconds"] for result in results],
                dtype=np.float64,
            )
        )
    ):
        raise ValueError("proxy summary runtime aggregation drift")
    _finite_number(runtime["wall_seconds"], label="proxy wall runtime", nonnegative=True)


def _proxy_evidence(
    proxy_dir: str | Path,
    *,
    require_checkpoints: bool,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    root = _regular_directory(proxy_dir, label="finalized proxy directory")
    checksums_path = _regular_file(root / "SHA256SUMS.json", label="proxy SHA256SUMS")
    if checksums_path.parent != root:
        raise ValueError("proxy SHA256SUMS escapes the finalized proxy directory")
    checksums = _canonical_json_file(checksums_path, label="proxy SHA256SUMS")
    if set(checksums) != {"groups", "run_inputs", "sample_manifest", "summary"}:
        raise ValueError("proxy SHA256SUMS file set drift")
    expected_checksum_files = {
        "groups": "groups.jsonl",
        "run_inputs": "run.inputs.json",
        "sample_manifest": "sample.manifest.json",
        "summary": "summary.json",
    }
    expected_files = set(_PROXY_FILES)
    expected_children = expected_files | ({"checkpoints"} if require_checkpoints else set())
    children = list(root.iterdir())
    if {child.name for child in children} != expected_children:
        missing = sorted(expected_children - {child.name for child in children})
        extra = sorted({child.name for child in children} - expected_children)
        raise ValueError(f"proxy directory coverage mismatch; missing={missing}, extra={extra}")
    for name in expected_files:
        child = _regular_file(root / name, label=f"proxy evidence {name}")
        if child.parent != root:
            raise ValueError(f"proxy evidence escapes its directory: {name}")
    for logical_name, filename in expected_checksum_files.items():
        descriptor = checksums[logical_name]
        path = root / filename
        if (
            not isinstance(descriptor, Mapping)
            or set(descriptor) != {"file", "sha256", "size_bytes"}
            or descriptor
            != {
                "file": filename,
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        ):
            raise ValueError(f"proxy SHA256SUMS descriptor drift: {logical_name}")

    figure_helpers = _load_figure_helpers()
    verify_summary = figure_helpers.verify_and_load_summary
    validate_metrics = figure_helpers._validated_metrics
    summary, provenance = verify_summary(root)
    if _canonical_json_file(root / "summary.json", label="proxy summary") != summary:
        raise ValueError("proxy summary changed during checksum verification")
    metrics = validate_metrics(summary)
    if (
        metrics["group_count"] != EXPERIMENT_001_SAMPLE_SIZE
        or metrics["module_count"] != QWEN35_08B_EXPECTED_QUANTIZED_MODULES
    ):
        raise ValueError("proxy summary inventory differs from the frozen experiment")

    if require_checkpoints:
        checkpoints = _regular_directory(root / "checkpoints", label="proxy checkpoints")
        checkpoint_children = list(checkpoints.iterdir())
        if len(checkpoint_children) != metrics["group_count"]:
            raise ValueError("proxy checkpoint coverage differs from the finalized group count")
        for child in checkpoint_children:
            regular = _regular_file(child, label="proxy group checkpoint")
            if regular.parent != checkpoints or regular.suffix != ".json":
                raise ValueError("proxy checkpoint inventory contains an unexpected entry")
    _recompute_proxy_summary(
        root,
        summary,
        require_checkpoints=require_checkpoints,
    )
    return root, summary, provenance


def _nll_figure_provenance(
    nll_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    helpers = _load_figure_helpers()
    verify_nll = helpers.verify_and_load_nll_result
    validate_metrics = helpers._validated_nll_metrics
    report, provenance = verify_nll(nll_root)
    return report, provenance, validate_metrics(report)


def _validate_figure_sources(
    *,
    figure_manifest: str | Path,
    policy_figure: str | Path,
    paired_figure: str | Path,
    nll_figure: str | Path,
    proxy_root: Path,
    nll_root: Path,
) -> tuple[Path, dict[str, Path], dict[str, Any]]:
    manifest_path = _regular_file(figure_manifest, label="figure manifest")
    figures = {
        "policy_worst_environment": _regular_file(
            policy_figure,
            label="proxy policy figure",
        ),
        "paired_evidence": _regular_file(paired_figure, label="proxy paired figure"),
        "heldout_nll_comparison": _regular_file(nll_figure, label="held-out NLL figure"),
    }
    if any(path.parent != manifest_path.parent for path in figures.values()):
        raise ValueError("all figure files must be adjacent to the figure manifest")
    if any(
        path.name != _FIGURE_LOCATIONS[logical_name].name for logical_name, path in figures.items()
    ):
        raise ValueError("figure logical names do not map to the frozen filenames")
    if {child.name for child in manifest_path.parent.iterdir()} != {
        "figure-manifest.json",
        *{path.name for path in _FIGURE_LOCATIONS.values()},
    }:
        raise ValueError("figure directory contains missing or extra entries")

    manifest = _canonical_json_file(manifest_path, label="figure manifest")
    expected_fields = {"data", "figures", "generator", "input", "nll_input", "schema"}
    if set(manifest) != expected_fields or manifest["schema"] != FIGURE_MANIFEST_SCHEMA:
        raise ValueError("figure manifest schema or fields drift")

    proxy_helpers = _load_figure_helpers()
    proxy_summary, proxy_provenance = proxy_helpers.verify_and_load_summary(proxy_root)
    proxy_metrics = proxy_helpers._validated_metrics(proxy_summary)
    nll_report, nll_provenance, nll_metrics = _nll_figure_provenance(nll_root)
    if manifest["input"] != proxy_provenance or manifest["nll_input"] != nll_provenance:
        raise ValueError("figure manifest source hashes do not bind the validated evidence")
    expected_data = {**proxy_metrics, "heldout_nll": nll_metrics}
    if manifest["data"] != expected_data:
        raise ValueError("figure manifest data differs from validated source metrics")
    if (
        nll_report["status"] != "pass"
        or nll_report["gates"]["publishable_model_nll_gate"] is not True
    ):
        raise ValueError("held-out NLL figures require a passing frozen publication gate")

    raw_descriptors = manifest["figures"]
    if not isinstance(raw_descriptors, Mapping) or set(raw_descriptors) != set(figures):
        raise ValueError("figure manifest coverage drift")
    for logical_name, path in figures.items():
        value = raw_descriptors[logical_name]
        if not isinstance(value, Mapping) or set(value) != {
            "file",
            "height_px",
            "sha256",
            "size_bytes",
            "width_px",
        }:
            raise ValueError(f"figure descriptor fields drift: {logical_name}")
        width, height = _png_dimensions(path)
        expected = {
            "file": path.name,
            "height_px": height,
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
            "width_px": width,
        }
        if dict(value) != expected:
            raise ValueError(f"figure descriptor disagrees with live bytes: {logical_name}")

    generator = manifest["generator"]
    repository_script = Path(__file__).resolve().parents[2] / "scripts" / "generate_figures.py"
    if not isinstance(generator, Mapping) or dict(generator) != {
        "matplotlib": "3.10.8",
        "script": "scripts/generate_figures.py",
        "script_sha256": sha256_file(repository_script),
    }:
        raise ValueError("figure generator identity differs from the current frozen source")
    return manifest_path, figures, manifest


def _regenerate_and_compare_figures(
    *,
    supplied_manifest: Mapping[str, Any],
    supplied_figures: Mapping[str, Path],
    proxy_root: Path,
    nll_root: Path,
) -> None:
    helpers = _load_figure_helpers()
    with tempfile.TemporaryDirectory(prefix="cliffquant-figure-rerun-") as temporary:
        output = Path(temporary) / "figures"
        regenerated = helpers.generate_figures(
            proxy_root,
            output,
            nll_input_dir=nll_root,
        )
        _manifest_path, regenerated_files, validated = _validate_figure_sources(
            figure_manifest=output / "figure-manifest.json",
            policy_figure=output / "proxy-policy-comparison.png",
            paired_figure=output / "proxy-paired-evidence.png",
            nll_figure=output / "heldout-nll-comparison.png",
            proxy_root=proxy_root,
            nll_root=nll_root,
        )
        if regenerated != validated or dict(supplied_manifest) != regenerated:
            raise ValueError("freshly regenerated figure manifest differs from supplied evidence")
        for logical_name, supplied in supplied_figures.items():
            if supplied.read_bytes() != regenerated_files[logical_name].read_bytes():
                raise ValueError(f"freshly regenerated figure bytes differ: {logical_name}")


def _collection_sources(
    *,
    calibration_manifest: str | Path,
    calibration_tokens: str | Path,
    calibration_sidecar: str | Path,
    heldout_manifest: str | Path,
    heldout_tokens: str | Path,
    heldout_sidecar: str | Path,
    verified_replay: Mapping[str, Any],
) -> tuple[Path, dict[str, Path], dict[str, Any]]:
    supplied = {
        "calibration.manifest.json": calibration_manifest,
        "calibration.sha256.json": calibration_sidecar,
        "calibration.tokens.npz": calibration_tokens,
        "heldout.manifest.json": heldout_manifest,
        "heldout.sha256.json": heldout_sidecar,
        "heldout.tokens.npz": heldout_tokens,
    }
    files: dict[str, Path] = {}
    for expected_name, path in supplied.items():
        regular = _regular_file(path, label=f"frozen collection file {expected_name}")
        if regular.name != expected_name:
            raise ValueError(f"frozen collection filename must be {expected_name}")
        files[expected_name] = regular
    parents = {path.parent for path in files.values()}
    if len(parents) != 1:
        raise ValueError("frozen collection files must share one directory")
    root = parents.pop()
    replay = validate_corpus_replay_pair(verified_replay)
    for phase in ("calibration", "heldout"):
        identity = replay[phase]
        expected = {
            f"{phase}.manifest.json": identity["manifest"],
            f"{phase}.sha256.json": identity["sidecar"],
            f"{phase}.tokens.npz": identity["token_archive"],
        }
        for name, descriptor in expected.items():
            path = files[name]
            if (
                descriptor["file"] != name
                or descriptor["sha256"] != sha256_file(path)
                or descriptor["size_bytes"] != path.stat().st_size
            ):
                raise ValueError(f"frozen corpus replay descriptor drift: {name}")
    return root, files, replay


def _require_publication_nll_gate(report: Mapping[str, Any]) -> None:
    gates = report.get("gates")
    if (
        report.get("status") != "pass"
        or not isinstance(gates, Mapping)
        or gates.get("publishable_model_nll_gate") is not True
        or not isinstance(gates.get("macro"), Mapping)
        or gates["macro"].get("pass") is not True
        or not isinstance(gates.get("environment"), Mapping)
        or gates["environment"].get("pass") is not True
    ):
        raise ValueError("held-out NLL evidence did not pass every frozen publication gate")


def _validate_live_nll_evidence(
    *,
    result_dir: Path,
    heldout_manifest: str | Path,
    absmax_checkpoint: Path,
    cliffquant_checkpoint: Path,
) -> dict[str, Any]:
    report = validate_heldout_nll_result(
        result_dir,
        heldout_manifest=heldout_manifest,
        absmax_checkpoint=absmax_checkpoint,
        cliffquant_checkpoint=cliffquant_checkpoint,
    )
    _require_publication_nll_gate(report)
    return report


def _require_deterministic_cuda_runtime(
    report: Mapping[str, Any],
    *,
    device: str,
    batch_size: int,
) -> None:
    if re.fullmatch(r"cuda:(0|[1-9][0-9]*)", device) is None:
        raise ValueError("publication NLL rerun requires an explicit CUDA device")
    runtime = report.get("runtime")
    if (
        not isinstance(runtime, Mapping)
        or runtime.get("device") != device
        or runtime.get("batch_size") != batch_size
        or runtime.get("dequantized_weight_cache") is not True
        or runtime.get("teacher_forcing") is not True
    ):
        raise ValueError("publication NLL runtime does not match the requested rerun")
    accelerator = runtime.get("accelerator")
    expected = {
        "cublas_workspace_config": DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG,
        "cudnn_allow_tf32": False,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "deterministic_algorithms": True,
        "deterministic_algorithms_warn_only": False,
        "float32_matmul_precision": "highest",
        "matmul_allow_tf32": False,
    }
    if not isinstance(accelerator, Mapping) or any(
        accelerator.get(field) != value for field, value in expected.items()
    ):
        raise ValueError("publication NLL runtime violates the deterministic CUDA contract")


def _compare_independent_nll_rerun(
    supplied: Mapping[str, Any],
    rerun: Mapping[str, Any],
) -> None:
    semantic_fields = {
        "checkpoints",
        "comparison",
        "corpus_replay",
        "evaluator",
        "gates",
        "heldout",
        "policies",
        "runtime",
        "schema",
        "status",
    }
    if {field: supplied[field] for field in semantic_fields} != {
        field: rerun[field] for field in semantic_fields
    }:
        raise ValueError("independent held-out NLL rerun semantic evidence differs")
    supplied_raw = supplied["raw_evidence"]
    rerun_raw = rerun["raw_evidence"]
    if (
        supplied_raw.get("file") != "heldout-nll.raw.npz"
        or rerun_raw.get("file") != "heldout-nll.raw.npz"
        or supplied_raw.get("arrays") != rerun_raw.get("arrays")
    ):
        raise ValueError("independent held-out NLL rerun raw array identities differ")


def _rerun_and_compare_nll(
    *,
    supplied_report: Mapping[str, Any],
    heldout_manifest: str | Path,
    absmax_checkpoint: Path,
    cliffquant_checkpoint: Path,
    gptqmodel_source: Path,
    device: str,
    batch_size: int,
) -> None:
    _require_deterministic_cuda_runtime(
        supplied_report,
        device=device,
        batch_size=batch_size,
    )
    with tempfile.TemporaryDirectory(prefix="cliffquant-nll-rerun-") as temporary:
        output = Path(temporary) / "result"
        compare_heldout_nll(
            cliffquant_checkpoint=cliffquant_checkpoint,
            absmax_checkpoint=absmax_checkpoint,
            heldout_manifest=heldout_manifest,
            gptqmodel_source=gptqmodel_source,
            output_dir=output,
            device=device,
            batch_size=batch_size,
        )
        rerun = _validate_live_nll_evidence(
            result_dir=output,
            heldout_manifest=heldout_manifest,
            absmax_checkpoint=absmax_checkpoint,
            cliffquant_checkpoint=cliffquant_checkpoint,
        )
        _require_deterministic_cuda_runtime(
            rerun,
            device=device,
            batch_size=batch_size,
        )
        _compare_independent_nll_rerun(supplied_report, rerun)


def _require_checkpoint_pair(
    *,
    absmax_identity: Mapping[str, Any],
    cliffquant_identity: Mapping[str, Any],
) -> None:
    if absmax_identity["policy"] != POLICY_ABSMAX:
        raise ValueError("AbsMax structural report has the wrong policy")
    if cliffquant_identity["policy"] != POLICY_CLIFFQUANT:
        raise ValueError("CliffQuant structural report has the wrong policy")
    if absmax_identity["sha256"] == cliffquant_identity["sha256"]:
        raise ValueError("AbsMax and CliffQuant checkpoint identities must be distinct")
    if _model_payload_sha256(absmax_identity) == _model_payload_sha256(cliffquant_identity):
        raise ValueError("AbsMax and CliffQuant model payload identities must be distinct")
    if (
        absmax_identity["scale_run"]["manifest_sha256"]
        == cliffquant_identity["scale_run"]["manifest_sha256"]
    ):
        raise ValueError("AbsMax and CliffQuant scale identities must be distinct")
    for field in ("base_model", "corpus_replay", "gptq", "gptqmodel"):
        if absmax_identity[field] != cliffquant_identity[field]:
            raise ValueError(f"checkpoint pair differs in shared {field} identity")


def _structural_tensor_identity(report: Mapping[str, Any]) -> dict[str, Any]:
    modules = report.get("modules")
    if not isinstance(modules, list):
        raise ValueError("structural report has no module tensor evidence")
    entries = []
    for module in modules:
        if not isinstance(module, Mapping):
            raise ValueError("structural report module tensor evidence drift")
        entry = {
            "module": module.get("module"),
            "qweight_sha256": module.get("qweight_sha256"),
            "scales_sha256": module.get("scales_sha256"),
        }
        if (
            not isinstance(entry["module"], str)
            or not entry["module"]
            or not _is_sha256(entry["qweight_sha256"])
            or not _is_sha256(entry["scales_sha256"])
        ):
            raise ValueError("structural report quantized tensor identity drift")
        entries.append(entry)
    entries.sort(key=lambda value: value["module"])
    names = [entry["module"] for entry in entries]
    if len(entries) != QWEN35_08B_EXPECTED_QUANTIZED_MODULES or len(names) != len(set(names)):
        raise ValueError("structural report quantized tensor inventory drift")
    return {
        "modules": entries,
        "sha256": sha256_bytes(canonical_json_bytes(entries)),
    }


def _require_distinct_structural_tensors(
    absmax_report: Mapping[str, Any],
    cliffquant_report: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    identities = {
        "absmax": _structural_tensor_identity(absmax_report),
        "cliffquant": _structural_tensor_identity(cliffquant_report),
    }
    abs_entries = identities["absmax"]["modules"]
    cliff_entries = identities["cliffquant"]["modules"]
    if [entry["module"] for entry in abs_entries] != [entry["module"] for entry in cliff_entries]:
        raise ValueError("structural reports quantify different module inventories")
    if all(
        abs_entry["qweight_sha256"] == cliff_entry["qweight_sha256"]
        and abs_entry["scales_sha256"] == cliff_entry["scales_sha256"]
        for abs_entry, cliff_entry in zip(abs_entries, cliff_entries, strict=True)
    ):
        raise ValueError(
            "AbsMax and CliffQuant have identical canonical qweight/scales tensor identities"
        )
    return identities


def _certificate_binding_equal(
    embedded: Any,
    certificate: Mapping[str, Any],
) -> bool:
    if not isinstance(embedded, Mapping):
        return False
    return {key: value for key, value in embedded.items() if key != "file"} == {
        key: value for key, value in certificate.items() if key != "file"
    }


def _validate_scale_calibration_source(
    calibration: Mapping[str, Any],
    *,
    identity: Mapping[str, Any],
    policy: str,
) -> None:
    source = calibration["source"]
    if not isinstance(source, Mapping) or set(source) != {
        "archive",
        "corpus",
        "metadata",
        "model",
        "phase",
        "sidecar_sha256",
    }:
        raise ValueError(f"{policy} scale-run calibration source fields drift")
    for field, expected_file, expected_sha in (
        (
            "archive",
            "calibration.diagonals.npz",
            calibration["archive_sha256"],
        ),
        (
            "metadata",
            "calibration.diagonals.json",
            calibration["metadata_sha256"],
        ),
    ):
        descriptor = source[field]
        if (
            not isinstance(descriptor, Mapping)
            or set(descriptor) != {"file", "sha256", "size_bytes"}
            or descriptor["file"] != expected_file
            or descriptor["sha256"] != expected_sha
            or type(descriptor["size_bytes"]) is not int
            or descriptor["size_bytes"] <= 0
        ):
            raise ValueError(f"{policy} scale-run calibration source {field} descriptor drift")
    replay = identity["corpus_replay"]["calibration"]
    expected_corpus = {
        "archive": replay["token_archive"],
        "manifest": replay["manifest"],
        "phase": "calibration",
        "sidecar_sha256": replay["sidecar"]["sha256"],
    }
    if (
        source["phase"] != "calibration"
        or not _is_sha256(source["sidecar_sha256"])
        or source["model"] != identity["base_model"]["source"]
        or source["corpus"] != expected_corpus
    ):
        raise ValueError(f"{policy} scale-run calibration source identity drift")


def _parse_scale_module(value: Any, *, policy: str) -> Any:
    if not isinstance(value, Mapping) or set(value) != {
        "branch",
        "in_features",
        "layer",
        "name",
        "out_features",
    }:
        raise ValueError(f"{policy} scale module object fields drift")
    return _parse_module(value)


def _validate_scale_manifest_offline(
    metadata: Mapping[str, Any],
    *,
    policy: str,
    identity: Mapping[str, Any],
    certificate: Mapping[str, Any],
    structural_modules: Sequence[str],
) -> None:
    expected_fields = {
        "base_fingerprint",
        "base_model",
        "calibration",
        "corpus_replay",
        "engine",
        "finished_at",
        "gptqmodel_revision",
        "modules",
        "policy",
        "protocol",
        "quantizer",
        "runtime",
        "schema",
        "started_at",
        "status",
    }
    if (
        set(metadata) != expected_fields
        or metadata["schema"] != SCALE_RUN_SCHEMA
        or metadata["status"] != "complete"
        or metadata["policy"] != policy
    ):
        raise ValueError(f"{policy} scale-run manifest schema, fields, or status drift")
    base_model = metadata["base_model"]
    if (
        not isinstance(base_model, Mapping)
        or set(base_model) != {"id", "revision", "source"}
        or base_model["id"] != BASE_MODEL_ID
        or base_model["revision"] != BASE_MODEL_REVISION
    ):
        raise ValueError(f"{policy} scale-run base-model identity drift")
    source = validate_model_snapshot_descriptor(
        base_model["source"],
        model_id=BASE_MODEL_ID,
        revision=BASE_MODEL_REVISION,
    )
    if (
        source != identity["base_model"]["source"]
        or metadata["base_fingerprint"] != _scale_source_binding_sha256(source)
        or metadata["gptqmodel_revision"] != QWEN35_GPTQMODEL_TREE_REVISION
        or metadata["corpus_replay"] != identity["corpus_replay"]
    ):
        raise ValueError(f"{policy} scale-run identity differs from its checkpoint")
    validate_corpus_replay_pair(metadata["corpus_replay"])
    calibration = metadata["calibration"]
    if (
        not isinstance(calibration, Mapping)
        or set(calibration)
        != {
            "archive_file",
            "archive_sha256",
            "environments",
            "metadata_file",
            "metadata_sha256",
            "source",
        }
        or calibration["archive_file"] != "calibration.diagonals.npz"
        or calibration["metadata_file"] != "calibration.diagonals.json"
        or not _is_sha256(calibration["archive_sha256"])
        or not _is_sha256(calibration["metadata_sha256"])
        or calibration["environments"] != list(FROZEN_ENVIRONMENTS)
        or not isinstance(calibration["source"], Mapping)
    ):
        raise ValueError(f"{policy} scale-run calibration contract drift")
    _validate_scale_calibration_source(
        calibration,
        identity=identity,
        policy=policy,
    )
    started_at = _utc_timestamp(
        metadata["started_at"],
        label=f"{policy} scale-run started_at",
    )
    finished_at = _utc_timestamp(
        metadata["finished_at"],
        label=f"{policy} scale-run finished_at",
    )
    if metadata["protocol"] != "research/EXPERIMENT_001_PROTOCOL.md" or finished_at < started_at:
        raise ValueError(f"{policy} scale-run protocol or timestamp drift")

    modules = metadata["modules"]
    if not isinstance(modules, list):
        raise ValueError(f"{policy} scale-run module coverage drift")
    module_specs = []
    for descriptor in modules:
        if not isinstance(descriptor, Mapping) or set(descriptor) != {
            "archive_sha256",
            "metadata_file",
            "metadata_sha256",
            "module",
            "policy",
            "provenance",
        }:
            raise ValueError(f"{policy} scale-run module descriptor fields drift")
        if (
            descriptor["policy"] != policy
            or not _is_sha256(descriptor["archive_sha256"])
            or not _is_sha256(descriptor["metadata_sha256"])
        ):
            raise ValueError(f"{policy} scale-run module descriptor identity drift")
        metadata_file = descriptor["metadata_file"]
        _safe_release_path(metadata_file, label=f"{policy} scale module metadata")
        metadata_parts = PurePosixPath(metadata_file).parts
        if (
            len(metadata_parts) != 3
            or metadata_parts[0] != "modules"
            or metadata_parts[-1] != "module.scales.json"
        ):
            raise ValueError(f"{policy} scale module metadata path drift")
        module = _parse_scale_module(descriptor["module"], policy=policy)
        readable = re.sub(r"[^a-zA-Z0-9._-]+", "-", module.name).strip("-")
        expected_slug = f"{readable}-{sha256_bytes(module.name.encode())[:12]}"
        if metadata_parts[1] != expected_slug:
            raise ValueError(f"{policy} scale module metadata slug drift")
        provenance = descriptor["provenance"]
        expected_provenance_fields = {
            "algorithm",
            "all_zero_exact_tie_shortcuts",
            "candidates_evaluated",
            "exact_solver_groups",
            "exhaustive_fallback_count",
            "fallback_reasons",
            "groups",
            "transition_counts",
        }
        if not isinstance(provenance, Mapping) or set(provenance) != expected_provenance_fields:
            raise ValueError(f"{policy} scale module provenance fields drift")
        groups = module.out_features * module.groups_per_row
        expected_algorithm = (
            "absmax_signed_range_nearest_fp16"
            if policy == POLICY_ABSMAX
            else "solve_breakpoint_exact"
        )
        integer_fields = (
            "all_zero_exact_tie_shortcuts",
            "candidates_evaluated",
            "exact_solver_groups",
            "exhaustive_fallback_count",
            "groups",
        )
        if (
            provenance["algorithm"] != expected_algorithm
            or any(type(provenance[field]) is not int for field in integer_fields)
            or provenance["groups"] != groups
            or provenance["candidates_evaluated"] < groups
            or not 0 <= provenance["all_zero_exact_tie_shortcuts"] <= groups
            or not 0 <= provenance["exhaustive_fallback_count"] <= groups
            or provenance["exact_solver_groups"] != (0 if policy == POLICY_ABSMAX else groups)
            or not isinstance(provenance["fallback_reasons"], Mapping)
            or not isinstance(provenance["transition_counts"], Mapping)
            or any(
                not isinstance(key, str) or type(count) is not int or count < 0
                for counter in (
                    provenance["fallback_reasons"],
                    provenance["transition_counts"],
                )
                for key, count in counter.items()
            )
            or sum(provenance["fallback_reasons"].values())
            != provenance["exhaustive_fallback_count"]
        ):
            raise ValueError(f"{policy} scale module provenance value drift")
        module_specs.append(module)
    validate_frozen_inventory(tuple(module_specs), strict=True)
    module_names = [module.name for module in module_specs]
    if tuple(sorted(module_names)) != tuple(structural_modules):
        raise ValueError(f"{policy} scale modules differ from structural tensor evidence")

    quantizer = metadata["quantizer"]
    expected_selection = (
        "absmax_signed_range_nearest_fp16" if policy == POLICY_ABSMAX else "solve_breakpoint_exact"
    )
    if not isinstance(quantizer, Mapping) or dict(quantizer) != {
        "bits": 4,
        "group_size": 128,
        "qmax": 7,
        "qmin": -8,
        "scale_dtype": "IEEE binary16 positive finite",
        "selection": expected_selection,
    }:
        raise ValueError(f"{policy} scale-run quantizer contract drift")
    engine = metadata["engine"]
    if (
        not isinstance(engine, Mapping)
        or set(engine)
        != {
            "certificate",
            "engine_source_sha256",
            "engine_sources",
            "engine_version",
            "solver_source_sha256",
        }
        or engine["engine_version"] != "cliffquant-full-model-scale-engine-v3"
        or not _is_sha256(engine["engine_source_sha256"])
        or not isinstance(engine["engine_sources"], Mapping)
        or not engine["engine_sources"]
        or any(
            not isinstance(name, str) or not _is_sha256(digest)
            for name, digest in engine["engine_sources"].items()
        )
        or engine["engine_source_sha256"]
        != sha256_bytes(canonical_json_bytes(engine["engine_sources"]))
        or not isinstance(engine["solver_source_sha256"], Mapping)
        or not engine["solver_source_sha256"]
        or any(
            not isinstance(name, str) or not _is_sha256(digest)
            for name, digest in engine["solver_source_sha256"].items()
        )
        or not _certificate_binding_equal(engine.get("certificate"), certificate)
    ):
        raise ValueError(f"{policy} scale-run solver certificate binding drift")
    runtime = metadata["runtime"]
    if (
        not isinstance(runtime, Mapping)
        or set(runtime)
        != {
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
        or runtime["device"] != "cpu"
        or not isinstance(runtime["numpy_build"], Mapping)
        or not isinstance(runtime["packages"], Mapping)
        or any(
            not isinstance(runtime[field], str) or not runtime[field]
            for field in (
                "implementation",
                "machine",
                "numpy",
                "platform",
                "python",
                "python_executable",
            )
        )
    ):
        raise ValueError(f"{policy} scale-run runtime contract drift")


def _validate_live_inputs(
    *,
    absmax_checkpoint: str | Path,
    cliffquant_checkpoint: str | Path,
    absmax_structural_report: str | Path,
    cliffquant_structural_report: str | Path,
    cliffquant_clean_text_report: str | Path,
    nll_result_dir: str | Path,
    proxy_result_dir: str | Path,
    solver_certificate: str | Path,
    calibration_manifest: str | Path,
    calibration_tokens: str | Path,
    calibration_sidecar: str | Path,
    heldout_manifest: str | Path,
    heldout_tokens: str | Path,
    heldout_sidecar: str | Path,
    absmax_scale_run_manifest: str | Path,
    cliffquant_scale_run_manifest: str | Path,
    figure_manifest: str | Path,
    proxy_policy_figure: str | Path,
    proxy_paired_figure: str | Path,
    heldout_nll_figure: str | Path,
    gptqmodel_source: str | Path,
    device: str,
    batch_size: int,
) -> dict[str, Any]:
    if re.fullmatch(r"cuda:(0|[1-9][0-9]*)", device) is None:
        raise ValueError("publication release requires an explicit CUDA device")
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("publication release batch_size must be a positive integer")
    absmax_checkpoint_root = _regular_directory(
        absmax_checkpoint,
        label="live AbsMax checkpoint",
    )
    cliffquant_checkpoint_root = _regular_directory(
        cliffquant_checkpoint,
        label="live CliffQuant checkpoint",
    )
    abs_report_source, _abs_sidecar = _report_source(
        absmax_structural_report,
        label="AbsMax structural report",
    )
    cliff_report_source, _cliff_sidecar = _report_source(
        cliffquant_structural_report,
        label="CliffQuant structural report",
    )
    text_report_source, _text_sidecar = _report_source(
        cliffquant_clean_text_report,
        label="clean CliffQuant text report",
    )

    abs_report = load_verification_report(
        abs_report_source,
        checkpoint_dir=absmax_checkpoint_root,
    )
    cliff_report = load_verification_report(
        cliff_report_source,
        checkpoint_dir=cliffquant_checkpoint_root,
    )
    text_report = load_verification_report(
        text_report_source,
        checkpoint_dir=cliffquant_checkpoint_root,
        require_clean_environment=True,
    )
    if abs_report["schema"] != VERIFICATION_SCHEMA or cliff_report["schema"] != VERIFICATION_SCHEMA:
        raise ValueError("both structural report inputs must be structural verification reports")
    if text_report["schema"] != RUNTIME_VERIFICATION_SCHEMA or text_report.get("kind") != "text":
        raise ValueError("clean CliffQuant runtime evidence must be a text smoke report")

    abs_identity = validate_exported_checkpoint_identity(abs_report["checkpoint_identity"])
    cliff_identity = validate_exported_checkpoint_identity(cliff_report["checkpoint_identity"])
    if text_report["checkpoint_identity"] != cliff_identity:
        raise ValueError("clean CliffQuant text report is not bound to the structural identity")
    _require_checkpoint_pair(
        absmax_identity=abs_identity,
        cliffquant_identity=cliff_identity,
    )
    tensor_identities = _require_distinct_structural_tensors(abs_report, cliff_report)

    nll_root = _regular_directory(nll_result_dir, label="held-out NLL result")
    nll_report = _validate_live_nll_evidence(
        result_dir=nll_root,
        heldout_manifest=heldout_manifest,
        absmax_checkpoint=absmax_checkpoint_root,
        cliffquant_checkpoint=cliffquant_checkpoint_root,
    )
    for role, identity in (("absmax", abs_identity), ("cliffquant", cliff_identity)):
        if nll_report["checkpoints"][role]["identity"] != identity:
            raise ValueError(f"held-out NLL {role} identity differs from structural evidence")
        if nll_report["checkpoints"][role]["model_payload_sha256"] != _model_payload_sha256(
            identity
        ):
            raise ValueError(f"held-out NLL {role} model payload identity drift")
    gptqmodel_root = _regular_directory(gptqmodel_source, label="GPTQModel source")
    _rerun_and_compare_nll(
        supplied_report=nll_report,
        heldout_manifest=heldout_manifest,
        absmax_checkpoint=absmax_checkpoint_root,
        cliffquant_checkpoint=cliffquant_checkpoint_root,
        gptqmodel_source=gptqmodel_root,
        device=device,
        batch_size=batch_size,
    )

    abs_scale_source = _regular_file(
        absmax_scale_run_manifest,
        label="AbsMax scale-run manifest",
    )
    cliff_scale_source = _regular_file(
        cliffquant_scale_run_manifest,
        label="CliffQuant scale-run manifest",
    )
    if abs_scale_source.name != "scale-run.json" or cliff_scale_source.name != "scale-run.json":
        raise ValueError("scale-run manifest inputs must be named scale-run.json")
    abs_scale = load_scale_run(abs_scale_source)
    cliff_scale = load_scale_run(cliff_scale_source)
    if abs_scale.policy != POLICY_ABSMAX or cliff_scale.policy != POLICY_CLIFFQUANT:
        raise ValueError("scale-run policies do not match their explicit roles")
    for scale, identity, role in (
        (abs_scale, abs_identity, "AbsMax"),
        (cliff_scale, cliff_identity, "CliffQuant"),
    ):
        if scale.manifest_sha256 != identity["scale_run"]["manifest_sha256"]:
            raise ValueError(f"{role} checkpoint is not bound to the supplied scale run")
        if scale.metadata["corpus_replay"] != identity["corpus_replay"]:
            raise ValueError(f"{role} scale run corpus identity drift")
        if scale.metadata["base_model"] != identity["base_model"]:
            raise ValueError(f"{role} scale run base-model identity drift")
    if abs_scale.manifest_sha256 == cliff_scale.manifest_sha256:
        raise ValueError("AbsMax and CliffQuant scale-run manifests must be distinct")

    certificate_source = _regular_file(solver_certificate, label="solver certificate")
    certificate = validate_solver_certificate(certificate_source)
    for scale, role in ((abs_scale, "AbsMax"), (cliff_scale, "CliffQuant")):
        if not _certificate_binding_equal(scale.metadata["engine"].get("certificate"), certificate):
            raise ValueError(f"{role} scale run is not bound to the supplied solver certificate")

    collection_root, collection_files, corpus_replay = _collection_sources(
        calibration_manifest=calibration_manifest,
        calibration_tokens=calibration_tokens,
        calibration_sidecar=calibration_sidecar,
        heldout_manifest=heldout_manifest,
        heldout_tokens=heldout_tokens,
        heldout_sidecar=heldout_sidecar,
        verified_replay=nll_report["corpus_replay"],
    )
    if corpus_replay != abs_identity["corpus_replay"]:
        raise ValueError("frozen collection replay differs from checkpoint identities")
    if nll_report["corpus_replay"] != corpus_replay:
        raise ValueError("held-out NLL corpus replay differs from frozen collection")

    proxy_root, proxy_summary, proxy_provenance = _proxy_evidence(
        proxy_result_dir,
        require_checkpoints=True,
    )
    run_inputs = _canonical_json_file(
        proxy_root / "run.inputs.json",
        label="proxy run inputs",
    )
    if not _certificate_binding_equal(run_inputs.get("solver_certificate"), certificate):
        raise ValueError("proxy evidence is not bound to the supplied solver certificate")

    figure_manifest_source, figure_sources, figure_data = _validate_figure_sources(
        figure_manifest=figure_manifest,
        policy_figure=proxy_policy_figure,
        paired_figure=proxy_paired_figure,
        nll_figure=heldout_nll_figure,
        proxy_root=proxy_root,
        nll_root=nll_root,
    )
    _regenerate_and_compare_figures(
        supplied_manifest=figure_data,
        supplied_figures=figure_sources,
        proxy_root=proxy_root,
        nll_root=nll_root,
    )

    sources: dict[str, Path] = {
        **{f"corpus/{name}": path for name, path in collection_files.items()},
        "figures/figure-manifest.json": figure_manifest_source,
        **{
            str(_FIGURE_LOCATIONS[logical_name]): path
            for logical_name, path in figure_sources.items()
        },
        **{f"nll/{name}": nll_root / name for name in _NLL_FILES},
        **{f"proxy/{name}": proxy_root / name for name in _PROXY_FILES},
        "scales/absmax/scale-run.json": abs_scale_source,
        "scales/cliffquant/scale-run.json": cliff_scale_source,
        "solver/certificate.json": certificate_source,
    }
    return {
        "absmax_identity": abs_identity,
        "absmax_checkpoint": absmax_checkpoint_root,
        "certificate": certificate,
        "cliffquant_checkpoint": cliffquant_checkpoint_root,
        "cliffquant_identity": cliff_identity,
        "corpus_replay": corpus_replay,
        "figure_manifest": figure_data,
        "nll_report": nll_report,
        "proxy_provenance": proxy_provenance,
        "proxy_summary": proxy_summary,
        "report_sources": {
            "absmax": abs_report_source,
            "cliffquant": cliff_report_source,
            "cliffquant_text": text_report_source,
        },
        "reports": {
            "absmax": abs_report,
            "cliffquant": cliff_report,
            "cliffquant_text": text_report,
        },
        "input_paths": tuple(
            sorted(
                {
                    absmax_checkpoint_root,
                    cliffquant_checkpoint_root,
                    gptqmodel_root,
                    nll_root,
                    proxy_root,
                    collection_root,
                    abs_scale_source.parent,
                    cliff_scale_source.parent,
                    certificate_source.parent,
                    figure_manifest_source.parent,
                    *(
                        path.parent
                        for path in (
                            abs_report_source,
                            cliff_report_source,
                            text_report_source,
                        )
                    ),
                },
                key=str,
            )
        ),
        "sources": sources,
        "tensor_identities": tensor_identities,
    }


def _validate_bundled_corpus(root: Path, corpus_replay: Mapping[str, Any]) -> None:
    replay = validate_corpus_replay_pair(corpus_replay)
    corpus_root = _release_child(root, "corpus", label="bundled corpus directory")
    for phase in ("calibration", "heldout"):
        manifest_name = f"{phase}.manifest.json"
        tokens_name = f"{phase}.tokens.npz"
        sidecar_name = f"{phase}.sha256.json"
        manifest_path = corpus_root / manifest_name
        tokens_path = corpus_root / tokens_name
        sidecar_path = corpus_root / sidecar_name
        manifest = _canonical_json_file(
            manifest_path,
            label=f"bundled {phase} corpus manifest",
        )
        sidecar = _canonical_json_file(
            sidecar_path,
            label=f"bundled {phase} corpus sidecar",
        )
        expected_sidecar = {
            "manifest": {
                "file": manifest_name,
                "sha256": sha256_file(manifest_path),
                "size_bytes": manifest_path.stat().st_size,
            },
            "tokens": {
                "file": tokens_name,
                "sha256": sha256_file(tokens_path),
                "size_bytes": tokens_path.stat().st_size,
            },
        }
        if sidecar != expected_sidecar:
            raise ValueError(f"bundled {phase} corpus sidecar drift")
        expected_replay = replay[phase]
        if (
            expected_replay["manifest"] != expected_sidecar["manifest"]
            or expected_replay["token_archive"] != expected_sidecar["tokens"]
            or expected_replay["sidecar"]
            != {
                "file": sidecar_name,
                "sha256": sha256_file(sidecar_path),
                "size_bytes": sidecar_path.stat().st_size,
            }
        ):
            raise ValueError(f"bundled {phase} corpus differs from replay identity")
        if (
            manifest.get("schema") != "cliffquant.corpus-manifest.v1"
            or manifest.get("phase") != phase
            or manifest.get("token_archive") != expected_sidecar["tokens"]
        ):
            raise ValueError(f"bundled {phase} corpus manifest contract drift")


def _semantic_bindings(validated: Mapping[str, Any]) -> dict[str, Any]:
    abs_identity = validated["absmax_identity"]
    cliff_identity = validated["cliffquant_identity"]
    text_report = validated["reports"]["cliffquant_text"]
    nll_report = validated["nll_report"]
    proxy_summary = validated["proxy_summary"]
    proxy_provenance = validated["proxy_provenance"]
    certificate = validated["certificate"]
    figure_manifest = validated["figure_manifest"]
    return {
        "checkpoints": {
            "absmax": {
                "checkpoint_identity_sha256": abs_identity["sha256"],
                "model_payload_sha256": _model_payload_sha256(abs_identity),
                "policy": abs_identity["policy"],
                "quantized_tensor_sha256": validated["tensor_identities"]["absmax"]["sha256"],
                "scale_run_sha256": abs_identity["scale_run"]["manifest_sha256"],
            },
            "cliffquant": {
                "checkpoint_identity_sha256": cliff_identity["sha256"],
                "model_payload_sha256": _model_payload_sha256(cliff_identity),
                "policy": cliff_identity["policy"],
                "quantized_tensor_sha256": validated["tensor_identities"]["cliffquant"]["sha256"],
                "scale_run_sha256": cliff_identity["scale_run"]["manifest_sha256"],
            },
        },
        "clean_cliffquant_text": {
            "clean_environment_sha256": text_report["clean_environment"]["sha256"],
            "generated_tokens": text_report["generated_tokens"],
            "sequence_sha256": text_report["sequence_sha256"],
            "status": text_report["status"],
        },
        "corpus_replay_sha256": validated["corpus_replay"]["sha256"],
        "figures": {
            "files": {
                logical_name: descriptor["sha256"]
                for logical_name, descriptor in sorted(figure_manifest["figures"].items())
            },
            "manifest_sha256": sha256_bytes(canonical_json_bytes(figure_manifest)),
        },
        "heldout_nll": {
            "comparison": nll_report["comparison"],
            "gates": nll_report["gates"],
            "raw_sha256": nll_report["raw_evidence"]["sha256"],
            "status": nll_report["status"],
        },
        "proxy": {
            "checksums_manifest_sha256": proxy_provenance["checksums_manifest"]["sha256"],
            "group_count": proxy_summary["group_count"],
            "module_count": proxy_summary["module_count"],
            "proxy_gate_passed": proxy_summary["proxy_gate_passed"],
            "summary_sha256": proxy_provenance["verified_files"]["summary"]["sha256"],
        },
        "solver_certificate": {
            "aggregate_fingerprint_sha256": certificate["aggregate_fingerprint_sha256"],
            "completed_cases": certificate["completed_cases"],
            "protocol_gate_passed": certificate["protocol_gate_passed"],
            "sha256": certificate["sha256"],
        },
    }


def _validate_bundled_semantics(root: Path) -> dict[str, Any]:
    reports = {
        role: _load_bundled_report(root, location) for role, location in _REPORT_LOCATIONS.items()
    }
    abs_report = reports["absmax"]
    cliff_report = reports["cliffquant"]
    text_report = reports["cliffquant_text"]
    if (
        abs_report.get("schema") != VERIFICATION_SCHEMA
        or cliff_report.get("schema") != VERIFICATION_SCHEMA
        or abs_report.get("status") != "pass"
        or cliff_report.get("status") != "pass"
    ):
        raise ValueError("bundled structural report status or schema drift")
    if (
        text_report.get("schema") != RUNTIME_VERIFICATION_SCHEMA
        or text_report.get("status") != "pass"
        or text_report.get("kind") != "text"
    ):
        raise ValueError("bundled clean text report status, schema, or kind drift")

    abs_identity = validate_exported_checkpoint_identity(abs_report["checkpoint_identity"])
    cliff_identity = validate_exported_checkpoint_identity(cliff_report["checkpoint_identity"])
    _require_checkpoint_pair(
        absmax_identity=abs_identity,
        cliffquant_identity=cliff_identity,
    )
    tensor_identities = _require_distinct_structural_tensors(abs_report, cliff_report)
    if text_report["checkpoint_identity"] != cliff_identity:
        raise ValueError("bundled clean text report checkpoint binding drift")
    _validate_structural_report(abs_report, checkpoint=abs_identity)
    _validate_structural_report(cliff_report, checkpoint=cliff_identity)
    _validate_runtime_report(
        text_report,
        checkpoint=cliff_identity,
        require_clean_environment=True,
    )

    nll_root = _release_child(root, "nll", label="bundled held-out NLL directory")
    nll_report = validate_heldout_nll_result(nll_root)
    _require_publication_nll_gate(nll_report)
    for role, identity in (("absmax", abs_identity), ("cliffquant", cliff_identity)):
        if nll_report["checkpoints"][role]["identity"] != identity or nll_report["checkpoints"][
            role
        ]["model_payload_sha256"] != _model_payload_sha256(identity):
            raise ValueError(f"bundled NLL {role} checkpoint binding drift")

    certificate_path = _release_child(
        root,
        "solver/certificate.json",
        label="bundled solver certificate",
    )
    certificate = validate_solver_certificate(certificate_path)
    scale_metadata: dict[str, dict[str, Any]] = {}
    for role, policy, identity in (
        ("absmax", POLICY_ABSMAX, abs_identity),
        ("cliffquant", POLICY_CLIFFQUANT, cliff_identity),
    ):
        relative = f"scales/{role}/scale-run.json"
        path = _release_child(root, relative, label=f"bundled {role} scale-run manifest")
        metadata = _canonical_json_file(path, label=f"bundled {role} scale-run manifest")
        if sha256_file(path) != identity["scale_run"]["manifest_sha256"]:
            raise ValueError(f"bundled {role} scale-run hash differs from checkpoint identity")
        _validate_scale_manifest_offline(
            metadata,
            policy=policy,
            identity=identity,
            certificate=certificate,
            structural_modules=tuple(
                entry["module"] for entry in tensor_identities[role]["modules"]
            ),
        )
        scale_metadata[role] = metadata
    if [item["module"] for item in scale_metadata["absmax"]["modules"]] != [
        item["module"] for item in scale_metadata["cliffquant"]["modules"]
    ]:
        raise ValueError("bundled scale-run module inventories differ")

    corpus_replay = validate_corpus_replay_pair(abs_identity["corpus_replay"])
    if (
        cliff_identity["corpus_replay"] != corpus_replay
        or nll_report["corpus_replay"] != corpus_replay
        or any(metadata["corpus_replay"] != corpus_replay for metadata in scale_metadata.values())
    ):
        raise ValueError("bundled evidence disagrees on frozen corpus replay identity")
    _validate_bundled_corpus(root, corpus_replay)

    proxy_root, proxy_summary, proxy_provenance = _proxy_evidence(
        _release_child(root, "proxy", label="bundled proxy directory"),
        require_checkpoints=False,
    )
    run_inputs = _canonical_json_file(
        proxy_root / "run.inputs.json",
        label="bundled proxy run inputs",
    )
    if not _certificate_binding_equal(run_inputs.get("solver_certificate"), certificate):
        raise ValueError("bundled proxy run is not bound to the solver certificate")

    figure_manifest_path = _release_child(
        root,
        "figures/figure-manifest.json",
        label="bundled figure manifest",
    )
    _figure_manifest_path, _figure_sources, figure_manifest = _validate_figure_sources(
        figure_manifest=figure_manifest_path,
        policy_figure=_release_child(
            root,
            str(_FIGURE_LOCATIONS["policy_worst_environment"]),
            label="bundled proxy policy figure",
        ),
        paired_figure=_release_child(
            root,
            str(_FIGURE_LOCATIONS["paired_evidence"]),
            label="bundled proxy paired figure",
        ),
        nll_figure=_release_child(
            root,
            str(_FIGURE_LOCATIONS["heldout_nll_comparison"]),
            label="bundled held-out NLL figure",
        ),
        proxy_root=proxy_root,
        nll_root=nll_root,
    )
    return {
        "absmax_identity": abs_identity,
        "certificate": certificate,
        "cliffquant_identity": cliff_identity,
        "corpus_replay": corpus_replay,
        "figure_manifest": figure_manifest,
        "nll_report": nll_report,
        "proxy_provenance": proxy_provenance,
        "proxy_summary": proxy_summary,
        "reports": reports,
        "tensor_identities": tensor_identities,
    }


def _require_release_location(path: Path) -> None:
    if (
        path.name != EXPERIMENT_NAME
        or path.parent.name != "results"
        or path.parent.parent.name != "research"
    ):
        raise ValueError("publication destination must end with research/results/experiment-001")


def _expected_release_directories() -> set[str]:
    directories: set[str] = set()
    for relative in {*_STATIC_PAYLOAD_PATHS, _RELEASE_MANIFEST, _RELEASE_SIDECAR}:
        parent = PurePosixPath(relative).parent
        while str(parent) != ".":
            directories.add(str(parent))
            parent = parent.parent
    return directories


def _scan_release_tree(root: Path) -> tuple[set[str], set[str]]:
    files: set[str] = set()
    directories: set[str] = set()
    for child in root.rglob("*"):
        relative = child.relative_to(root).as_posix()
        mode = child.lstat().st_mode
        if _is_link_or_reparse(child, mode):
            raise ValueError(f"release tree contains a symlink or reparse point: {relative}")
        if stat.S_ISREG(mode):
            resolved = child.resolve(strict=True)
            if resolved != child.absolute():
                raise ValueError(f"release file resolves away from its lexical path: {relative}")
            files.add(relative)
        elif stat.S_ISDIR(mode):
            directories.add(relative)
        else:
            raise ValueError(f"release tree contains a non-regular entry: {relative}")
    return files, directories


def _payload_descriptors(root: Path) -> list[dict[str, Any]]:
    descriptors = [
        _descriptor(
            _release_child(root, relative, label="release payload"),
            relative=relative,
        )
        for relative in sorted(_STATIC_PAYLOAD_PATHS)
    ]
    return descriptors


def _validate_release_manifest_and_inventory(
    root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    files, directories = _scan_release_tree(root)
    expected_files = {*_STATIC_PAYLOAD_PATHS, _RELEASE_MANIFEST, _RELEASE_SIDECAR}
    if files != expected_files:
        missing = sorted(expected_files - files)
        extra = sorted(files - expected_files)
        raise ValueError(f"release file coverage mismatch; missing={missing}, extra={extra}")
    expected_directories = _expected_release_directories()
    if directories != expected_directories:
        missing = sorted(expected_directories - directories)
        extra = sorted(directories - expected_directories)
        raise ValueError(f"release directory coverage mismatch; missing={missing}, extra={extra}")

    manifest_path = root / _RELEASE_MANIFEST
    sidecar_path = root / _RELEASE_SIDECAR
    manifest_payload = manifest_path.read_bytes()
    manifest = _strict_json_object(manifest_payload, label="publication release manifest")
    if manifest_payload != canonical_json_bytes(manifest):
        raise ValueError("publication release manifest is not canonical JSON")
    if set(manifest) != {"bindings", "experiment", "files", "schema", "status"}:
        raise ValueError("publication release manifest fields drift")
    if (
        manifest["schema"] != RELEASE_SCHEMA
        or manifest["experiment"] != EXPERIMENT_NAME
        or manifest["status"] != "pass"
    ):
        raise ValueError("publication release manifest schema, experiment, or status drift")

    raw_files = manifest["files"]
    if not isinstance(raw_files, list):
        raise ValueError("publication release file inventory must be a list")
    descriptors = [
        _validate_descriptor(value, label=f"release file {index}")
        for index, value in enumerate(raw_files)
    ]
    paths = [descriptor["path"] for descriptor in descriptors]
    if (
        paths != sorted(paths)
        or len(paths) != len(set(paths))
        or set(paths) != _STATIC_PAYLOAD_PATHS
    ):
        raise ValueError("publication release file inventory is not exact and canonical")
    for descriptor in descriptors:
        path = _release_child(root, descriptor["path"], label="release file")
        if (
            path.stat().st_size != descriptor["size_bytes"]
            or sha256_file(path) != descriptor["sha256"]
        ):
            raise ValueError(f"publication release file hash drift: {descriptor['path']}")

    sidecar = _canonical_json_file(sidecar_path, label="publication release sidecar")
    expected_sidecar = {
        "manifest": {
            "file": _RELEASE_MANIFEST,
            "sha256": sha256_bytes(manifest_payload),
            "size_bytes": len(manifest_payload),
        },
        "schema": RELEASE_SIDECAR_SCHEMA,
    }
    if sidecar != expected_sidecar:
        raise ValueError("publication release sidecar drift")
    return manifest, descriptors


def _validate_release(
    release_dir: str | Path,
    *,
    enforce_location: bool,
) -> dict[str, Any]:
    root = _regular_directory(release_dir, label="publication release")
    if enforce_location:
        _require_release_location(root)
    manifest, descriptors = _validate_release_manifest_and_inventory(root)
    validated = _validate_bundled_semantics(root)
    expected_bindings = _semantic_bindings(validated)
    if manifest["bindings"] != expected_bindings:
        raise ValueError("publication release bindings drift from bundled evidence")
    final_manifest, final_descriptors = _validate_release_manifest_and_inventory(root)
    if final_manifest != manifest or final_descriptors != descriptors:
        raise ValueError("publication release bytes changed during semantic validation")
    return json.loads(canonical_json_bytes(final_manifest))


def validate_publication_release(release_dir: str | Path) -> dict[str, Any]:
    """Validate bundled bytes and internal consistency, not live computation origin."""

    return _validate_release(release_dir, enforce_location=True)


def _copy_regular(source: Path, destination: Path) -> None:
    source = _regular_file(source, label="publication source file")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    if destination.stat().st_size != source.stat().st_size or sha256_file(
        destination
    ) != sha256_file(source):
        raise OSError(f"publication copy verification failed: {source}")


def _copy_report(source: Path, destination: Path) -> None:
    source = _regular_file(source, label="publication verification report")
    payload = source.read_bytes()
    report = _strict_json_object(payload, label=f"verification report {source.name}")
    if payload != canonical_json_bytes(report):
        raise ValueError(f"verification report is not canonical JSON: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(payload)
    destination.with_name(f"{destination.stem}.sha256.json").write_bytes(
        _verification_sidecar_payload(payload)
    )


def _require_absent_path(path: Path, *, label: str) -> None:
    try:
        path.lstat()
    except FileNotFoundError:
        return
    raise FileExistsError(f"refusing to overwrite {label}: {path}")


def _require_disjoint_destination(destination: Path, inputs: Sequence[Path]) -> None:
    destination_text = os.path.normcase(str(destination))
    for source in inputs:
        source_text = os.path.normcase(str(source))
        try:
            common = os.path.commonpath((destination_text, source_text))
        except ValueError:
            continue
        if common in {destination_text, source_text}:
            raise ValueError(f"publication destination overlaps validated input path: {source}")


def _rename_directory_noreplace(source: Path, destination: Path) -> None:
    """Atomically publish a directory without replacing an existing leaf."""

    _require_absent_path(destination, label="publication destination")
    if os.name == "nt":
        os.rename(source, destination)
        return

    if os.name != "posix":
        raise RuntimeError("atomic no-replace publication is unsupported on this platform")

    import ctypes

    libc = ctypes.CDLL(None, use_errno=True)
    at_fdcwd = -100
    if sys.platform.startswith("linux"):
        rename_no_replace = getattr(libc, "renameat2", None)
        flag = 1  # RENAME_NOREPLACE
        missing_message = "this Linux runtime lacks atomic no-replace publication"
    elif sys.platform == "darwin":
        rename_no_replace = getattr(libc, "renameatx_np", None)
        flag = 4  # RENAME_EXCL
        missing_message = "this macOS runtime lacks atomic no-replace publication"
    else:
        raise RuntimeError("atomic no-replace publication is unsupported on this POSIX platform")
    if rename_no_replace is None:
        raise RuntimeError(missing_message)
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
        os.fsencode(source),
        at_fdcwd,
        os.fsencode(destination),
        flag,
    )
    if result == 0:
        return
    error = ctypes.get_errno()
    if error in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(f"refusing to overwrite publication destination: {destination}")
    raise OSError(error, os.strerror(error), str(destination))


def build_publication_release(
    *,
    absmax_checkpoint: str | Path,
    cliffquant_checkpoint: str | Path,
    absmax_structural_report: str | Path,
    cliffquant_structural_report: str | Path,
    cliffquant_clean_text_report: str | Path,
    nll_result_dir: str | Path,
    proxy_result_dir: str | Path,
    solver_certificate: str | Path,
    calibration_manifest: str | Path,
    calibration_tokens: str | Path,
    calibration_sidecar: str | Path,
    heldout_manifest: str | Path,
    heldout_tokens: str | Path,
    heldout_sidecar: str | Path,
    absmax_scale_run_manifest: str | Path,
    cliffquant_scale_run_manifest: str | Path,
    figure_manifest: str | Path,
    proxy_policy_figure: str | Path,
    proxy_paired_figure: str | Path,
    heldout_nll_figure: str | Path,
    gptqmodel_source: str | Path,
    destination: str | Path,
    device: str = "cuda:0",
    batch_size: int = 1,
) -> dict[str, Any]:
    """Validate all live gates, then atomically create the small release bundle."""

    destination_path = Path(os.path.abspath(destination))
    _require_release_location(destination_path)
    _reject_symlink_components(destination_path, label="publication destination")
    _require_absent_path(destination_path, label="publication destination")
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_components(destination_path, label="publication destination")
    _require_absent_path(destination_path, label="publication destination")
    destination_parent = _regular_directory(
        destination_path.parent,
        label="publication destination parent",
    )
    if destination_parent != destination_path.parent:
        raise ValueError("publication destination parent resolves unexpectedly")
    destination_path = destination_parent / destination_path.name

    validated = _validate_live_inputs(
        absmax_checkpoint=absmax_checkpoint,
        cliffquant_checkpoint=cliffquant_checkpoint,
        absmax_structural_report=absmax_structural_report,
        cliffquant_structural_report=cliffquant_structural_report,
        cliffquant_clean_text_report=cliffquant_clean_text_report,
        nll_result_dir=nll_result_dir,
        proxy_result_dir=proxy_result_dir,
        solver_certificate=solver_certificate,
        calibration_manifest=calibration_manifest,
        calibration_tokens=calibration_tokens,
        calibration_sidecar=calibration_sidecar,
        heldout_manifest=heldout_manifest,
        heldout_tokens=heldout_tokens,
        heldout_sidecar=heldout_sidecar,
        absmax_scale_run_manifest=absmax_scale_run_manifest,
        cliffquant_scale_run_manifest=cliffquant_scale_run_manifest,
        figure_manifest=figure_manifest,
        proxy_policy_figure=proxy_policy_figure,
        proxy_paired_figure=proxy_paired_figure,
        heldout_nll_figure=heldout_nll_figure,
        gptqmodel_source=gptqmodel_source,
        device=device,
        batch_size=batch_size,
    )
    _require_disjoint_destination(
        destination_path,
        validated.get("input_paths", ()),
    )
    live_bindings = _semantic_bindings(validated)

    stage = Path(
        tempfile.mkdtemp(
            prefix=f".{EXPERIMENT_NAME}.",
            dir=destination_parent,
        )
    ).resolve(strict=True)
    try:
        for relative, source in sorted(validated["sources"].items()):
            if relative not in _STATIC_PAYLOAD_PATHS:
                raise RuntimeError(f"internal publication source path drift: {relative}")
            _copy_regular(
                source,
                _release_child(stage, relative, label="staged publication file"),
            )
        for role, source in validated["report_sources"].items():
            _copy_report(
                source,
                _release_child(
                    stage,
                    str(_REPORT_LOCATIONS[role]),
                    label="staged verification report",
                ),
            )

        staged_files, staged_directories = _scan_release_tree(stage)
        if staged_files != _STATIC_PAYLOAD_PATHS:
            missing = sorted(_STATIC_PAYLOAD_PATHS - staged_files)
            extra = sorted(staged_files - _STATIC_PAYLOAD_PATHS)
            raise RuntimeError(f"staged payload coverage drift; missing={missing}, extra={extra}")
        if staged_directories != _expected_release_directories():
            raise RuntimeError("staged payload directory coverage drift")

        bundled = _validate_bundled_semantics(stage)
        bundled_bindings = _semantic_bindings(bundled)
        if bundled_bindings != live_bindings:
            raise ValueError("copied publication evidence differs from live validated inputs")
        if (
            describe_exported_checkpoint(validated["absmax_checkpoint"])
            != validated["absmax_identity"]
            or describe_exported_checkpoint(validated["cliffquant_checkpoint"])
            != validated["cliffquant_identity"]
        ):
            raise ValueError("live checkpoint bytes changed during publication construction")
        manifest = {
            "bindings": bundled_bindings,
            "experiment": EXPERIMENT_NAME,
            "files": _payload_descriptors(stage),
            "schema": RELEASE_SCHEMA,
            "status": "pass",
        }
        manifest_payload = canonical_json_bytes(manifest)
        (stage / _RELEASE_MANIFEST).write_bytes(manifest_payload)
        sidecar = {
            "manifest": {
                "file": _RELEASE_MANIFEST,
                "sha256": sha256_bytes(manifest_payload),
                "size_bytes": len(manifest_payload),
            },
            "schema": RELEASE_SIDECAR_SCHEMA,
        }
        (stage / _RELEASE_SIDECAR).write_bytes(canonical_json_bytes(sidecar))
        _validate_release(stage, enforce_location=False)
        _rename_directory_noreplace(stage, destination_path)
    finally:
        if stage.exists():
            if stage.parent != destination_parent:
                raise RuntimeError("refusing to clean an unexpected publication staging path")
            shutil.rmtree(stage)

    return validate_publication_release(destination_path)
