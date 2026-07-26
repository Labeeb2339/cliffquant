"""Path-free byte identity for a pinned Hugging Face safetensors snapshot."""

from __future__ import annotations

import json
import os
import re
import struct
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .provenance import canonical_json_bytes, sha256_bytes, sha256_file

MODEL_SNAPSHOT_SCHEMA = "cliffquant.model-snapshot.v1"
_SHA256 = re.compile(r"[0-9a-f]{64}")


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _load_strict_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON number: {token}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read strict {label}: {path.name}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return value


def resolve_snapshot_file(root: Path, relative: str, *, label: str) -> Path:
    """Resolve a snapshot basename, permitting only the matching HF blob store."""

    raw = Path(relative)
    if (
        not relative
        or raw.is_absolute()
        or raw.drive
        or raw.name != relative
        or any(part in {"", ".", ".."} for part in raw.parts)
    ):
        raise ValueError(f"{label} must be a safe snapshot basename")
    lexical = Path(os.path.abspath(root / raw))
    resolved_root = root.resolve()
    if lexical.parent != resolved_root:
        raise ValueError(f"{label} escapes the pinned snapshot")
    if not lexical.is_file():
        raise ValueError(f"missing {label}: {relative}")
    resolved = lexical.resolve()
    if resolved != lexical and resolved_root not in resolved.parents:
        try:
            blob_root = (resolved_root.parents[1] / "blobs").resolve(strict=True)
        except (IndexError, OSError) as exc:
            raise ValueError(f"{label} symlink has no matching HF blob store") from exc
        if resolved != blob_root and blob_root not in resolved.parents:
            raise ValueError(f"{label} symlink escapes the matching HF blob store")
    return lexical


def _file_descriptor(path: Path) -> dict[str, Any]:
    return {
        "file": path.name,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _safetensors_keys(path: Path) -> frozenset[str]:
    try:
        with path.open("rb") as handle:
            prefix = handle.read(8)
            if len(prefix) != 8:
                raise ValueError(f"truncated safetensors prefix: {path.name}")
            header_size = struct.unpack("<Q", prefix)[0]
            if header_size <= 0 or header_size > path.stat().st_size - 8:
                raise ValueError(f"invalid safetensors header length: {path.name}")
            payload = handle.read(header_size)
    except OSError as exc:
        raise ValueError(f"cannot read safetensors header: {path.name}") from exc
    try:
        header = json.loads(
            payload.decode("utf-8").rstrip(" "),
            object_pairs_hook=_strict_object,
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid safetensors header: {path.name}") from exc
    if not isinstance(header, dict):
        raise ValueError(f"safetensors header must be an object: {path.name}")
    keys = frozenset(key for key in header if key != "__metadata__")
    if not keys:
        raise ValueError(f"safetensors shard contains no tensors: {path.name}")
    return keys


def describe_model_snapshot(
    snapshot_dir: str | Path,
    *,
    model_id: str,
    revision: str,
) -> dict[str, Any]:
    """Hash config, index, and every indexed shard of one pinned snapshot."""

    if not isinstance(model_id, str) or not model_id:
        raise ValueError("model_id must be non-empty text")
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("revision must be a lowercase 40-character commit SHA")
    root = Path(snapshot_dir).resolve()
    if not root.is_dir() or root.name != revision:
        raise ValueError("snapshot directory must exist and be named by the pinned revision")

    config_path = resolve_snapshot_file(root, "config.json", label="model config")
    config = _load_strict_json(config_path, label="model config")
    if config.get("model_type") != "qwen3_5":
        raise ValueError("frozen model config type drift")

    index_path = resolve_snapshot_file(
        root,
        "model.safetensors.index.json",
        label="safetensors index",
    )
    index = _load_strict_json(index_path, label="safetensors index")
    if set(index) != {"metadata", "weight_map"}:
        raise ValueError("safetensors index fields drift")
    metadata = index["metadata"]
    if (
        not isinstance(metadata, dict)
        or set(metadata) != {"total_size"}
        or type(metadata["total_size"]) is not int
        or metadata["total_size"] <= 0
    ):
        raise ValueError("safetensors index metadata is invalid")
    weight_map = index["weight_map"]
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("safetensors index weight_map is empty")
    if not all(
        isinstance(key, str) and key and isinstance(value, str) and value.endswith(".safetensors")
        for key, value in weight_map.items()
    ):
        raise ValueError("safetensors weight_map must map tensor names to shard basenames")

    shard_names = sorted(set(weight_map.values()))
    lexical_safetensors = sorted(
        path.name
        for path in root.iterdir()
        if path.is_file() and path.name.endswith(".safetensors")
    )
    if lexical_safetensors != shard_names:
        raise ValueError("snapshot safetensors files disagree with the index shard set")

    shards: list[dict[str, Any]] = []
    observed_keys: set[str] = set()
    for shard_name in shard_names:
        shard = resolve_snapshot_file(root, shard_name, label="safetensors shard")
        actual_keys = _safetensors_keys(shard)
        expected_keys = frozenset(
            tensor_name
            for tensor_name, mapped_shard in weight_map.items()
            if mapped_shard == shard_name
        )
        if actual_keys != expected_keys:
            raise ValueError(f"safetensors index/header key mismatch for {shard_name}")
        if observed_keys & actual_keys:
            raise ValueError("tensor key appears in more than one safetensors shard")
        observed_keys.update(actual_keys)
        shards.append(_file_descriptor(shard))
    if observed_keys != set(weight_map):
        raise ValueError("safetensors index does not cover every shard tensor")

    payload = {
        "config": _file_descriptor(config_path),
        "id": model_id,
        "index": _file_descriptor(index_path),
        "revision": revision,
        "schema": MODEL_SNAPSHOT_SCHEMA,
        "shards": shards,
        "tensor_count": len(weight_map),
    }
    return {
        **payload,
        "fingerprint_sha256": sha256_bytes(canonical_json_bytes(payload)),
    }


def validate_model_snapshot_descriptor(
    value: Any,
    *,
    model_id: str,
    revision: str,
) -> dict[str, Any]:
    """Validate a path-free descriptor without requiring the local snapshot."""

    expected_fields = {
        "config",
        "fingerprint_sha256",
        "id",
        "index",
        "revision",
        "schema",
        "shards",
        "tensor_count",
    }
    if not isinstance(value, Mapping) or set(value) != expected_fields:
        raise ValueError("model snapshot descriptor fields drift")
    if (
        value["schema"] != MODEL_SNAPSHOT_SCHEMA
        or value["id"] != model_id
        or value["revision"] != revision
    ):
        raise ValueError("model snapshot descriptor identity drift")

    def validate_file(descriptor: Any, *, expected_name: str | None = None) -> None:
        if not isinstance(descriptor, Mapping) or set(descriptor) != {
            "file",
            "sha256",
            "size_bytes",
        }:
            raise ValueError("model snapshot file descriptor fields drift")
        name = descriptor["file"]
        if (
            not isinstance(name, str)
            or not name
            or Path(name).name != name
            or (expected_name is not None and name != expected_name)
        ):
            raise ValueError("model snapshot file descriptor name drift")
        if not isinstance(descriptor["sha256"], str) or not _SHA256.fullmatch(descriptor["sha256"]):
            raise ValueError("model snapshot file descriptor hash is invalid")
        if type(descriptor["size_bytes"]) is not int or descriptor["size_bytes"] <= 0:
            raise ValueError("model snapshot file descriptor size is invalid")

    validate_file(value["config"], expected_name="config.json")
    validate_file(value["index"], expected_name="model.safetensors.index.json")
    shards = value["shards"]
    if not isinstance(shards, list) or not shards:
        raise ValueError("model snapshot shard descriptor list is empty")
    for descriptor in shards:
        validate_file(descriptor)
        if not descriptor["file"].endswith(".safetensors"):
            raise ValueError("model snapshot shard filename drift")
    shard_names = [descriptor["file"] for descriptor in shards]
    if shard_names != sorted(set(shard_names)):
        raise ValueError("model snapshot shard descriptors are not unique and sorted")
    if type(value["tensor_count"]) is not int or value["tensor_count"] <= 0:
        raise ValueError("model snapshot tensor count is invalid")
    payload = {key: value[key] for key in expected_fields if key != "fingerprint_sha256"}
    expected_fingerprint = sha256_bytes(canonical_json_bytes(payload))
    if value["fingerprint_sha256"] != expected_fingerprint:
        raise ValueError("model snapshot descriptor fingerprint mismatch")
    return dict(value)
