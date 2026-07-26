from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from cliffquant.model_snapshot import (
    describe_model_snapshot,
    validate_model_snapshot_descriptor,
)
from cliffquant.provenance import canonical_json_bytes

_REVISION = "a" * 40
_MODEL_ID = "owner/model"


def _write_safetensors(path: Path, keys: tuple[str, ...]) -> None:
    header = {}
    payload = bytearray()
    for offset, key in enumerate(keys):
        header[key] = {
            "data_offsets": [offset, offset + 1],
            "dtype": "U8",
            "shape": [1],
        }
        payload.append(offset % 256)
    encoded = json.dumps(header, separators=(",", ":"), sort_keys=True).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


def _snapshot(tmp_path: Path) -> Path:
    root = tmp_path / "models--owner--model" / "snapshots" / _REVISION
    root.mkdir(parents=True)
    (root / "config.json").write_bytes(
        canonical_json_bytes({"architectures": ["Qwen"], "model_type": "qwen3_5"})
    )
    _write_safetensors(root / "model-00001-of-00001.safetensors", ("a", "b"))
    (root / "model.safetensors.index.json").write_bytes(
        canonical_json_bytes(
            {
                "metadata": {"total_size": 2},
                "weight_map": {
                    "a": "model-00001-of-00001.safetensors",
                    "b": "model-00001-of-00001.safetensors",
                },
            }
        )
    )
    return root


def test_descriptor_binds_config_index_and_every_shard(tmp_path: Path) -> None:
    root = _snapshot(tmp_path)
    descriptor = describe_model_snapshot(
        root,
        model_id=_MODEL_ID,
        revision=_REVISION,
    )
    assert descriptor["tensor_count"] == 2
    assert [item["file"] for item in descriptor["shards"]] == ["model-00001-of-00001.safetensors"]
    assert (
        validate_model_snapshot_descriptor(
            descriptor,
            model_id=_MODEL_ID,
            revision=_REVISION,
        )
        == descriptor
    )
    assert str(tmp_path) not in str(descriptor)


def test_descriptor_rejects_unindexed_safetensors_file(tmp_path: Path) -> None:
    root = _snapshot(tmp_path)
    _write_safetensors(root / "unindexed.safetensors", ("extra",))
    with pytest.raises(ValueError, match="index shard set"):
        describe_model_snapshot(root, model_id=_MODEL_ID, revision=_REVISION)


def test_descriptor_rejects_index_header_disagreement(tmp_path: Path) -> None:
    root = _snapshot(tmp_path)
    index_path = root / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["weight_map"]["missing"] = "model-00001-of-00001.safetensors"
    index_path.write_bytes(canonical_json_bytes(index))
    with pytest.raises(ValueError, match="index/header key mismatch"):
        describe_model_snapshot(root, model_id=_MODEL_ID, revision=_REVISION)


def test_descriptor_rejects_traversal_and_tampered_fingerprint(tmp_path: Path) -> None:
    root = _snapshot(tmp_path)
    index_path = root / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["weight_map"]["a"] = "../outside.safetensors"
    index_path.write_bytes(canonical_json_bytes(index))
    with pytest.raises(
        ValueError,
        match=r"snapshot safetensors files|safe snapshot basename",
    ):
        describe_model_snapshot(root, model_id=_MODEL_ID, revision=_REVISION)

    root = _snapshot(tmp_path / "second")
    descriptor = describe_model_snapshot(root, model_id=_MODEL_ID, revision=_REVISION)
    descriptor["fingerprint_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        validate_model_snapshot_descriptor(
            descriptor,
            model_id=_MODEL_ID,
            revision=_REVISION,
        )
