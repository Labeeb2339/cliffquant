"""Explicit compatibility data for the frozen Experiment 001 corpus contract."""

from __future__ import annotations

from typing import Any

# This is the exact collection contract embedded in the released Experiment 001
# calibration and held-out manifests. Version 0.1.1 changes only how the protocol
# file is located after installation; the frozen corpus evidence remains immutable.
EXPERIMENT_001_COLLECTION_CONTRACT: dict[str, Any] = {
    "schema": "cliffquant.collection-contract.v1",
    "sources": {
        "activation_statistics": {
            "file": "activation_stats.py",
            "sha256": "55a5f859f8c36793447a1967377e454350941675f58b014ff51e9dd726c1779f",
            "size_bytes": 14_733,
        },
        "collection_entrypoint": {
            "file": "calibration_cli.py",
            "sha256": "fb5185457453c3293c06e5d76281523de75f06a9e86bfed824295908f952424a",
            "size_bytes": 13_239,
        },
        "corpus_contract": {
            "file": "corpus.py",
            "sha256": "444c7b3ed5013a486915b30d51079ad32749fc15d105263295843281d001d0bf",
            "size_bytes": 31_666,
        },
        "dataset_viewer": {
            "file": "dataset_viewer.py",
            "sha256": "12abdd48977dc81dd207babd8538ac04254bf476b27e8a3ffada9368a94e15a0",
            "size_bytes": 22_621,
        },
        "model_snapshot": {
            "file": "model_snapshot.py",
            "sha256": "9665f986a5c567ee5a573a4876513041c1218dfdacb2c40a0ff8f51d9e47d58d",
            "size_bytes": 10_296,
        },
        "protocol": {
            "file": "EXPERIMENT_001_PROTOCOL.md",
            "sha256": "8fb027b720798401a10f3eeef5501e924de5e619c4f5b152bbdf61079c345cb0",
            "size_bytes": 17_663,
        },
        "provenance": {
            "file": "provenance.py",
            "sha256": "061f5ab960bccc14e2f597d895141397b51dc263762b98b3ced448f8e65e9294",
            "size_bytes": 7_758,
        },
        "qwen35_inventory": {
            "file": "qwen35_modules.py",
            "sha256": "5fd2ed0c90e0ae62baa433bac0f888f7dd1c4c421a8ab3f08cd72c312d5ba86b",
            "size_bytes": 3_283,
        },
    },
}

# Filled with the final v0.1.1 corpus.py identity after the portability change.
# Keeping this outside corpus.py avoids a self-referential file hash.
PORTABLE_CORPUS_SOURCE_SHA256 = "b7783f2472e8dee85c915d37bd2c82cb7be60f4f45ce3e4e0e96cf48b1247ceb"
PORTABLE_CORPUS_SOURCE_SIZE_BYTES = 33_391
