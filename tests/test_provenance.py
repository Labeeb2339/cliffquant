from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from cliffquant.provenance import tokenizer_metadata


class _Tokenizer:
    name_or_path = r"C:\Users\private\.cache\huggingface\snapshot"
    vocab_size = 10
    bos_token_id = None
    eos_token_id = 2
    pad_token_id = None
    unk_token_id = 1
    vocab_files_names: ClassVar[dict[str, str]] = {}


def test_logical_tokenizer_identity_hides_the_local_snapshot_path() -> None:
    metadata = tokenizer_metadata(
        _Tokenizer(),
        model_revision="a" * 40,
        explicit_files={},
        logical_name_or_path="owner/model",
    )

    assert metadata["name_or_path"] == "owner/model"
    assert "Users" not in str(metadata)
    assert str(Path.home()) not in str(metadata)
