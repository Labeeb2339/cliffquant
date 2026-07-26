"""Independent structural and runtime verification for CliffQuant GPTQ exports."""

from __future__ import annotations

import json
import re
import stat
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .full_model_artifacts import (
    BASE_MODEL_ID,
    BASE_MODEL_REVISION,
    BITS,
    FROZEN_POLICIES,
    GROUP_SIZE,
    SafetensorsWeightSource,
    ScaleRun,
    load_scale_run,
    resolve_hf_snapshot_file,
    validate_corpus_replay_pair,
    verify_frozen_corpus_pair,
)
from .gptq_pack import (
    INT4_LOGICAL_ZERO,
    dequantize_gptq_int4,
    make_g_idx,
    pack_gptq_v1_qzeros,
    unpack_int4_qweight,
)
from .gptqmodel_export import (
    EXPORT_SCHEMA,
    PRESERVED_AUXILIARY_FILES,
    fp16_values_from_bits,
    load_installed_gptqmodel_contract,
    load_pinned_gptqmodel_contract,
    quantize_rows,
)
from .model_snapshot import validate_model_snapshot_descriptor
from .provenance import (
    canonical_json_bytes,
    runtime_metadata,
    sha256_bytes,
    sha256_file,
)
from .qwen35_modules import (
    QWEN35_08B_EXPECTED_QUANTIZED_MODULES,
    QWEN35_GPTQMODEL_TREE_REVISION,
)

VERIFICATION_SCHEMA = "cliffquant.gptq-verification.v2"
RUNTIME_VERIFICATION_SCHEMA = "cliffquant.gptq-runtime-verification.v1"
CHECKPOINT_IDENTITY_SCHEMA = "cliffquant.gptq-checkpoint-identity.v1"
VERIFICATION_SIDECAR_SCHEMA = "cliffquant.verification-report-sidecar.v1"
VERIFIER_IDENTITY_SCHEMA = "cliffquant.verifier-identity.v1"
CLEAN_ENVIRONMENT_SCHEMA = "cliffquant.clean-verification-environment.v1"
CLEAN_BUILD_SOURCE_SCHEMA = "cliffquant.clean-build-source.v1"
VERIFICATION_LOCK_SHA256 = "7306bc57d022196092ae878fbad343f1c6dc7473ac9e3a461a995e0f64d6f4fe"
HUB_RELEASE_ENVELOPE_FILES = (
    "README.md",
    "LICENSE",
    ".gitattributes",
    "assets/proxy-policy-comparison.png",
    "assets/heldout-nll-comparison.png",
)

_VERIFIER_SOURCES = (
    "full_model_artifacts.py",
    "full_model_verify.py",
    "gptq_pack.py",
    "gptqmodel_export.py",
    "model_snapshot.py",
    "provenance.py",
    "qwen35_modules.py",
)

_RUNTIME_PACKAGES = (
    "accelerate",
    "datasets",
    "gptqmodel",
    "huggingface-hub",
    "numpy",
    "pillow",
    "safetensors",
    "tokenizers",
    "torch",
    "torchao",
    "torchvision",
    "transformers",
)

_AUXILIARY_FILES = PRESERVED_AUXILIARY_FILES

_FROZEN_VERIFICATION_REQUIREMENTS = (
    "accelerate==1.14.0",
    "aiohappyeyeballs==2.7.1",
    "aiohttp==3.14.3",
    "aiosignal==1.4.0",
    "annotated-doc==0.0.4",
    "anyio==4.14.2",
    "attrs==26.1.0",
    "certifi==2026.7.22",
    "charset-normalizer==3.4.9",
    "click==8.4.2",
    "colorama==0.4.6",
    "datasets==5.0.0",
    "Defuser==0.0.24",
    "Device-SMI==0.5.6",
    "dill==0.4.1",
    "filelock==3.32.0",
    "frozenlist==1.8.0",
    "fsspec==2026.4.0",
    "h11==0.16.0",
    "hf-xet==1.5.2",
    "httpcore==1.0.9",
    "httpx==0.28.1",
    "huggingface_hub==1.24.0",
    "idna==3.18",
    "Jinja2==3.1.6",
    "LogBar==0.4.4",
    "markdown-it-py==4.2.0",
    "MarkupSafe==3.0.3",
    "maturin==1.14.1",
    "mdurl==0.1.2",
    "mpmath==1.3.0",
    "multidict==6.7.1",
    "multiprocess==0.70.19",
    "networkx==3.6.1",
    "ninja==1.13.0",
    "numpy==2.2.6",
    "packaging==26.2",
    "pandas==3.0.5",
    "pillow==12.3.0",
    "propcache==0.5.2",
    "protobuf==7.35.1",
    "psutil==7.2.2",
    "pyarrow==25.0.0",
    "Pygments==2.20.0",
    "PyPcre==0.4.0",
    "python-dateutil==2.9.0.post0",
    "PyYAML==6.0.3",
    "regex==2026.7.19",
    "requests==2.34.2",
    "rich==15.0.0",
    "safetensors==0.8.0",
    "shellingham==1.5.4",
    "six==1.17.0",
    "sympy==1.14.0",
    "threadpoolctl==3.6.0",
    "TokeNicer==0.0.14",
    "tokenizers==0.22.2",
    "torch==2.11.0+cu128",
    "torchao==0.17.0",
    "torchvision==0.26.0+cu128",
    "tqdm==4.69.1",
    "transformers==5.14.1",
    "typer==0.27.0",
    "typing_extensions==4.16.0",
    "tzdata==2026.3",
    "urllib3==2.7.0",
    "xxhash==3.8.1",
    "yarl==1.24.5",
)


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _file_descriptor(path: Path, *, logical_name: str | None = None) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"verification evidence file does not exist: {path}")
    return {
        "file": logical_name or path.name,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _validate_file_descriptor(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {"file", "sha256", "size_bytes"}:
        raise ValueError(f"{label} descriptor fields drift")
    name = value["file"]
    relative = Path(name) if isinstance(name, str) else None
    if (
        not isinstance(name, str)
        or not name
        or relative is None
        or relative.is_absolute()
        or relative.drive
        or name != relative.as_posix()
        or not relative.parts
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise ValueError(f"{label} descriptor path drift")
    if not _is_sha256(value["sha256"]):
        raise ValueError(f"{label} descriptor hash drift")
    if type(value["size_bytes"]) is not int or value["size_bytes"] < 0:
        raise ValueError(f"{label} descriptor size drift")
    return dict(value)


def _validate_descriptor_list(value: Any, *, label: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} descriptor list is empty")
    descriptors = [
        _validate_file_descriptor(descriptor, label=f"{label} file") for descriptor in value
    ]
    names = [descriptor["file"] for descriptor in descriptors]
    if names != sorted(names) or len(names) != len(set(names)):
        raise ValueError(f"{label} descriptor list is not unique and canonical")
    return descriptors


def _package_inventory(lines: Any, *, label: str) -> dict[str, str]:
    if not isinstance(lines, (list, tuple)) or not lines:
        raise ValueError(f"{label} package inventory is empty")
    inventory: dict[str, str] = {}
    for item in lines:
        if (
            not isinstance(item, str)
            or item.count("==") != 1
            or " @ " in item
            or item.startswith("-e ")
        ):
            raise ValueError(f"{label} package is not exactly version-pinned: {item!r}")
        name, version = item.split("==", 1)
        normalized = re.sub(r"[-_.]+", "-", name).casefold()
        if not normalized or not version or normalized in inventory:
            raise ValueError(f"{label} package inventory has a duplicate or invalid item: {item!r}")
        inventory[normalized] = version
    return inventory


def _expected_clean_environment_packages() -> dict[str, str]:
    expected = _package_inventory(
        _FROZEN_VERIFICATION_REQUIREMENTS,
        label="frozen verification lock",
    )
    expected.update(
        {
            "cliffquant": "0.1.0",
            "gptqmodel": "7.3.4",
            "pip": "26.1.2",
            "setuptools": "81.0.0",
            "wheel": "0.47.0",
        }
    )
    return expected


def _verifier_identity() -> dict[str, Any]:
    package_root = Path(__file__).resolve().parent
    files = [
        _file_descriptor(package_root / name, logical_name=f"cliffquant/{name}")
        for name in _VERIFIER_SOURCES
    ]
    payload = {
        "files": files,
        "schema": VERIFIER_IDENTITY_SCHEMA,
    }
    return {
        **payload,
        "sha256": sha256_bytes(canonical_json_bytes(payload)),
    }


def _validate_verifier_identity(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {"files", "schema", "sha256"}:
        raise ValueError("verifier identity fields drift")
    if value["schema"] != VERIFIER_IDENTITY_SCHEMA:
        raise ValueError("verifier identity schema drift")
    files = value["files"]
    if not isinstance(files, list) or len(files) != len(_VERIFIER_SOURCES):
        raise ValueError("verifier source coverage drift")
    validated = [
        _validate_file_descriptor(descriptor, label="verifier source") for descriptor in files
    ]
    names = [descriptor["file"] for descriptor in validated]
    expected_names = [f"cliffquant/{name}" for name in _VERIFIER_SOURCES]
    if names != expected_names:
        raise ValueError("verifier source order drift")
    payload = {"files": validated, "schema": VERIFIER_IDENTITY_SCHEMA}
    if value["sha256"] != sha256_bytes(canonical_json_bytes(payload)):
        raise ValueError("verifier source fingerprint mismatch")
    return json.loads(canonical_json_bytes(dict(value)))


def _verification_runtime(device: str) -> dict[str, Any]:
    return runtime_metadata(
        device=device,
        package_names=(
            "accelerate",
            "datasets",
            "gptqmodel",
            "huggingface-hub",
            "numpy",
            "pillow",
            "safetensors",
            "tokenizers",
            "torch",
            "torchao",
            "torchvision",
            "transformers",
        ),
    )


def _sidecar_path(report_path: Path) -> Path:
    return report_path.with_name(f"{report_path.stem}.sha256.json")


def _write_canonical_report(report: Mapping[str, Any], report_path: str | Path) -> None:
    destination = Path(report_path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes(dict(report))
    descriptor = {
        "file": destination.name,
        "sha256": sha256_bytes(payload),
        "size_bytes": len(payload),
    }
    sidecar = {
        "report": descriptor,
        "schema": VERIFICATION_SIDECAR_SCHEMA,
    }
    sidecar_payload = canonical_json_bytes(sidecar)
    sidecar_path = _sidecar_path(destination)
    for path, expected in ((destination, payload), (sidecar_path, sidecar_payload)):
        if path.exists() and path.read_bytes() != expected:
            raise FileExistsError(f"refusing to overwrite different verification evidence: {path}")
        if not path.exists():
            path.write_bytes(expected)


def _load_canonical_report(path: str | Path, *, expected_schema: str) -> dict[str, Any]:
    report_path = Path(path).resolve()
    payload = report_path.read_bytes()
    report = json.loads(payload)
    if not isinstance(report, dict) or report.get("schema") != expected_schema:
        raise ValueError(f"unsupported verification report schema: {report_path}")
    if payload != canonical_json_bytes(report):
        raise ValueError(f"verification report is not canonical JSON: {report_path}")
    sidecar_path = _sidecar_path(report_path)
    sidecar_payload = sidecar_path.read_bytes()
    sidecar = json.loads(sidecar_payload)
    expected_sidecar = {
        "report": {
            "file": report_path.name,
            "sha256": sha256_bytes(payload),
            "size_bytes": len(payload),
        },
        "schema": VERIFICATION_SIDECAR_SCHEMA,
    }
    if sidecar != expected_sidecar or sidecar_payload != canonical_json_bytes(sidecar):
        raise ValueError(f"verification report sidecar mismatch: {sidecar_path}")
    return report


def _is_git_object_id(value: Any) -> bool:
    return (
        isinstance(value, str) and re.fullmatch(r"[0-9a-f]{40}(?:[0-9a-f]{24})?", value) is not None
    )


def validate_clean_build_source_identity(value: Any) -> dict[str, Any]:
    """Validate the complete path-free source and wheel provenance report."""

    fields = {"cliffquant", "dependencies", "gptqmodel", "schema", "sha256"}
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("clean-build source report fields drift")
    report = dict(value)
    if report["schema"] != CLEAN_BUILD_SOURCE_SCHEMA:
        raise ValueError("clean-build source schema drift")

    cliffquant = report["cliffquant"]
    if not isinstance(cliffquant, Mapping) or set(cliffquant) != {
        "commit",
        "tree",
        "wheel",
    }:
        raise ValueError("CliffQuant clean-build source fields drift")
    if not _is_git_object_id(cliffquant["commit"]) or not _is_git_object_id(cliffquant["tree"]):
        raise ValueError("CliffQuant clean-build Git identity drift")
    cliffquant_wheel = _validate_file_descriptor(
        cliffquant["wheel"],
        label="CliffQuant clean-build wheel",
    )
    cliffquant_wheel_name = cliffquant_wheel["file"].casefold()
    if (
        (
            cliffquant_wheel_name != "cliffquant-0.1.0.whl"
            and not cliffquant_wheel_name.startswith("cliffquant-0.1.0-")
        )
        or not cliffquant_wheel_name.endswith(".whl")
        or cliffquant_wheel["size_bytes"] <= 0
    ):
        raise ValueError("CliffQuant clean-build wheel identity drift")

    gptqmodel = report["gptqmodel"]
    if not isinstance(gptqmodel, Mapping) or set(gptqmodel) != {
        "commit",
        "repository",
        "tree",
        "wheel",
    }:
        raise ValueError("GPTQModel clean-build source fields drift")
    if (
        gptqmodel["repository"] != "https://github.com/ModelCloud/GPTQModel.git"
        or gptqmodel["commit"] != QWEN35_GPTQMODEL_TREE_REVISION
        or not _is_git_object_id(gptqmodel["tree"])
    ):
        raise ValueError("GPTQModel clean-build source identity drift")
    gptqmodel_wheel = _validate_file_descriptor(
        gptqmodel["wheel"],
        label="GPTQModel clean-build wheel",
    )
    gptqmodel_wheel_name = gptqmodel_wheel["file"].casefold()
    if (
        (
            gptqmodel_wheel_name != "gptqmodel-7.3.4.whl"
            and not gptqmodel_wheel_name.startswith("gptqmodel-7.3.4-")
        )
        or not gptqmodel_wheel_name.endswith(".whl")
        or gptqmodel_wheel["size_bytes"] <= 0
    ):
        raise ValueError("GPTQModel clean-build wheel identity drift")

    dependencies = report["dependencies"]
    dependency_fields = {"files", "hashed_lock", "mode", "sha256"}
    if not isinstance(dependencies, Mapping) or set(dependencies) != dependency_fields:
        raise ValueError("clean-build dependency identity fields drift")
    if dependencies["mode"] != "hashed-wheelhouse":
        raise ValueError("clean-build dependencies are not a hashed wheelhouse")
    hashed_lock = _validate_file_descriptor(
        dependencies["hashed_lock"],
        label="clean-build hashed lock",
    )
    lock_path = Path(hashed_lock["file"])
    if (
        lock_path.parts[0] != "requirements"
        or lock_path.suffix.casefold() != ".txt"
        or hashed_lock["size_bytes"] <= 0
    ):
        raise ValueError("clean-build hashed-lock identity drift")
    raw_wheelhouse = dependencies["files"]
    if not isinstance(raw_wheelhouse, list) or not raw_wheelhouse:
        raise ValueError("clean-build wheelhouse descriptor list is empty")
    wheelhouse = [
        _validate_file_descriptor(descriptor, label="clean-build wheelhouse file")
        for descriptor in raw_wheelhouse
    ]
    wheel_names = [descriptor["file"] for descriptor in wheelhouse]
    if wheel_names != sorted(wheel_names, key=lambda name: (name.casefold(), name)) or len(
        wheel_names
    ) != len(set(wheel_names)):
        raise ValueError("clean-build wheelhouse descriptor list is not unique and canonical")
    if any(
        not descriptor["file"].casefold().endswith(".whl") or descriptor["size_bytes"] <= 0
        for descriptor in wheelhouse
    ):
        raise ValueError("clean-build wheelhouse descriptor drift")
    dependency_payload = {
        "files": wheelhouse,
        "hashed_lock": hashed_lock,
        "mode": "hashed-wheelhouse",
    }
    if dependencies["sha256"] != sha256_bytes(canonical_json_bytes(dependency_payload)):
        raise ValueError("clean-build wheelhouse aggregate mismatch")

    payload = {key: report[key] for key in fields if key != "sha256"}
    if report["sha256"] != sha256_bytes(canonical_json_bytes(payload)):
        raise ValueError("clean-build source report fingerprint mismatch")
    return json.loads(canonical_json_bytes(report))


def validate_clean_environment_identity(value: Any) -> dict[str, Any]:
    """Validate the path-free contents of one clean-environment report."""

    if not isinstance(value, Mapping):
        raise ValueError("clean-environment report must be an object")
    report = dict(value)
    fields = {
        "cliffquant_wheel",
        "created_from_empty_directory",
        "gptqmodel",
        "gptqmodel_wheel",
        "lock",
        "packages",
        "pip_check",
        "python",
        "schema",
        "sha256",
        "source_build",
        "status",
    }
    if set(report) != fields:
        raise ValueError("clean-environment report fields drift")
    if report["created_from_empty_directory"] is not True or report["status"] != "pass":
        raise ValueError("clean-environment creation did not pass")
    for field in ("cliffquant_wheel", "gptqmodel_wheel", "lock"):
        _validate_file_descriptor(report[field], label=field.replace("_", " "))
    if (
        report["lock"]["file"] != "requirements/verification-cu128.txt"
        or report["lock"]["sha256"] != VERIFICATION_LOCK_SHA256
    ):
        raise ValueError("clean-environment lock identity drift")
    gptqmodel = report["gptqmodel"]
    if (
        not isinstance(gptqmodel, Mapping)
        or set(gptqmodel) != {"repository", "revision"}
        or gptqmodel["repository"] != "https://github.com/ModelCloud/GPTQModel.git"
        or gptqmodel["revision"] != QWEN35_GPTQMODEL_TREE_REVISION
    ):
        raise ValueError("clean-environment GPTQModel source drift")
    source_build = validate_clean_build_source_identity(report["source_build"])
    if (
        report["cliffquant_wheel"] != source_build["cliffquant"]["wheel"]
        or report["gptqmodel_wheel"] != source_build["gptqmodel"]["wheel"]
        or gptqmodel["repository"] != source_build["gptqmodel"]["repository"]
        or gptqmodel["revision"] != source_build["gptqmodel"]["commit"]
    ):
        raise ValueError("clean-environment source-build binding drift")
    packages = report["packages"]
    if not isinstance(packages, list) or packages != sorted(packages, key=str.casefold):
        raise ValueError("clean-environment package lock drift")
    if _package_inventory(packages, label="clean environment") != (
        _expected_clean_environment_packages()
    ):
        raise ValueError("clean-environment package inventory differs from the frozen lock")
    if report["pip_check"] != "No broken requirements found.":
        raise ValueError("clean-environment pip check did not pass")
    python = report["python"]
    if (
        not isinstance(python, Mapping)
        or set(python) != {"implementation", "version"}
        or python["implementation"] != "CPython"
        or python["version"] != "3.11.15"
    ):
        raise ValueError("clean-environment Python runtime drift")
    payload = {key: report[key] for key in fields if key != "sha256"}
    if report["sha256"] != sha256_bytes(canonical_json_bytes(payload)):
        raise ValueError("clean-environment report fingerprint mismatch")
    return json.loads(canonical_json_bytes(report))


def load_clean_environment_identity(path: str | Path) -> dict[str, Any]:
    """Load a canonical clean-environment report and verify its self-identity."""

    report = _load_canonical_report(path, expected_schema=CLEAN_ENVIRONMENT_SCHEMA)
    return validate_clean_environment_identity(report)


def _require_report_outside_checkpoint(
    report_path: str | Path,
    checkpoint_root: Path,
) -> None:
    destination = Path(report_path).resolve()
    if destination == checkpoint_root or checkpoint_root in destination.parents:
        raise ValueError("verification reports must be written outside the immutable checkpoint")


class TensorCheckpoint:
    """Strict local safetensors index with CPU tensor access."""

    def __init__(
        self,
        root: str | Path,
        *,
        allow_hf_blob_links: bool = False,
    ) -> None:
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(f"checkpoint directory does not exist: {self.root}")
        index_candidate = self.root / "model.safetensors.index.json"
        single_candidate = self.root / "model.safetensors"
        if index_candidate.is_file():
            index_path = self._safe_file(
                index_candidate.name,
                label="safetensors index",
                allow_hf_blob_links=allow_hf_blob_links,
            )
            data = json.loads(index_path.read_text(encoding="utf-8"))
            mapping = data.get("weight_map")
            if not isinstance(mapping, dict) or not mapping:
                raise ValueError(f"invalid safetensors weight map: {index_path}")
            if not all(
                isinstance(key, str) and isinstance(value, str) for key, value in mapping.items()
            ):
                raise TypeError("safetensors weight map must map strings to strings")
            self.index_path: Path | None = index_path
            self._weight_map = dict(mapping)
        elif single_candidate.is_file():
            single_path = self._safe_file(
                single_candidate.name,
                label="safetensors checkpoint",
                allow_hf_blob_links=allow_hf_blob_links,
            )
            try:
                from safetensors import safe_open
            except ImportError as exc:
                raise RuntimeError("safetensors is required for checkpoint verification") from exc
            with safe_open(single_path, framework="pt", device="cpu") as handle:
                tensor_keys = handle.keys()
                self._weight_map = {str(key): single_path.name for key in tensor_keys}
            self.index_path = None
        else:
            raise FileNotFoundError(f"checkpoint has no standard model safetensors: {self.root}")
        self.keys = frozenset(self._weight_map)
        self._shards: dict[str, Path] = {}
        for name in sorted(set(self._weight_map.values())):
            candidate = self._safe_file(
                name,
                label="safetensors shard",
                allow_hf_blob_links=allow_hf_blob_links,
            )
            self._shards[name] = candidate

        try:
            from safetensors import safe_open
        except ImportError as exc:
            raise RuntimeError("safetensors is required for checkpoint verification") from exc
        actual_locations: dict[str, list[str]] = {}
        shard_keys: dict[str, frozenset[str]] = {}
        for name, path in self._shards.items():
            with safe_open(path, framework="pt", device="cpu") as handle:
                keys = frozenset(str(key) for key in handle.keys())  # noqa: SIM118
            shard_keys[name] = keys
            for key in keys:
                actual_locations.setdefault(key, []).append(name)
        duplicated = sorted(key for key, names in actual_locations.items() if len(names) != 1)
        if duplicated:
            raise ValueError(f"safetensors keys occur in multiple shards: {duplicated}")
        actual_keys = frozenset(actual_locations)
        if actual_keys != self.keys:
            missing = sorted(self.keys - actual_keys)
            extra = sorted(actual_keys - self.keys)
            raise ValueError(
                f"safetensors index coverage mismatch; missing={missing}, extra={extra}"
            )
        misplaced = sorted(
            key for key, shard_name in self._weight_map.items() if key not in shard_keys[shard_name]
        )
        if misplaced:
            raise ValueError(f"safetensors index maps keys to the wrong shard: {misplaced}")

    def _safe_file(
        self,
        relative: str,
        *,
        label: str,
        allow_hf_blob_links: bool,
    ) -> Path:
        if allow_hf_blob_links:
            return resolve_hf_snapshot_file(self.root, relative, label=label)
        candidate = (self.root / relative).resolve()
        if self.root != candidate and self.root not in candidate.parents:
            raise ValueError(f"{label} escapes checkpoint directory: {relative!r}")
        if not candidate.is_file():
            raise FileNotFoundError(f"missing {label}: {candidate}")
        return candidate

    @property
    def file_hashes(self) -> list[dict[str, Any]]:
        files = [
            {
                "file": path.relative_to(self.root).as_posix(),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for path in sorted(self._shards.values())
        ]
        if self.index_path is not None:
            files.append(
                {
                    "file": self.index_path.name,
                    "sha256": sha256_file(self.index_path),
                    "size_bytes": self.index_path.stat().st_size,
                }
            )
        return sorted(files, key=lambda entry: entry["file"])

    def read(self, key: str) -> Any:
        try:
            from safetensors import safe_open
        except ImportError as exc:
            raise RuntimeError("safetensors is required for checkpoint verification") from exc
        try:
            path = self._shards[self._weight_map[key]]
        except KeyError as exc:
            raise KeyError(f"tensor is absent from checkpoint: {key}") from exc
        with safe_open(path, framework="pt", device="cpu") as handle:
            return handle.get_tensor(key).detach().cpu().contiguous()


def tensor_sha256(tensor: Any) -> str:
    """Hash any torch tensor, including bfloat16, without numeric conversion."""

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for tensor hashing") from exc
    value = tensor.detach().cpu().contiguous()
    raw_view = value.view(torch.uint8)
    raw = raw_view.numpy().tobytes(order="C")
    header = canonical_json_bytes(
        {
            "dtype": str(value.dtype),
            "shape": [int(dimension) for dimension in value.shape],
        }
    )
    return sha256_bytes(header + raw)


def _numpy(tensor: Any, *, float64: bool = False) -> NDArray[np.generic]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for checkpoint verification") from exc
    value = tensor.detach().cpu()
    if float64:
        value = value.to(dtype=torch.float64)
    return np.ascontiguousarray(value.numpy())


def _require_regular_checkpoint_file(
    checkpoint: Path,
    logical_name: str,
    *,
    label: str,
) -> Path:
    candidate = checkpoint / logical_name
    for parent in candidate.parents:
        if parent == checkpoint:
            break
        if parent.is_symlink():
            raise ValueError(
                f"{label} must have regular non-symlink parents inside the checkpoint: "
                f"{logical_name}"
            )
    try:
        mode = candidate.lstat().st_mode
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"{label} is missing: {logical_name}") from exc
    if candidate.is_symlink() or not stat.S_ISREG(mode):
        raise ValueError(
            f"{label} must be a regular non-symlink file inside the checkpoint: {logical_name}"
        )
    resolved = candidate.resolve(strict=True)
    if checkpoint != resolved and checkpoint not in resolved.parents:
        raise ValueError(f"{label} escapes checkpoint: {logical_name}")
    return resolved


def _validate_hub_release_envelope(checkpoint: Path) -> dict[str, Any]:
    for logical_name in HUB_RELEASE_ENVELOPE_FILES:
        try:
            _require_regular_checkpoint_file(
                checkpoint,
                logical_name,
                label="Hub publication envelope file",
            )
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"required Hub publication envelope file is missing: {logical_name}"
            ) from exc
    return {
        "files": list(HUB_RELEASE_ENVELOPE_FILES),
        "identity_scope": "model-payload-only",
        "path_policy": "required-regular-non-symlink-in-root",
        "publication_metadata_bytes_verified": False,
        "status": "path-policy-verified",
    }


def _verify_export_manifest(
    checkpoint: Path,
    *,
    hub_release: bool = False,
) -> dict[str, Any]:
    checkpoint = checkpoint.resolve()
    candidate_path = checkpoint / "cliffquant_export.json"
    if candidate_path.is_symlink():
        raise ValueError("export manifest must be a regular file inside the checkpoint")
    path = candidate_path.resolve(strict=True)
    if path.parent != checkpoint:
        raise ValueError("export manifest escapes checkpoint")
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict) or metadata.get("schema") != EXPORT_SCHEMA:
        raise ValueError("checkpoint lacks a supported CliffQuant export manifest")
    policy = metadata.get("policy")
    if policy not in FROZEN_POLICIES:
        raise ValueError("export manifest is missing an immutable policy tag")
    if metadata.get("base_model", {}).get("id") != BASE_MODEL_ID:
        raise ValueError("export manifest base-model id drift")
    if metadata.get("base_model", {}).get("revision") != BASE_MODEL_REVISION:
        raise ValueError("export manifest base-model revision drift")
    base_source = validate_model_snapshot_descriptor(
        metadata.get("base_model", {}).get("source"),
        model_id=BASE_MODEL_ID,
        revision=BASE_MODEL_REVISION,
    )
    if metadata.get("gptqmodel", {}).get("revision") != QWEN35_GPTQMODEL_TREE_REVISION:
        raise ValueError("export manifest GPTQModel revision drift")
    gptq = metadata.get("gptq")
    if not isinstance(gptq, dict) or {
        "bits": gptq.get("bits"),
        "format": gptq.get("format"),
        "group_size": gptq.get("group_size"),
        "method": gptq.get("method"),
        "sym": gptq.get("sym"),
        "zero": gptq.get("zero"),
    } != {
        "bits": BITS,
        "format": "gptq",
        "group_size": GROUP_SIZE,
        "method": "gptq",
        "sym": True,
        "zero": INT4_LOGICAL_ZERO,
    }:
        raise ValueError("export manifest GPTQ contract drift")
    scale_run = metadata.get("scale_run")
    if not isinstance(scale_run, dict) or not isinstance(scale_run.get("manifest_sha256"), str):
        raise ValueError("export manifest scale-run identity is missing")
    corpus_replay = validate_corpus_replay_pair(metadata.get("corpus_replay"))

    descriptors = metadata.get("files")
    if not isinstance(descriptors, list):
        raise ValueError("export manifest files must be a list")
    described_files: set[str] = set()
    for descriptor in descriptors:
        if not isinstance(descriptor, dict) or not isinstance(descriptor.get("file"), str):
            raise ValueError("invalid export file descriptor")
        if descriptor["file"] in described_files:
            raise ValueError(f"duplicate export file descriptor: {descriptor['file']}")
        described_files.add(descriptor["file"])
        candidate = _require_regular_checkpoint_file(
            checkpoint,
            descriptor["file"],
            label="export manifest payload file",
        )
        if sha256_file(candidate) != descriptor.get("sha256"):
            raise ValueError(f"export file hash mismatch: {candidate}")
        if candidate.stat().st_size != descriptor.get("size_bytes"):
            raise ValueError(f"export file size mismatch: {descriptor['file']}")
    actual_files = {
        candidate.relative_to(checkpoint).as_posix()
        for candidate in checkpoint.rglob("*")
        if (candidate.is_file() or candidate.is_symlink()) and candidate.name != path.name
    }
    if hub_release:
        _validate_hub_release_envelope(checkpoint)
        actual_payload_files = actual_files - set(HUB_RELEASE_ENVELOPE_FILES)
    else:
        actual_payload_files = actual_files
    if described_files != actual_payload_files:
        missing = sorted(described_files - actual_payload_files)
        extra = sorted(actual_payload_files - described_files)
        raise ValueError(f"export file coverage mismatch; missing={missing}, extra={extra}")

    quantize_config_path = checkpoint / "quantize_config.json"
    quantize_config = json.loads(quantize_config_path.read_text(encoding="utf-8"))
    if not isinstance(quantize_config, dict):
        raise ValueError("quantize_config must contain a JSON object")
    method = str(quantize_config.get("quant_method", quantize_config.get("method", ""))).lower()
    format_name = str(
        quantize_config.get("format", quantize_config.get("checkpoint_format", ""))
    ).lower()
    if method != "gptq" or format_name != "gptq":
        raise ValueError("quantize_config is not standard GPTQ-v1")
    meta = quantize_config.get("meta")
    embedded = meta.get("cliffquant") if isinstance(meta, dict) else None
    if not isinstance(embedded, dict):
        raise ValueError("quantize_config has no embedded CliffQuant identity")
    if (
        embedded.get("policy") != policy
        or embedded.get("scale_run_sha256") != scale_run["manifest_sha256"]
        or embedded.get("base_snapshot_sha256") != base_source["fingerprint_sha256"]
        or embedded.get("corpus_replay_sha256") != corpus_replay["sha256"]
        or embedded.get("base_model") != BASE_MODEL_ID
        or embedded.get("base_revision") != BASE_MODEL_REVISION
        or embedded.get("gptqmodel_revision") != QWEN35_GPTQMODEL_TREE_REVISION
    ):
        raise ValueError("quantize_config embedded CliffQuant identity drift")
    return metadata


def _checkpoint_identity(
    checkpoint: Path,
    export_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    base_model = export_manifest["base_model"]
    base_source = validate_model_snapshot_descriptor(
        base_model["source"],
        model_id=BASE_MODEL_ID,
        revision=BASE_MODEL_REVISION,
    )
    corpus_replay = validate_corpus_replay_pair(export_manifest["corpus_replay"])
    export_descriptor = _file_descriptor(
        checkpoint / "cliffquant_export.json",
        logical_name="cliffquant_export.json",
    )
    described_files = [
        _validate_file_descriptor(value, label="checkpoint file")
        for value in export_manifest["files"]
    ]
    described_files.sort(key=lambda value: value["file"])
    all_files = [export_descriptor, *described_files]
    scale_run = export_manifest["scale_run"]
    payload = {
        "base_model": {
            "id": BASE_MODEL_ID,
            "revision": BASE_MODEL_REVISION,
            "source": base_source,
        },
        "checkpoint_files": all_files,
        "checkpoint_files_sha256": sha256_bytes(canonical_json_bytes(all_files)),
        "corpus_replay": corpus_replay,
        "export_manifest": export_descriptor,
        "gptq": dict(export_manifest["gptq"]),
        "gptqmodel": dict(export_manifest["gptqmodel"]),
        "policy": export_manifest["policy"],
        "scale_run": {
            "manifest_sha256": scale_run["manifest_sha256"],
            "path": scale_run["path"],
        },
        "schema": CHECKPOINT_IDENTITY_SCHEMA,
    }
    return {
        **payload,
        "sha256": sha256_bytes(canonical_json_bytes(payload)),
    }


def validate_exported_checkpoint_identity(value: Any) -> dict[str, Any]:
    """Validate a path-free identity for every byte in one exported checkpoint."""

    fields = {
        "base_model",
        "checkpoint_files",
        "checkpoint_files_sha256",
        "corpus_replay",
        "export_manifest",
        "gptq",
        "gptqmodel",
        "policy",
        "scale_run",
        "schema",
        "sha256",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("exported checkpoint identity fields drift")
    if value["schema"] != CHECKPOINT_IDENTITY_SCHEMA:
        raise ValueError("exported checkpoint identity schema drift")
    base_model = value["base_model"]
    if (
        not isinstance(base_model, Mapping)
        or set(base_model) != {"id", "revision", "source"}
        or base_model["id"] != BASE_MODEL_ID
        or base_model["revision"] != BASE_MODEL_REVISION
    ):
        raise ValueError("exported checkpoint base-model identity drift")
    validate_model_snapshot_descriptor(
        base_model["source"],
        model_id=BASE_MODEL_ID,
        revision=BASE_MODEL_REVISION,
    )
    validate_corpus_replay_pair(value["corpus_replay"])
    export_manifest = _validate_file_descriptor(
        value["export_manifest"],
        label="export manifest",
    )
    if export_manifest["file"] != "cliffquant_export.json":
        raise ValueError("export manifest filename drift")
    files = value["checkpoint_files"]
    if not isinstance(files, list) or not files:
        raise ValueError("exported checkpoint file list is empty")
    validated_files = [
        _validate_file_descriptor(descriptor, label="checkpoint file") for descriptor in files
    ]
    names = [descriptor["file"] for descriptor in validated_files]
    if names != ["cliffquant_export.json", *sorted(names[1:])] or len(names) != len(set(names)):
        raise ValueError("exported checkpoint files are not unique and canonical")
    if validated_files[0] != export_manifest:
        raise ValueError("export manifest descriptor disagrees with checkpoint file list")
    if value["checkpoint_files_sha256"] != sha256_bytes(canonical_json_bytes(validated_files)):
        raise ValueError("exported checkpoint file aggregate mismatch")
    if value["policy"] not in FROZEN_POLICIES:
        raise ValueError("exported checkpoint policy drift")
    gptq = value["gptq"]
    if not isinstance(gptq, Mapping) or dict(gptq) != {
        "backend": "GPTQ_TORCH",
        "bits": BITS,
        "desc_act": False,
        "format": "gptq",
        "group_size": GROUP_SIZE,
        "method": "gptq",
        "pack_dtype": "int32",
        "sym": True,
        "zero": INT4_LOGICAL_ZERO,
    }:
        raise ValueError("exported checkpoint GPTQ contract drift")
    gptqmodel = value["gptqmodel"]
    if (
        not isinstance(gptqmodel, Mapping)
        or gptqmodel.get("revision") != QWEN35_GPTQMODEL_TREE_REVISION
        or not isinstance(gptqmodel.get("version"), str)
        or not gptqmodel["version"]
    ):
        raise ValueError("exported checkpoint GPTQModel identity drift")
    scale_run = value["scale_run"]
    if (
        not isinstance(scale_run, Mapping)
        or set(scale_run) != {"manifest_sha256", "path"}
        or not _is_sha256(scale_run["manifest_sha256"])
        or not isinstance(scale_run["path"], str)
        or Path(scale_run["path"]).name != scale_run["path"]
    ):
        raise ValueError("exported checkpoint scale-run identity drift")
    payload = {key: value[key] for key in fields if key != "sha256"}
    if value["sha256"] != sha256_bytes(canonical_json_bytes(payload)):
        raise ValueError("exported checkpoint identity fingerprint mismatch")
    return json.loads(canonical_json_bytes(dict(value)))


def describe_exported_checkpoint(
    checkpoint_dir: str | Path,
    *,
    hub_release: bool = False,
) -> dict[str, Any]:
    """Re-hash the immutable payload of a strict or explicitly staged Hub export."""

    checkpoint = Path(checkpoint_dir).resolve()
    export_manifest = _verify_export_manifest(checkpoint, hub_release=hub_release)
    identity = _checkpoint_identity(checkpoint, export_manifest)
    return validate_exported_checkpoint_identity(identity)


def _validate_runtime_identity(value: Any, *, device: str) -> dict[str, Any]:
    base_fields = {
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
    fields = base_fields | ({"accelerator"} if device.startswith("cuda") else set())
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("verification runtime fields drift")
    runtime = dict(value)
    if runtime["device"] != device:
        raise ValueError("verification runtime device drift")
    for key in ("implementation", "numpy", "platform", "python", "python_executable"):
        if not isinstance(runtime[key], str) or not runtime[key]:
            raise ValueError(f"verification runtime {key} drift")
    if not isinstance(runtime["machine"], str) or not isinstance(runtime["numpy_build"], Mapping):
        raise ValueError("verification runtime platform metadata drift")
    executable = Path(runtime["python_executable"])
    if executable.is_absolute() or executable.name != runtime["python_executable"]:
        raise ValueError("verification runtime executable path drift")
    packages = runtime["packages"]
    if (
        not isinstance(packages, Mapping)
        or set(packages) != set(_RUNTIME_PACKAGES)
        or not all(
            value is None or (isinstance(value, str) and value) for value in packages.values()
        )
    ):
        raise ValueError("verification runtime package identity drift")
    if runtime["numpy"] != packages["numpy"]:
        raise ValueError("verification runtime NumPy identity drift")
    if device.startswith("cuda"):
        accelerator = runtime["accelerator"]
        base_accelerator_fields = {
            "available",
            "cuda_runtime",
            "deterministic_algorithms",
            "deterministic_algorithms_warn_only",
        }
        if not isinstance(accelerator, Mapping):
            raise ValueError("verification runtime accelerator drift")
        available = accelerator.get("available")
        expected_accelerator_fields = base_accelerator_fields
        if available is True:
            expected_accelerator_fields |= {
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
        if (
            type(available) is not bool
            or set(accelerator) != expected_accelerator_fields
            or type(accelerator["deterministic_algorithms"]) is not bool
            or type(accelerator["deterministic_algorithms_warn_only"]) is not bool
        ):
            raise ValueError("verification runtime accelerator fields drift")
    return json.loads(canonical_json_bytes(runtime))


def _validate_current_verifier(value: Any) -> dict[str, Any]:
    verifier = _validate_verifier_identity(value)
    if verifier != _verifier_identity():
        raise ValueError("verification report was produced by different verifier source bytes")
    return verifier


def _validate_clean_runtime_binding(
    clean_environment: Mapping[str, Any],
    runtime: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
) -> None:
    clean_packages = _package_inventory(
        clean_environment["packages"],
        label="clean environment",
    )
    runtime_packages = runtime["packages"]
    for name, version in runtime_packages.items():
        normalized = re.sub(r"[-_.]+", "-", name).casefold()
        if version != clean_packages.get(normalized):
            raise ValueError(f"runtime package differs from clean environment: {name}")
    if (
        runtime["implementation"] != clean_environment["python"]["implementation"]
        or runtime["python"] != clean_environment["python"]["version"]
    ):
        raise ValueError("runtime Python differs from clean environment")
    if (
        checkpoint["gptqmodel"]["revision"] != clean_environment["gptqmodel"]["revision"]
        or checkpoint["gptqmodel"]["version"] != clean_packages["gptqmodel"]
    ):
        raise ValueError("checkpoint GPTQModel identity differs from clean environment")


def _validate_structural_report(
    report: Mapping[str, Any],
    *,
    checkpoint: Mapping[str, Any],
) -> None:
    fields = {
        "auxiliary_files",
        "base_checkpoint_files",
        "checkpoint_files",
        "checkpoint_identity",
        "corpus_replay",
        "export_manifest_sha256",
        "modules",
        "policy",
        "preserved_non_target_tensor_count",
        "preserved_non_target_tensors",
        "runtime",
        "scale_run_sha256",
        "schema",
        "status",
        "summary",
        "verifier",
    }
    if set(report) != fields:
        raise ValueError("structural verification report fields drift")
    if (
        report["policy"] != checkpoint["policy"]
        or report["export_manifest_sha256"] != checkpoint["export_manifest"]["sha256"]
        or report["scale_run_sha256"] != checkpoint["scale_run"]["manifest_sha256"]
        or report["corpus_replay"] != checkpoint["corpus_replay"]
    ):
        raise ValueError("structural report checkpoint binding drift")
    validate_corpus_replay_pair(report["corpus_replay"])
    _validate_current_verifier(report["verifier"])
    _validate_runtime_identity(report["runtime"], device="cpu")

    base_files = _validate_descriptor_list(
        report["base_checkpoint_files"],
        label="base checkpoint",
    )
    checkpoint_files = _validate_descriptor_list(
        report["checkpoint_files"],
        label="packed checkpoint",
    )
    del base_files
    live_files = {item["file"]: item for item in checkpoint["checkpoint_files"]}
    if any(live_files.get(item["file"]) != item for item in checkpoint_files):
        raise ValueError("structural checkpoint tensor-file identity drift")

    auxiliary = report["auxiliary_files"]
    if not isinstance(auxiliary, list) or not auxiliary:
        raise ValueError("structural auxiliary-file evidence is empty")
    auxiliary_names: list[str] = []
    for item in auxiliary:
        if not isinstance(item, Mapping) or set(item) != {
            "base_sha256",
            "file",
            "output_sha256",
        }:
            raise ValueError("structural auxiliary-file evidence fields drift")
        descriptor = _validate_file_descriptor(
            {
                "file": item["file"],
                "sha256": item["output_sha256"],
                "size_bytes": live_files.get(item["file"], {}).get("size_bytes"),
            },
            label="structural auxiliary",
        )
        if (
            not _is_sha256(item["base_sha256"])
            or item["base_sha256"] != item["output_sha256"]
            or live_files.get(item["file"]) != descriptor
        ):
            raise ValueError("structural auxiliary-file identity drift")
        auxiliary_names.append(item["file"])
    expected_auxiliary_order = [name for name in _AUXILIARY_FILES if name in set(auxiliary_names)]
    if auxiliary_names != expected_auxiliary_order or len(auxiliary_names) != len(
        set(auxiliary_names)
    ):
        raise ValueError("structural auxiliary-file evidence is not unique and canonical")
    if not any(name.startswith("tokenizer") for name in auxiliary_names) or not any(
        "processor" in name for name in auxiliary_names
    ):
        raise ValueError("structural tokenizer or processor evidence is missing")

    modules = report["modules"]
    if not isinstance(modules, list) or len(modules) != QWEN35_08B_EXPECTED_QUANTIZED_MODULES:
        raise ValueError("structural module evidence coverage drift")
    module_names: list[str] = []
    signed_codes_checked = 0
    for item in modules:
        if not isinstance(item, Mapping) or set(item) != {
            "dequant_values_checked",
            "g_idx_sha256",
            "module",
            "qweight_sha256",
            "qzeros_sha256",
            "scales_sha256",
            "signed_codes_checked",
        }:
            raise ValueError("structural module evidence fields drift")
        if (
            not isinstance(item["module"], str)
            or not item["module"]
            or type(item["dequant_values_checked"]) is not int
            or item["dequant_values_checked"] <= 0
            or type(item["signed_codes_checked"]) is not int
            or item["signed_codes_checked"] <= 0
            or not all(
                _is_sha256(item[key])
                for key in (
                    "g_idx_sha256",
                    "qweight_sha256",
                    "qzeros_sha256",
                    "scales_sha256",
                )
            )
        ):
            raise ValueError("structural module evidence value drift")
        module_names.append(item["module"])
        signed_codes_checked += item["signed_codes_checked"]
    if len(module_names) != len(set(module_names)):
        raise ValueError("structural module evidence contains duplicate modules")

    preserved = report["preserved_non_target_tensors"]
    if not isinstance(preserved, list):
        raise ValueError("structural preserved-tensor evidence drift")
    preserved_keys: list[str] = []
    for item in preserved:
        if (
            not isinstance(item, Mapping)
            or set(item) != {"key", "sha256"}
            or not isinstance(item["key"], str)
            or not item["key"]
            or not _is_sha256(item["sha256"])
        ):
            raise ValueError("structural preserved-tensor evidence fields drift")
        preserved_keys.append(item["key"])
    if preserved_keys != sorted(preserved_keys) or len(preserved_keys) != len(set(preserved_keys)):
        raise ValueError("structural preserved-tensor evidence is not unique and canonical")
    if type(report["preserved_non_target_tensor_count"]) is not int or report[
        "preserved_non_target_tensor_count"
    ] != len(preserved):
        raise ValueError("structural preserved-tensor count drift")
    expected_summary = {
        "dense_target_weights_present": 0,
        "module_buffer_sets": len(modules),
        "non_target_tensors_exact": len(preserved),
        "signed_codes_checked": signed_codes_checked,
    }
    if report["summary"] != expected_summary:
        raise ValueError("structural verification summary drift")


def _validate_runtime_report(
    report: Mapping[str, Any],
    *,
    checkpoint: Mapping[str, Any],
    require_clean_environment: bool,
) -> None:
    kind = report.get("kind")
    common_fields = {
        "backend",
        "checkpoint_identity",
        "clean_environment",
        "device",
        "generated_tokens",
        "kind",
        "max_new_tokens",
        "prompt",
        "prompt_sha256",
        "runtime",
        "schema",
        "score_steps",
        "scores_sha256",
        "sequence_sha256",
        "status",
        "verifier",
    }
    expected_fields = common_fields | (
        {"decoded_text"} if kind == "text" else {"image", "image_sha256"}
    )
    if kind not in {"text", "image-text"} or set(report) != expected_fields:
        raise ValueError("runtime verification report fields drift")
    prompt = report["prompt"]
    if (
        report["backend"] != "GPTQ_TORCH"
        or not isinstance(report["device"], str)
        or not report["device"]
        or not isinstance(prompt, str)
        or not prompt
        or report["prompt_sha256"] != sha256_bytes(prompt.encode("utf-8"))
        or type(report["max_new_tokens"]) is not int
        or report["max_new_tokens"] <= 0
        or type(report["generated_tokens"]) is not int
        or report["generated_tokens"] <= 0
        or type(report["score_steps"]) is not int
        or report["score_steps"] <= 0
        or report["score_steps"] > report["max_new_tokens"]
        or report["generated_tokens"] != report["score_steps"]
        or not _is_sha256(report["sequence_sha256"])
        or not isinstance(report["scores_sha256"], list)
        or len(report["scores_sha256"]) != report["score_steps"]
        or not all(_is_sha256(value) for value in report["scores_sha256"])
    ):
        raise ValueError("runtime smoke-test evidence drift")
    if (
        kind == "text"
        and report["decoded_text"] is not None
        and not isinstance(report["decoded_text"], str)
    ):
        raise ValueError("runtime decoded text drift")
    if kind == "image-text":
        image = _validate_file_descriptor(report["image"], label="smoke image")
        if report["image_sha256"] != image["sha256"]:
            raise ValueError("image smoke input identity drift")

    _validate_current_verifier(report["verifier"])
    runtime = _validate_runtime_identity(report["runtime"], device=report["device"])
    if runtime["packages"]["gptqmodel"] != checkpoint["gptqmodel"]["version"]:
        raise ValueError("runtime GPTQModel version differs from checkpoint")
    clean_environment = report["clean_environment"]
    if clean_environment is None:
        if require_clean_environment:
            raise ValueError("runtime report is not bound to a clean-environment report")
    else:
        clean_environment = validate_clean_environment_identity(clean_environment)
        _validate_clean_runtime_binding(clean_environment, runtime, checkpoint)


def load_verification_report(
    path: str | Path,
    *,
    checkpoint_dir: str | Path,
    require_clean_environment: bool = False,
    hub_release: bool = False,
) -> dict[str, Any]:
    """Validate a canonical report against live checkpoint and verifier bytes."""

    candidate = Path(path).resolve()
    checkpoint_root = Path(checkpoint_dir).resolve()
    _require_report_outside_checkpoint(candidate, checkpoint_root)
    try:
        schema = json.loads(candidate.read_bytes()).get("schema")
    except (AttributeError, json.JSONDecodeError) as exc:
        raise ValueError(f"verification report is not a JSON object: {candidate}") from exc
    if schema not in {VERIFICATION_SCHEMA, RUNTIME_VERIFICATION_SCHEMA}:
        raise ValueError(f"unsupported verification report schema: {schema!r}")
    report = _load_canonical_report(candidate, expected_schema=schema)
    if report.get("status") != "pass":
        raise ValueError("verification report did not pass")
    checkpoint = validate_exported_checkpoint_identity(report.get("checkpoint_identity"))
    live_checkpoint = describe_exported_checkpoint(checkpoint_root, hub_release=hub_release)
    if checkpoint != live_checkpoint:
        raise ValueError("verification report checkpoint differs from live checkpoint bytes")
    if schema == VERIFICATION_SCHEMA:
        _validate_structural_report(report, checkpoint=checkpoint)
    else:
        _validate_runtime_report(
            report,
            checkpoint=checkpoint,
            require_clean_environment=require_clean_environment,
        )
    if not hub_release:
        return report
    return {
        "publication_envelope": _validate_hub_release_envelope(checkpoint_root),
        "report": report,
        "status": "model-payload-verified",
    }


def _verify_auxiliary_files(base: Path, output: Path) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for name in _AUXILIARY_FILES:
        base_file = base / name
        if not base_file.is_file():
            continue
        output_file = output / name
        if not output_file.is_file():
            raise ValueError(f"GPTQModel writer did not preserve auxiliary file: {name}")
        base_sha256 = sha256_file(base_file)
        output_sha256 = sha256_file(output_file)
        if output_sha256 != base_sha256:
            raise ValueError(f"GPTQModel writer changed preserved auxiliary file: {name}")
        evidence.append(
            {
                "base_sha256": base_sha256,
                "file": name,
                "output_sha256": output_sha256,
            }
        )
    if not any(entry["file"].startswith("tokenizer") for entry in evidence):
        raise ValueError("no tokenizer artifacts were preserved")
    if not any("processor" in entry["file"] for entry in evidence):
        raise ValueError("no processor artifacts were preserved")
    return evidence


def _verify_module_buffers(
    *,
    base: TensorCheckpoint,
    output: TensorCheckpoint,
    scale_run: ScaleRun,
    row_chunk_size: int,
    dequant_blocks_per_module: int,
) -> list[dict[str, Any]]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for checkpoint verification") from exc
    evidence: list[dict[str, Any]] = []
    for module in scale_run.modules:
        prefix = module.name
        dense_key = f"{prefix}.weight"
        if dense_key in output.keys:
            raise ValueError(f"target dense weight survived GPTQ export: {dense_key}")
        required = {
            "g_idx": f"{prefix}.g_idx",
            "qweight": f"{prefix}.qweight",
            "qzeros": f"{prefix}.qzeros",
            "scales": f"{prefix}.scales",
        }
        if any(key not in output.keys for key in required.values()):
            missing = sorted(key for key in required.values() if key not in output.keys)
            raise ValueError(f"incomplete GPTQ buffer set for {prefix}: {missing}")
        qweight_tensor = output.read(required["qweight"])
        qzeros_tensor = output.read(required["qzeros"])
        scales_tensor = output.read(required["scales"])
        g_idx_tensor = output.read(required["g_idx"])
        if qweight_tensor.dtype != torch.int32 or qzeros_tensor.dtype != torch.int32:
            raise TypeError(f"GPTQ packed buffers must be int32 for {prefix}")
        if g_idx_tensor.dtype != torch.int32:
            raise TypeError(f"GPTQ g_idx must be int32 for {prefix}")
        if scales_tensor.dtype != torch.float16:
            raise TypeError(f"GPTQ scales must be FP16 for {prefix}")

        qweight = _numpy(qweight_tensor)
        qzeros = _numpy(qzeros_tensor)
        scales = _numpy(scales_tensor)
        g_idx = _numpy(g_idx_tensor)
        expected_bits = scale_run.artifacts[prefix].export_scale_bits
        expected_scales = fp16_values_from_bits(expected_bits)
        np.testing.assert_array_equal(scales.view(np.uint16), expected_scales.T.view(np.uint16))
        np.testing.assert_array_equal(g_idx, make_g_idx(module.in_features, GROUP_SIZE))
        np.testing.assert_array_equal(
            qzeros,
            pack_gptq_v1_qzeros(
                num_groups=module.groups_per_row,
                out_features=module.out_features,
                logical_zero=INT4_LOGICAL_ZERO,
            ),
        )

        dense = base.read(dense_key)
        for start in range(0, module.out_features, row_chunk_size):
            stop = min(start + row_chunk_size, module.out_features)
            dense_rows = _numpy(dense[start:stop], float64=True)
            expected_codes = quantize_rows(dense_rows, expected_scales[start:stop])
            actual_codes = unpack_int4_qweight(qweight[:, start:stop])
            np.testing.assert_array_equal(
                actual_codes,
                expected_codes,
                err_msg=f"signed-code parity failed for {prefix} rows {start}:{stop}",
            )

        block_count = min(dequant_blocks_per_module, module.out_features // 8)
        if block_count <= 0:
            raise ValueError(f"module is too narrow for GPTQ dequant verification: {prefix}")
        seed = int(sha256_bytes(prefix.encode())[:16], 16)
        generator = np.random.default_rng(seed)
        block_indices = np.sort(
            generator.choice(module.out_features // 8, size=block_count, replace=False)
        )
        dequant_samples = 0
        for block_index in block_indices:
            start = int(block_index) * 8
            stop = start + 8
            reconstructed = dequantize_gptq_int4(
                qweight[:, start:stop],
                qzeros[:, start // 8 : stop // 8],
                scales[:, start:stop],
                g_idx,
            )
            codes = unpack_int4_qweight(qweight[:, start:stop]).astype(np.float32)
            expected = codes * scales[:, start:stop][g_idx].T.astype(np.float32)
            np.testing.assert_array_equal(reconstructed, expected)
            dequant_samples += int(reconstructed.size)
        evidence.append(
            {
                "dequant_values_checked": dequant_samples,
                "g_idx_sha256": tensor_sha256(g_idx_tensor),
                "module": prefix,
                "qweight_sha256": tensor_sha256(qweight_tensor),
                "qzeros_sha256": tensor_sha256(qzeros_tensor),
                "scales_sha256": tensor_sha256(scales_tensor),
                "signed_codes_checked": module.in_features * module.out_features,
            }
        )
    return evidence


def verify_full_model_structure(
    *,
    base_model_dir: str | Path,
    scale_run_dir: str | Path,
    corpus_directory: str | Path,
    checkpoint_dir: str | Path,
    report_path: str | Path,
    row_chunk_size: int = 32,
    dequant_blocks_per_module: int = 2,
) -> dict[str, Any]:
    """Verify every target and preserved tensor in a written GPTQ checkpoint."""

    if type(row_chunk_size) is not int or row_chunk_size <= 0:
        raise ValueError("row_chunk_size must be a positive integer")
    if type(dequant_blocks_per_module) is not int or dequant_blocks_per_module <= 0:
        raise ValueError("dequant_blocks_per_module must be a positive integer")
    base_root = Path(base_model_dir).resolve()
    if base_root.name != BASE_MODEL_REVISION:
        raise ValueError("base_model_dir is not the pinned Hugging Face snapshot")
    # Re-hash pinned shards and verify the scale run before comparing tensors.
    base_source = SafetensorsWeightSource(base_root)
    scale_run = load_scale_run(scale_run_dir, strict_inventory=True)
    corpus_replay = verify_frozen_corpus_pair(corpus_directory)
    scale_corpus_replay = validate_corpus_replay_pair(scale_run.metadata.get("corpus_replay"))
    if corpus_replay != scale_corpus_replay:
        raise ValueError("scale run and replayed corpus bundles differ")
    expected_fingerprint = sha256_bytes(canonical_json_bytes(dict(base_source.provenance)))
    if scale_run.metadata.get("base_fingerprint") != expected_fingerprint:
        raise ValueError("scale run and base checkpoint fingerprints differ")

    checkpoint_root = Path(checkpoint_dir).resolve()
    _require_report_outside_checkpoint(report_path, checkpoint_root)
    export_manifest = _verify_export_manifest(checkpoint_root)
    checkpoint_identity = validate_exported_checkpoint_identity(
        _checkpoint_identity(checkpoint_root, export_manifest)
    )
    if export_manifest["base_model"]["source"] != dict(base_source.provenance):
        raise ValueError("export manifest and live base snapshot bytes differ")
    if export_manifest.get("scale_run", {}).get("manifest_sha256") != scale_run.manifest_sha256:
        raise ValueError("export manifest and scale-run hash differ")
    if export_manifest.get("policy") != scale_run.policy:
        raise ValueError("export manifest and scale-run policy differ")
    if validate_corpus_replay_pair(export_manifest.get("corpus_replay")) != corpus_replay:
        raise ValueError("export manifest and replayed corpus bundles differ")
    base = TensorCheckpoint(base_root, allow_hf_blob_links=True)
    output = TensorCheckpoint(checkpoint_root)

    target_weights = {module.weight_key for module in scale_run.modules}
    quant_buffers = {
        f"{module.name}.{suffix}"
        for module in scale_run.modules
        for suffix in ("g_idx", "qweight", "qzeros", "scales")
    }
    expected_output_keys = (base.keys - target_weights) | quant_buffers
    if output.keys != expected_output_keys:
        missing = sorted(expected_output_keys - output.keys)
        extra = sorted(output.keys - expected_output_keys)
        raise ValueError(f"checkpoint tensor-key mismatch; missing={missing}, extra={extra}")

    preserved: list[dict[str, Any]] = []
    for key in sorted(base.keys - target_weights):
        before = base.read(key)
        after = output.read(key)
        if before.dtype != after.dtype or tuple(before.shape) != tuple(after.shape):
            raise ValueError(f"preserved tensor metadata changed: {key}")
        if not bool(__import__("torch").equal(before, after)):
            raise ValueError(f"preserved non-target tensor changed: {key}")
        preserved.append({"key": key, "sha256": tensor_sha256(before)})

    modules = _verify_module_buffers(
        base=base,
        output=output,
        scale_run=scale_run,
        row_chunk_size=row_chunk_size,
        dequant_blocks_per_module=dequant_blocks_per_module,
    )
    if len(modules) != QWEN35_08B_EXPECTED_QUANTIZED_MODULES:
        raise RuntimeError("structural verifier did not check exactly 150 modules")
    report = {
        "auxiliary_files": _verify_auxiliary_files(base_root, checkpoint_root),
        "base_checkpoint_files": base.file_hashes,
        "checkpoint_files": output.file_hashes,
        "checkpoint_identity": checkpoint_identity,
        "corpus_replay": corpus_replay,
        "export_manifest_sha256": sha256_file(checkpoint_root / "cliffquant_export.json"),
        "modules": modules,
        "policy": scale_run.policy,
        "preserved_non_target_tensor_count": len(preserved),
        "preserved_non_target_tensors": preserved,
        "scale_run_sha256": scale_run.manifest_sha256,
        "schema": VERIFICATION_SCHEMA,
        "status": "pass",
        "summary": {
            "dense_target_weights_present": 0,
            "module_buffer_sets": len(modules),
            "non_target_tensors_exact": len(preserved),
            "signed_codes_checked": sum(item["signed_codes_checked"] for item in modules),
        },
        "runtime": _verification_runtime("cpu"),
        "verifier": _verifier_identity(),
    }
    _write_canonical_report(report, report_path)
    return report


def load_fresh_gptq_torch(
    *,
    checkpoint_dir: str | Path,
    gptqmodel_source: str | Path | None = None,
    device: str,
    cache_dequantized_weights: bool = False,
) -> tuple[Any, Any]:
    """Load a checkpoint through installed or explicit pinned-source GPTQ_TORCH.

    GPTQModel's TorchLinear kernel compiles its dequantizer with Inductor during
    ``post_init``.  The pinned Windows CUDA environment has no Triton runtime.
    Load through the unchanged standard backend first, then strictly restore the
    original bound method carried by PyTorch's ``__wrapped__`` contract.
    """

    if type(cache_dequantized_weights) is not bool:
        raise ValueError("cache_dequantized_weights must be a boolean")
    contract = (
        load_installed_gptqmodel_contract()
        if gptqmodel_source is None
        else load_pinned_gptqmodel_contract(gptqmodel_source)
    )
    try:
        wrapper = contract.GPTQModel.load(
            str(Path(checkpoint_dir).resolve()),
            device=device,
            backend=contract.BACKEND.GPTQ_TORCH,
            trust_remote_code=False,
            local_files_only=True,
        )
    except Exception as exc:
        raise RuntimeError("fresh GPTQ_TORCH checkpoint load failed") from exc
    quantized = [
        (name, module)
        for name, module in wrapper.model.named_modules()
        if isinstance(module, contract.BaseQuantLinear)
    ]
    if len(quantized) != QWEN35_08B_EXPECTED_QUANTIZED_MODULES:
        raise RuntimeError(
            f"fresh GPTQ_TORCH load found {len(quantized)} quantized modules, expected 150"
        )
    eager_dequantizers: list[tuple[str, Any, Any]] = []
    for name, module in quantized:
        compiled = getattr(module, "dequantize_weight", None)
        eager = getattr(compiled, "__wrapped__", None)
        if (
            type(module) is not contract.TorchLinear
            or not callable(compiled)
            or getattr(compiled, "__self__", None) is not None
            or not callable(eager)
            or getattr(eager, "__wrapped__", None) is not None
            or getattr(eager, "__self__", None) is not module
            or getattr(eager, "__func__", None)
            is not contract.TorchLinear.dequantize_weight
        ):
            raise RuntimeError(
                "fresh GPTQ_TORCH load exposed an unexpected compiled TorchLinear "
                f"dequantizer: {name}"
            )
        eager_dequantizers.append((name, module, eager))
    for name, module, eager in eager_dequantizers:
        module.dequantize_weight = eager
        restored = module.dequantize_weight
        if (
            getattr(restored, "__self__", None) is not module
            or getattr(restored, "__func__", None)
            is not contract.TorchLinear.dequantize_weight
            or getattr(restored, "__wrapped__", None) is not None
        ):
            raise RuntimeError(
                f"fresh GPTQ_TORCH eager dequantizer restoration failed: {name}"
            )
    expected_cache_control = getattr(contract.TorchLinear, "enable_weight_cache", None)
    for name, module in quantized:
        cache_control = getattr(module, "enable_weight_cache", None)
        if (
            not callable(cache_control)
            or getattr(cache_control, "__self__", None) is not module
            or getattr(cache_control, "__func__", None) is not expected_cache_control
        ):
            raise RuntimeError(f"fresh GPTQ_TORCH weight-cache configuration failed: {name}")
        configured = cache_control(cache_dequantized_weights)
        if (
            configured is not module
            or getattr(module, "_cache_enabled", None) is not cache_dequantized_weights
        ):
            raise RuntimeError(f"fresh GPTQ_TORCH weight-cache configuration failed: {name}")
        if not cache_dequantized_weights and getattr(module, "_cached_weights", None):
            raise RuntimeError(f"fresh GPTQ_TORCH disabled weight cache was not empty: {name}")
    return contract, wrapper


def _move_inputs(inputs: Mapping[str, Any], device: str) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in inputs.items():
        to = getattr(value, "to", None)
        moved[key] = to(device) if callable(to) else value
    return moved


def _generate_once(
    wrapper: Any,
    inputs: Mapping[str, Any],
    *,
    max_new_tokens: int,
) -> tuple[Any, list[Any]]:
    generator = getattr(wrapper, "generate", None)
    if not callable(generator):
        generator = getattr(wrapper.model, "generate", None)
    if not callable(generator):
        raise RuntimeError("loaded GPTQModel wrapper exposes no generate method")
    output = generator(
        **inputs,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        output_scores=True,
        return_dict_in_generate=True,
        use_cache=True,
    )
    sequences = getattr(output, "sequences", None)
    scores = list(getattr(output, "scores", ()))
    if sequences is None:
        raise RuntimeError("generation returned no token sequences")
    return sequences, scores


def _check_generation(
    first: tuple[Any, list[Any]],
    second: tuple[Any, list[Any]],
) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for runtime verification") from exc
    first_sequences, first_scores = first
    second_sequences, second_scores = second
    if not torch.equal(first_sequences, second_sequences):
        raise ValueError("greedy generation is not deterministic across repeated calls")
    if len(first_scores) != len(second_scores):
        raise ValueError("repeated greedy generations returned different score counts")
    if not first_scores:
        raise ValueError("generation returned no score tensors")
    for score in [*first_scores, *second_scores]:
        if not bool(torch.all(torch.isfinite(score)).item()):
            raise ValueError("generation produced non-finite logits")
    return {
        "generated_tokens": len(first_scores),
        "score_steps": len(first_scores),
        "sequence_sha256": tensor_sha256(first_sequences),
        "scores_sha256": [tensor_sha256(score) for score in first_scores],
    }


def run_text_generation_smoke(
    *,
    checkpoint_dir: str | Path,
    gptqmodel_source: str | Path | None = None,
    report_path: str | Path,
    prompt: str,
    environment_report: str | Path | None = None,
    device: str = "cuda:0",
    max_new_tokens: int = 16,
) -> dict[str, Any]:
    """Fresh-load GPTQ_TORCH and check finite deterministic greedy text output."""

    if not prompt:
        raise ValueError("text smoke prompt must be non-empty")
    if type(max_new_tokens) is not int or max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be a positive integer")
    checkpoint_root = Path(checkpoint_dir).resolve()
    _require_report_outside_checkpoint(report_path, checkpoint_root)
    checkpoint_identity = describe_exported_checkpoint(checkpoint_root)
    contract, wrapper = load_fresh_gptq_torch(
        checkpoint_dir=checkpoint_root,
        gptqmodel_source=gptqmodel_source,
        device=device,
    )
    if (
        contract.revision != checkpoint_identity["gptqmodel"]["revision"]
        or contract.version != checkpoint_identity["gptqmodel"]["version"]
    ):
        raise ValueError("runtime GPTQModel identity differs from the exported checkpoint")
    tokenizer = getattr(wrapper, "tokenizer", None)
    if tokenizer is None:
        tokenizer = getattr(wrapper, "processor", None)
    tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    if not callable(tokenizer):
        raise RuntimeError("fresh GPTQModel load did not preserve a callable tokenizer")
    inputs = _move_inputs(tokenizer(prompt, return_tensors="pt"), device)
    first = _generate_once(wrapper, inputs, max_new_tokens=max_new_tokens)
    second = _generate_once(wrapper, inputs, max_new_tokens=max_new_tokens)
    evidence = _check_generation(first, second)
    decode = getattr(tokenizer, "decode", None)
    decoded = (
        decode(first[0][0].detach().cpu(), skip_special_tokens=True) if callable(decode) else None
    )
    report = {
        **evidence,
        "backend": "GPTQ_TORCH",
        "checkpoint_identity": checkpoint_identity,
        "clean_environment": (
            load_clean_environment_identity(environment_report)
            if environment_report is not None
            else None
        ),
        "decoded_text": decoded,
        "device": device,
        "kind": "text",
        "max_new_tokens": max_new_tokens,
        "prompt": prompt,
        "prompt_sha256": sha256_bytes(prompt.encode("utf-8")),
        "runtime": _verification_runtime(device),
        "schema": RUNTIME_VERIFICATION_SCHEMA,
        "status": "pass",
        "verifier": _verifier_identity(),
    }
    _write_canonical_report(report, report_path)
    return report


def run_image_generation_smoke(
    *,
    checkpoint_dir: str | Path,
    gptqmodel_source: str | Path | None = None,
    image_path: str | Path,
    report_path: str | Path,
    prompt: str,
    environment_report: str | Path | None = None,
    device: str = "cuda:0",
    max_new_tokens: int = 16,
) -> dict[str, Any]:
    """Separately check one finite deterministic image-text generation."""

    if not prompt:
        raise ValueError("image smoke prompt must be non-empty")
    path = Path(image_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"image smoke input does not exist: {path}")
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required for the image-text smoke") from exc
    checkpoint_root = Path(checkpoint_dir).resolve()
    _require_report_outside_checkpoint(report_path, checkpoint_root)
    checkpoint_identity = describe_exported_checkpoint(checkpoint_root)
    contract, wrapper = load_fresh_gptq_torch(
        checkpoint_dir=checkpoint_root,
        gptqmodel_source=gptqmodel_source,
        device=device,
    )
    if (
        contract.revision != checkpoint_identity["gptqmodel"]["revision"]
        or contract.version != checkpoint_identity["gptqmodel"]["version"]
    ):
        raise ValueError("runtime GPTQModel identity differs from the exported checkpoint")
    processor = getattr(wrapper, "processor", None)
    if not callable(processor):
        raise RuntimeError("fresh GPTQModel load did not preserve a callable processor")
    apply_chat_template = getattr(processor, "apply_chat_template", None)
    if not callable(apply_chat_template):
        raise RuntimeError("fresh GPTQModel processor exposes no multimodal chat template")
    with Image.open(path) as image:
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image.convert("RGB")},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        inputs = apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
    if not isinstance(inputs, Mapping) or not {
        "attention_mask",
        "image_grid_thw",
        "input_ids",
        "pixel_values",
    }.issubset(inputs):
        raise RuntimeError("multimodal chat template returned incomplete model inputs")
    moved = _move_inputs(inputs, device)
    first = _generate_once(wrapper, moved, max_new_tokens=max_new_tokens)
    second = _generate_once(wrapper, moved, max_new_tokens=max_new_tokens)
    report = {
        **_check_generation(first, second),
        "backend": "GPTQ_TORCH",
        "checkpoint_identity": checkpoint_identity,
        "clean_environment": (
            load_clean_environment_identity(environment_report)
            if environment_report is not None
            else None
        ),
        "device": device,
        "image": _file_descriptor(path),
        "image_sha256": sha256_file(path),
        "kind": "image-text",
        "max_new_tokens": max_new_tokens,
        "prompt": prompt,
        "prompt_sha256": sha256_bytes(prompt.encode("utf-8")),
        "runtime": _verification_runtime(device),
        "schema": RUNTIME_VERIFICATION_SCHEMA,
        "status": "pass",
        "verifier": _verifier_identity(),
    }
    _write_canonical_report(report, report_path)
    return report
