"""Pinned GPTQModel adapter and standard GPTQ-v1 CliffQuant exporter.

GPTQModel's high-level ``pack_model`` helper is intentionally not used.  The
pinned contract is exercised one module at a time through
``create_quant_module`` and ``pack_module`` so every input and serialized
buffer can be checked immediately.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import inspect
import json
import os
import subprocess
import sys
import sysconfig
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .full_model_artifacts import (
    BASE_MODEL_ID,
    BASE_MODEL_REVISION,
    BITS,
    GROUP_SIZE,
    QMAX,
    QMIN,
    SafetensorsWeightSource,
    ScaleRun,
    load_scale_run,
    validate_corpus_replay_pair,
    verify_frozen_corpus_pair,
)
from .gptq_pack import (
    INT4_LOGICAL_ZERO,
    make_g_idx,
    pack_gptq_v1_qzeros,
    pack_int4_qweight,
    unpack_int4_qweight,
)
from .provenance import (
    canonical_array_sha256,
    canonical_json_bytes,
    runtime_metadata,
    sha256_bytes,
    sha256_file,
)
from .qwen35_modules import (
    QWEN35_08B_EXPECTED_QUANTIZED_MODULES,
    QWEN35_GPTQMODEL_TREE_REVISION,
    discover_qwen35_quantized_modules,
)

EXPORT_SCHEMA = "cliffquant.gptq-export.v1"
PINNED_GPTQMODEL_VERSION = "7.3.4"

_GPTQMODEL_MODULES = {
    "auto": "gptqmodel.models.auto",
    "backend": "gptqmodel.utils.backend",
    "config": "gptqmodel.quantization.config",
    "constants": "gptqmodel.models._const",
    "model_utils": "gptqmodel.utils.model",
    "package": "gptqmodel",
    "qlinear_base": "gptqmodel.nn_modules.qlinear",
    "torch_linear": "gptqmodel.nn_modules.qlinear.torch",
}
_REQUIRED_CREATE_PARAMETERS = {
    "backend",
    "bits",
    "desc_act",
    "device",
    "dynamic",
    "format",
    "group_size",
    "linear_cls",
    "lm_head_name",
    "module",
    "name",
    "pack_dtype",
    "submodule",
    "sym",
}
_REQUIRED_PACK_PARAMETERS = {
    "layers",
    "lock",
    "name",
    "qModules",
    "q_g_idx",
    "q_scales",
    "q_zeros",
    "quant_linear_cls",
    "quantize_config",
}


@dataclass(frozen=True, slots=True)
class GPTQModelContract:
    """Exact GPTQModel objects imported from one verified package root."""

    source_root: Path
    revision: str
    version: str
    GPTQModel: Any
    QuantizeConfig: Any
    METHOD: Any
    FORMAT: Any
    BACKEND: Any
    DEVICE: Any
    TorchLinear: Any
    BaseQuantLinear: Any
    create_quant_module: Any
    pack_module: Any


def _git_revision(root: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"GPTQModel source must be a git checkout: {completed.stderr.strip()}")
    return completed.stdout.strip().lower()


def _require_clean_git_checkout(root: Path) -> None:
    completed = subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"cannot verify GPTQModel checkout cleanliness: {completed.stderr.strip()}"
        )
    if completed.stdout.strip():
        raise RuntimeError(
            "GPTQModel checkout has tracked or untracked source drift; "
            "use a clean checkout of the frozen revision"
        )


def _ensure_windows_utf8_console() -> None:
    if os.name != "nt":
        return
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="backslashreplace")


def _is_within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _import_gptqmodel_modules(*, label: str) -> dict[str, Any]:
    _ensure_windows_utf8_console()
    try:
        return {
            logical_name: importlib.import_module(module_name)
            for logical_name, module_name in _GPTQMODEL_MODULES.items()
        }
    except Exception as exc:
        raise RuntimeError(
            f"failed to import {label} GPTQModel and its runtime dependencies"
        ) from exc


def _require_module_origins(
    modules: dict[str, Any],
    *,
    roots: set[Path],
    label: str,
) -> None:
    for logical_name, module in modules.items():
        raw_origin = getattr(module, "__file__", None)
        if not isinstance(raw_origin, str):
            raise RuntimeError(f"{label} GPTQModel module has no file origin: {logical_name}")
        origin = Path(raw_origin).resolve()
        if not any(_is_within(origin, root) for root in roots):
            raise RuntimeError(
                f"{label} GPTQModel module resolved outside the verified package root: "
                f"{logical_name}"
            )


def _contract_from_modules(
    *,
    modules: dict[str, Any],
    revision: str,
    source_root: Path,
) -> GPTQModelContract:
    model_utils = modules["model_utils"]
    create_signature = inspect.signature(model_utils.create_quant_module)
    if not _REQUIRED_CREATE_PARAMETERS.issubset(create_signature.parameters):
        raise RuntimeError("pinned create_quant_module contract is incomplete or drifted")
    pack_signature = inspect.signature(model_utils.pack_module)
    if not _REQUIRED_PACK_PARAMETERS.issubset(pack_signature.parameters):
        raise RuntimeError("pinned pack_module contract is incomplete or drifted")
    package = modules["package"]
    return GPTQModelContract(
        source_root=source_root,
        revision=revision,
        version=str(getattr(package, "__version__", "unknown")),
        GPTQModel=modules["auto"].GPTQModel,
        QuantizeConfig=modules["config"].QuantizeConfig,
        METHOD=modules["config"].METHOD,
        FORMAT=modules["config"].FORMAT,
        BACKEND=modules["backend"].BACKEND,
        DEVICE=modules["constants"].DEVICE,
        TorchLinear=modules["torch_linear"].TorchLinear,
        BaseQuantLinear=modules["qlinear_base"].BaseQuantLinear,
        create_quant_module=model_utils.create_quant_module,
        pack_module=model_utils.pack_module,
    )


def load_pinned_gptqmodel_contract(source_root: str | Path) -> GPTQModelContract:
    """Import GPTQModel only from the exact frozen source revision."""

    root = Path(source_root).resolve()
    if not (root / "gptqmodel").is_dir():
        raise FileNotFoundError(f"GPTQModel package directory is absent from {root}")
    revision = _git_revision(root)
    if revision != QWEN35_GPTQMODEL_TREE_REVISION:
        raise RuntimeError(
            "GPTQModel source revision drift: "
            f"expected {QWEN35_GPTQMODEL_TREE_REVISION}, got {revision}"
        )
    _require_clean_git_checkout(root)
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    already_loaded = sys.modules.get("gptqmodel")
    if already_loaded is not None:
        loaded_path = Path(already_loaded.__file__).resolve()
        if root not in loaded_path.parents:
            raise RuntimeError(
                "a different GPTQModel package is already imported; start a fresh process"
            )
    modules = _import_gptqmodel_modules(label="pinned-checkout")
    _require_module_origins(modules, roots={root}, label="pinned-checkout")
    return _contract_from_modules(
        modules=modules,
        revision=revision,
        source_root=root,
    )


def load_installed_gptqmodel_contract() -> GPTQModelContract:
    """Import GPTQModel only from the active venv's installed distribution."""

    prefix = Path(sys.prefix).resolve()
    base_prefix = Path(getattr(sys, "base_prefix", sys.prefix)).resolve()
    if prefix == base_prefix:
        raise RuntimeError(
            "installed GPTQModel verification requires an active virtual environment"
        )
    roots = {
        Path(value).resolve()
        for key in ("purelib", "platlib")
        if (value := sysconfig.get_path(key))
    }
    if not roots or any(not _is_within(root, prefix) for root in roots):
        raise RuntimeError("active virtual environment site-packages identity drift")
    try:
        distribution = importlib.metadata.distribution("GPTQModel")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError("GPTQModel is not installed in the active virtual environment") from exc
    if distribution.version != PINNED_GPTQMODEL_VERSION:
        raise RuntimeError(
            "installed GPTQModel version drift: "
            f"expected {PINNED_GPTQMODEL_VERSION}, got {distribution.version}"
        )
    spec = importlib.util.find_spec("gptqmodel")
    if spec is None or spec.origin is None:
        raise RuntimeError("installed GPTQModel package has no import origin")
    origin = Path(spec.origin).resolve()
    if not any(_is_within(origin, root) for root in roots):
        raise RuntimeError("installed GPTQModel resolves outside active venv site-packages")
    distribution_init = Path(distribution.locate_file("gptqmodel/__init__.py")).resolve()
    if distribution_init != origin:
        raise RuntimeError("installed GPTQModel import does not match its distribution metadata")
    modules = _import_gptqmodel_modules(label="installed")
    _require_module_origins(modules, roots=roots, label="installed")
    contract = _contract_from_modules(
        modules=modules,
        revision=QWEN35_GPTQMODEL_TREE_REVISION,
        source_root=origin.parent,
    )
    if contract.version != PINNED_GPTQMODEL_VERSION:
        raise RuntimeError(
            "installed GPTQModel package version drift: "
            f"expected {PINNED_GPTQMODEL_VERSION}, got {contract.version}"
        )
    return contract


def fp16_values_from_bits(bits: NDArray[np.uint16]) -> NDArray[np.float16]:
    """Return an FP16 array preserving every stored scale bit exactly."""

    array = np.ascontiguousarray(bits, dtype="<u2")
    if np.any(array == 0) or np.any(array > 0x7BFF):
        raise ValueError("scale bits must encode positive finite binary16 values")
    return array.view("<f2").copy()


def quantize_rows(
    weights: NDArray[np.generic],
    scales_out_group: NDArray[np.float16],
    *,
    group_size: int = GROUP_SIZE,
) -> NDArray[np.int8]:
    """Apply the frozen RNE-plus-clipping quantizer to dense output rows."""

    dense = np.asarray(weights, dtype=np.float64)
    scales = np.asarray(scales_out_group, dtype=np.float16)
    if dense.ndim != 2 or scales.ndim != 2:
        raise ValueError("weights and scales must be rank two")
    if dense.shape[0] != scales.shape[0]:
        raise ValueError("weights and scales must have the same output-row count")
    if dense.shape[1] != scales.shape[1] * group_size:
        raise ValueError("scale group count does not match dense input width")
    expanded = np.repeat(scales.astype(np.float64), group_size, axis=1)
    with np.errstate(divide="raise", invalid="raise", over="raise"):
        rounded = np.rint(dense / expanded)
    return np.clip(rounded, QMIN, QMAX).astype(np.int8)


def _tensor_numpy(tensor: Any, *, floating_dtype: Any | None = None) -> NDArray[np.generic]:
    value = tensor.detach().cpu()
    if floating_dtype is not None:
        value = value.to(dtype=floating_dtype)
    return np.ascontiguousarray(value.numpy())


def _make_quantize_config(contract: GPTQModelContract, scale_run: ScaleRun) -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for GPTQ export") from exc
    qcfg = contract.QuantizeConfig(
        bits=BITS,
        group_size=GROUP_SIZE,
        desc_act=False,
        sym=True,
        method=contract.METHOD.GPTQ,
        format=contract.FORMAT.GPTQ,
        pack_dtype=torch.int32,
        pack_impl="original",
        offload_to_disk=False,
    )
    raw_corpus_replay = getattr(scale_run, "metadata", {}).get("corpus_replay")
    corpus_replay_sha256 = (
        validate_corpus_replay_pair(raw_corpus_replay)["sha256"]
        if raw_corpus_replay is not None
        else None
    )
    raw_base_model = getattr(scale_run, "metadata", {}).get("base_model")
    raw_base_source = raw_base_model.get("source") if isinstance(raw_base_model, dict) else None
    base_snapshot_sha256 = (
        raw_base_source.get("fingerprint_sha256") if isinstance(raw_base_source, dict) else None
    )
    qcfg.meta_set(
        "cliffquant",
        {
            "base_model": BASE_MODEL_ID,
            "base_revision": BASE_MODEL_REVISION,
            "base_snapshot_sha256": base_snapshot_sha256,
            "corpus_replay_sha256": corpus_replay_sha256,
            "exact_solver": "solve_breakpoint_exact",
            "gptqmodel_revision": contract.revision,
            "protocol": "experiment-001",
            "policy": scale_run.policy,
            "scale_run_sha256": scale_run.manifest_sha256,
            "scale_selection": (
                "multi-environment minimax diagonal reconstruction proxy"
                if scale_run.policy == "cliffquant_minimax"
                else "signed-range AbsMax scale rounded to nearest positive finite FP16"
            ),
        },
    )
    return qcfg


def _load_dense_wrapper(
    contract: GPTQModelContract,
    *,
    base_model_dir: Path,
    quantize_config: Any,
) -> Any:
    try:
        wrapper = contract.GPTQModel.load(
            str(base_model_dir),
            quantize_config=quantize_config,
            device="cpu",
            backend=contract.BACKEND.GPTQ_TORCH,
            trust_remote_code=False,
            local_files_only=True,
            torch_dtype="auto",
        )
    except Exception as exc:
        raise RuntimeError(
            "the pinned GPTQModel high-level loader could not materialize the dense "
            "Qwen3.5 wrapper; export is aborted rather than writing a substitute format"
        ) from exc
    if not hasattr(wrapper, "model") or not hasattr(wrapper, "save_quantized"):
        raise RuntimeError("GPTQModel loader returned an incompatible wrapper")
    if getattr(wrapper, "processor", None) is None:
        raise RuntimeError("Qwen3.5 processor was not preserved by the GPTQModel wrapper")
    return wrapper


def _restore_non_target_tensors_exact(
    *,
    core_model: Any,
    base_source: SafetensorsWeightSource,
    target_weights: set[str],
) -> dict[str, Any]:
    """Restore every non-quantized state tensor to its exact base bytes and dtype."""

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for exact non-target restoration") from exc

    current_state = core_model.state_dict(keep_vars=True)
    expected_keys = base_source.keys - target_weights
    missing = sorted(expected_keys - set(current_state))
    if missing:
        raise ValueError(f"dense wrapper omitted preserved base tensors: {missing}")

    restored: list[dict[str, Any]] = []
    for key in sorted(expected_keys):
        current = current_state[key]
        if current.device.type != "cpu":
            raise RuntimeError(f"non-target tensor must remain on CPU before export: {key}")
        with base_source._open(key) as handle:
            exact = handle.get_tensor(key).detach().cpu().contiguous()
        if tuple(current.shape) != tuple(exact.shape):
            raise ValueError(f"dense wrapper changed non-target tensor shape: {key}")
        if current.dtype != exact.dtype or not bool(torch.equal(current.detach(), exact)):
            previous_dtype = str(current.dtype)
            current.data = exact
            if current.dtype != exact.dtype or not bool(torch.equal(current.detach(), exact)):
                raise RuntimeError(f"failed to restore exact non-target tensor: {key}")
            restored.append(
                {
                    "from_dtype": previous_dtype,
                    "key": key,
                    "shape": [int(value) for value in exact.shape],
                    "to_dtype": str(exact.dtype),
                }
            )

    return {
        "checked_tensor_count": len(expected_keys),
        "restored_tensor_count": len(restored),
        "restored_tensors": restored,
    }


def _verify_packed_module(
    *,
    dense_module: Any,
    quant_module: Any,
    scale_bits: NDArray[np.uint16],
    row_chunk_size: int,
) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exporter already requires torch
        raise RuntimeError("PyTorch is required for packed-module verification") from exc
    qweight = _tensor_numpy(quant_module.qweight)
    qzeros = _tensor_numpy(quant_module.qzeros)
    stored_scales = _tensor_numpy(quant_module.scales)
    stored_g_idx = _tensor_numpy(quant_module.g_idx)
    expected_scales = fp16_values_from_bits(scale_bits)
    if stored_scales.shape != expected_scales.T.shape:
        raise ValueError("GPTQModel serialized scale shape drift")
    np.testing.assert_array_equal(
        stored_scales.view(np.uint16),
        expected_scales.T.view(np.uint16),
        err_msg="GPTQModel changed CliffQuant FP16 scale bits",
    )
    expected_g_idx = make_g_idx(dense_module.in_features, GROUP_SIZE)
    np.testing.assert_array_equal(stored_g_idx, expected_g_idx)
    expected_qzeros = pack_gptq_v1_qzeros(
        num_groups=expected_scales.shape[1],
        out_features=dense_module.out_features,
        logical_zero=INT4_LOGICAL_ZERO,
    )
    np.testing.assert_array_equal(qzeros, expected_qzeros)

    if row_chunk_size <= 0:
        raise ValueError("pack verification row_chunk_size must be positive")
    for start in range(0, dense_module.out_features, row_chunk_size):
        stop = min(start + row_chunk_size, dense_module.out_features)
        dense_rows = _tensor_numpy(
            dense_module.weight[start:stop],
            floating_dtype=torch.float64,
        )
        expected_codes = quantize_rows(dense_rows, expected_scales[start:stop])
        expected_packed = pack_int4_qweight(expected_codes)
        np.testing.assert_array_equal(
            qweight[:, start:stop],
            expected_packed,
            err_msg=f"GPTQModel packed-code mismatch at rows {start}:{stop}",
        )
        np.testing.assert_array_equal(
            unpack_int4_qweight(qweight[:, start:stop]),
            expected_codes,
        )
    return {
        "g_idx_sha256": canonical_array_sha256(stored_g_idx),
        "qweight_sha256": canonical_array_sha256(qweight),
        "qzeros_sha256": canonical_array_sha256(qzeros),
        "scales_sha256": canonical_array_sha256(stored_scales),
    }


def _pack_one_module(
    contract: GPTQModelContract,
    *,
    core_model: Any,
    dense_module: Any,
    name: str,
    scale_bits: NDArray[np.uint16],
    quantize_config: Any,
    lm_head_name: str,
    verify_row_chunk_size: int,
) -> tuple[Any, dict[str, Any]]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for GPTQ export") from exc
    weight = dense_module.weight
    if weight.device.type != "cpu" or weight.device.type == "meta":
        raise RuntimeError(f"dense module must be materialized on CPU before packing: {name}")
    scales = fp16_values_from_bits(scale_bits)
    q_scales = torch.from_numpy(scales.copy()).to(dtype=torch.float16, device="cpu")
    q_zeros = torch.full(
        q_scales.shape,
        INT4_LOGICAL_ZERO,
        dtype=torch.float16,
        device="cpu",
    )
    q_g_idx = torch.from_numpy(make_g_idx(dense_module.in_features, GROUP_SIZE)).to(
        dtype=torch.int32,
        device="cpu",
    )

    try:
        contract.create_quant_module(
            name=name,
            linear_cls=contract.TorchLinear,
            bits=BITS,
            desc_act=False,
            dynamic=None,
            group_size=GROUP_SIZE,
            module=core_model,
            submodule=dense_module,
            sym=True,
            device=contract.DEVICE.CPU,
            lm_head_name=lm_head_name,
            pack_dtype=torch.int32,
            format=contract.FORMAT.GPTQ,
            backend=contract.BACKEND.GPTQ_TORCH,
            register_buffers=True,
            dtype=weight.dtype,
        )
        quant_module = core_model.get_submodule(name)
        if not isinstance(quant_module, contract.BaseQuantLinear):
            raise TypeError("create_quant_module did not install a BaseQuantLinear")
        packer_label = contract.pack_module(
            name=name,
            qModules={name: quant_module},
            q_scales=q_scales,
            q_zeros=q_zeros,
            q_g_idx=q_g_idx,
            layers={name: dense_module},
            quant_linear_cls=contract.TorchLinear,
            lock=threading.Lock(),
            quantize_config=quantize_config,
        )
    except Exception as exc:
        raise RuntimeError(
            f"pinned GPTQModel create/pack contract failed for {name}; "
            "export is aborted rather than using pack_model or a custom checkpoint layout"
        ) from exc
    hashes = _verify_packed_module(
        dense_module=dense_module,
        quant_module=quant_module,
        scale_bits=scale_bits,
        row_chunk_size=verify_row_chunk_size,
    )
    return quant_module, {"packer": packer_label, **hashes}


def _output_file_hashes(root: Path) -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "cliffquant_export.json":
            files.append(
                {
                    "file": path.relative_to(root).as_posix(),
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
            )
    return files


def export_full_model_gptq(
    *,
    base_model_dir: str | Path,
    scale_run_dir: str | Path,
    corpus_directory: str | Path,
    gptqmodel_source: str | Path,
    output_dir: str | Path,
    verify_row_chunk_size: int = 32,
) -> dict[str, Any]:
    """Pack and write the frozen model through standard GPTQModel GPTQ-v1."""

    base_path = Path(base_model_dir).resolve()
    destination = Path(output_dir).resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite export destination: {destination}")
    if type(verify_row_chunk_size) is not int or verify_row_chunk_size <= 0:
        raise ValueError("verify_row_chunk_size must be a positive integer")

    scale_run = load_scale_run(scale_run_dir, strict_inventory=True)
    corpus_replay = verify_frozen_corpus_pair(corpus_directory)
    scale_corpus_replay = validate_corpus_replay_pair(scale_run.metadata.get("corpus_replay"))
    if corpus_replay != scale_corpus_replay:
        raise ValueError("scale run was not produced from these exact replayed corpus bundles")
    base_source = SafetensorsWeightSource(base_path)
    if scale_run.metadata.get("base_fingerprint") != sha256_bytes(
        canonical_json_bytes(dict(base_source.provenance))
    ):
        raise ValueError("scale run was not produced from this exact pinned base checkpoint")
    contract = load_pinned_gptqmodel_contract(gptqmodel_source)
    qcfg = _make_quantize_config(contract, scale_run)
    wrapper = _load_dense_wrapper(contract, base_model_dir=base_path, quantize_config=qcfg)
    core_model = wrapper.model
    discovered = discover_qwen35_quantized_modules(core_model)
    if len(discovered) != QWEN35_08B_EXPECTED_QUANTIZED_MODULES:
        raise ValueError("loaded Qwen3.5 model does not have the frozen 150-module inventory")
    inventory = [(item.name, item.in_features, item.out_features) for item in discovered]
    del discovered
    expected = {module.name: module for module in scale_run.modules}
    if [name for name, _in_features, _out_features in inventory] != [
        module.name for module in scale_run.modules
    ]:
        raise ValueError("loaded model module order or identity differs from the scale run")

    packed: list[dict[str, Any]] = []
    for name, in_features, out_features in inventory:
        spec = expected[name]
        if (out_features, in_features) != (spec.out_features, spec.in_features):
            raise ValueError(f"loaded module shape drift for {name}")
        dense_module = core_model.get_submodule(name)
        _quant_module, evidence = _pack_one_module(
            contract,
            core_model=core_model,
            dense_module=dense_module,
            name=name,
            scale_bits=scale_run.artifacts[name].export_scale_bits,
            quantize_config=qcfg,
            lm_head_name=str(getattr(wrapper, "lm_head", "lm_head")),
            verify_row_chunk_size=verify_row_chunk_size,
        )
        packed.append({"module": name, **evidence})

    non_target_preservation = _restore_non_target_tensors_exact(
        core_model=core_model,
        base_source=base_source,
        target_weights={module.weight_key for module in scale_run.modules},
    )
    wrapper.quantized = True
    wrapper.load_quantized_model = False
    wrapper.quantize_config = qcfg
    wrapper.qlinear_kernel = contract.TorchLinear
    stage = destination.with_name(f".{destination.name}.partial-{uuid.uuid4().hex}")
    stage.mkdir(parents=True, exist_ok=False)
    try:
        wrapper.save_quantized(
            str(stage),
            safetensors_metadata={
                "cliffquant": "experiment-001-fp16-scale-search",
                "cliffquant_policy": scale_run.policy,
                "cliffquant_scale_run_sha256": scale_run.manifest_sha256,
                "gptqmodel_revision": contract.revision,
            },
            meta_quantizer="cliffquant:0.1.0",
        )
    except Exception as exc:
        raise RuntimeError(
            f"GPTQModel writer failed; partial output is retained for diagnosis at {stage}"
        ) from exc

    config_path = stage / "config.json"
    quant_config_path = stage / "quantize_config.json"
    if not config_path.is_file() or not quant_config_path.is_file():
        raise RuntimeError(f"GPTQModel writer omitted required configuration files in {stage}")
    quant_config = json.loads(quant_config_path.read_text(encoding="utf-8"))
    method = str(quant_config.get("quant_method", quant_config.get("method", ""))).lower()
    format_name = str(quant_config.get("format", quant_config.get("checkpoint_format", ""))).lower()
    if method != "gptq" or format_name != "gptq":
        raise RuntimeError(
            "writer did not serialize standard method=gptq and format=gptq; export is blocked"
        )
    meta = quant_config.get("meta")
    embedded_identity = meta.get("cliffquant") if isinstance(meta, dict) else None
    if not isinstance(embedded_identity, dict):
        raise RuntimeError("writer omitted CliffQuant provenance from quantize_config meta")
    if (
        embedded_identity.get("policy") != scale_run.policy
        or embedded_identity.get("scale_run_sha256") != scale_run.manifest_sha256
        or embedded_identity.get("base_snapshot_sha256")
        != base_source.provenance["fingerprint_sha256"]
        or embedded_identity.get("corpus_replay_sha256") != corpus_replay["sha256"]
        or embedded_identity.get("base_model") != BASE_MODEL_ID
        or embedded_identity.get("base_revision") != BASE_MODEL_REVISION
        or embedded_identity.get("gptqmodel_revision") != contract.revision
    ):
        raise RuntimeError("writer changed the embedded CliffQuant policy or provenance identity")

    manifest = {
        "base_model": {
            "id": BASE_MODEL_ID,
            "revision": BASE_MODEL_REVISION,
            "source": dict(base_source.provenance),
        },
        "corpus_replay": corpus_replay,
        "files": _output_file_hashes(stage),
        "gptq": {
            "backend": "GPTQ_TORCH",
            "bits": BITS,
            "desc_act": False,
            "format": "gptq",
            "group_size": GROUP_SIZE,
            "method": "gptq",
            "pack_dtype": "int32",
            "sym": True,
            "zero": INT4_LOGICAL_ZERO,
        },
        "gptqmodel": {
            "revision": contract.revision,
            "version": contract.version,
        },
        "modules": packed,
        "non_target_preservation": non_target_preservation,
        "policy": scale_run.policy,
        "runtime": runtime_metadata(device="cpu"),
        "scale_run": {
            "manifest_sha256": scale_run.manifest_sha256,
            "path": scale_run.manifest_path.name,
        },
        "schema": EXPORT_SCHEMA,
        "status": "packed-unverified",
    }
    (stage / "cliffquant_export.json").write_bytes(canonical_json_bytes(manifest))
    os.replace(stage, destination)
    return manifest


def run_gptqmodel_pack_parity(
    *,
    gptqmodel_source: str | Path,
    seed: int = 2339,
) -> dict[str, Any]:
    """Reproduce synthetic parity between GPTQModel and the local reference."""

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for the GPTQModel pack-parity check") from exc
    contract = load_pinned_gptqmodel_contract(gptqmodel_source)
    generator = np.random.default_rng(seed)
    # The pinned GPTQ-v1 conversion packs zero points in 32-output-channel
    # words, so keep the synthetic contract fixture on that real alignment.
    out_features = 32
    in_features = 256
    groups = in_features // GROUP_SIZE
    scales = generator.uniform(0.01, 0.2, size=(out_features, groups)).astype(np.float16)
    codes = generator.integers(QMIN, QMAX + 1, size=(out_features, in_features), dtype=np.int8)
    weights = codes.astype(np.float32) * np.repeat(scales.astype(np.float32), GROUP_SIZE, axis=1)

    class TinyRoot(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(
                in_features, out_features, bias=False, dtype=torch.float16
            )
            self.linear.weight.data.copy_(torch.from_numpy(weights).to(dtype=torch.float16))

    root = TinyRoot().cpu()
    dense = root.linear
    dummy_scale_run = type(
        "ScaleRunIdentity",
        (),
        {
            "manifest_sha256": "0" * 64,
            "policy": "cliffquant_minimax",
        },
    )()
    qcfg = _make_quantize_config(contract, dummy_scale_run)
    contract.create_quant_module(
        name="linear",
        linear_cls=contract.TorchLinear,
        bits=BITS,
        desc_act=False,
        dynamic=None,
        group_size=GROUP_SIZE,
        module=root,
        submodule=dense,
        sym=True,
        device=contract.DEVICE.CPU,
        lm_head_name="lm_head",
        pack_dtype=torch.int32,
        format=contract.FORMAT.GPTQ,
        backend=contract.BACKEND.GPTQ_TORCH,
        register_buffers=True,
        dtype=torch.float16,
    )
    quant_module = root.linear
    contract.pack_module(
        name="linear",
        qModules={"linear": quant_module},
        q_scales=torch.from_numpy(scales.copy()),
        q_zeros=torch.full(scales.shape, INT4_LOGICAL_ZERO, dtype=torch.float16),
        q_g_idx=torch.from_numpy(make_g_idx(in_features, GROUP_SIZE)),
        layers={"linear": dense},
        quant_linear_cls=contract.TorchLinear,
        lock=threading.Lock(),
        quantize_config=qcfg,
    )
    actual_qweight = _tensor_numpy(quant_module.qweight)
    actual_qzeros = _tensor_numpy(quant_module.qzeros)
    actual_scales = _tensor_numpy(quant_module.scales)
    actual_g_idx = _tensor_numpy(quant_module.g_idx)
    stored_dense = _tensor_numpy(dense.weight, floating_dtype=torch.float64)
    expected_codes = quantize_rows(stored_dense, scales)
    expected_qweight = pack_int4_qweight(expected_codes)
    expected_qzeros = pack_gptq_v1_qzeros(num_groups=groups, out_features=out_features)
    np.testing.assert_array_equal(actual_qweight, expected_qweight)
    np.testing.assert_array_equal(actual_qzeros, expected_qzeros)
    np.testing.assert_array_equal(actual_scales.view(np.uint16), scales.T.view(np.uint16))
    np.testing.assert_array_equal(actual_g_idx, make_g_idx(in_features, GROUP_SIZE))
    np.testing.assert_array_equal(unpack_int4_qweight(actual_qweight), expected_codes)
    return {
        "g_idx_sha256": canonical_array_sha256(actual_g_idx),
        "gptqmodel_revision": contract.revision,
        "gptqmodel_version": contract.version,
        "qweight_sha256": canonical_array_sha256(actual_qweight),
        "qzeros_sha256": canonical_array_sha256(actual_qzeros),
        "scales_sha256": canonical_array_sha256(actual_scales),
        "seed": seed,
        "status": "pass",
    }
