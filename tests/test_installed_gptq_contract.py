from __future__ import annotations

import types
from pathlib import Path
from typing import Any

import pytest

import cliffquant.full_model_verify as verification
import cliffquant.gptqmodel_export as adapter
from cliffquant.qwen35_modules import QWEN35_GPTQMODEL_TREE_REVISION
from scripts.verify_full_model_gptq import _parser


def _create_quant_module(
    *,
    name: object,
    linear_cls: object,
    bits: object,
    desc_act: object,
    dynamic: object,
    group_size: object,
    module: object,
    submodule: object,
    sym: object,
    device: object,
    lm_head_name: object,
    pack_dtype: object,
    format: object,
    backend: object,
) -> None:
    del (
        name,
        linear_cls,
        bits,
        desc_act,
        dynamic,
        group_size,
        module,
        submodule,
        sym,
        device,
        lm_head_name,
        pack_dtype,
        format,
        backend,
    )


def _pack_module(
    *,
    name: object,
    qModules: object,
    q_scales: object,
    q_zeros: object,
    q_g_idx: object,
    layers: object,
    quant_linear_cls: object,
    lock: object,
    quantize_config: object,
) -> None:
    del (
        name,
        qModules,
        q_scales,
        q_zeros,
        q_g_idx,
        layers,
        quant_linear_cls,
        lock,
        quantize_config,
    )


def _fake_modules(origin: Path) -> dict[str, Any]:
    def module(**values: object) -> types.SimpleNamespace:
        return types.SimpleNamespace(__file__=str(origin), **values)

    return {
        "auto": module(GPTQModel=object()),
        "backend": module(BACKEND=object()),
        "config": module(QuantizeConfig=object(), METHOD=object(), FORMAT=object()),
        "constants": module(DEVICE=object()),
        "model_utils": module(
            create_quant_module=_create_quant_module,
            pack_module=_pack_module,
        ),
        "package": module(__version__="7.3.4"),
        "qlinear_base": module(BaseQuantLinear=object()),
        "torch_linear": module(TorchLinear=object()),
    }


def test_installed_contract_requires_pinned_distribution_inside_active_venv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prefix = tmp_path / "venv"
    site_packages = prefix / "Lib" / "site-packages"
    package_init = site_packages / "gptqmodel" / "__init__.py"
    package_init.parent.mkdir(parents=True)
    package_init.write_text("# installed fixture\n", encoding="utf-8")
    distribution = types.SimpleNamespace(
        version="7.3.4",
        locate_file=lambda relative: site_packages / relative,
    )
    monkeypatch.setattr(adapter.sys, "prefix", str(prefix))
    monkeypatch.setattr(adapter.sys, "base_prefix", str(tmp_path / "base-python"))
    monkeypatch.setattr(
        adapter.sysconfig,
        "get_path",
        lambda key: str(site_packages) if key in {"purelib", "platlib"} else None,
    )
    monkeypatch.setattr(adapter.importlib.metadata, "distribution", lambda _name: distribution)
    monkeypatch.setattr(
        adapter.importlib.util,
        "find_spec",
        lambda _name: types.SimpleNamespace(origin=str(package_init)),
    )
    monkeypatch.setattr(
        adapter,
        "_import_gptqmodel_modules",
        lambda *, label: _fake_modules(package_init),
    )

    contract = adapter.load_installed_gptqmodel_contract()

    assert contract.version == "7.3.4"
    assert contract.revision == QWEN35_GPTQMODEL_TREE_REVISION
    assert contract.source_root == package_init.parent


def test_broken_installed_distribution_cannot_fall_back_to_checkout_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_called = False

    def broken_installed() -> None:
        raise RuntimeError("installed GPTQModel import failed")

    def forbidden_source(_source: str | Path) -> None:
        nonlocal source_called
        source_called = True
        raise AssertionError("source fallback must not run")

    monkeypatch.setattr(verification, "load_installed_gptqmodel_contract", broken_installed)
    monkeypatch.setattr(verification, "load_pinned_gptqmodel_contract", forbidden_source)

    with pytest.raises(RuntimeError, match="installed GPTQModel import failed"):
        verification.load_fresh_gptq_torch(
            checkpoint_dir=tmp_path,
            gptqmodel_source=None,
            device="cpu",
        )

    assert source_called is False


def test_smoke_cli_requires_one_explicit_gptqmodel_mode() -> None:
    parser = _parser()
    common = [
        "text",
        "--checkpoint",
        "checkpoint",
        "--report",
        "report.json",
    ]

    with pytest.raises(SystemExit):
        parser.parse_args(common)
    installed = parser.parse_args([*common, "--installed-gptqmodel"])
    assert installed.installed_gptqmodel is True
    assert installed.gptqmodel_source is None
    source = parser.parse_args([*common, "--gptqmodel-source", "source"])
    assert source.installed_gptqmodel is False
    assert source.gptqmodel_source == Path("source")
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                *common,
                "--installed-gptqmodel",
                "--gptqmodel-source",
                "source",
            ]
        )
