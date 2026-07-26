from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from cliffquant.clean_environment import (
    _activate_venv_environment,
    _build_source_report,
    _clone_clean_source,
    _dependency_install_command,
    _installed_package_inventory,
    _isolated_pip_command,
    _isolated_python_command,
    _lock_is_fully_hashed,
    _package_map,
    _require_clean_repository,
    _require_disjoint,
    _require_unchanged_wheelhouse,
    _resolve_git_executable,
    _sanitized_environment,
    _wheelhouse_identity,
)
from cliffquant.full_model_verify import CLEAN_BUILD_SOURCE_SCHEMA
from cliffquant.provenance import canonical_json_bytes, sha256_bytes


def test_sanitized_environment_drops_process_injection_and_rehomes_caches(
    tmp_path: Path,
) -> None:
    source = {
        "Path": "trusted-tools",
        "SYSTEMROOT": r"C:\Windows",
        "HTTPS_PROXY": "http://proxy.invalid",
        "PYTHONPATH": "malicious-source",
        "PYTHONHOME": "malicious-runtime",
        "PIP_INDEX_URL": "https://untrusted.invalid/simple",
        "PIP_CONFIG_FILE": "untrusted.ini",
        "GIT_DIR": "other-repository",
        "GIT_CONFIG_COUNT": "1",
        "CONDA_PREFIX": "other-environment",
        "UV_PROJECT_ENVIRONMENT": "other-venv",
        "VIRTUAL_ENV": "other-venv",
        "TORCHINDUCTOR_CACHE_DIR": r"C:\shared-torch-cache",
        "SECRET_TOKEN": "must-not-cross-boundary",
    }

    environment = _sanitized_environment(source, workspace=tmp_path)

    assert environment["PATH"] == "trusted-tools"
    assert environment["SYSTEMROOT"] == r"C:\Windows"
    assert environment["HTTPS_PROXY"] == "http://proxy.invalid"
    for forbidden in (
        "PYTHONPATH",
        "PYTHONHOME",
        "PIP_INDEX_URL",
        "PIP_CONFIG_FILE",
        "GIT_DIR",
        "GIT_CONFIG_COUNT",
        "CONDA_PREFIX",
        "UV_PROJECT_ENVIRONMENT",
        "VIRTUAL_ENV",
        "SECRET_TOKEN",
    ):
        assert forbidden not in environment
    assert environment["GIT_CONFIG_NOSYSTEM"] == "1"
    assert environment["GIT_CONFIG_GLOBAL"] == os.devnull
    assert environment["PYTHONHASHSEED"] == "2339"
    assert Path(environment["HF_HOME"]).is_relative_to(tmp_path)
    assert Path(environment["TORCH_HOME"]).is_relative_to(tmp_path)
    assert Path(environment["TORCHINDUCTOR_CACHE_DIR"]).is_relative_to(tmp_path)
    assert environment["TORCHINDUCTOR_CACHE_DIR"] == str(
        tmp_path / "cache" / "torch-inductor"
    )
    assert source["PYTHONPATH"] == "malicious-source"


def test_venv_activation_and_all_python_entrypoints_are_isolated(tmp_path: Path) -> None:
    root = tmp_path / "venv"
    environment = _activate_venv_environment({"PATH": "system-tools"}, root)

    assert environment["VIRTUAL_ENV"] == str(root)
    assert environment["PATH"].split(os.pathsep)[0] == str(
        root / ("Scripts" if os.name == "nt" else "bin")
    )
    assert _isolated_python_command(Path("python"), "-c", "pass") == [
        Path("python"),
        "-I",
        "-c",
        "pass",
    ]
    assert _isolated_pip_command(Path("python"), "check") == [
        Path("python"),
        "-I",
        "-m",
        "pip",
        "--isolated",
        "check",
    ]


def test_hashed_wheelhouse_command_is_offline_and_binary_only(tmp_path: Path) -> None:
    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir()
    lock = tmp_path / "lock.txt"
    digest = "a" * 64
    lock.write_text(
        "--require-hashes\n"
        "--only-binary=:all:\n"
        "example-package==1.2.3 \\\n"
        f"    --hash=sha256:{digest}\n",
        encoding="utf-8",
    )

    assert _lock_is_fully_hashed(lock.read_text(encoding="utf-8").splitlines())
    assert _package_map(lock.read_text(encoding="utf-8").splitlines()) == {
        "example-package": "1.2.3"
    }
    command = _dependency_install_command(
        Path("python"),
        lock=lock,
        wheelhouse=wheelhouse,
        allow_unhashed_lock=False,
    )
    assert command[:6] == [
        Path("python"),
        "-I",
        "-m",
        "pip",
        "--isolated",
        "install",
    ]
    for option in ("--no-index", "--require-hashes", "--only-binary=:all:"):
        assert option in command
    assert "--quiet" in command
    assert command[command.index("--find-links") + 1] == wheelhouse


def test_publication_dependency_install_refuses_unhashed_or_online_mode(
    tmp_path: Path,
) -> None:
    lock = tmp_path / "lock.txt"
    lock.write_text(
        "--require-hashes\n--only-binary=:all:\nnumpy==2.2.6\n",
        encoding="utf-8",
    )
    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir()

    with pytest.raises(ValueError, match="hash every package"):
        _dependency_install_command(
            Path("python"),
            lock=lock,
            wheelhouse=wheelhouse,
            allow_unhashed_lock=False,
        )
    with pytest.raises(ValueError, match="hashed lock plus an external wheelhouse"):
        _dependency_install_command(
            Path("python"),
            lock=lock,
            wheelhouse=None,
            allow_unhashed_lock=False,
        )
    compatibility = _dependency_install_command(
        Path("python"),
        lock=lock,
        wheelhouse=None,
        allow_unhashed_lock=True,
    )
    assert "--require-hashes" not in compatibility
    assert compatibility[-2:] == ["--requirement", lock]


@pytest.mark.parametrize(
    "directive",
    [
        "--find-links https://attacker.invalid/wheels",
        "--find-links C:/untrusted/wheels",
        "--index-url https://attacker.invalid/simple",
        "--extra-index-url https://attacker.invalid/simple",
        "--trusted-host attacker.invalid",
        "--requirement C:/untrusted/extra.txt",
    ],
)
def test_publication_dependency_install_rejects_every_noncanonical_lock_directive(
    tmp_path: Path,
    directive: str,
) -> None:
    lock = tmp_path / "lock.txt"
    lock.write_text(
        f"--require-hashes\n--only-binary=:all:\n{directive}\ndemo==1.0 --hash=sha256:{'a' * 64}\n",
        encoding="utf-8",
    )
    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir()

    assert not _lock_is_fully_hashed(lock.read_text(encoding="utf-8").splitlines())
    with pytest.raises(ValueError, match="contain no other directives"):
        _dependency_install_command(
            Path("python"),
            lock=lock,
            wheelhouse=wheelhouse,
            allow_unhashed_lock=False,
        )


@pytest.mark.parametrize(
    "prefix",
    [
        "",
        "--require-hashes\n",
        "--only-binary=:all:\n--require-hashes\n",
        "--require-hashes\n--require-hashes\n--only-binary=:all:\n",
    ],
)
def test_publication_dependency_install_requires_exact_canonical_directive_prefix(
    tmp_path: Path,
    prefix: str,
) -> None:
    lock = tmp_path / "lock.txt"
    lock.write_text(
        prefix + f"demo==1.0 --hash=sha256:{'a' * 64}\n",
        encoding="utf-8",
    )
    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir()

    with pytest.raises(ValueError, match="begin with exactly"):
        _dependency_install_command(
            Path("python"),
            lock=lock,
            wheelhouse=wheelhouse,
            allow_unhashed_lock=False,
        )


def test_import_inventory_uses_isolated_probe_and_rejects_non_venv_origins(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_run(
        command: list[str | Path],
        *,
        cwd: Path,
        environment: dict[str, str],
        capture: bool = False,
    ) -> str:
        captured.update(
            {
                "capture": capture,
                "command": command,
                "cwd": cwd,
                "environment": environment,
            }
        )
        return json.dumps(
            {
                "origins": {
                    "cliffquant": "__init__.py",
                    "gptqmodel": "__init__.py",
                },
                "packages": ["cliffquant==0.1.0", "GPTQModel==7.3.4"],
            }
        )

    monkeypatch.setattr("cliffquant.clean_environment._run", fake_run)
    packages = _installed_package_inventory(
        Path("python"),
        environment_root=tmp_path / "venv",
        cwd=tmp_path,
        environment={"PATH": "tools"},
    )

    command = captured["command"]
    assert isinstance(command, list)
    assert command[1] == "-I"
    assert "import escaped clean venv" in str(command[3])
    compile(str(command[3]), "<clean-environment-probe>", "exec")
    assert packages == ["cliffquant==0.1.0", "GPTQModel==7.3.4"]


def _new_git_repository(root: Path, git: Path) -> None:
    root.mkdir()
    (root / "tracked.txt").write_text("frozen\n", encoding="utf-8")
    subprocess.run([git, "init", root], check=True, capture_output=True)
    subprocess.run(
        [git, "-C", root, "config", "core.autocrlf", "false"],
        check=True,
        capture_output=True,
    )
    subprocess.run([git, "-C", root, "add", "tracked.txt"], check=True, capture_output=True)
    subprocess.run(
        [
            git,
            "-C",
            root,
            "-c",
            "user.name=CliffQuant Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-m",
            "frozen",
        ],
        check=True,
        capture_output=True,
    )


def test_clean_source_identity_and_detached_clone_are_exact(tmp_path: Path) -> None:
    git = _resolve_git_executable(os.environ)
    source = tmp_path / "source"
    clone = tmp_path / "clone"
    _new_git_repository(source, git)
    environment = _sanitized_environment(os.environ)

    identity = _require_clean_repository(
        source,
        git=git,
        cwd=tmp_path,
        environment=environment,
    )
    assert set(identity) == {"commit", "tree"}
    assert (
        _clone_clean_source(
            source,
            clone,
            identity=identity,
            git=git,
            cwd=tmp_path,
            environment=environment,
        )
        == identity
    )

    (source / "untracked.txt").write_text("dirty\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="completely clean"):
        _require_clean_repository(
            source,
            git=git,
            cwd=tmp_path,
            environment=environment,
        )


def test_source_build_report_binds_commits_trees_and_wheels_without_paths(
    tmp_path: Path,
) -> None:
    cliffquant_wheel = tmp_path / "cliffquant-0.1.0.whl"
    gptqmodel_wheel = tmp_path / "gptqmodel-7.3.4.whl"
    cliffquant_wheel.write_bytes(b"cliffquant")
    gptqmodel_wheel.write_bytes(b"gptqmodel")
    report = _build_source_report(
        cliffquant_identity={"commit": "1" * 40, "tree": "2" * 40},
        cliffquant_wheel=cliffquant_wheel,
        dependency_identity={"mode": "test"},
        gptqmodel_identity={"commit": "3" * 40, "tree": "4" * 40},
        gptqmodel_wheel=gptqmodel_wheel,
    )

    assert report["schema"] == CLEAN_BUILD_SOURCE_SCHEMA
    assert report["cliffquant"]["commit"] == "1" * 40
    assert report["dependencies"] == {"mode": "test"}
    assert report["gptqmodel"]["tree"] == "4" * 40
    assert str(tmp_path) not in str(report)
    payload = {key: value for key, value in report.items() if key != "sha256"}
    assert report["sha256"] == sha256_bytes(canonical_json_bytes(payload))


def test_wheelhouse_identity_binds_lock_and_every_wheel_without_paths(
    tmp_path: Path,
) -> None:
    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir()
    (wheelhouse / "Beta-2-py3-none-any.whl").write_bytes(b"beta")
    (wheelhouse / "alpha-1-py3-none-any.whl").write_bytes(b"alpha")
    hashed_lock = tmp_path / "verification-cu128-hashed.txt"
    hashed_lock.write_text("alpha==1 --hash=sha256:" + "a" * 64 + "\n", encoding="utf-8")

    identity = _wheelhouse_identity(
        wheelhouse,
        hashed_lock=hashed_lock,
        hashed_lock_logical_name="requirements/verification-cu128-hashed.txt",
    )

    assert identity["mode"] == "hashed-wheelhouse"
    assert [item["file"] for item in identity["files"]] == [
        "alpha-1-py3-none-any.whl",
        "Beta-2-py3-none-any.whl",
    ]
    assert identity["hashed_lock"]["file"] == "requirements/verification-cu128-hashed.txt"
    assert str(tmp_path) not in str(identity)


def test_wheelhouse_identity_rejects_every_non_wheel_entry(tmp_path: Path) -> None:
    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir()
    (wheelhouse / "alpha-1-py3-none-any.whl").write_bytes(b"alpha")
    (wheelhouse / "README.txt").write_text("not installable\n", encoding="utf-8")
    hashed_lock = tmp_path / "verification-cu128-hashed.txt"
    hashed_lock.write_text("alpha==1 --hash=sha256:" + "a" * 64 + "\n", encoding="utf-8")

    with pytest.raises(
        ValueError,
        match=r"only regular in-directory wheel files: README\.txt",
    ):
        _wheelhouse_identity(
            wheelhouse,
            hashed_lock=hashed_lock,
            hashed_lock_logical_name="requirements/verification-cu128-hashed.txt",
        )


def test_wheelhouse_identity_rejects_symlinked_wheel_escape(tmp_path: Path) -> None:
    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir()
    outside = tmp_path / "checkpoint"
    outside.mkdir()
    target = outside / "demo-1-py3-none-any.whl"
    target.write_bytes(b"outside")
    link = wheelhouse / target.name
    try:
        link.symlink_to(target)
    except OSError as exc:
        pytest.skip(f"wheel symlink creation is unavailable: {exc}")
    hashed_lock = tmp_path / "verification-cu128-hashed.txt"
    hashed_lock.write_text("demo==1 --hash=sha256:" + "a" * 64 + "\n", encoding="utf-8")

    assert link.resolve().parent == outside
    with pytest.raises(ValueError, match="regular in-directory wheel files"):
        _wheelhouse_identity(
            wheelhouse,
            hashed_lock=hashed_lock,
            hashed_lock_logical_name="requirements/verification-cu128-hashed.txt",
        )


def test_external_wheelhouse_identity_must_not_change_during_install(
    tmp_path: Path,
) -> None:
    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir()
    wheel = wheelhouse / "alpha-1-py3-none-any.whl"
    wheel.write_bytes(b"before")
    hashed_lock = tmp_path / "verification-cu128-hashed.txt"
    hashed_lock.write_text("alpha==1 --hash=sha256:" + "a" * 64 + "\n", encoding="utf-8")
    before = _wheelhouse_identity(
        wheelhouse,
        hashed_lock=hashed_lock,
        hashed_lock_logical_name="requirements/verification-cu128-hashed.txt",
    )

    wheel.write_bytes(b"after")
    after = _wheelhouse_identity(
        wheelhouse,
        hashed_lock=hashed_lock,
        hashed_lock_logical_name="requirements/verification-cu128-hashed.txt",
    )

    with pytest.raises(RuntimeError, match="changed during installation"):
        _require_unchanged_wheelhouse(before, after)


def test_path_boundaries_are_rejected_in_both_directions(tmp_path: Path) -> None:
    parent = tmp_path / "parent"
    child = parent / "child"

    with pytest.raises(ValueError, match="must not overlap"):
        _require_disjoint(parent, child, label="paths")
    with pytest.raises(ValueError, match="must not overlap"):
        _require_disjoint(child, parent, label="paths")

    wheelhouse = tmp_path / "wheelhouse"
    for protected in (
        wheelhouse / "checkpoint",
        wheelhouse / "output",
        wheelhouse / "workspace",
    ):
        with pytest.raises(ValueError, match="must not overlap"):
            _require_disjoint(wheelhouse, protected, label="external wheelhouse")
