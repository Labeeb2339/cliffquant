"""Create a new pinned venv and run publication-gate GPTQ smoke checks."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .full_model_verify import (
    CLEAN_BUILD_SOURCE_SCHEMA,
    CLEAN_ENVIRONMENT_SCHEMA,
    _file_descriptor,
    _write_canonical_report,
    validate_clean_build_source_identity,
)
from .provenance import canonical_json_bytes, sha256_bytes, sha256_file
from .qwen35_modules import QWEN35_GPTQMODEL_TREE_REVISION

PINNED_PYTHON_VERSION = "3.11.15"
PINNED_PIP_VERSION = "26.1.2"
PINNED_SETUPTOOLS_VERSION = "81.0.0"
PINNED_WHEEL_VERSION = "0.47.0"
PINNED_GPTQMODEL_VERSION = "7.3.4"
GPTQMODEL_REPOSITORY = "https://github.com/ModelCloud/GPTQModel.git"

_ALLOWED_ENVIRONMENT_KEYS = frozenset(
    {
        "ALLUSERSPROFILE",
        "ALL_PROXY",
        "APPDATA",
        "COMSPEC",
        "CUDA_HOME",
        "CUDA_PATH",
        "CUDA_VISIBLE_DEVICES",
        "CURL_CA_BUNDLE",
        "HOMEDRIVE",
        "HOMEPATH",
        "HOME",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "LANG",
        "LC_ALL",
        "LD_LIBRARY_PATH",
        "LIBRARY_PATH",
        "LOCALAPPDATA",
        "NO_PROXY",
        "NUMBER_OF_PROCESSORS",
        "NVIDIA_VISIBLE_DEVICES",
        "OS",
        "PATH",
        "PATHEXT",
        "PROCESSOR_ARCHITECTURE",
        "PROCESSOR_IDENTIFIER",
        "PROGRAMDATA",
        "PROGRAMFILES",
        "PROGRAMFILES(X86)",
        "PROGRAMW6432",
        "REQUESTS_CA_BUNDLE",
        "SSL_CERT_FILE",
        "SYSTEMDRIVE",
        "SYSTEMROOT",
        "TEMP",
        "TMP",
        "TMPDIR",
        "USERPROFILE",
        "WINDIR",
    }
)
_SHA_PATTERN = re.compile(r"[0-9a-f]{40}(?:[0-9a-f]{24})?")
_REQUIREMENT_HASH = re.compile(r"--hash=sha256:[0-9a-f]{64}")
_CANONICAL_HASHED_LOCK_DIRECTIVES = (
    "--require-hashes",
    "--only-binary=:all:",
)


def _run(
    command: Sequence[str | Path],
    *,
    cwd: Path,
    environment: dict[str, str],
    capture: bool = False,
) -> str:
    completed = subprocess.run(
        [str(item) for item in command],
        cwd=cwd,
        env=environment,
        check=False,
        capture_output=capture,
        text=True,
        encoding="utf-8",
    )
    if completed.returncode != 0:
        detail = ""
        if capture:
            detail = f"\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        raise RuntimeError(
            f"clean-environment command failed ({completed.returncode}): "
            f"{' '.join(str(item) for item in command)}{detail}"
        )
    return completed.stdout.strip() if capture else ""


def _sanitized_environment(
    source: Mapping[str, str],
    *,
    workspace: Path | None = None,
) -> dict[str, str]:
    """Copy only host facts required for networking, CUDA, and process startup."""

    environment = {
        key.upper(): value
        for key, value in source.items()
        if key.upper() in _ALLOWED_ENVIRONMENT_KEYS
    }
    environment.setdefault("PATH", os.defpath)
    environment.update(
        {
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "GCM_INTERACTIVE": "Never",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_TERMINAL_PROMPT": "0",
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PIP_NO_INPUT": "1",
            "PYTHONHASHSEED": "2339",
            "PYTHONIOENCODING": "utf-8",
            "PYTHONNOUSERSITE": "1",
            "PYTHONUTF8": "1",
        }
    )
    if workspace is not None:
        cache_root = workspace / "cache"
        temporary_root = workspace / "tmp"
        environment.update(
            {
                "HF_HOME": str(cache_root / "huggingface"),
                "TEMP": str(temporary_root),
                "TMP": str(temporary_root),
                "TMPDIR": str(temporary_root),
                "TORCH_EXTENSIONS_DIR": str(cache_root / "torch-extensions"),
                "TORCH_HOME": str(cache_root / "torch"),
                "TORCHINDUCTOR_CACHE_DIR": str(cache_root / "torch-inductor"),
                "XDG_CACHE_HOME": str(cache_root),
            }
        )
    return environment


def _venv_scripts(environment_root: Path) -> Path:
    return environment_root / ("Scripts" if os.name == "nt" else "bin")


def _activate_venv_environment(
    environment: Mapping[str, str],
    environment_root: Path,
) -> dict[str, str]:
    """Prepend only the new venv while retaining the sanitized system tool path."""

    result = dict(environment)
    scripts = _venv_scripts(environment_root)
    inherited_path = result.get("PATH", "")
    result["PATH"] = (
        str(scripts) if not inherited_path else f"{scripts}{os.pathsep}{inherited_path}"
    )
    result["VIRTUAL_ENV"] = str(environment_root)
    return result


def _isolated_python_command(
    python: str | Path,
    *arguments: str | Path,
) -> list[str | Path]:
    return [python, "-I", *arguments]


def _isolated_pip_command(
    python: str | Path,
    *arguments: str | Path,
) -> list[str | Path]:
    return _isolated_python_command(python, "-m", "pip", "--isolated", *arguments)


def _resolve_git_executable(source: Mapping[str, str]) -> Path:
    source_path = next(
        (value for key, value in source.items() if key.upper() == "PATH"),
        None,
    )
    candidate = shutil.which("git", path=source_path)
    if candidate is None:
        raise FileNotFoundError("Git executable is required for clean verification")
    resolved = Path(candidate).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"resolved Git executable does not exist: {resolved}")
    return resolved


def _venv_python(environment_root: Path) -> Path:
    return _venv_scripts(environment_root) / ("python.exe" if os.name == "nt" else "python")


def _require_python(
    python: Path,
    *,
    cwd: Path,
    environment: dict[str, str],
) -> dict[str, str]:
    raw = _run(
        _isolated_python_command(
            python,
            "-c",
            (
                "import json, platform; "
                "print(json.dumps({'implementation': platform.python_implementation(), "
                "'version': platform.python_version()}))"
            ),
        ),
        cwd=cwd,
        environment=environment,
        capture=True,
    )
    identity = json.loads(raw)
    if identity != {"implementation": "CPython", "version": PINNED_PYTHON_VERSION}:
        raise RuntimeError(
            f"clean verification requires CPython {PINNED_PYTHON_VERSION}, got {identity!r}"
        )
    return identity


def _one_wheel(wheel_directory: Path, pattern: str) -> Path:
    matches = sorted(wheel_directory.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"expected one wheel matching {pattern}, found {matches}")
    return matches[0]


def _is_within(path: Path, parent: Path) -> bool:
    return path == parent or parent in path.parents


def _require_disjoint(path: Path, other: Path, *, label: str) -> None:
    if _is_within(path, other) or _is_within(other, path):
        raise ValueError(f"{label} must not overlap")


def _requirement_records(lines: Sequence[str]) -> list[str]:
    records: list[str] = []
    pending = ""
    for raw_line in lines:
        value = raw_line.strip()
        if not value or value.startswith("#"):
            continue
        continued = value.endswith("\\")
        if continued:
            value = value[:-1].rstrip()
        pending = f"{pending} {value}".strip()
        if not continued:
            records.append(pending)
            pending = ""
    if pending:
        raise ValueError("verification lock ends with an incomplete continuation")
    return records


def _package_map(lines: Sequence[str]) -> dict[str, str]:
    packages: dict[str, str] = {}
    for value in _requirement_records(lines):
        if value.startswith("--"):
            continue
        requirement, *options = value.split()
        if (
            requirement.count("==") != 1
            or " @ " in value
            or requirement.startswith("-e")
            or any(_REQUIREMENT_HASH.fullmatch(option) is None for option in options)
        ):
            raise ValueError(f"package is not exactly version-pinned: {value}")
        name, version = requirement.split("==", 1)
        normalized_name = re.sub(r"[-_.]+", "-", name).casefold()
        if not normalized_name or not version or normalized_name in packages:
            raise ValueError(f"duplicate or invalid package pin: {value}")
        packages[normalized_name] = version
    return packages


def _lock_is_fully_hashed(lines: Sequence[str]) -> bool:
    try:
        _require_canonical_hashed_lock(lines)
    except ValueError:
        return False
    return True


def _require_canonical_hashed_lock(lines: Sequence[str]) -> list[str]:
    records = _requirement_records(lines)
    directive_count = len(_CANONICAL_HASHED_LOCK_DIRECTIVES)
    if tuple(records[:directive_count]) != _CANONICAL_HASHED_LOCK_DIRECTIVES or any(
        value.startswith("-") for value in records[directive_count:]
    ):
        raise ValueError(
            "verification hashed lock must begin with exactly --require-hashes and "
            "--only-binary=:all: and contain no other directives"
        )
    package_records = records[directive_count:]
    if not package_records or not all(
        any(_REQUIREMENT_HASH.fullmatch(option) for option in value.split()[1:])
        for value in package_records
    ):
        raise ValueError("verification lock must hash every package for wheelhouse mode")
    return package_records


def _git_output(
    git: Path,
    repository: Path,
    arguments: Sequence[str | Path],
    *,
    cwd: Path,
    environment: dict[str, str],
) -> str:
    return _run(
        [git, "-C", repository, *arguments],
        cwd=cwd,
        environment=environment,
        capture=True,
    )


def _require_clean_repository(
    repository: Path,
    *,
    git: Path,
    cwd: Path,
    environment: dict[str, str],
    expected_commit: str | None = None,
) -> dict[str, str]:
    """Return a path-free Git identity only for an exact clean checkout."""

    commit = _git_output(
        git,
        repository,
        ("rev-parse", "--verify", "HEAD^{commit}"),
        cwd=cwd,
        environment=environment,
    )
    tree = _git_output(
        git,
        repository,
        ("rev-parse", "--verify", "HEAD^{tree}"),
        cwd=cwd,
        environment=environment,
    )
    if _SHA_PATTERN.fullmatch(commit) is None or _SHA_PATTERN.fullmatch(tree) is None:
        raise RuntimeError("source repository returned an invalid commit or tree identity")
    if expected_commit is not None and commit != expected_commit:
        raise RuntimeError(
            f"source checkout revision drift: expected {expected_commit}, got {commit}"
        )
    status = _git_output(
        git,
        repository,
        ("status", "--porcelain=v1", "--untracked-files=all"),
        cwd=cwd,
        environment=environment,
    )
    if status:
        raise RuntimeError("publication source repository must be completely clean")
    return {"commit": commit, "tree": tree}


def _clone_clean_source(
    source: Path,
    destination: Path,
    *,
    identity: Mapping[str, str],
    git: Path,
    cwd: Path,
    environment: dict[str, str],
) -> dict[str, str]:
    _run(
        [git, "clone", "--no-hardlinks", "--no-checkout", "--", source, destination],
        cwd=cwd,
        environment=environment,
    )
    _run(
        [git, "-C", destination, "checkout", "--detach", identity["commit"]],
        cwd=cwd,
        environment=environment,
    )
    cloned = _require_clean_repository(
        destination,
        git=git,
        cwd=cwd,
        environment=environment,
        expected_commit=identity["commit"],
    )
    if cloned != dict(identity):
        raise RuntimeError("detached source clone tree does not match its origin")
    return cloned


def _dependency_install_command(
    python: Path,
    *,
    lock: Path,
    wheelhouse: Path | None,
    allow_unhashed_lock: bool,
) -> list[str | Path]:
    command = _isolated_pip_command(
        python,
        "install",
        "--quiet",
        "--no-cache-dir",
        "--no-deps",
    )
    lines = lock.read_text(encoding="utf-8").splitlines()
    if wheelhouse is None:
        if not allow_unhashed_lock:
            raise ValueError(
                "publication verification requires a committed hashed lock plus "
                "an external wheelhouse"
            )
    else:
        if not wheelhouse.is_dir():
            raise FileNotFoundError(f"verification wheelhouse does not exist: {wheelhouse}")
        _require_canonical_hashed_lock(lines)
        command.extend(
            [
                "--no-index",
                "--find-links",
                wheelhouse,
                "--require-hashes",
                "--only-binary=:all:",
            ]
        )
    command.extend(["--requirement", lock])
    return command


def _installed_package_inventory(
    python: Path,
    *,
    environment_root: Path,
    cwd: Path,
    environment: dict[str, str],
) -> list[str]:
    """Inventory metadata and prove imports resolve inside the new venv."""

    probe = (
        "import importlib.metadata as m, importlib.util, json, pathlib, sys, sysconfig; "
        "expected=pathlib.Path(sys.argv[1]).resolve(); "
        "prefix=pathlib.Path(sys.prefix).resolve(); "
        "roots={pathlib.Path(sysconfig.get_path(k)).resolve() for k in ('purelib','platlib')}; "
        "inside=lambda p:any(p==r or r in p.parents for r in roots); "
        "origins={}; "
        "exec(\"for name in ('cliffquant','gptqmodel'):\\n"
        " spec=importlib.util.find_spec(name)\\n"
        " if spec is None or spec.origin is None: raise RuntimeError(f'missing import: {name}')\\n"
        " origin=pathlib.Path(spec.origin).resolve()\\n"
        " if not inside(origin): raise RuntimeError(f'import escaped clean venv: {name}')\\n"
        ' origins[name]=origin.name"); '
        "packages=sorted([f\"{d.metadata['Name']}=={d.version}\" for d in m.distributions() "
        "if d.metadata.get('Name')], key=str.casefold); "
        "assert prefix==expected, f'venv prefix drift: {prefix}'; "
        "print(json.dumps({'origins':origins,'packages':packages}, sort_keys=True))"
    )
    raw = _run(
        _isolated_python_command(python, "-c", probe, environment_root),
        cwd=cwd,
        environment=environment,
        capture=True,
    )
    try:
        result = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError("clean import/package probe returned invalid JSON") from exc
    if (
        not isinstance(result, dict)
        or set(result) != {"origins", "packages"}
        or not isinstance(result["origins"], dict)
        or set(result["origins"]) != {"cliffquant", "gptqmodel"}
        or not isinstance(result["packages"], list)
        or not all(isinstance(item, str) for item in result["packages"])
    ):
        raise RuntimeError("clean import/package probe returned invalid fields")
    return result["packages"]


def _build_source_report(
    *,
    cliffquant_identity: Mapping[str, str],
    cliffquant_wheel: Path,
    dependency_identity: Mapping[str, Any],
    gptqmodel_identity: Mapping[str, str],
    gptqmodel_wheel: Path,
) -> dict[str, Any]:
    payload = {
        "cliffquant": {
            **dict(cliffquant_identity),
            "wheel": _file_descriptor(cliffquant_wheel),
        },
        "dependencies": dict(dependency_identity),
        "gptqmodel": {
            **dict(gptqmodel_identity),
            "repository": GPTQMODEL_REPOSITORY,
            "wheel": _file_descriptor(gptqmodel_wheel),
        },
        "schema": CLEAN_BUILD_SOURCE_SCHEMA,
    }
    return {**payload, "sha256": sha256_bytes(canonical_json_bytes(payload))}


def _wheelhouse_identity(
    wheelhouse: Path,
    *,
    hashed_lock: Path,
    hashed_lock_logical_name: str,
) -> dict[str, Any]:
    wheelhouse_root = wheelhouse.resolve(strict=True)
    if not wheelhouse_root.is_dir():
        raise FileNotFoundError(f"verification wheelhouse does not exist: {wheelhouse_root}")
    entries = sorted(
        wheelhouse_root.iterdir(),
        key=lambda item: (item.name.casefold(), item.name),
    )
    if not entries:
        raise ValueError("verification wheelhouse contains no wheels")
    invalid: list[str] = []
    resolved_entries: list[tuple[str, Path]] = []
    for entry in entries:
        if entry.is_symlink() or not entry.is_file() or entry.suffix.casefold() != ".whl":
            invalid.append(entry.name)
            continue
        resolved = entry.resolve(strict=True)
        if resolved.parent != wheelhouse_root:
            invalid.append(entry.name)
            continue
        resolved_entries.append((entry.name, resolved))
    if invalid:
        raise ValueError(
            "verification wheelhouse must contain only regular in-directory wheel files: "
            + ", ".join(invalid)
        )
    files = [
        _file_descriptor(path, logical_name=logical_name) for logical_name, path in resolved_entries
    ]
    payload = {
        "files": files,
        "hashed_lock": _file_descriptor(
            hashed_lock,
            logical_name=hashed_lock_logical_name,
        ),
        "mode": "hashed-wheelhouse",
    }
    return {**payload, "sha256": sha256_bytes(canonical_json_bytes(payload))}


def _require_unchanged_wheelhouse(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> None:
    if dict(before) != dict(after):
        raise RuntimeError("external verification wheelhouse changed during installation")


def _build_environment_report(
    *,
    cliffquant_wheel: Path,
    gptqmodel_wheel: Path,
    lock_path: Path,
    packages: list[str],
    pip_check: str,
    python_identity: dict[str, str],
    source_build: Mapping[str, Any],
) -> dict[str, Any]:
    validated_source_build = validate_clean_build_source_identity(source_build)
    cliffquant_descriptor = _file_descriptor(cliffquant_wheel)
    gptqmodel_descriptor = _file_descriptor(gptqmodel_wheel)
    if (
        cliffquant_descriptor != validated_source_build["cliffquant"]["wheel"]
        or gptqmodel_descriptor != validated_source_build["gptqmodel"]["wheel"]
    ):
        raise ValueError("clean-environment wheels differ from clean-build source report")
    payload = {
        "cliffquant_wheel": cliffquant_descriptor,
        "created_from_empty_directory": True,
        "gptqmodel": {
            "repository": GPTQMODEL_REPOSITORY,
            "revision": QWEN35_GPTQMODEL_TREE_REVISION,
        },
        "gptqmodel_wheel": gptqmodel_descriptor,
        "lock": _file_descriptor(
            lock_path,
            logical_name="requirements/verification-cu128.txt",
        ),
        "packages": packages,
        "pip_check": pip_check,
        "python": python_identity,
        "schema": CLEAN_ENVIRONMENT_SCHEMA,
        "source_build": validated_source_build,
        "status": "pass",
    }
    return {
        **payload,
        "sha256": sha256_bytes(canonical_json_bytes(payload)),
    }


def run_clean_verification(
    *,
    checkpoint_dir: str | Path,
    output_directory: str | Path,
    work_directory: str | Path,
    python_executable: str | Path,
    lock_path: str | Path,
    prompt: str,
    device: str,
    max_new_tokens: int,
    image_path: str | Path | None = None,
    image_prompt: str = "Describe this image briefly.",
    hashed_lock_path: str | Path | None = None,
    wheelhouse: str | Path | None = None,
    allow_unhashed_lock: bool = False,
) -> dict[str, Any]:
    """Create a clean venv from a complete lock and persist runtime evidence."""

    checkpoint = Path(checkpoint_dir).resolve()
    output = Path(output_directory).resolve()
    workspace = Path(work_directory).resolve()
    python = Path(python_executable).resolve()
    lock = Path(lock_path).resolve()
    repository = Path(__file__).resolve().parents[2]
    frozen_lock = repository / "requirements" / "verification-cu128.txt"
    requested_hashed_lock = (
        Path(hashed_lock_path).resolve() if hashed_lock_path is not None else None
    )
    requested_wheelhouse = Path(wheelhouse).resolve() if wheelhouse is not None else None
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"checkpoint directory does not exist: {checkpoint}")
    if not python.is_file():
        raise FileNotFoundError(f"Python executable does not exist: {python}")
    if not lock.is_file():
        raise FileNotFoundError(f"verification lock does not exist: {lock}")
    if requested_hashed_lock is not None and not requested_hashed_lock.is_file():
        raise FileNotFoundError(f"hashed verification lock does not exist: {requested_hashed_lock}")
    if requested_wheelhouse is not None and not requested_wheelhouse.is_dir():
        raise FileNotFoundError(f"verification wheelhouse does not exist: {requested_wheelhouse}")
    if workspace.exists():
        raise FileExistsError(
            f"clean verification work directory must not exist before launch: {workspace}"
        )
    if not frozen_lock.is_file() or sha256_file(lock) != sha256_file(frozen_lock):
        raise ValueError("verification lock bytes differ from the frozen repository lock")
    _require_disjoint(output, workspace, label="evidence output and disposable workspace")
    _require_disjoint(checkpoint, workspace, label="immutable checkpoint and clean workspace")
    _require_disjoint(output, checkpoint, label="evidence output and immutable checkpoint")
    _require_disjoint(workspace, repository, label="clean workspace and source repository")
    if requested_wheelhouse is not None:
        _require_disjoint(
            requested_wheelhouse,
            checkpoint,
            label="external wheelhouse and immutable checkpoint",
        )
        _require_disjoint(
            requested_wheelhouse,
            output,
            label="external wheelhouse and evidence output",
        )
        _require_disjoint(
            requested_wheelhouse,
            workspace,
            label="external wheelhouse and disposable workspace",
        )
    if not prompt:
        raise ValueError("text smoke prompt must be non-empty")
    if type(max_new_tokens) is not int or max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be a positive integer")
    if type(allow_unhashed_lock) is not bool:
        raise TypeError("allow_unhashed_lock must be a boolean")

    host_environment = dict(os.environ)
    git = _resolve_git_executable(host_environment)
    bootstrap_environment = _sanitized_environment(host_environment)
    _require_python(python, cwd=repository, environment=bootstrap_environment)
    cliffquant_identity = _require_clean_repository(
        repository,
        git=git,
        cwd=repository,
        environment=bootstrap_environment,
    )
    try:
        lock_relative = lock.relative_to(repository)
    except ValueError as exc:
        if not allow_unhashed_lock:
            raise ValueError(
                "verification lock must be inside the clean source repository"
            ) from exc
        lock_relative = None
    hashed_lock_relative: Path | None = None
    if requested_hashed_lock is not None:
        try:
            hashed_lock_relative = requested_hashed_lock.relative_to(repository)
        except ValueError as exc:
            raise ValueError(
                "hashed verification lock must be inside the clean source repository"
            ) from exc
    if not allow_unhashed_lock and (hashed_lock_relative is None or requested_wheelhouse is None):
        raise ValueError(
            "publication verification requires a committed hashed lock plus an external wheelhouse"
        )

    workspace.mkdir(parents=True)
    (workspace / "tmp").mkdir()
    output.mkdir(parents=True, exist_ok=True)
    environment_root = workspace / "venv"
    cliffquant_source = workspace / "cliffquant-source"
    gptqmodel_source = workspace / "gptqmodel-source"
    wheel_directory = workspace / "wheels"
    wheel_directory.mkdir()
    environment = _sanitized_environment(host_environment, workspace=workspace)

    _clone_clean_source(
        repository,
        cliffquant_source,
        identity=cliffquant_identity,
        git=git,
        cwd=workspace,
        environment=environment,
    )
    install_lock = cliffquant_source / lock_relative if lock_relative is not None else lock
    install_wheelhouse = requested_wheelhouse
    install_hashed_lock = (
        cliffquant_source / hashed_lock_relative if hashed_lock_relative is not None else None
    )
    if sha256_file(install_lock) != sha256_file(lock):
        raise RuntimeError("detached source lock differs from the preflight lock")
    wheelhouse_identity_before = (
        _wheelhouse_identity(
            install_wheelhouse,
            hashed_lock=install_hashed_lock,
            hashed_lock_logical_name=hashed_lock_relative.as_posix(),
        )
        if install_wheelhouse is not None
        and install_hashed_lock is not None
        and hashed_lock_relative is not None
        else None
    )

    _run(
        _isolated_python_command(python, "-m", "venv", environment_root),
        cwd=workspace,
        environment=environment,
    )
    clean_python = _venv_python(environment_root)
    environment = _activate_venv_environment(environment, environment_root)
    python_identity = _require_python(
        clean_python,
        cwd=workspace,
        environment=environment,
    )
    if allow_unhashed_lock:
        _run(
            _isolated_pip_command(
                clean_python,
                "install",
                "--no-cache-dir",
                f"pip=={PINNED_PIP_VERSION}",
                f"setuptools=={PINNED_SETUPTOOLS_VERSION}",
                f"wheel=={PINNED_WHEEL_VERSION}",
            ),
            cwd=workspace,
            environment=environment,
        )
    _run(
        _dependency_install_command(
            clean_python,
            lock=install_hashed_lock or install_lock,
            wheelhouse=install_wheelhouse,
            allow_unhashed_lock=allow_unhashed_lock,
        ),
        cwd=workspace,
        environment=environment,
    )
    _run([git, "init", gptqmodel_source], cwd=workspace, environment=environment)
    _run(
        [git, "-C", gptqmodel_source, "remote", "add", "origin", GPTQMODEL_REPOSITORY],
        cwd=workspace,
        environment=environment,
    )
    _run(
        [
            git,
            "-C",
            gptqmodel_source,
            "fetch",
            "--depth",
            "1",
            "origin",
            QWEN35_GPTQMODEL_TREE_REVISION,
        ],
        cwd=workspace,
        environment=environment,
    )
    _run(
        [git, "-C", gptqmodel_source, "checkout", "--detach", "FETCH_HEAD"],
        cwd=workspace,
        environment=environment,
    )
    gptqmodel_identity = _require_clean_repository(
        gptqmodel_source,
        git=git,
        cwd=workspace,
        environment=environment,
        expected_commit=QWEN35_GPTQMODEL_TREE_REVISION,
    )
    _run(
        _isolated_pip_command(
            clean_python,
            "wheel",
            "--no-cache-dir",
            "--no-deps",
            "--no-build-isolation",
            "--wheel-dir",
            wheel_directory,
            gptqmodel_source,
        ),
        cwd=workspace,
        environment=environment,
    )
    _run(
        _isolated_pip_command(
            clean_python,
            "wheel",
            "--no-cache-dir",
            "--no-deps",
            "--no-build-isolation",
            "--wheel-dir",
            wheel_directory,
            cliffquant_source,
        ),
        cwd=workspace,
        environment=environment,
    )
    gptqmodel_wheel = _one_wheel(wheel_directory, "gptqmodel-*.whl")
    cliffquant_wheel = _one_wheel(wheel_directory, "cliffquant-*.whl")
    _run(
        _isolated_pip_command(
            clean_python,
            "install",
            "--no-cache-dir",
            "--no-deps",
            gptqmodel_wheel,
            cliffquant_wheel,
        ),
        cwd=workspace,
        environment=environment,
    )
    if (
        install_wheelhouse is not None
        and install_hashed_lock is not None
        and hashed_lock_relative is not None
        and wheelhouse_identity_before is not None
    ):
        wheelhouse_identity_after = _wheelhouse_identity(
            install_wheelhouse,
            hashed_lock=install_hashed_lock,
            hashed_lock_logical_name=hashed_lock_relative.as_posix(),
        )
        _require_unchanged_wheelhouse(
            wheelhouse_identity_before,
            wheelhouse_identity_after,
        )
    _require_clean_repository(
        cliffquant_source,
        git=git,
        cwd=workspace,
        environment=environment,
        expected_commit=cliffquant_identity["commit"],
    )
    _require_clean_repository(
        gptqmodel_source,
        git=git,
        cwd=workspace,
        environment=environment,
        expected_commit=QWEN35_GPTQMODEL_TREE_REVISION,
    )

    pip_check = _run(
        _isolated_pip_command(clean_python, "check"),
        cwd=workspace,
        environment=environment,
        capture=True,
    )
    packages = _installed_package_inventory(
        clean_python,
        environment_root=environment_root,
        cwd=workspace,
        environment=environment,
    )
    actual_packages = _package_map(packages)
    exact_packages = _package_map(install_lock.read_text(encoding="utf-8").splitlines())
    bootstrap_packages = {
        "pip": PINNED_PIP_VERSION,
        "setuptools": PINNED_SETUPTOOLS_VERSION,
        "wheel": PINNED_WHEEL_VERSION,
    }
    local_packages = {
        "cliffquant": "0.1.0",
        "gptqmodel": PINNED_GPTQMODEL_VERSION,
    }
    required_packages = {**bootstrap_packages, **local_packages}
    conflicting = {
        name: exact_packages[name]
        for name, version in required_packages.items()
        if name in exact_packages and exact_packages[name] != version
    }
    if conflicting:
        raise RuntimeError(f"verification lock has conflicting tool pins: {conflicting}")
    expected_installation_packages = {**exact_packages, **bootstrap_packages}
    if install_hashed_lock is not None:
        installation_packages = _package_map(
            install_hashed_lock.read_text(encoding="utf-8").splitlines()
        )
        if installation_packages != expected_installation_packages:
            raise RuntimeError(
                "hashed installation lock package set differs from the exact-version contract"
            )
    expected_packages = {**expected_installation_packages, **local_packages}
    if actual_packages != expected_packages:
        missing = sorted(set(expected_packages) - set(actual_packages))
        extra = sorted(set(actual_packages) - set(expected_packages))
        changed = sorted(
            name
            for name in set(actual_packages) & set(expected_packages)
            if actual_packages[name] != expected_packages[name]
        )
        raise RuntimeError(
            "clean environment package set differs from the frozen lock; "
            f"missing={missing}, extra={extra}, changed={changed}"
        )

    source_report_path = output / "clean-build-source.json"
    dependency_identity = (
        wheelhouse_identity_before
        if wheelhouse_identity_before is not None
        else {
            "exact_lock": _file_descriptor(
                install_lock,
                logical_name="requirements/verification-cu128.txt",
            ),
            "mode": "unhashed-test-compatibility",
        }
    )
    source_report = _build_source_report(
        cliffquant_identity=cliffquant_identity,
        cliffquant_wheel=cliffquant_wheel,
        dependency_identity=dependency_identity,
        gptqmodel_identity=gptqmodel_identity,
        gptqmodel_wheel=gptqmodel_wheel,
    )
    _write_canonical_report(source_report, source_report_path)
    environment_report_path = output / "clean-environment.json"
    environment_report = _build_environment_report(
        cliffquant_wheel=cliffquant_wheel,
        gptqmodel_wheel=gptqmodel_wheel,
        lock_path=lock,
        packages=packages,
        pip_check=pip_check,
        python_identity=python_identity,
        source_build=source_report,
    )
    _write_canonical_report(environment_report, environment_report_path)

    verifier_script = cliffquant_source / "scripts" / "verify_full_model_gptq.py"
    text_report = output / "text-smoke.json"
    _run(
        _isolated_python_command(
            clean_python,
            verifier_script,
            "text",
            "--checkpoint",
            checkpoint,
            "--installed-gptqmodel",
            "--report",
            text_report,
            "--environment-report",
            environment_report_path,
            "--prompt",
            prompt,
            "--device",
            device,
            "--max-new-tokens",
            str(max_new_tokens),
        ),
        cwd=workspace,
        environment=environment,
    )
    result = {
        "clean_environment_report": str(environment_report_path),
        "source_build_report": str(source_report_path),
        "text_report": str(text_report),
    }
    if image_path is not None:
        image = Path(image_path).resolve()
        image_report = output / "image-smoke.json"
        _run(
            _isolated_python_command(
                clean_python,
                verifier_script,
                "image",
                "--checkpoint",
                checkpoint,
                "--installed-gptqmodel",
                "--image",
                image,
                "--report",
                image_report,
                "--environment-report",
                environment_report_path,
                "--prompt",
                image_prompt,
                "--device",
                device,
                "--max-new-tokens",
                str(max_new_tokens),
            ),
            cwd=workspace,
            environment=environment,
        )
        result["image_report"] = str(image_report)
    return result


def default_python_executable() -> Path:
    """Return the current interpreter for CLI defaults and explicit reporting."""

    return Path(sys.executable).resolve()
