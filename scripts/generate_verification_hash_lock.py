"""Generate a deterministic, wheel-only hash lock from exact version pins."""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import (
    InvalidWheelFilename,
    canonicalize_name,
    parse_wheel_filename,
)
from packaging.version import InvalidVersion, Version

_INDEX_OPTIONS = ("--index-url", "--extra-index-url")
_SDIST_SUFFIXES = (".tar.gz", ".tar.bz2", ".tar.xz", ".tgz", ".zip")


@dataclass(frozen=True, slots=True)
class ExactPin:
    """A normalized package name and an exact PEP 440 version."""

    name: str
    version: Version

    @property
    def requirement(self) -> str:
        return f"{self.name}=={self.version}"


def _strip_inline_comment(line: str) -> str:
    return re.split(r"\s+#", line, maxsplit=1)[0].strip()


def _parse_exact_pin(value: str, *, source: str) -> ExactPin:
    try:
        requirement = Requirement(value)
    except InvalidRequirement as exc:
        raise ValueError(f"{source} is not a valid requirement: {value!r}") from exc
    if requirement.url is not None:
        raise ValueError(f"{source} must not use a direct URL: {value!r}")
    if requirement.extras:
        raise ValueError(f"{source} must not select extras: {value!r}")
    if requirement.marker is not None:
        raise ValueError(f"{source} must not use an environment marker: {value!r}")
    specifiers = tuple(requirement.specifier)
    if len(specifiers) != 1 or specifiers[0].operator != "==":
        raise ValueError(f"{source} is not exactly version-pinned: {value!r}")
    version_text = specifiers[0].version
    if "*" in version_text:
        raise ValueError(f"{source} must not use a wildcard version: {value!r}")
    try:
        version = Version(version_text)
    except InvalidVersion as exc:
        raise ValueError(f"{source} has an invalid version: {value!r}") from exc
    return ExactPin(
        name=str(canonicalize_name(requirement.name)),
        version=version,
    )


def _parse_index_directive(value: str, *, line_number: int) -> tuple[str, str]:
    for option in _INDEX_OPTIONS:
        if value.startswith(f"{option}="):
            url = value[len(option) + 1 :].strip()
            break
        if value.startswith(f"{option} "):
            option_and_url = value.split(maxsplit=1)
            if len(option_and_url) != 2:
                break
            url = option_and_url[1].strip()
            break
    else:
        raise ValueError(f"unsupported directive on requirements line {line_number}: {value!r}")
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"index URL on requirements line {line_number} is invalid: {url!r}")
    if any(character.isspace() for character in url):
        raise ValueError(
            f"index URL on requirements line {line_number} contains whitespace: {url!r}"
        )
    return option, url


def _read_source_requirements(
    path: Path,
) -> tuple[dict[str, ExactPin], tuple[tuple[str, str], ...]]:
    if not path.is_file():
        raise FileNotFoundError(f"exact-version requirements file does not exist: {path}")
    pins: dict[str, ExactPin] = {}
    directives: list[tuple[str, str]] = []
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        value = _strip_inline_comment(raw_line)
        if not value or value.startswith("#"):
            continue
        if value.endswith("\\"):
            raise ValueError(
                f"line continuations are not supported on requirements line {line_number}"
            )
        if value.startswith("-"):
            directive = _parse_index_directive(value, line_number=line_number)
            if directive in directives:
                raise ValueError(
                    f"duplicate index directive on requirements line {line_number}: {value!r}"
                )
            directives.append(directive)
            continue
        pin = _parse_exact_pin(
            value,
            source=f"requirements line {line_number}",
        )
        if pin.name in pins:
            raise ValueError(f"duplicate package pin in requirements: {pin.name}")
        pins[pin.name] = pin
    if not pins:
        raise ValueError("exact-version requirements file contains no package pins")
    primary_indexes = [item for item in directives if item[0] == "--index-url"]
    if len(primary_indexes) > 1:
        raise ValueError("requirements must contain at most one --index-url directive")
    return pins, tuple(directives)


def _merge_bootstrap_pins(
    pins: dict[str, ExactPin],
    bootstrap_requirements: Sequence[str],
) -> dict[str, ExactPin]:
    merged = dict(pins)
    for index, value in enumerate(bootstrap_requirements, start=1):
        pin = _parse_exact_pin(value, source=f"bootstrap requirement {index}")
        if pin.name in merged:
            raise ValueError(f"bootstrap package duplicates an existing package pin: {pin.name}")
        merged[pin.name] = pin
    return merged


def _looks_like_sdist(path: Path) -> bool:
    lower_name = path.name.casefold()
    return any(lower_name.endswith(suffix) for suffix in _SDIST_SUFFIXES)


def _collect_wheels(
    wheelhouse: Path,
    pins: dict[str, ExactPin],
) -> dict[str, Path]:
    if not wheelhouse.is_dir():
        raise FileNotFoundError(f"wheelhouse directory does not exist: {wheelhouse}")
    matched: dict[str, list[Path]] = {}
    for path in sorted(
        (entry for entry in wheelhouse.iterdir() if entry.is_file()),
        key=lambda item: (item.name.casefold(), item.name),
    ):
        if path.suffix.casefold() != ".whl":
            kind = "source distribution" if _looks_like_sdist(path) else "non-wheel file"
            raise ValueError(f"wheelhouse contains a {kind}: {path.name}")
        try:
            wheel_name, wheel_version, _build, _tags = parse_wheel_filename(path.name)
        except InvalidWheelFilename as exc:
            raise ValueError(f"invalid wheel filename: {path.name}") from exc
        name = str(canonicalize_name(wheel_name))
        expected = pins.get(name)
        if expected is None:
            raise ValueError(f"wheelhouse contains an extra package wheel: {path.name}")
        if wheel_version != expected.version:
            raise ValueError(f"wheel version does not match {expected.requirement}: {path.name}")
        matched.setdefault(name, []).append(path)
    duplicate_names = sorted(name for name, paths in matched.items() if len(paths) != 1)
    if duplicate_names:
        details = ", ".join(
            f"{name} ({', '.join(path.name for path in matched[name])})" for name in duplicate_names
        )
        raise ValueError(f"wheelhouse contains duplicate wheels: {details}")
    missing = sorted(set(pins) - set(matched))
    if missing:
        raise ValueError(f"wheelhouse is missing wheels for: {', '.join(missing)}")
    return {name: paths[0] for name, paths in matched.items()}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_directives(directives: Sequence[tuple[str, str]]) -> list[str]:
    primary = sorted(url for option, url in directives if option == "--index-url")
    extra = sorted(url for option, url in directives if option == "--extra-index-url")
    return [
        *(f"--index-url {url}" for url in primary),
        *(f"--extra-index-url {url}" for url in extra),
    ]


def _render_lock(
    pins: dict[str, ExactPin],
    wheels: dict[str, Path],
    directives: Sequence[tuple[str, str]],
) -> str:
    lines = [
        "# Generated by scripts/generate_verification_hash_lock.py; do not edit.",
        "--require-hashes",
        "--only-binary=:all:",
        *_canonical_directives(directives),
        "",
    ]
    for name in sorted(pins):
        lines.append(f"{pins[name].requirement} --hash=sha256:{_sha256_file(wheels[name])}")
    return "\n".join(lines) + "\n"


def generate_hash_lock(
    *,
    requirements_path: str | Path,
    wheelhouse: str | Path,
    output_path: str | Path,
    bootstrap_requirements: Sequence[str] = (),
) -> str:
    """Validate a complete wheelhouse and atomically write its deterministic lock."""

    requirements = Path(requirements_path).resolve()
    wheel_directory = Path(wheelhouse).resolve()
    output = Path(output_path).resolve()
    if output == requirements:
        raise ValueError("output path must not overwrite the exact-version requirements file")
    if output.is_relative_to(wheel_directory):
        raise ValueError("output path must be outside the wheelhouse")
    source_pins, directives = _read_source_requirements(requirements)
    pins = _merge_bootstrap_pins(source_pins, bootstrap_requirements)
    wheels = _collect_wheels(wheel_directory, pins)
    rendered = _render_lock(pins, wheels, directives)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
            temporary_path = Path(handle.name)
        os.replace(temporary_path, output)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return rendered


def _parser() -> argparse.ArgumentParser:
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--requirements",
        type=Path,
        default=repository / "requirements" / "verification-cu128.txt",
        help="input file containing only exact package pins and index directives",
    )
    parser.add_argument("--wheelhouse", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--bootstrap",
        action="append",
        default=[],
        metavar="PACKAGE==VERSION",
        help="exact bootstrap pin to include; repeat for each bootstrap package",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    rendered = generate_hash_lock(
        requirements_path=args.requirements,
        wheelhouse=args.wheelhouse,
        output_path=args.output,
        bootstrap_requirements=args.bootstrap,
    )
    package_count = sum(
        1 for line in rendered.splitlines() if line and not line.startswith(("#", "--"))
    )
    print(f"wrote {package_count} hashed wheel pins to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
