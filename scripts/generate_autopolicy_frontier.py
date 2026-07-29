"""Generate a measured AutoPolicy logical-byte/proxy frontier."""

from __future__ import annotations

import argparse
import json
import math
import os
import struct
import tempfile
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from cliffquant.autopolicy_artifacts import (
    ROBUST_OBJECTIVE,
    load_candidate_bundle,
    solve_candidate_bundle,
)
from cliffquant.autopolicy_experiment001 import (
    DENSE16_CANDIDATE,
    W4_CANDIDATE,
    W8_CANDIDATE,
)
from cliffquant.provenance import canonical_json_bytes, sha256_file

FRONTIER_SCHEMA = "cliffquant.autopolicy-frontier.v1"
FIGURE_PROVENANCE_SCHEMA = "cliffquant.autopolicy-figure-provenance.v1"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-figure", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--max-frontier-states", type=int, default=250_000)
    return parser


def _budgets(bundle: dict) -> list[int]:
    w4 = int(bundle["totals"][W4_CANDIDATE])
    w8 = int(bundle["totals"][W8_CANDIDATE])
    dense = int(bundle["totals"][DENSE16_CANDIDATE])
    interpolated = {
        round(w4 + fraction * (dense - w4))
        for fraction in (0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0)
    }
    return sorted({w4, w8, dense, *interpolated})


def generate(
    candidate_path: Path,
    *,
    max_frontier_states: int,
) -> dict:
    bundle = load_candidate_bundle(candidate_path)
    points = []
    for budget in _budgets(bundle):
        plan = solve_candidate_bundle(
            bundle,
            budget_bytes=budget,
            objective=ROBUST_OBJECTIVE,
            max_frontier_states=max_frontier_states,
        )
        counts = Counter(decision.candidate_id for decision in plan.decisions)
        points.append(
            {
                "budget_bytes": budget,
                "used_bytes": plan.used_bytes,
                "remaining_bytes": plan.remaining_bytes,
                "logical_bits_per_target_weight": (
                    plan.used_bytes * 8 / bundle["scope"]["weight_count"]
                ),
                "worst_environment_distortion": plan.worst_environment_distortion,
                "per_environment_distortion": {
                    environment: value
                    for environment, value in zip(
                        plan.environments,
                        plan.per_environment_distortion,
                        strict=True,
                    )
                },
                "selection_counts": {
                    candidate_id: counts.get(candidate_id, 0)
                    for candidate_id in (
                        W4_CANDIDATE,
                        W8_CANDIDATE,
                        DENSE16_CANDIDATE,
                    )
                },
                "transitions_evaluated": plan.transitions_evaluated,
                "peak_frontier_states": plan.peak_frontier_states,
            }
        )
    baseline = points[0]["worst_environment_distortion"]
    for point in points:
        point["worst_environment_distortion_reduction_vs_all_w4"] = (
            0.0 if baseline == 0.0 else 1.0 - point["worst_environment_distortion"] / baseline
        )
    return {
        "schema": FRONTIER_SCHEMA,
        "status": "complete",
        "candidate_sha256": sha256_file(candidate_path),
        "candidate_fingerprint_sha256": bundle["fingerprint_sha256"],
        "candidate_profile": bundle["profile"],
        "objective": ROBUST_OBJECTIVE,
        "budget_scope": bundle["scope"]["budget_scope"],
        "allocation_granularity": bundle["scope"].get("allocation_granularity", "matrix"),
        "allocation_unit_count": len(bundle["units"]),
        "checkpoint_export": False,
        "points": points,
    }


def validate_frontier(payload: dict) -> None:
    """Validate the portable AutoPolicy frontier summary."""

    expected_keys = {
        "schema",
        "status",
        "candidate_sha256",
        "candidate_fingerprint_sha256",
        "candidate_profile",
        "objective",
        "budget_scope",
        "allocation_granularity",
        "allocation_unit_count",
        "checkpoint_export",
        "points",
    }
    if set(payload) != expected_keys:
        raise ValueError("frontier keys do not match the v1 schema")
    if payload["schema"] != FRONTIER_SCHEMA or payload["status"] != "complete":
        raise ValueError("unsupported or incomplete AutoPolicy frontier")
    if payload["objective"] != ROBUST_OBJECTIVE:
        raise ValueError("frontier objective must be robust-minimax")
    if payload["checkpoint_export"] is not False:
        raise ValueError("frontier must not claim checkpoint export")
    unit_count = payload["allocation_unit_count"]
    if type(unit_count) is not int or unit_count <= 0:
        raise ValueError("allocation_unit_count must be a positive integer")
    points = payload["points"]
    if not isinstance(points, list) or len(points) < 2:
        raise ValueError("frontier must contain at least two points")
    previous_budget = -1
    for point in points:
        budget = point.get("budget_bytes")
        used = point.get("used_bytes")
        if type(budget) is not int or type(used) is not int or not 0 < used <= budget:
            raise ValueError("frontier point has invalid logical-byte values")
        if budget <= previous_budget:
            raise ValueError("frontier budgets must be strictly increasing")
        previous_budget = budget
        proxy = point.get("worst_environment_distortion")
        if (
            isinstance(proxy, bool)
            or not isinstance(proxy, (int, float))
            or not math.isfinite(float(proxy))
            or proxy < 0
        ):
            raise ValueError("frontier point has invalid distortion")
        counts = point.get("selection_counts")
        if not isinstance(counts, dict) or sum(counts.values()) != unit_count:
            raise ValueError("frontier selection counts do not match allocation units")


def _atomic_replace(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _plot(payload: dict, output: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("install CliffQuant with the 'figures' extra") from exc

    points = payload["points"]
    x = [point["used_bytes"] / (1024**2) for point in points]
    y = [point["worst_environment_distortion"] for point in points]
    counts = {
        candidate: [point["selection_counts"][candidate] for point in points]
        for candidate in (W4_CANDIDATE, W8_CANDIDATE, DENSE16_CANDIDATE)
    }

    plt.style.use("dark_background")
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(11.4, 7.4),
        sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.0]},
    )
    figure.patch.set_facecolor("#081015")
    for axis in axes:
        axis.set_facecolor("#081015")
        axis.grid(color="#29424b", alpha=0.38, linewidth=0.8)
        axis.spines[["top", "right"]].set_visible(False)

    axes[0].plot(x, y, color="#7ce7d8", marker="o", linewidth=2.2, markersize=5)
    axes[0].set_ylabel("Worst-environment proxy")
    axes[0].set_yscale("symlog", linthresh=1.0, linscale=0.7)

    bottom = [0] * len(points)
    colors = {
        W4_CANDIDATE: "#377f86",
        W8_CANDIDATE: "#efb65b",
        DENSE16_CANDIDATE: "#d66bb1",
    }
    labels = {
        W4_CANDIDATE: "W4 CliffQuant",
        W8_CANDIDATE: "W8 AbsMax",
        DENSE16_CANDIDATE: "BF16 reference",
    }
    for candidate in (W4_CANDIDATE, W8_CANDIDATE, DENSE16_CANDIDATE):
        axes[1].bar(
            x,
            counts[candidate],
            bottom=bottom,
            width=max(x) * 0.025,
            color=colors[candidate],
            label=labels[candidate],
        )
        bottom = [
            previous + current for previous, current in zip(bottom, counts[candidate], strict=True)
        ]
    axes[1].set_xlabel("Selected logical target payload (MiB)")
    unit_label = (
        "Transformer blocks"
        if payload["allocation_granularity"] == "transformer-block"
        else "Allocation units"
    )
    axes[1].set_ylabel(unit_label)
    axes[1].legend(
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.16),
    )
    figure.suptitle(
        "CliffQuant AutoPolicy 001",
        x=0.09,
        y=0.975,
        ha="left",
        fontsize=17,
        weight="bold",
    )
    figure.text(
        0.09,
        0.93,
        "Exact robust transformer-block allocation on measured W4, W8, and BF16 candidates",
        color="#9eb8c1",
        fontsize=10,
    )
    figure.text(
        0.09,
        0.018,
        "Planning proxy; not checkpoint size, runtime memory, latency, or downstream quality.",
        color="#8da6af",
        fontsize=8,
    )
    figure.subplots_adjust(left=0.09, right=0.98, top=0.88, bottom=0.11, hspace=0.2)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".png",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
        figure.savefig(
            temporary_path,
            dpi=180,
            facecolor=figure.get_facecolor(),
            format="png",
        )
        os.replace(temporary_path, output)
    finally:
        plt.close(figure)
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _png_dimensions(path: Path) -> tuple[int, int]:
    header = path.read_bytes()[:24]
    if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"not a PNG file: {path}")
    return struct.unpack(">II", header[16:24])


def _json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _relative_file(path: Path, *, manifest_path: Path) -> str:
    return Path(os.path.relpath(path.resolve(), manifest_path.parent.resolve())).as_posix()


def build_provenance_manifest(
    *,
    candidate_path: Path,
    frontier_path: Path,
    figure_path: Path,
    manifest_path: Path,
    candidate_fingerprint_sha256: str,
) -> dict:
    """Bind the checked-in AutoPolicy graph to its generator and exact inputs."""

    import matplotlib

    width, height = _png_dimensions(figure_path)
    generator_path = Path(__file__).resolve()
    return {
        "schema": FIGURE_PROVENANCE_SCHEMA,
        "generator": {
            "file": "scripts/generate_autopolicy_frontier.py",
            "matplotlib": matplotlib.__version__,
            "sha256": sha256_file(generator_path),
        },
        "input": {
            "candidate_fingerprint_sha256": candidate_fingerprint_sha256,
            "file": _relative_file(candidate_path, manifest_path=manifest_path),
            "sha256": sha256_file(candidate_path),
            "size_bytes": candidate_path.stat().st_size,
        },
        "outputs": {
            "figure": {
                "file": _relative_file(figure_path, manifest_path=manifest_path),
                "height_px": height,
                "sha256": sha256_file(figure_path),
                "size_bytes": figure_path.stat().st_size,
                "width_px": width,
            },
            "frontier": {
                "file": _relative_file(frontier_path, manifest_path=manifest_path),
                "sha256": sha256_file(frontier_path),
                "size_bytes": frontier_path.stat().st_size,
            },
        },
    }


def verify_provenance_manifest(manifest_path: Path) -> dict:
    """Verify every path, size, and digest recorded for the AutoPolicy graph."""

    try:
        payload = json.loads(
            manifest_path.read_text(encoding="utf-8"),
            object_pairs_hook=_json_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON number in {manifest_path}: {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid AutoPolicy figure provenance: {manifest_path}") from exc
    if not isinstance(payload, dict) or payload.get("schema") != FIGURE_PROVENANCE_SCHEMA:
        raise ValueError("unsupported AutoPolicy figure provenance manifest")
    root = manifest_path.parent.resolve()
    records = [
        payload.get("input"),
        payload.get("outputs", {}).get("figure"),
        payload.get("outputs", {}).get("frontier"),
    ]
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("AutoPolicy figure provenance record is missing")
        relative = record.get("file")
        expected_size = record.get("size_bytes")
        expected_sha256 = record.get("sha256")
        if (
            not isinstance(relative, str)
            or Path(relative).is_absolute()
            or type(expected_size) is not int
            or expected_size < 0
            or not isinstance(expected_sha256, str)
            or len(expected_sha256) != 64
        ):
            raise ValueError("invalid AutoPolicy figure provenance record")
        path = (root / relative).resolve()
        if not path.is_file():
            raise ValueError(f"missing AutoPolicy figure provenance file: {relative}")
        if path.stat().st_size != expected_size or sha256_file(path) != expected_sha256:
            raise ValueError(f"AutoPolicy figure provenance mismatch: {relative}")
    figure = payload["outputs"]["figure"]
    width, height = _png_dimensions((root / figure["file"]).resolve())
    if figure.get("width_px") != width or figure.get("height_px") != height:
        raise ValueError("AutoPolicy figure dimensions do not match provenance")

    generator = payload.get("generator")
    if not isinstance(generator, dict):
        raise ValueError("AutoPolicy figure generator provenance is missing")
    generator_path = Path(__file__).resolve()
    if generator.get("file") != "scripts/generate_autopolicy_frontier.py" or generator.get(
        "sha256"
    ) != sha256_file(generator_path):
        raise ValueError("AutoPolicy figure generator does not match provenance")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    payload = generate(
        args.candidates,
        max_frontier_states=args.max_frontier_states,
    )
    validate_frontier(payload)
    _atomic_replace(args.output_json, canonical_json_bytes(payload))
    _plot(payload, args.output_figure)
    manifest = build_provenance_manifest(
        candidate_path=args.candidates,
        frontier_path=args.output_json,
        figure_path=args.output_figure,
        manifest_path=args.output_manifest,
        candidate_fingerprint_sha256=payload["candidate_fingerprint_sha256"],
    )
    _atomic_replace(args.output_manifest, canonical_json_bytes(manifest))
    verify_provenance_manifest(args.output_manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
