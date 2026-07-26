from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

MATPLOTLIB_VERSION = "3.10.8"
SUMMARY_SCHEMA = "cliffquant.experiment-001.proxy-summary.v1"
FIGURE_MANIFEST_SCHEMA = "cliffquant.figures.v1"

BACKGROUND = "#081018"
FOREGROUND = "#E8F0F5"
MUTED = "#9AABB7"
GRID = "#263642"
CLIFF = "#65DCC2"
ABS_MAX = "#9DAEBA"
POOLED = "#7796C8"
TIE = "#687985"
LOSS = "#D99A67"

POLICIES = (
    ("absmax", "AbsMax", ABS_MAX),
    ("pooled_wmse", "Pooled-WMSE", POOLED),
    ("cliffquant_minimax", "CliffQuant minimax", CLIFF),
)
COMPARISONS = (
    ("absmax", "vs AbsMax"),
    ("pooled_wmse", "vs Pooled-WMSE"),
)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _parse_json(payload: bytes, *, source: str) -> dict[str, Any]:
    try:
        decoded = json.loads(payload, object_pairs_hook=_json_object)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"invalid JSON in {source}: {exc}") from exc
    if not isinstance(decoded, dict):
        raise ValueError(f"{source} must contain a JSON object")
    return decoded


def _checked_member(root: Path, member: str) -> Path:
    relative = Path(member)
    if (
        not member
        or relative.is_absolute()
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise ValueError(f"unsafe checksummed path: {member!r}")
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError(f"checksummed path escapes the input directory: {member!r}")
    if not candidate.is_file():
        raise ValueError(f"checksummed file is missing or not regular: {member!r}")
    return candidate


def _require_int(value: Any, *, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}")
    return value


def _require_float(value: Any, *, field: str, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        qualifier = "finite and positive" if positive else "finite"
        raise ValueError(f"{field} must be {qualifier}")
    return result


def verify_and_load_summary(input_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify every SHA256SUMS entry before parsing the summary."""
    root = input_dir.resolve()
    checksums_path = root / "SHA256SUMS.json"
    if not checksums_path.is_file():
        raise ValueError(f"missing checksum manifest: {checksums_path}")

    checksums_bytes = checksums_path.read_bytes()
    checksums = _parse_json(checksums_bytes, source="SHA256SUMS.json")
    if not checksums:
        raise ValueError("SHA256SUMS.json must not be empty")

    verified: dict[str, dict[str, Any]] = {}
    seen_files: set[str] = set()
    for logical_name in sorted(checksums):
        entry = checksums[logical_name]
        if not isinstance(entry, Mapping):
            raise ValueError(f"checksum entry {logical_name!r} must be an object")
        filename = entry.get("file")
        expected_hash = entry.get("sha256")
        expected_size = entry.get("size_bytes")
        if not isinstance(filename, str):
            raise ValueError(f"checksum entry {logical_name!r} has no valid file")
        if filename in seen_files:
            raise ValueError(f"duplicate checksummed file: {filename!r}")
        seen_files.add(filename)
        if (
            not isinstance(expected_hash, str)
            or len(expected_hash) != 64
            or any(character not in "0123456789abcdef" for character in expected_hash)
        ):
            raise ValueError(f"checksum entry {logical_name!r} has no canonical SHA-256")
        size = _require_int(
            expected_size,
            field=f"SHA256SUMS.{logical_name}.size_bytes",
            minimum=0,
        )
        path = _checked_member(root, filename)
        actual_size = path.stat().st_size
        actual_hash = _sha256_file(path)
        if actual_size != size or actual_hash != expected_hash:
            raise ValueError(
                f"checksum verification failed for {filename}: "
                f"expected {expected_hash}/{size}, got {actual_hash}/{actual_size}"
            )
        verified[logical_name] = {
            "file": filename,
            "sha256": actual_hash,
            "size_bytes": actual_size,
        }

    summary_entry = verified.get("summary")
    if summary_entry is None or summary_entry["file"] != "summary.json":
        raise ValueError("SHA256SUMS.json must bind the summary key to summary.json")
    summary_path = _checked_member(root, "summary.json")
    summary_bytes = summary_path.read_bytes()
    if (
        len(summary_bytes) != summary_entry["size_bytes"]
        or _sha256_bytes(summary_bytes) != summary_entry["sha256"]
    ):
        raise ValueError("summary.json changed after checksum verification")

    provenance = {
        "checksums_manifest": {
            "file": "SHA256SUMS.json",
            "sha256": _sha256_bytes(checksums_bytes),
            "size_bytes": len(checksums_bytes),
        },
        "verified_files": verified,
    }
    return _parse_json(summary_bytes, source="summary.json"), provenance


def _validated_metrics(summary: Mapping[str, Any]) -> dict[str, Any]:
    if summary.get("schema") != SUMMARY_SCHEMA:
        raise ValueError(f"summary schema must be {SUMMARY_SCHEMA!r}")
    if summary.get("proxy_gate_passed") is not True:
        raise ValueError("figures are only generated from a finalized passing proxy summary")

    group_count = _require_int(summary.get("group_count"), field="group_count", minimum=1)
    module_count = _require_int(summary.get("module_count"), field="module_count", minimum=1)
    environments = summary.get("environments")
    if (
        not isinstance(environments, list)
        or not environments
        or any(not isinstance(item, str) or not item for item in environments)
    ):
        raise ValueError("environments must be a non-empty list of names")

    raw_policies = summary.get("policies")
    if not isinstance(raw_policies, Mapping):
        raise ValueError("policies must be an object")
    policy_values: dict[str, float] = {}
    for policy, _, _ in POLICIES:
        entry = raw_policies.get(policy)
        if not isinstance(entry, Mapping):
            raise ValueError(f"missing policy metrics: {policy}")
        policy_values[policy] = _require_float(
            entry.get("module_macro_heldout_max_loss"),
            field=f"policies.{policy}.module_macro_heldout_max_loss",
            positive=True,
        )

    raw_comparisons = summary.get("comparisons")
    if not isinstance(raw_comparisons, Mapping):
        raise ValueError("comparisons must be an object")
    comparison_values: dict[str, dict[str, Any]] = {}
    for baseline, _ in COMPARISONS:
        entry = raw_comparisons.get(baseline)
        if not isinstance(entry, Mapping) or entry.get("gate_passed") is not True:
            raise ValueError(f"comparison {baseline!r} is missing or did not pass")
        counts = entry.get("wins_ties_losses")
        if not isinstance(counts, Mapping):
            raise ValueError(f"comparison {baseline!r} has no win/tie/loss counts")
        wins = _require_int(counts.get("wins"), field=f"{baseline}.wins")
        ties = _require_int(counts.get("ties"), field=f"{baseline}.ties")
        losses = _require_int(counts.get("losses"), field=f"{baseline}.losses")
        if wins + ties + losses != group_count:
            raise ValueError(f"comparison {baseline!r} counts do not sum to group_count")

        ci = entry.get("bootstrap_95_ci")
        if not isinstance(ci, list) or len(ci) != 2:
            raise ValueError(f"comparison {baseline!r} has no two-sided bootstrap CI")
        lower = _require_float(ci[0], field=f"{baseline}.bootstrap_95_ci[0]", positive=True)
        upper = _require_float(ci[1], field=f"{baseline}.bootstrap_95_ci[1]", positive=True)
        mean = _require_float(
            entry.get("paired_mean_improvement"),
            field=f"{baseline}.paired_mean_improvement",
            positive=True,
        )
        macro_difference = _require_float(
            entry.get("baseline_minus_cliff_module_macro"),
            field=f"{baseline}.baseline_minus_cliff_module_macro",
            positive=True,
        )
        if not lower <= mean <= upper:
            raise ValueError(f"comparison {baseline!r} mean is outside its bootstrap CI")
        if not math.isclose(mean, macro_difference, rel_tol=1e-10, abs_tol=1e-15):
            raise ValueError(f"comparison {baseline!r} paired means disagree")

        method = entry.get("bootstrap_method")
        if not isinstance(method, Mapping):
            raise ValueError(f"comparison {baseline!r} has no bootstrap method")
        replicates = _require_int(
            method.get("replicates"),
            field=f"{baseline}.bootstrap_method.replicates",
            minimum=1,
        )
        comparison_values[baseline] = {
            "bootstrap_95_ci": [lower, upper],
            "bootstrap_replicates": replicates,
            "losses": losses,
            "mean_improvement": mean,
            "ties": ties,
            "wins": wins,
        }

    return {
        "comparison": comparison_values,
        "environment_count": len(environments),
        "group_count": group_count,
        "module_count": module_count,
        "policy": policy_values,
    }


def _plotting() -> tuple[Any, Any, Any]:
    import matplotlib

    if matplotlib.__version__ != MATPLOTLIB_VERSION:
        raise RuntimeError(
            f"figure generation requires matplotlib=={MATPLOTLIB_VERSION}; "
            f"found {matplotlib.__version__}"
        )
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    return matplotlib, plt, ticker


def _apply_style(plt: Any) -> None:
    plt.rcParams.update(
        {
            "axes.edgecolor": GRID,
            "axes.facecolor": BACKGROUND,
            "axes.labelcolor": MUTED,
            "axes.titlecolor": FOREGROUND,
            "figure.facecolor": BACKGROUND,
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "savefig.facecolor": BACKGROUND,
            "text.color": FOREGROUND,
            "xtick.color": MUTED,
            "ytick.color": FOREGROUND,
        }
    )


def _finish_axis(axis: Any, *, grid_axis: str = "x") -> None:
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.grid(axis=grid_axis, color=GRID, linewidth=0.8, alpha=0.75)
    axis.set_axisbelow(True)
    axis.tick_params(length=0, pad=8)


def _save_figure(figure: Any, destination: Path, *, width: int, height: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    dpi = 200
    figure.set_size_inches(width / dpi, height / dpi)
    temporary = destination.with_name(f".{destination.stem}.tmp.png")
    figure.savefig(
        temporary,
        dpi=dpi,
        facecolor=BACKGROUND,
        format="png",
        metadata={"Software": "CliffQuant reproducible figure generator"},
    )
    os.replace(temporary, destination)


def _policy_figure(metrics: Mapping[str, Any], destination: Path) -> None:
    _, plt, ticker = _plotting()
    _apply_style(plt)
    figure, axis = plt.subplots()
    figure.subplots_adjust(left=0.245, right=0.93, top=0.72, bottom=0.2)

    values = [float(metrics["policy"][key]) for key, _, _ in POLICIES]
    scaled = [value * 1e4 for value in values]
    names = [name for _, name, _ in POLICIES]
    colors = [color for _, _, color in POLICIES]
    positions = list(range(len(POLICIES)))

    axis.barh(positions, scaled, height=0.5, color=colors, edgecolor="none")
    axis.set_yticks(positions, labels=names)
    axis.invert_yaxis()
    axis.set_xlim(0.0, max(scaled) * 1.72)
    axis.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5, min_n_ticks=4))
    axis.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
    axis.set_xlabel("Weighted MSE (x 10⁻⁴) · lower is better", labelpad=14)
    _finish_axis(axis)

    baseline = values[0]
    for index, ((policy, _, _), raw_value) in enumerate(zip(POLICIES, values, strict=True)):
        suffix = "baseline"
        if index:
            reduction = 100.0 * (baseline - raw_value) / baseline
            suffix = f"{reduction:.1f}% lower than AbsMax"
        if policy == "cliffquant_minimax":
            pooled_reduction = 100.0 * (values[1] - raw_value) / values[1]
            suffix = (
                f"{pooled_reduction:.2f}% lower than Pooled-WMSE\n"
                f"{reduction:.1f}% lower than AbsMax"
            )
        axis.text(
            max(scaled) * 1.025,
            index,
            f"{raw_value:.6g}  ·  {suffix}",
            va="center",
            ha="left",
            color=FOREGROUND,
            fontsize=11,
            fontweight="medium",
        )

    figure.text(
        0.075,
        0.91,
        "Worst-environment held-out proxy error",
        fontsize=23,
        fontweight="medium",
        color=FOREGROUND,
    )
    figure.text(
        0.075,
        0.855,
        "Module-macro mean of each group's maximum weighted MSE across held-out environments",
        fontsize=11.5,
        color=MUTED,
    )
    figure.text(
        0.075,
        0.075,
        (
            f"Experiment 001 proxy · {metrics['group_count']:,} groups · "
            f"{metrics['module_count']:,} modules · "
            f"{metrics['environment_count']} environments"
        ),
        fontsize=10.5,
        color=MUTED,
    )

    _save_figure(figure, destination, width=2400, height=1350)
    plt.close(figure)


def _paired_figure(metrics: Mapping[str, Any], destination: Path) -> None:
    _, plt, ticker = _plotting()
    _apply_style(plt)
    figure = plt.figure()
    grid = figure.add_gridspec(
        2,
        1,
        height_ratios=(1.0, 1.0),
        left=0.23,
        right=0.93,
        top=0.78,
        bottom=0.18,
        hspace=0.72,
    )
    outcome_axis = figure.add_subplot(grid[0])
    bootstrap_axis = figure.add_subplot(grid[1])

    labels: list[str] = []
    for row, (baseline, display_name) in enumerate(COMPARISONS):
        comparison = metrics["comparison"][baseline]
        counts = (
            int(comparison["wins"]),
            int(comparison["ties"]),
            int(comparison["losses"]),
        )
        labels.append(f"{display_name}\n{counts[0]:,} W · {counts[1]:,} T · {counts[2]:,} L")
        left = 0.0
        for name, count, color in zip(
            ("Better", "Tie", "Worse"),
            counts,
            (CLIFF, TIE, LOSS),
            strict=True,
        ):
            percentage = 100.0 * count / metrics["group_count"]
            outcome_axis.barh(
                row,
                percentage,
                left=left,
                height=0.5,
                color=color,
                edgecolor=BACKGROUND,
                linewidth=1.0,
            )
            if percentage >= 7.0:
                outcome_axis.text(
                    left + percentage / 2.0,
                    row,
                    f"{name} {percentage:.1f}%",
                    ha="center",
                    va="center",
                    color=BACKGROUND if name == "Better" else FOREGROUND,
                    fontsize=10.5,
                    fontweight="medium",
                )
            left += percentage

    outcome_axis.set_yticks(range(len(COMPARISONS)), labels=labels)
    outcome_axis.invert_yaxis()
    outcome_axis.set_xlim(0.0, 100.0)
    outcome_axis.xaxis.set_major_locator(ticker.MultipleLocator(20))
    outcome_axis.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=100, decimals=0))
    outcome_axis.set_xlabel("Share of evaluated groups", labelpad=12)
    outcome_axis.set_title(
        "Paired group outcomes · W/T/L = CliffQuant better / exact tie / worse",
        loc="left",
        pad=18,
        fontsize=13,
        fontweight="medium",
    )
    _finish_axis(outcome_axis)

    means: list[float] = []
    lowers: list[float] = []
    uppers: list[float] = []
    for baseline, _ in COMPARISONS:
        comparison = metrics["comparison"][baseline]
        means.append(float(comparison["mean_improvement"]) * 1e6)
        lowers.append(float(comparison["bootstrap_95_ci"][0]) * 1e6)
        uppers.append(float(comparison["bootstrap_95_ci"][1]) * 1e6)

    rows = list(range(len(COMPARISONS)))
    for row, (mean, lower, upper) in enumerate(zip(means, lowers, uppers, strict=True)):
        bootstrap_axis.errorbar(
            mean,
            row,
            xerr=[[mean - lower], [upper - mean]],
            fmt="o",
            color=CLIFF,
            ecolor=CLIFF,
            elinewidth=2.0,
            capsize=5,
            capthick=2.0,
            markersize=8,
            markeredgecolor=BACKGROUND,
            markeredgewidth=1.2,
            zorder=3,
        )
        bootstrap_axis.text(
            upper * 1.12,
            row,
            f"{mean:.4g}  [{lower:.4g}, {upper:.4g}]",
            ha="left",
            va="center",
            color=FOREGROUND,
            fontsize=10.5,
            fontweight="medium",
        )

    bootstrap_axis.set_yticks(rows, labels=[label for _, label in COMPARISONS])
    bootstrap_axis.invert_yaxis()
    bootstrap_axis.set_xscale("log")
    minimum = min(lowers)
    maximum = max(uppers)
    bootstrap_axis.set_xlim(minimum / 1.55, maximum * 2.25)
    bootstrap_axis.xaxis.set_major_locator(ticker.LogLocator(base=10.0, subs=(1.0, 2.0, 5.0)))
    bootstrap_axis.xaxis.set_major_formatter(ticker.FormatStrFormatter("%g"))
    bootstrap_axis.xaxis.set_minor_formatter(ticker.NullFormatter())
    bootstrap_axis.set_xlabel(
        "Baseline minus CliffQuant module-macro weighted MSE "
        "(x 10⁻⁶, log scale) · positive favors CliffQuant",
        labelpad=12,
    )
    bootstrap_replicates = {
        int(metrics["comparison"][baseline]["bootstrap_replicates"]) for baseline, _ in COMPARISONS
    }
    if len(bootstrap_replicates) != 1:
        raise ValueError("comparisons use inconsistent bootstrap replicate counts")
    replicates = bootstrap_replicates.pop()
    bootstrap_axis.set_title(
        f"Module-stratified bootstrap mean improvement · 95% CI · {replicates:,} replicates",
        loc="left",
        pad=18,
        fontsize=13,
        fontweight="medium",
    )
    _finish_axis(bootstrap_axis)

    figure.text(
        0.07,
        0.925,
        "Held-out paired evidence",
        fontsize=23,
        fontweight="medium",
        color=FOREGROUND,
    )
    figure.text(
        0.07,
        0.875,
        "Every group is evaluated under the same frozen calibration and held-out contract",
        fontsize=11.5,
        color=MUTED,
    )
    figure.text(
        0.07,
        0.022,
        (
            f"Experiment 001 proxy · {metrics['group_count']:,} paired groups across "
            f"{metrics['module_count']:,} modules"
        ),
        fontsize=10.5,
        color=MUTED,
    )

    _save_figure(figure, destination, width=2400, height=1500)
    plt.close(figure)


def _png_dimensions(path: Path) -> tuple[int, int]:
    header = path.read_bytes()[:24]
    if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"not a PNG file: {path}")
    return struct.unpack(">II", header[16:24])


def generate_figures(input_dir: Path, output_dir: Path) -> dict[str, Any]:
    summary, provenance = verify_and_load_summary(input_dir)
    metrics = _validated_metrics(summary)

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    destinations = {
        "policy_worst_environment": output_dir / "proxy-policy-comparison.png",
        "paired_evidence": output_dir / "proxy-paired-evidence.png",
    }
    _policy_figure(metrics, destinations["policy_worst_environment"])
    _paired_figure(metrics, destinations["paired_evidence"])

    outputs: dict[str, dict[str, Any]] = {}
    for logical_name, path in sorted(destinations.items()):
        width, height = _png_dimensions(path)
        outputs[logical_name] = {
            "file": path.name,
            "height_px": height,
            "sha256": _sha256_file(path),
            "size_bytes": path.stat().st_size,
            "width_px": width,
        }

    manifest = {
        "data": metrics,
        "figures": outputs,
        "generator": {
            "matplotlib": MATPLOTLIB_VERSION,
            "script": "scripts/generate_figures.py",
            "script_sha256": _sha256_file(Path(__file__).resolve()),
        },
        "input": provenance,
        "schema": FIGURE_MANIFEST_SCHEMA,
    }
    manifest_path = output_dir / "figure-manifest.json"
    manifest_payload = (
        json.dumps(manifest, sort_keys=True, ensure_ascii=True, separators=(",", ":")) + "\n"
    ).encode()
    temporary_manifest = output_dir / ".figure-manifest.tmp.json"
    temporary_manifest.write_bytes(manifest_payload)
    os.replace(temporary_manifest, manifest_path)
    return manifest


def _parser() -> argparse.ArgumentParser:
    repository_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Generate reproducible README figures from verified Experiment 001 artifacts."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=repository_root / "artifacts" / "experiment-001" / "proxy",
        help="Directory containing SHA256SUMS.json and the finalized proxy artifacts.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repository_root / "figures",
        help="Destination for PNG figures and figure-manifest.json.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    manifest = generate_figures(arguments.input_dir, arguments.output_dir)
    for figure in manifest["figures"].values():
        print(f"{figure['sha256']}  {figure['file']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
