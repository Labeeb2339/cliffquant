"""Install a built distribution away from the checkout and exercise the real CLI."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
import venv
from collections.abc import Sequence
from pathlib import Path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--expected-version", default="0.2.0")
    return parser


def _run(command: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> None:
    try:
        subprocess.run(
            command,
            check=True,
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        details = "\n".join(part for part in (exc.stdout.strip(), exc.stderr.strip()) if part)
        raise RuntimeError(
            f"command failed with exit code {exc.returncode}: {command!r}\n{details}"
        ) from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    artifact = args.artifact.resolve()
    if not artifact.is_file():
        raise FileNotFoundError(f"distribution artifact does not exist: {artifact}")

    with tempfile.TemporaryDirectory(prefix="cliffquant-installed-smoke-") as temporary:
        root = Path(temporary).resolve()
        environment_dir = root / "venv"
        output = root / "output"
        venv.EnvBuilder(with_pip=True).create(environment_dir)
        installed_python = (
            environment_dir / "Scripts" / "python.exe"
            if os.name == "nt"
            else environment_dir / "bin" / "python"
        )
        _run(
            [
                str(installed_python),
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                str(artifact),
            ],
            cwd=root,
        )
        _run([str(installed_python), "-m", "pip", "check"], cwd=root)

        env = os.environ.copy()
        env["PYTHONNOUSERSITE"] = "1"
        env.pop("PYTHONPATH", None)
        env["PYTHONSAFEPATH"] = "1"
        import_check = (
            "from pathlib import Path; "
            "import cliffquant; "
            "from cliffquant.corpus import collection_contract, "
            "collection_contract_is_supported; "
            f"environment=Path({str(environment_dir)!r}).resolve(); "
            "loaded=Path(cliffquant.__file__).resolve(); "
            "assert loaded.is_relative_to(environment), (loaded, environment); "
            f"assert cliffquant.__version__ == {args.expected_version!r}, "
            "(cliffquant.__version__, loaded); "
            "contract=collection_contract(); "
            "assert collection_contract_is_supported(contract); "
            "assert contract['sources']['protocol'] == "
            "{'file':'EXPERIMENT_001_PROTOCOL.md',"
            "'sha256':'8fb027b720798401a10f3eeef5501e924de5e619c4f5b152bbdf61079c345cb0',"
            "'size_bytes':17663}"
        )
        _run([str(installed_python), "-I", "-c", import_check], cwd=root, env=env)

        installed_cli = (
            environment_dir / "Scripts" / "cliffquant.exe"
            if os.name == "nt"
            else environment_dir / "bin" / "cliffquant"
        )
        _run([str(installed_cli), "--version"], cwd=root, env=env)
        _run([str(installed_cli), "autopolicy", "--help"], cwd=root, env=env)

        autopolicy_dir = root / "autopolicy"
        autopolicy_dir.mkdir()
        candidate_path = autopolicy_dir / "candidates.json"
        plan_path = autopolicy_dir / "plan.json"
        fixture_code = (
            "from pathlib import Path; "
            "from cliffquant.autopolicy_artifacts import "
            "CANDIDATE_SCHEMA,write_candidate_bundle; "
            f"path=Path({str(candidate_path)!r}); "
            "write_candidate_bundle(path,{"
            "'schema':CANDIDATE_SCHEMA,'status':'complete','profile':'wheel-smoke',"
            "'scope':{'budget_scope':'logical-target-weight-and-scale-payload',"
            "'checkpoint_export':False},"
            "'source':{'kind':'synthetic'},'environments':['general','code'],"
            "'distortion_metric':{'name':'synthetic','additive_across_units':True,"
            "'downstream_metric':False},"
            "'candidate_definitions':{'dense16':{},'w4':{}},"
            "'units':[{'unit_id':'layer.0','weight_key':'layer.0.weight',"
            "'shape':[8,128],'group_count':8,'candidates':["
            "{'candidate_id':'dense16','logical_bytes':2048,"
            "'groupwise_max_distortion':0.0,"
            "'per_environment_distortion':{'general':0.0,'code':0.0},"
            "'metadata':{'bits':16}},"
            "{'candidate_id':'w4','logical_bytes':528,"
            "'groupwise_max_distortion':2.0,"
            "'per_environment_distortion':{'general':2.0,'code':1.0},"
            "'metadata':{'bits':4}}],"
            "'evidence':{'kind':'synthetic'}}],"
            "'totals':{'dense16':2048,'w4':528},"
            "'builder':{'name':'wheel-smoke'}})"
        )
        _run([str(installed_python), "-I", "-c", fixture_code], cwd=root, env=env)
        _run(
            [
                str(installed_cli),
                "autopolicy",
                "solve",
                "--candidates",
                str(candidate_path),
                "--budget-bytes",
                "528",
                "--output",
                str(plan_path),
            ],
            cwd=root,
            env=env,
        )
        _run(
            [
                str(installed_cli),
                "autopolicy",
                "verify",
                "--candidates",
                str(candidate_path),
                "--plan",
                str(plan_path),
            ],
            cwd=root,
            env=env,
        )
        if not plan_path.is_file():
            raise ValueError("installed AutoPolicy CLI did not create a plan")

        _run(
            [
                str(installed_python),
                "-I",
                "-m",
                "cliffquant.calibration_cli",
                "--dry-run",
                "--output-dir",
                str(output),
            ],
            cwd=root,
            env=env,
        )

        summary_path = output / "collection-summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("dry_run") is not True:
            raise ValueError("installed calibration CLI did not report a dry run")
        if set(summary.get("phases", {})) != {"calibration", "heldout"}:
            raise ValueError("installed calibration CLI did not produce both phases")
        required = (
            output / "calibration.manifest.json",
            output / "heldout.manifest.json",
        )
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise ValueError(f"installed calibration CLI outputs are missing: {missing}")
        contracts = [
            json.loads(path.read_text(encoding="utf-8"))["source_loading"]["contract"]
            for path in required
        ]
        if contracts[0] != contracts[1]:
            raise ValueError("installed calibration CLI phases recorded different contracts")
        sources = contracts[0]["sources"]
        expected_protocol = {
            "file": "EXPERIMENT_001_PROTOCOL.md",
            "sha256": "8fb027b720798401a10f3eeef5501e924de5e619c4f5b152bbdf61079c345cb0",
            "size_bytes": 17_663,
        }
        if sources.get("protocol") != expected_protocol:
            raise ValueError("installed calibration CLI protocol identity drifted")
        expected_corpus = {
            "file": "corpus.py",
            "sha256": "b7783f2472e8dee85c915d37bd2c82cb7be60f4f45ce3e4e0e96cf48b1247ceb",
            "size_bytes": 33_391,
        }
        if sources.get("corpus_contract") != expected_corpus:
            raise ValueError("installed calibration CLI corpus source identity drifted")

    print(
        json.dumps(
            {
                "artifact": artifact.name,
                "artifact_sha256": _sha256(artifact),
                "expected_version": args.expected_version,
                "status": "pass",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
