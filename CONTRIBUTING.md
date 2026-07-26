# Contributing to CliffQuant

I welcome focused bug fixes, tests, reproducibility improvements, and clearly
scoped research extensions. For a large change, please open an issue first so
the design and evidence requirements can be agreed before implementation.

## Set up a development environment

CliffQuant requires Python 3.11 or newer.

```bash
python -m venv .venv
```

Activate the environment with `.venv\Scripts\Activate.ps1` in PowerShell or
`source .venv/bin/activate` on macOS and Linux, then install the project:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev,figures]"
```

## Validate a change

Run the same checks used in continuous integration:

```bash
python -m ruff check .
python -m ruff format --check .
python -m pytest --basetemp .pytest-local-contrib
python -m build
```

Use `python -m ruff format .` to apply the formatter.

## Keep results auditable

- Add a regression test for behavior changes.
- Keep calibration, held-out evaluation, and solver certification separate.
- Support quantitative statements with a reproducible script, protocol, and
  raw checksummed result where applicable.
- Report negative or incomplete results plainly; do not broaden a result beyond
  the metric and model that were tested.
- Do not commit secrets, private data, local caches, environments, or build
  outputs.

Keep each pull request narrow enough to review and explain any intentional
change to deterministic behavior or evidence provenance.
