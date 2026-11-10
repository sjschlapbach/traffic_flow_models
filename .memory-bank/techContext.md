# Tech Context — traffic_flow_models

> Update when dependencies or environment change.

## Python & install

- Requires **Python 3.13+** (`requires-python = ">=3.13"`); CI runs a single-version matrix
  `[3.13]` — no legacy Python support to maintain.
- Editable install: `pip install -e ".[dev]"`. Package lives under `src/` via
  `package-dir = {"" = "src"}` in `pyproject.toml`.
- No `requirements.txt` — all dependencies are declared in `pyproject.toml`.

## Runtime dependencies (pinned, `[project.dependencies]`)

| Package | Version | Role |
| --- | --- | --- |
| `numpy` | `2.3.4` | Array operations |
| `scipy` | `1.16.3` | Calibration: `least_squares` (Trust-Region-Reflective) solver, `stats.qmc.LatinHypercube` for multi-start search |
| `matplotlib` | `3.10.7` | Network/result plotting |
| `casadi` | `3.7.2` | **Core numerical engine.** All CTM/METANET state updates are built as symbolic `SX` computation graphs, compiled once per network topology into a `casadi.Function`, then evaluated numerically each timestep. See `systemPatterns.md`. |
| `networkx` | `3.5` | Graph layout for `Network.plot` topology plots |
| `opencv-python` | `4.13.0.92` | AVI video export of simulation results (`cv2.VideoWriter`) |
| `tqdm` | `4.67.1` | Progress bars |

## Dev dependencies (`[project.optional-dependencies].dev`)

- `pytest==9.0.0` — test runner (see Testing conventions below).
- `pylint==4.0.4` — CI runs `pylint src --errors-only` / `pylint tests --errors-only` only; this
  is **not** a full style/convention gate, just error-severity findings.
- `git-cliff==2.11.0` — changelog generation from conventional commits (see
  `.agent_docs/INTEGRATIONS.md`).
- `mkdocs-material>=9.5.5`, `mkdocstrings[python]>=0.25` — docs site (also available standalone
  via the `docs` extra, used by CI's docs-only install job).
- **Black is a hard CI gate** (`psf/black@stable` in `lint.yml`, no `--check` bypass) even though
  it is *not* listed in `pyproject.toml`'s dev extras — install it separately
  (`pip install black`) if formatting locally.

## Testing conventions

- Flat `tests/` directory at repo root (no co-location with `src/`), no `conftest.py`, no shared
  fixtures beyond pytest's own built-ins.
- File naming: `<module>_test.py` (e.g. `ctm_test.py`, `network_test.py`).
- The suite needs no external tools. Tests that write results use pytest's `tmp_path`.
- Run: `pytest` (no markers/flags needed; no coverage reporting configured).

## CI workflows (`.github/workflows/`)

All trigger on push/PR to `master`/`dev`, except the two tag-triggered release workflows.

| Workflow | Purpose |
| --- | --- |
| `python_testing.yml` | Runs `pytest` — the only workflow that runs the `tests/` suite |
| `lint.yml` | `psf/black@stable` (hard gate) + `pylint src/tests --errors-only` |
| `build.yml` | `python -m build --sdist --wheel` — build sanity check only, no tests |
| `docs_build.yml` | `mkdocs build` on every push/PR |
| `docs_release.yml` | Tag-triggered `mkdocs gh-deploy --force` to GitHub Pages |
| `release.yml` | Tag-triggered (`v*.*.*`): validates tag == `pyproject.toml` version == changelog version, builds, publishes to PyPI via `twine`, commits `CHANGELOG.md` back to `master`, creates a GitHub Release |
| `scripts_basics.yml` | Smoke-tests `src.demo.plot_demo_network` |
| `scripts_ctm_simulation.yml` / `scripts_metanet_simulation.yml` | Smoke-test demo scenarios A, B, C, C1–C4, D for CTM/METANET respectively |
| `scripts_metanet_calibration.yml` | Smoke-tests `src.demo.metanet_calibration` |

## Release mechanics

1. Bump version via `git-cliff --bump` (updates `pyproject.toml`), commit only the version change
   (not the changelog — CI generates that).
2. Tag `vX.Y.Z` and push the tag.
3. `release.yml` validates versions match across tag/`pyproject.toml`/changelog, builds, publishes
   to PyPI, commits the generated `CHANGELOG.md` back to `master`, creates the GitHub Release.
4. **Never hand-edit `CHANGELOG.md`.** Agents should not initiate a release unless explicitly
   instructed to (already stated in `AGENTS.md`).

## See also

- `projectbrief.md` — what this project is and why
- `.agent_docs/INTEGRATIONS.md` — git-cliff/MkDocs config-pattern detail
