# Suggested Commands

Commands assume a POSIX shell. The maintainer's machine is macOS (zsh, BSD userland, not GNU).
On Windows, activate with `venv\Scripts\activate`.

**Always run everything inside the local `venv/` (git-ignored), from the repository root.**
Python **3.13** (`requires-python = ">=3.13"`, CI uses 3.13). Name `python3.13` explicitly when
you create the venv, because plain `python3` may be a different version (Homebrew's is 3.14 on the
maintainer's Mac).

## Setup

```bash
python3.13 -m venv venv          # only if venv/ does not exist yet
source venv/bin/activate
pip install -e ".[dev]"          # editable install: runtime + pytest, pylint, mypy + stubs,
                                 # git-cliff, mkdocs
pip install black                # black is a CI gate but is not in the dev extra
```

Every shell (each agent Bash call is a fresh one) starts with `source venv/bin/activate`.

## Test, format, lint

```bash
pytest                           # full suite, must pass (no flags, no coverage config)
pytest tests/network_test.py -k packing   # one file / pattern
black .                          # format after every code change
black --check .                  # what CI checks
pylint src --errors-only         # CI gate (same for tests)
pylint src                       # full report: new code adds no warnings
mypy                             # strict type check of the package (pyproject.toml config)
```

The stub pins in the dev extra follow the runtime pins: `scipy-stubs` 1.16.3.x for scipy
1.16.3, `types-networkx` 3.5.x, `types-tqdm` 4.67.x, and `numpy-typing-compat` 20251206.2.3
(the numpy 2.3 line; a newer release would upgrade numpy). Bump them together with the
library.

## Demos

```bash
python -m src.demo.ctm_simulation --scenario A --no-plot      # scenarios A B C C1-C4 D E E1 E2
python -m src.demo.metanet_simulation --scenario C1 --generate-video
python -m src.demo.plot_demo_network --no-plot
python -m src.demo.metanet_calibration                        # --parameter-search: LHS multi-start
```

Outputs go to `results/` (git-ignored); delete test runs afterwards.

## Docs

```bash
mkdocs serve                     # local preview
mkdocs build                     # what docs_build.yml runs
```

MkDocs Material, `mkdocstrings` python handler (`paths: [src]`, Google docstrings), MathJax via
`pymdownx.arithmatex`. Nav lives in `mkdocs.yml` and mirrors `docs/`.

## CI (`.github/workflows/`, push/PR to `master`/`dev`)

| Workflow | Runs |
|---|---|
| `python_testing.yml` | `pytest` |
| `lint.yml` | jobs `format` (`psf/black@stable`), `lint` (`pylint src/tests --errors-only`), `typecheck` (`mypy`) |
| `build.yml` | `python -m build --sdist --wheel`, then checks the wheel contains `traffic_flow_models/py.typed` |
| `docs_build.yml` | `mkdocs build` |
| `scripts_basics.yml` | `src.demo.plot_demo_network --no-plot` |
| `scripts_ctm_simulation.yml` / `scripts_metanet_simulation.yml` | demo scenarios A, B, C, C1–C4, D |
| `scripts_metanet_calibration.yml` | `src.demo.metanet_calibration` |
| `docs_release.yml` (tag) | `mkdocs gh-deploy --force` to GitHub Pages |
| `release.yml` (tag `v*.*.*`) | checks tag == `pyproject.toml` == changelog version, builds, publishes to PyPI, commits `CHANGELOG.md` to `master`, creates a GitHub Release |

## Release (maintainers only, never run by an agent)

**Agents never run a release command, even when asked to prepare or start a release.** The
maintainers run these by hand: `git-cliff` (any form, incl. `--bump`), `git tag`, `git push`
(incl. tags), `twine`, `mkdocs gh-deploy`, `gh release`. In Claude Code they are denied in
`.claude/settings.json`. An agent may only describe the steps. `python -m build` (local wheel
build, no upload) is fine for checking packaging.

Manual steps, for reference:
1. `git-cliff --bump` bumps `pyproject.toml`. Commit only the version change.
2. Tag `vX.Y.Z` and push the tag. `release.yml` does the rest.

git-cliff config is `cliff.toml`: conventional commits (`feat`, `enhance`, `fix`, `doc`, `perf`,
`refactor`, `style`, `test`, `chore`/`ci`, `revert`), `breaking_always_bump_major = true`,
`chore(release|deps|pr|pull)` skipped. **Never hand-edit `CHANGELOG.md`.**

## Git

Agents never commit or push; the human working with the agent does (AGENTS.md). In Claude Code,
`git commit`, `git push`, `git worktree` and the `EnterWorktree` tool are also denied in
`.claude/settings.json`.
