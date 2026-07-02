# AGENTS.md — traffic_flow_models Coding Agent Instructions

> **Compatibility:** This file is read automatically by OpenCode, OpenAI Codex, GitHub Copilot,
> Aider, and Claude Code. All agents operating on this repository MUST follow every rule here.

---

## Repository Overview

This repository implements **traffic_flow_models** — a Python 3.13 library for simulating and
analyzing macroscopic traffic flow on highway networks. It implements two widely-used traffic
flow models, the **Cell Transmission Model (CTM)** and **METANET**, along with network
infrastructure components (motorway links, nodes, onramps, offramps, origins, destinations),
ramp metering controllers (ALINEA, METALINE, fixed-rate), performance metrics, and visualization
tools. It also includes experimental SUMO microsimulation pipeline components and macroscopic
model calibration tools.

The package is published to PyPI as `traffic-flow-models` and documented via MkDocs at
`https://sjschlapbach.github.io/traffic_flow_models/`.

**Primary language:** Python 3.13
**Key runtime dependencies:** numpy, scipy, matplotlib, osmnx, casadi, networkx, opencv-python,
tqdm (see `pyproject.toml`)
**Key dev dependencies:** pytest, pylint, black, git-cliff, mkdocs-material, mkdocstrings

---

## Documentation Files — MANDATORY MAINTENANCE

The following files are the **single source of truth** for the repository's state.
Every agent **MUST** keep them up to date. Failure to do so is treated as an incomplete task.

| File                          | Purpose                                                  | Update trigger                           |
| ----------------------------- | -------------------------------------------------------- | ----------------------------------------- |
| `README.md`                   | Package overview, usage examples, project structure, release process | When public API, usage patterns, or entry points change |
| `.agent_docs/STRUCTURE.md`    | Annotated directory tree with module responsibilities    | Any file/folder added, moved, or removed |
| `.agent_docs/DECISIONS.md`    | Architectural Decision Records (ADRs)                    | Any non-trivial tech or design choice    |
| `.agent_docs/INTEGRATIONS.md` | External tools, simulators, and their config patterns    | When external dependency changes         |
| `.agent_docs/scratchpad.md`   | Per-session working memory (tasks, learnings, dead ends) | Continuously during every session        |
| `docs/`                       | **User-facing MkDocs documentation** (deployed to GitHub Pages) | When models, network components, controllers, calibration, or the pipeline change |
| `CHANGELOG.md`                | Generated automatically via `git-cliff` from conventional commits | Never edit by hand — see Release Process section below |

> `.agent_docs/` and `.memory-bank/` are populated — start every session by reading
> `.memory-bank/` (see the "Memory Bank" section below) and skim `.agent_docs/STRUCTURE.md` and
> `.agent_docs/scratchpad.md` for recent context before making changes.

### Update Protocol (MANDATORY)

At the **end of every task**, before considering it complete, you MUST:

1. **`README.md`** — Ensure usage examples, the "Project Structure" section, and installation
   instructions match the current code. If you changed public API surface (new classes exported
   from `src/traffic_flow_models/__init__.py`, changed constructor signatures used in examples,
   new demo scripts), update the corresponding examples.
2. **`.agent_docs/STRUCTURE.md`** — Update the directory tree if any files or folders were added,
   removed, or had their responsibilities changed.
3. **`.agent_docs/DECISIONS.md`** — Add an ADR if a non-obvious architectural or algorithmic choice
   was made.
4. **`.agent_docs/scratchpad.md`** — Update task progress, mark completed sub-tasks, and record any
   learnings or dead ends encountered during the session.
5. **`docs/`** — Update the relevant MkDocs pages (`docs/api/`, `docs/models/`, `docs/concepts/`,
   `docs/control/`, `docs/simulation/`, `docs/experimental/`, `docs/getting-started/`) if you
   changed components, setup instructions, interfaces, or architecture. These pages are deployed
   to GitHub Pages and must stay in sync with the implementation.

> If you are unsure whether a change warrants a docs update, err on the side of updating.
> A one-line entry is always better than silence.
>
> **Special note on README:** This file is your primary communication with users about what the
> package does and how to use it. If you change the public API, usage patterns, or installation
> steps, the README must change. Keep it in sync with the actual code — a stale README damages
> credibility and wastes user time.

---

## Project Layout

```
traffic_flow_models/
├── src/
│   ├── traffic_flow_models/        – Main installable package
│   │   ├── __init__.py             – Public API re-exports (keep in sync with new modules)
│   │   ├── model/                  – Traffic flow model implementations
│   │   │   ├── ctm.py              – Cell Transmission Model (first-order)
│   │   │   ├── metanet.py          – METANET model (second-order)
│   │   │   └── helpers.py          – Shared model helper functions
│   │   ├── network/                – Network topology and simulation orchestration
│   │   │   ├── network.py          – Network class: builds/validates topology, runs simulation
│   │   │   ├── simulation.py       – Simulation class: time-stepping, metrics, visualization
│   │   │   ├── cell.py             – Discrete highway segment (CTM/METANET state unit)
│   │   │   ├── motorway_link.py    – Mainline link composed of cells
│   │   │   ├── onramp.py / offramp.py – Ramp links, optionally attached to a controller
│   │   │   ├── node.py             – Junction connecting incoming/outgoing links
│   │   │   ├── origin.py / destination.py – Network boundary conditions
│   │   │   └── calibrator.py       – Parameter calibration against ground-truth data
│   │   ├── controller/             – Ramp metering control strategies
│   │   │   ├── flow_controller.py  – Base controller interface
│   │   │   ├── alinea.py           – ALINEA local feedback ramp metering
│   │   │   ├── metaline.py         – METALINE coordinated ramp metering
│   │   │   └── custom_controller.py – Template/interface for user-defined controllers
│   │   ├── simulator/              – SUMO microsimulation integration (experimental)
│   │   │   ├── sumo_simulation.py  – SUMO microsimulation runner
│   │   │   └── sumo_pipeline.py    – Pipeline for importing/benchmarking real-world networks
│   │   └── arbitrator/             – Demand/turning-rate aggregation and loop detector tooling
│   │       ├── network_arbitrator.py
│   │       ├── demand_aggregator.py
│   │       ├── turning_rate_aggregator.py
│   │       ├── backbone_aggregator.py
│   │       ├── loop_detector_generator.py
│   │       └── aggregation_helpers.py
│   └── demo/                       – Demo scripts and example scenarios
│       ├── ctm_simulation.py       – CTM usage example
│       ├── metanet_simulation.py   – METANET usage example
│       ├── metanet_calibration.py  – Calibration workflow example
│       ├── plot_demo_network.py    – Network topology visualization example
│       ├── scenarios.py            – Scenario definitions
│       ├── scenarios/              – Named scenario configs (scenario_a .. scenario_e, example)
│       └── run_pipeline.py         – SUMO pipeline usage example
├── tests/                          – pytest test suite (flat, one file per module/feature)
├── docs/                           – User-facing MkDocs documentation (deployed to GitHub Pages)
│   ├── getting-started/            – Installation, quickstart
│   ├── concepts/                   – Network and state-space concepts
│   ├── models/                     – CTM, METANET, model comparison
│   ├── control/                    – Ramp metering strategies
│   ├── simulation/                 – Running simulations, metrics, visualization
│   ├── experimental/               – Calibration, SUMO pipeline
│   ├── api/                        – API reference pages (mkdocstrings)
│   └── development/                – Contributing guide, changelog
├── results/                        – Generated simulation outputs (not authoritative source)
├── .agent_docs/                    – LLM-maintained documentation (see above)
│   ├── STRUCTURE.md
│   ├── DECISIONS.md
│   ├── INTEGRATIONS.md
│   └── scratchpad.md
├── .memory-bank/                   – Persistent cross-session project context
│   ├── projectbrief.md
│   ├── systemPatterns.md
│   ├── techContext.md
│   ├── activeContext.md
│   └── progress.md
├── .github/workflows/              – CI: testing, linting, docs build/release, package build/release
├── pyproject.toml                  – Package metadata, dependencies (runtime + dev + docs extras)
├── mkdocs.yml                      – MkDocs site configuration
├── cliff.toml                      – git-cliff changelog generation configuration
├── .pylintrc                       – Pylint configuration
├── .gitignore
├── AGENTS.md                       – This file
└── CLAUDE.md                       – Claude-specific quick reference
```

- All installable source code lives under `src/traffic_flow_models/`. Demo/example scripts live
  under `src/demo/` and are not part of the installed package.
- Tests live in the flat `tests/` directory at the repo root, not co-located with source.
- Do **not** create new top-level directories without explicit instruction.
- Use relative imports within `src/traffic_flow_models/` submodules; when adding a new public
  class or function, re-export it from `src/traffic_flow_models/__init__.py` and add it to
  `__all__` so it is importable as `from traffic_flow_models import X`.
- After any structural change, update the "Project Layout" section above and the README's
  "Project Structure" section.

---

## Hard Requirements

These rules apply to **every task** and must never be skipped.

### Python Virtual Environment

Always check for a virtual environment before running any Python command.

- If `venv/` exists at the project root, activate it first:
  ```bash
  source venv/bin/activate
  ```
- If no virtual environment exists and one is needed, create it with Python 3.13:
  ```bash
  python3.13 -m venv venv
  source venv/bin/activate
  pip install -e ".[dev]"
  ```
- Never install packages globally. Always install into the active virtual environment.
- The package uses editable installs (`pip install -e .`), not a pinned `requirements.txt`.
  Runtime and dev dependencies are declared in `pyproject.toml` under `[project.dependencies]`
  and `[project.optional-dependencies]`. If you add a new dependency, add it there with a pinned
  version, matching the existing style (`package==x.y.z`).

### SUMO (only relevant for `simulator/` and pipeline work)

The experimental SUMO integration (`src/traffic_flow_models/simulator/`) requires SUMO to be
installed separately (not a Python dependency) with `netconvert` and `sumo` available on `PATH`.
Do not assume SUMO is available in every environment — guard SUMO-dependent tests/scripts
accordingly and consult existing patterns in `sumo_simulation.py` / `sumo_pipeline.py`.

---

## Memory Bank — MANDATORY

All files in `.memory-bank/` MUST be read at the start of every session and updated
at the end. They are the primary source of project context — treat them as your
persistent working memory across sessions.

- `projectbrief.md` — read-only unless scope changes
- `systemPatterns.md` — update when architecture changes
- `techContext.md` — update when dependencies or environment changes
- `activeContext.md` — update every session (current focus + recent changes)
- `progress.md` — update every session (what works, what's broken)

## Code Quality

- Follow existing code style and naming conventions exactly — do not diverge without instruction.
- Format code with **Black** before finishing a task:
  ```bash
  black .
  ```
  CI (`lint.yml`) runs `black --check` and will fail on unformatted code.
- Obey the lint rules in `.pylintrc`. Run `pylint src/` and fix all warnings before considering a
  task complete.
- Write docstrings for all public functions, classes, and modules.
- Add inline comments only when the _why_ is not obvious from the code.
- Avoid magic numbers and strings — use named constants.
- Validate inputs at system boundaries (file I/O, scenario/config parsing, calibration entry
  points).
- Prefer explicit over implicit; readable over clever.
- Use type annotations on all new function signatures.

---

## Testing

- Write tests for every non-trivial function, covering edge cases and numerical edge cases
  (e.g. congestion/free-flow regime transitions, capacity limits, degenerate network topologies).
- Place tests in the flat `tests/` directory at the repo root, named `<module>_test.py` or
  `test_<module>.py` to match the existing convention for that area.
- Run the full test suite before and after every change:
  ```bash
  pytest
  ```
- Do not consider a task complete until all tests pass.
- `pytest` is declared as a dev dependency in `pyproject.toml` (`pip install -e ".[dev]"`).

---

## Security & Data Integrity

- Never hardcode absolute file paths — use `pathlib.Path` and relative paths.
- Never commit generated output files (plots, videos, result JSON/CSV under `results/`) — verify
  they are covered by `.gitignore` before adding new output locations.
- Sanitize all external inputs (scenario/config files, CLI arguments, imported SUMO network data)
  before passing them to model or calibration algorithms.

---

## Communication & Reasoning

- Think step-by-step before writing code or making changes, especially for traffic flow model
  internals (CTM/METANET equations) and calibration logic.
- If requirements are ambiguous, ask one focused clarifying question before proceeding.
- When proposing a non-obvious algorithmic approach, briefly explain the reasoning and complexity.
- If multiple valid approaches exist, list them with trade-offs (numerical stability, performance,
  correctness) before choosing.
- Flag potential risks explicitly — especially numerical stability issues (e.g. CFL-condition
  violations in CTM, METANET solver divergence), degenerate network geometry, or calibration
  optimizer (`casadi`/`scipy`) failures to converge.
- At the end of each task, summarize what was changed and why.
- Do not hallucinate APIs for `numpy`, `scipy`, `casadi`, `networkx`, `osmnx`, or `matplotlib` —
  verify method signatures before using.

---

## Release Process

Releases are handled via `git-cliff` (conventional commits → changelog) and GitHub Actions
(`release.yml`, `build.yml`). Do not hand-edit `CHANGELOG.md`. See the "Release" section of
`README.md` for the full step-by-step process. Agents should not initiate a release unless
explicitly instructed to.

---

## Scratchpad Format

When updating `.agent_docs/scratchpad.md`, use this structure for each session:

```markdown
## Task: <short description>

### Plan

- [ ] Step 1
- [ ] Step 2

### Learnings

- YYYY-MM-DD: <learning>

### Decisions

- <decision>: <rationale>

### Dead Ends

- <approach>: <why it failed>

### Open Questions

- <question>
```

Do not delete previous entries between sessions — accumulate knowledge over time.
