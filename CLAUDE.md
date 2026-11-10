# CLAUDE.md — traffic_flow_models Coding Guidelines

This file provides quick reference guidance for working on **traffic_flow_models**, a Python
library implementing macroscopic traffic flow models (CTM, METANET) and ramp metering
controllers. For comprehensive agent instructions, code quality requirements, and architectural
decisions, always refer to **AGENTS.md**. You NEVER need to commit your changes, this will be
done manually by the project maintainer. You are only responsible for following the rules and
updating the documentation.

## Quick Reference

**Read these first:**

- `AGENTS.md` — Complete agent instructions (mandatory for all work)
- `README.md` — Package overview, usage examples, project structure
- `.memory-bank/` — Persistent context about the project state and decisions
- `.agent_docs/STRUCTURE.md` — Current directory structure and module responsibilities
- `.agent_docs/DECISIONS.md` — Architectural Decision Records (ADRs)

> `.agent_docs/` and `.memory-bank/` are populated — read `.memory-bank/` at the start of every
> session and skim `.agent_docs/STRUCTURE.md`/`.agent_docs/scratchpad.md` for recent context.

## Essential Rules

1. **Always activate the virtual environment:**

   ```bash
   source venv/bin/activate
   ```

   If it doesn't exist: `python3.13 -m venv venv && source venv/bin/activate && pip install -e ".[dev]"`

2. **Update documentation on every task:**
   - `README.md` — If public API, usage patterns, or the project structure change (keep in sync!)
   - `.agent_docs/STRUCTURE.md` — After adding/moving/removing files or folders
   - `.agent_docs/DECISIONS.md` — After any non-trivial architectural choice
   - `.agent_docs/scratchpad.md` — Session progress and learnings (mandatory)
   - `docs/` — User-facing MkDocs pages (deployed to GitHub Pages); update when models, network
     components, controllers, or calibration change

3. **Update `.memory-bank/` at session end:**
   - `.memory-bank/activeContext.md` — Current focus and recent changes
   - `.memory-bank/progress.md` — What works, what's broken
   - `.memory-bank/systemPatterns.md` — When architecture changes

4. **Run tests before finishing a task:**

   ```bash
   pytest
   ```

   All tests must pass.

5. **Follow the code style:**
   - Type annotations on all function signatures
   - Docstrings for public functions, classes, and modules
   - Inline comments only for the _why_, not the _what_
   - Follow existing naming conventions exactly
   - Format with `black .` and run `pylint src/`, fixing all warnings

## Project Structure

```
traffic_flow_models/
├── src/
│   ├── traffic_flow_models/    – Main installable package
│   │   ├── __init__.py         – Public API re-exports
│   │   ├── model/               – CTM and METANET model implementations
│   │   ├── network/             – Network topology, simulation, calibration
│   │   └── controller/          – Ramp metering controllers (ALINEA, METALINE, custom)
│   └── demo/                   – Demo scripts and example scenarios
├── tests/                      – pytest test suite (flat, root level)
├── docs/                       – User-facing MkDocs documentation (GitHub Pages)
├── results/                    – Generated simulation outputs
├── .agent_docs/                – LLM-maintained documentation
│   ├── STRUCTURE.md            – Directory tree and responsibilities
│   ├── DECISIONS.md            – Architectural Decision Records
│   ├── INTEGRATIONS.md         – External tools and integrations
│   └── scratchpad.md           – Per-session working memory
├── .memory-bank/                – Persistent project context
├── pyproject.toml               – Package metadata and dependencies (runtime, dev, docs extras)
├── mkdocs.yml                   – MkDocs site configuration
├── AGENTS.md                    – Complete agent instructions
└── CLAUDE.md                    – This file
```

## Common Tasks

### Add New Code

1. Create/modify files under `src/traffic_flow_models/` as needed
2. Add type annotations and docstrings
3. Re-export new public classes/functions from `src/traffic_flow_models/__init__.py`
   (and add to `__all__`)
4. Update `.agent_docs/STRUCTURE.md` if adding new files/directories
5. Run `pytest` to verify

### Fix a Bug

1. Write a test that reproduces the bug
2. Fix the bug
3. Verify all tests pass
4. Update `.agent_docs/DECISIONS.md` if the fix involved non-obvious reasoning
5. Update `.agent_docs/scratchpad.md` with what you learned

### Refactor Code

1. Plan the refactor (use AGENTS.md for architectural guidance)
2. Preserve all existing behavior
3. Update `.agent_docs/STRUCTURE.md` and the "Project Layout" in AGENTS.md if module
   responsibilities change
4. Run full test suite
5. Update `.agent_docs/DECISIONS.md` with the rationale

## Key Technologies

- **Python 3.13** — Required version
- **numpy / scipy** — Numerical computing for CTM/METANET state updates
- **casadi** — Optimization backend for model calibration
- **networkx** — Graph layout for network topology plots
- **matplotlib / opencv-python** — Visualization and video export
- **pytest** — Testing framework
- **black / pylint** — Formatting and linting
- **git-cliff** — Automated changelog generation for releases
- **mkdocs-material / mkdocstrings** — Documentation site generation

## Contact & Help

For issues with Claude Code itself, see `/help` or report at:
https://github.com/anthropics/claude-code/issues

For project-specific questions, refer to AGENTS.md or check the `.memory-bank/` context.

---

**Everything below this line is reference. For the authoritative source, see AGENTS.md.**

### From AGENTS.md: Hard Requirements

- **Virtual environment** — Always required. Create with `python3.13 -m venv venv` and
  `pip install -e ".[dev]"`
- **Type annotations** — All function signatures must have type hints
- **Docstrings** — Required for all public functions, classes, and modules
- **Code quality** — Format with `black .`, run `pylint src/`, fix all warnings
- **Testing** — Write tests for every non-trivial function; run `pytest` before finishing a task
- **Documentation** — Keep `.agent_docs/STRUCTURE.md`, `.agent_docs/DECISIONS.md`,
  `.agent_docs/scratchpad.md`, and the `docs/` MkDocs pages up to date at task end
- **No absolute paths** — Always use `pathlib.Path` and relative paths
- **Security** — Sanitize all external inputs before passing to model/calibration algorithms

### From AGENTS.md: Memory Bank Files

All files in `.memory-bank/` MUST be read at the start of every session and updated at the end:

- `projectbrief.md` — Read-only (project scope and goals)
- `systemPatterns.md` — Update when architecture changes
- `techContext.md` — Update when dependencies or environment changes
- `activeContext.md` — Update every session (focus + recent changes)
- `progress.md` — Update every session (what works + what's broken)

### From AGENTS.md: Scratchpad Format

When updating `.agent_docs/scratchpad.md`, document each session with:

```markdown
## Task: <description>

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

Do not delete previous entries — accumulate knowledge over time.

### From AGENTS.md: Release Process

Releases use `git-cliff` for changelog generation and GitHub Actions for build/publish. Never
hand-edit `CHANGELOG.md`. Do not initiate a release unless explicitly instructed.
