# AGENTS.md

Working protocol for **every coding agent** on this repository (Claude Code, Codex, Copilot,
Cursor, Aider, Gemini, …). Claude Code loads this file through `CLAUDE.md`, which adds only
Claude-specific tooling.

## Project

**traffic_flow_models** is a Python 3.13 library for macroscopic traffic flow simulation on
highway networks: CTM and METANET models, network components, ramp-metering controllers (ALINEA,
METALINE, fixed rate, custom) and experimental METANET calibration. It is published on PyPI and
documented with MkDocs. The details live in the project memories (below), not here.

## Keep the code simple (guiding principle)

Prefer the simplest solution that works and do not over-engineer. Follow the style already in the
code (see the `code_style_and_conventions` memory). No speculative abstraction, and no options or
configuration nobody asked for.

**No inline lint/type suppressions without asking (strict):** never add `# pylint: disable=...`
or `# type: ignore` on your own. Ask the user first whether the rule should be suppressed on that
line, disabled repo-wide in `.pylintrc`, or the code fixed instead.

**Comments (strict):** every inline `#` comment starts **lower-case**, with no exceptions; reword
rather than start with a capitalised identifier. Comment only where it adds meaning the code does
not already carry (why, modelling assumption, unit, convention).

**Comments describe the current code, never its history (strict):** no comment, docstring,
README line, `docs/` page or memory may mention past changes ("copied", "moved from", "now", "no
longer", "previously", "was"). Describe what the code does and why. History goes only in
`.ai/PROGRESS.md` / `.ai/DECISIONS.md`.

## Environment

Always run everything inside the local `venv/` (`source venv/bin/activate`) from the repository
root. Setup and all commands are in the `suggested_commands` memory. **`pytest` must pass, all
Python code must be formatted with black (`black --check .`), and `pylint src --errors-only` must
be clean (CI runs all three); new code adds no pylint warnings.**

## Git (hard rules)

- **Agents never run `git commit` or `git push`.** The human working with the agent commits and
  pushes. Leave changes in the working tree.
- **Never create a git worktree.** Apply all changes directly to the checked-out branch: `.ai/`
  is git-ignored and does not follow into a worktree, so tracking would split across copies.
- Never hand-edit `CHANGELOG.md` (git-cliff generates it). Do not start a release unless asked.

## Project knowledge (where to look)

Knowledge is split to avoid duplication, and keeping it current is part of every task:

- **Memories** (`.serena/memories/*.md`, committed): the reference knowledge and the **primary
  source of truth**:

  | Memory | Contents |
  |---|---|
  | `project_overview` | scope, stability tiers, fragile areas |
  | `codebase_structure` | every module, its classes and invariants |
  | `code_style_and_conventions` | formatting, typing, docstrings, tests |
  | `suggested_commands` | setup, test/lint, demos, docs, CI, release |
  | `task_completion_checklist` | the definition of done |
  | `tools_and_skills` | serena, context7, other tools |
  | `ai_tracking_system` | how this split is enforced |

  They are plain markdown. With the [serena](https://github.com/oraios/serena) MCP server, use
  `list_memories` / `read_memory` / `write_memory`; without it, read and edit the files directly.
- **`.ai/`** (git-ignored, local): the append-only logs `DECISIONS.md` (ADRs, newest first) and
  `PROGRESS.md` (dated log, newest first). See `.ai/README.md`. On a fresh clone `.ai/` does not
  exist; create both files on your first task.
- **This file**: the protocol and pointers only. It deliberately does **not** restate the memory
  content.

## Working protocol

**Before starting a task**

1. Read the relevant memories. Start with `project_overview`, then `codebase_structure`,
   `code_style_and_conventions`, `suggested_commands`, `tools_and_skills`.
2. Read `.ai/DECISIONS.md` (if present) for the decisions and constraints that bind your work.
3. Check `.ai/PROGRESS.md` (if present) for in-flight work and recent changes.

**While working**, follow `code_style_and_conventions`. Check library APIs (numpy, scipy,
casadi, networkx, matplotlib) against current docs instead of writing them from memory. If you
deviate from a convention, record why in `.ai/DECISIONS.md`.

**Before finishing a task (definition of done)**, update whichever apply:

- **Memories**: update `codebase_structure`, `code_style_and_conventions`, `suggested_commands`
  or `project_overview` if structure, conventions, commands or scope changed. A new module,
  model, network component or controller means `codebase_structure` needs a line; do not defer it.
- **Documentation**: if the public API, usage or installation changed, update `README.md` and the
  matching `docs/` MkDocs pages in the same task.
- `.ai/DECISIONS.md`: add an ADR when you made a non-obvious or systematic choice (algorithm,
  modelling assumption, API change, parameter value).
- `.ai/PROGRESS.md`: **always** append a dated entry summarizing what changed and why.
- Work through the `task_completion_checklist` memory.

**A stale memory or log is a bug**, so fix it in the same task.
