# AI Tracking System (how project knowledge is split)

Project knowledge is split so that no fact is stored twice:

- **Serena memories** (this store, `.serena/memories/`, committed to git) hold the **reference
  knowledge** and are the single source of truth: `mem:project_overview`, `mem:codebase_structure`,
  `mem:code_style_and_conventions`, `mem:suggested_commands`, `mem:tools_and_skills`,
  `mem:task_completion_checklist`. Read them with serena's `read_memory`/`list_memories`, or
  directly as `.serena/memories/*.md` files if serena is unavailable.

- **`.ai/`** (git-ignored) holds only what serena memory does not hold well, the append-only logs:
  - `.ai/DECISIONS.md`: Architectural Decision Records (ADRs), newest first.
  - `.ai/PROGRESS.md`: dated progress log (session work, learnings, dead ends, open questions,
    in-flight branches), newest first.
  - `.ai/README.md`: describes this split.

- **`AGENTS.md`** holds the working protocol for every coding agent: what to read before a task
  and what to update after. It points here and does not repeat the content. **`CLAUDE.md`**
  imports it (`@AGENTS.md`) and adds only the Claude Code specifics (serena plugin, hooks).
- On a fresh clone `.ai/` does not exist yet; the first task creates `.ai/DECISIONS.md` and
  `.ai/PROGRESS.md`.

## Enforcement (Claude Code only, `.claude/settings.json`)

Other agents have no hooks and follow the protocol in AGENTS.md by hand.

- A `SessionStart` hook prints a one-line reminder to read the memories and `.ai/` before working.
- A `Stop` hook (`.claude/hooks/ai-tracking-reminder.sh`) blocks the end of a turn when a
  **relevant file** changed but tracking was not updated.
  - **Relevant:** `*.py` (outside `.ai/`/`.serena/`), `pyproject.toml`, `mkdocs.yml` and
    `.github/workflows/*`. These always require a tracking update. Changes confined to other files
    (docs, README, …) should still be reflected in tracking per AGENTS.md, but the hook does
    **not** enforce them.
  - **Detection by mtime, not `git status`:** `.ai/` is git-ignored, so git never lists `.ai/`
    edits. The hook compares mtimes. Tracking counts as current when `.ai/DECISIONS.md`,
    `.ai/PROGRESS.md` or some `.serena/memories/*.md` was modified at least as recently as the
    newest changed relevant file. A deleted relevant file counts with the mtime of its nearest
    existing parent directory. Touching a tracking file, or committing the change, clears the
    block.
