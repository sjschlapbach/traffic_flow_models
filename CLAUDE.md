# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

The full working protocol for every coding agent lives in `AGENTS.md`, imported here:

@AGENTS.md

## Claude Code specifics

Configured in `.claude/settings.json` (committed):

- **serena and context7 start automatically**: both plugins are enabled for this project in
  `enabledPlugins`. Use serena's `list_memories` / `read_memory` / `write_memory` for the
  memories and its symbol tools for code navigation and editing. If the serena tools are missing,
  edit `.serena/memories/*.md` as plain files and tell the user the plugin did not start. Details
  are in the `tools_and_skills` memory.
- **Hooks**: a `SessionStart` hook reminds you to read the memories. A `Stop` hook
  (`.claude/hooks/ai-tracking-reminder.sh`) blocks if `*.py`, `pyproject.toml`, `mkdocs.yml` or a
  CI workflow changed but neither `.ai/` nor `.serena/memories/` was updated afterwards.
- **Denied**: `git commit`, `git push`, `git worktree` and the `EnterWorktree` tool. Background-job
  isolation is off (`worktree.bgIsolation: "none"`).
