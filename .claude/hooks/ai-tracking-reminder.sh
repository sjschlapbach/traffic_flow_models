#!/usr/bin/env bash
# Stop hook: require a tracking update when a relevant file changes.
#
# Tracking = .ai/ (decision + progress logs) and .serena/memories/ (reference
# knowledge). Relevant = source code (*.py) plus the files the serena memories
# document directly: pyproject.toml (dependencies), mkdocs.yml (docs site) and
# .github/workflows/ (CI). Changes confined to other files (docs, README, ...)
# should still be reflected in tracking per AGENTS.md, but are deliberately not
# enforced here.
#
# .ai/ is git-ignored, so tracking freshness is detected by file mtime rather
# than `git status` (which never lists ignored files): tracking counts as current
# when some tracking file was modified at least as recently as the newest changed
# relevant file. Self-terminating: update .ai/ or a serena memory (or commit the
# change) and the condition is false, so Claude stops normally.
set -euo pipefail

repo="${CLAUDE_PROJECT_DIR:-$(pwd)}"
cd "$repo" 2>/dev/null || exit 0

# seconds-since-epoch mtime for a file: BSD stat (macOS) first, GNU stat as fallback.
# gnu `stat -f` means --file-system and prints a non-numeric value, so only digits count
mtime() {
  local m
  m="$(stat -f %m "$1" 2>/dev/null || true)"
  [[ "$m" =~ ^[0-9]+$ ]] || m="$(stat -c %Y "$1" 2>/dev/null || true)"
  [[ "$m" =~ ^[0-9]+$ ]] || m=0
  echo "$m"
}

status="$(git status --porcelain --untracked-files=all 2>/dev/null || true)"
# not a git repo, or a clean tree: nothing to enforce
[ -z "$status" ] && exit 0

# strip the "XY " porcelain prefix and any rename "old -> new" arrow (keeping the
# new path), then keep the relevant files outside the tracking stores
changed="$(printf '%s\n' "$status" | sed 's/^...//; s/.* -> //; s/^"//; s/"$//' \
  | grep -E '\.py$|^pyproject\.toml$|^mkdocs\.yml$|^\.github/workflows/' \
  | grep -vE '^(\.ai/|\.serena/)' || true)"
[ -z "$changed" ] && exit 0

# newest mtime among the changed relevant files. a deleted file has no mtime, but
# removing it bumps the mtime of its directory, so the nearest existing parent stands in
newest_change=0
while IFS= read -r f; do
  if [ -f "$f" ]; then
    m="$(mtime "$f")"
  else
    d="$(dirname "$f")"
    while [ ! -d "$d" ]; do d="$(dirname "$d")"; done
    m="$(mtime "$d")"
  fi
  [ "$m" -gt "$newest_change" ] && newest_change="$m"
done <<< "$changed"

# newest mtime among the tracking files
newest_track=0
for f in .ai/DECISIONS.md .ai/PROGRESS.md .serena/memories/*.md; do
  [ -f "$f" ] || continue
  m="$(mtime "$f")"
  [ "$m" -gt "$newest_track" ] && newest_track="$m"
done

# tracking touched at least as recently as the newest relevant change: nothing to do
[ "$newest_track" -ge "$newest_change" ] && exit 0

cat <<'JSON'
{"decision":"block","reason":"Code, dependencies, docs config or CI changed but project tracking wasn't updated this session. Per AGENTS.md: append a dated entry to .ai/PROGRESS.md (always), update the relevant serena memory via write_memory if structure/conventions/commands/scope changed (codebase_structure for any new or moved module), add an ADR to .ai/DECISIONS.md for notable choices, and update README.md / docs/ if the public API or behaviour changed; then finish. If the change genuinely warrants no tracking update, add a one-line note in .ai/PROGRESS.md saying so (or commit the work), then finish."}
JSON
exit 0
