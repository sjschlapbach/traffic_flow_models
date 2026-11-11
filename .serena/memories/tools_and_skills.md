# Tools & Skills

The plugins and skills below are Claude Code's. Other agents can run the same serena MCP server
themselves (`uvx --from git+https://github.com/oraios/serena serena start-mcp-server`, project =
repository root) or read `.serena/memories/*.md` as plain files.

Prefer these over ad-hoc approaches:

- **serena MCP**: semantic code intelligence and the memory store. It is enabled for this project
  via `enabledPlugins` in `.claude/settings.json`; it is off in the user's global settings, so the
  project setting is what starts it. Memory tools: `list_memories`, `read_memory`, `write_memory`.
  Code tools: `get_symbols_overview`, `find_symbol`, `find_referencing_symbols`, and symbol-level
  edits (`replace_symbol_body`, `rename_symbol`, `safe_delete_symbol`). Use them on the big files
  (`simulation.py`, `calibrator.py`, `network.py`, `metanet.py`) instead of reading whole files.
- **context7 MCP**: up-to-date library docs. Call `resolve-library-id`, then `query-docs`. Use it
  before any version-sensitive `numpy`, `scipy` (`least_squares`, `qmc`), `casadi`, `networkx`,
  `matplotlib`, `opencv` or `mkdocs` API. Do not write library calls from memory.
- **dataviz skill**: before you change or add a plot.
- **run skill**: actually execute a demo to confirm the real output.
- **code-review / simplify skills**: on the working diff (correctness / cleanup).

The read-only serena/context7 tools, `write_memory`, venv activation, `pytest`, `black`, `pylint`,
`pip install -e`, `pip show/list/freeze` and `mkdocs build` are pre-approved in
`.claude/settings.json`, so they run without permission prompts.

If the serena tools are missing, read and edit `.serena/memories/*.md` as plain files and tell the
user that the plugin did not start (check `/mcp`; serena runs via `uvx`).
