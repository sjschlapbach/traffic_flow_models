# Task Completion Checklist

When you finish a task:

1. **Tests:** inside the `venv/`, `pytest` must pass. New non-trivial code gets tests in
   `tests/<module>_test.py`, including numerical edge cases. For a bug fix, first write a test that
   reproduces it.
2. **Format + lint:** run `black .`, then `black --check .` must report every file unchanged;
   `pylint src --errors-only` and `pylint tests --errors-only` must be clean (CI gate); `pylint src`
   must show no new warnings in the code you touched; `mypy` must report no issues (strict).
3. **Demos:** if model, network, controller or calibration behaviour changed, run the affected demo
   from `mem:suggested_commands`. CI smoke-tests every demo.
4. **Public API:** export a new public class or function from `src/traffic_flow_models/__init__.py`
   and add it to `__all__`. Do not reorder positional constructor parameters.
5. **Dependencies:** pin a new third-party import in `pyproject.toml` in the same task.
6. **Docs:** if the public API, usage or installation changed, update `README.md` (usage examples,
   project structure) and the matching `docs/` MkDocs pages (`api/`, `models/`, `concepts/`,
   `control/`, `simulation/`, `experimental/`, `getting-started/`). Docstrings feed `docs/api/`.
   Run `mkdocs build` if you edited `docs/` or `mkdocs.yml`.
7. **Update tracking:**
   - Update the relevant **serena memory** (serena `write_memory`, or edit the file directly) if reference facts changed: `mem:codebase_structure`
     (a new or moved module, class, controller or model, always), `mem:code_style_and_conventions`
     (style), `mem:suggested_commands` (commands, CI, dependencies), `mem:project_overview`
     (scope, stability tiers, fragile areas).
   - Append a dated entry to `.ai/PROGRESS.md`, **always**.
   - Add an ADR to `.ai/DECISIONS.md` if you made a non-obvious choice (algorithm, modelling
     assumption, API change, parameter value).
8. **Do not commit or push** (the human working with the agent does that). Never hand-edit `CHANGELOG.md`, and never run a
   release command (`git-cliff`, `git tag`, `twine`, `mkdocs gh-deploy`, `gh release`); releases are
   manual-only for the maintainers (`mem:suggested_commands`).

See `mem:ai_tracking_system` for how project knowledge is split.
