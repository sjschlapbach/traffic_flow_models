# Integrations — traffic_flow_models

> External tools, simulators, and their config patterns. Update when an external dependency
> changes. Deliberately excludes pip-installed Python libraries like CasADi/numpy/scipy — those
> belong in `.memory-bank/techContext.md`; this file is for things with their own CLI/binary/
> config surface outside the Python packaging system.

## git-cliff (changelog automation)

Config: `cliff.toml`. `conventional_commits = true`, `filter_unconventional = true`,
`require_conventional = false`. Commit-prefix → changelog-section mapping: `feat`→Features,
`enhance`→Enhancements, `fix`→Bug Fixes, `doc`→Documentation, `perf`→Performance,
`refactor`→Refactor, `style`→Styling, `test`→Testing, `chore`/`ci`→Miscellaneous Tasks,
`revert`→Revert, commit body matching `.*security`→Security, everything else→Other.
`skip = true` parsers exclude `chore(release): prepare for`, `chore(deps...)`, `chore(pr)`,
`chore(pull)` commits from the changelog entirely. `sort_commits = "oldest"`,
`topo_order_commits = true`, `initial_tag = "0.1.2"`, `breaking_always_bump_major = true`.

Used both for `git-cliff --bump` (local version-bump step before tagging a release) and inside
`release.yml` (`git-cliff --bump --output CHANGELOG.md`, tag-triggered). See the release mechanics
already summarized in `.memory-bank/techContext.md` — never hand-edit `CHANGELOG.md`.

## MkDocs Material + mkdocstrings (docs site)

Config: `mkdocs.yml`. Theme: `material` (indigo palette, light/dark toggle, tabs/sections
navigation, search, code copy/annotate). `mkdocstrings` python handler config: `paths: [src]`,
Google docstring style, `show_source: true`, members ordered by source. MathJax enabled via
`pymdownx.arithmatex` for rendering model equations. Nav mirrors the `docs/` directory structure
(see `.agent_docs/STRUCTURE.md`).

Built by `.github/workflows/docs_build.yml` (`mkdocs build`, every push/PR — verifies the site
builds) and deployed by `docs_release.yml` (`mkdocs gh-deploy --force`, tag-triggered, publishes
to GitHub Pages at `https://sjschlapbach.github.io/traffic_flow_models/`). `docs/` is user-facing
and must be kept in sync with the implementation per `AGENTS.md`'s mandatory update protocol.
