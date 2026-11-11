# Code Style & Conventions

Follow the conventions of the existing code so that new code reads the same.

- **Formatting: black is required** (default settings, line length 88; CI uses
  `psf/black@stable`). Run `black .` after every code change. Black owns the formatting, so do not
  hand-format against it.
- **Linting: pylint.** Config lives in `.pylintrc` (only `cv2` false-positive handling). The CI gate
  is `pylint src --errors-only` and `pylint tests --errors-only`. The full `pylint src` score is
  below 10, mostly from line-too-long and too-many-* warnings in existing code. New or changed
  code must add no new warnings; fix the warnings you touch.
- **No inline lint/type suppressions without asking (strict).** Before you add any
  `# pylint: disable=...` or `# type: ignore`, **stop and ask the user** whether to suppress
  the rule on that line, disable it repo-wide in `.pylintrc`, or fix the code. The default is to
  fix the code. Leave the existing `# type: ignore`s in place.
- **Typing: strict mypy is required.** `mypy` (config in `pyproject.toml`: `strict = true`,
  `warn_unreachable`, package `traffic_flow_models` only) must report no issues; CI job
  `typecheck`. The package ships `py.typed`, so its annotations are public API for users' type
  checkers. Every function and method is annotated.
  - **Avoid `Any`/`object`** wherever the real type is known. Accepted exceptions: raw parsed
    JSON (`JSONDict`), networkx edge attribute dicts, and everything casadi (it ships no types,
    so `casadi.SX` is `Any` to mypy, an override in `pyproject.toml` ignores its missing
    imports).
  - **Shared types live in `traffic_flow_models/types.py`** as simple `type` aliases (`Link`,
    `RampController`, `PackedVector`, `TimeProfile`, `TurningRateProfile`, `TimeSeriesById`,
    `JSONDict`) plus two small TypedDicts (`ModelOptions`, `ParameterSearchRun`). Use them
    instead of re-spelling the union or callable; add a new one only when a type is passed
    between several modules. Keep it simple: unions and aliases, no Protocols or class
    hierarchies when a union works.
  - Typing style is mixed: built-in generics and `X | None` in most modules, `typing`'s
    `Optional`/`Tuple`/`Union` in some (`model/helpers.py`, `calibrator.py`). Match the
    surrounding file, and prefer the built-in style in new code. `Literal` for string enums,
    `NDArray[np.float64]` for numpy arrays, `ArrayLike` for anything passed to `np.asarray`.
  - For imports needed only in annotations, use `if TYPE_CHECKING:` with a quoted annotation
    (`onramp: "Onramp"`); this avoids circular imports between `network/` and `controller/`.
  - The `# type: ignore`s on `casadi.SX.sym(...)` in `ctm.py`/`metanet.py` are for pyright
    (Pylance), which infers casadi's wrong SWIG signatures. mypy's unused-ignore warning is off
    for those two modules only. Keep them.
- **Imports:** subpackage `__init__.py` files use relative imports; modules import other package
  modules absolutely (`from traffic_flow_models.network.onramp import Onramp`). Tests import the
  public API (`from traffic_flow_models import ...`).
- **Docstrings: Google style** (`Args:` / `Returns:` / `Raises:`). mkdocstrings renders them
  into `docs/api/`, so they are user-facing. Give units where relevant (veh/h, veh/km/lane, km,
  h). Required on public modules, classes and functions.
- **Comments (strict):** every inline `#` comment starts **lower-case**, with no exceptions. If
  it would start with a class name, constant or acronym, reword it. Comment only where it adds
  meaning the code does not already carry: the why, a modelling assumption, a unit, an index
  convention. Do not restate the next line and do not leave commented-out code.
- **Comments describe the current code only (strict).** Never mention history or the change that
  produced the code: no "copied", "moved from", "now", "no longer", "previously", "changed to",
  "was", "fixed". This also holds for docstrings, config-file comments, README, `docs/` and the
  serena memories. History belongs only in `.ai/PROGRESS.md` / `.ai/DECISIONS.md` and commit
  messages. When code changes, re-read the nearby comments and fix any that no longer match.
- **CasADi first:** model equations are symbolic `SX` graphs. Use `casadi.fmin`/`fmax`/`if_else`,
  never Python `min`/`max`/`if` on symbolic values. Change physics in the graph construction
  (`ctm.py`/`metanet.py`), not in a numeric loop.
- **Input validation:** constructors and boundary entry points raise `ValueError` with a short
  message on invalid input (negative gain, setpoint, index, …). Sanitize external inputs (results
  and ground-truth JSON, config files, CLI args) before they reach a model or the calibrator.
- **Paths:** never hardcode absolute paths. Public APIs take file paths as `str`
  (`save_to_json(filepath: str)`, `results_dir: str | None`), and the package builds them with
  `os.path.join` / `os.makedirs`. Generated output goes under `results/` (git-ignored), never
  into the repo.
- **Named constants** instead of magic numbers or strings. Prefer explicit over clever.
- **No functions inside functions (strict).** Never define a `def` inside a function or method,
  and no closure factories (a function or lambda that builds and returns another function). Put
  helpers at module level (private `_name`) or make them a method / `@staticmethod`, and pass
  what they need as arguments; to bind a value per call site use
  `functools.partial(_module_helper, name=value)`. A short inline lambda used directly as a
  value or argument (`key=lambda x: ...`) is fine. Known exceptions in existing code: nested
  helpers in `network.py` (`nodes_with_outgoing_link`, `nodes_with_incoming_link`,
  `_link_style`, `_ensure`, `_draw_group`), `simulation.py` (`_extend_time`, `_interp_2d`,
  `_interp_1d`, and the outer-lambda turning-rate default in
  `_augment_network_initialization`), `calibrator.py` (`residual_func`) and `tests/metanet_test.py`
  (`run_model`); lift them out when you change that code.
- **Public API stability:** never reorder or rename positional constructor parameters of exported
  classes. Add new parameters as keyword arguments with defaults at the end.
- **Dependencies:** pinned in `pyproject.toml` (`package==x.y.z`); no `requirements.txt`.
- **Tests:** flat `tests/`, `<module>_test.py`. Tests are grouped as methods of `class Test<Thing>`
  classes (plain `def test_*` functions only in `controller_test.py`). Assertions use `np.isclose`,
  `np.testing.assert_array_equal`/`assert_array_almost_equal`/`assert_allclose`, and
  `pytest.raises` for validation errors. Files go to `tmp_path`. Every non-trivial function gets
  tests, including the numerical edge cases (free-flow/congestion transitions, capacity limits,
  degenerate topologies).
- **Numerical risks:** call out CFL violations, METANET divergence or oscillation, degenerate
  geometry and optimizer non-convergence explicitly when a change could cause them. Check the
  signatures of numpy/scipy/casadi/networkx/matplotlib calls (context7, `mem:tools_and_skills`)
  rather than writing them from memory.
- **Commits** use Conventional Commits (`feat`, `fix`, `chore`, `doc`, `refactor`, …; `feat!` or
  a `BREAKING CHANGE:` footer for breaking changes) because git-cliff builds the changelog from
  them. Agents never commit; the human does.
