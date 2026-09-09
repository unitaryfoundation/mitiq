# AGENTS.md

Orientation for AI coding agents working in the Mitiq repository.
Humans should read [`CONTRIBUTING.md`](CONTRIBUTING.md) — this file is the condensed,
agent-facing version of it plus the conventions that aren't written down elsewhere.

**Policy:** contributors may use whatever tools they like, but **there must be a human in
the loop**. Don't open PRs or push to `main` without the maintainer explicitly asking.

**This file is the source of truth for agent instructions.** If you learn something about
this repo worth persisting, add it here rather than starting a `CLAUDE.md`, `.cursorrules`,
or an editor-specific rules file. Claude Code does not read `AGENTS.md` on its own; to wire
it up locally, either add a `CLAUDE.md` containing `See @AGENTS.md for project
instructions.` or run `/import`. Neither is checked in.

## What Mitiq is

An open-source toolkit for quantum error mitigation (QEM). The core idea running through
the whole codebase: a user hands Mitiq a circuit and an *executor* (a function that runs a
circuit and returns a result), and Mitiq returns an error-mitigated expectation value.
Circuits may come from any supported frontend; internally everything is Cirq.

## Environment and commands

Mitiq uses [`uv`](https://docs.astral.sh/uv/). **Never** call bare `pip`, `python`,
`pytest`, `ruff`, or `mypy` — prefix with `uv run`, or use the `make` targets (which
already handle it).

| Task | Command |
| --- | --- |
| Install everything | `make install` (= `uv sync --all-extras --all-groups`) |
| Lint + format check | `make check-format` |
| Auto-fix | `make format` |
| Type check | `make check-types` (mypy over `mitiq/`) |
| Both of the above | `make check-all` |
| Test one module | `uv run pytest mitiq/zne -x -q` |
| Full suite (no pyQuil) | `make test` |
| Everything incl. pyQuil | `make test-all` |
| Docs | `make docs-lite` while writing; see [Documentation](#documentation) |

Before saying a change is done, `make check-all` must pass and the tests for the touched
module must pass. That's exactly what the `.git-hooks/pre-commit` hook enforces.

### Known environment limitations

- **pyQuil tests need a running QVM + quilc** (`docker run --rm -idt -p 5000:5000
  rigetti/qvm -S` and `... -p 5555:5555 rigetti/quilc -R`). `make test` deliberately
  excludes `mitiq/interface/mitiq_pyquil`. Failures there without Docker running are
  environmental, not regressions.
- **`make docs` is slow** — it executes notebooks. Prefer `make docs-lite` while iterating.
  If the environment is missing docs dependencies, fall back to
  `uv run --group docs sphinx-build -b html docs/source docs/build`. Some notebook
  execution errors (pyQuil/QVM, optional heavy deps) are pre-existing; check whether they
  reproduce on `main` before treating one as caused by your change.

## Repository layout

```
mitiq/
  <technique>/          one directory per QEM technique, named by acronym
    __init__.py         public exports for the technique
    <technique>.py      the main entry points
    tests/              tests live beside the code they test
  experimental/         techniques not covered by semantic versioning
  interface/            frontend <-> Cirq conversion, one mitiq_<frontend> pkg each
  benchmarks/           circuit generators used by tests and calibration
  calibration/          picks technique + parameters for a given backend
  executor/             Executor: batching, caching, result-type handling
  observable/           PauliString, Observable
  typing.py             QPROGRAM, QuantumResult, MeasurementResult, SUPPORTED_PROGRAM_TYPES
  utils.py              shared helpers, incl. qem_methods()
docs/source/            Sphinx + MyST; guide/, examples/, apidoc.md
scripts/                benchmarking scripts, not part of the package
```

Stable techniques: `cdr`, `ddd`, `lre`, `pec`, `pt`, `qse`, `raw`, `rem`, `zne`.
Experimental: `mitiq.experimental.{deb, pea, shadows, trex, vd}`.

Each technique generally exposes a `execute_with_<x>` (all-in-one) plus a
`construct_circuits` / `combine_results` pair (two-stage, for users who run circuits
themselves). Follow that shape when adding one.

## Conventions that will bite you

- **Every source file starts with the GPL v3 copyright header.** Copy it from a
  neighbouring file.
- **Line length is 79.** Ruff enforces `E`, `F`, and `I` (isort). `__init__.py` is
  excluded from Ruff.
- **Full type annotations are required** — mypy runs with `disallow_untyped_defs`,
  `disallow_incomplete_defs`, and `disallow_any_generics`. Tests are excluded
  (`mitiq.*.tests*`).
- **Google-style docstrings** (PEP 257), with `Args:` / `Returns:` / `Raises:` sections.
- **Tests are nested**, in a `tests/` directory beside the module. The lone exception is
  anything needing a QVM, which goes in `mitiq/interface/mitiq_pyquil/tests`.
- **Frontend-agnostic code goes through `mitiq/interface/conversions.py`.** Use
  `convert_to_mitiq` / `convert_from_mitiq`, or decorate with
  `@accept_qprogram_and_validate` (converts to Cirq, runs your function, converts back,
  and checks the result is consistent) rather than special-casing frontends.
- **Experimental modules warn on import** (`FutureWarning`) and old top-level paths
  (`mitiq.pea`, `mitiq.shadows`, `mitiq.vd`) are shims that `raise ImportError` pointing
  at `mitiq.experimental.*`. Don't "fix" those by re-adding the old module.
- **`INTEGRATIONS.txt`** lists the supported frontends and is read by tooling — update it
  if you add one.
- Python support is `>=3.11,<3.13`.

## What good code looks like here

Trimmed from `mitiq/zne/zne.py` — annotations on every argument and the return, a
Google-style docstring, and ``double backticks`` for code references (autodoc renders the
docstring as RST, so single backticks come out as italics):

```python
def construct_circuits(
    circuit: QPROGRAM,
    scale_factors: list[float],
    scale_method: Callable[[QPROGRAM, float], QPROGRAM] = fold_gates_at_random,
) -> list[QPROGRAM]:
    """Given a circuit, scale_factors and a scale_method, outputs a list
       of circuits that will be used in ZNE.

    Args:
        circuit: The input circuit to execute with ZNE.
        scale_factors: An array of noise scale factors.
        scale_method: The function for scaling the noise of a quantum circuit.
            A list of built-in functions can be found in ``mitiq.zne.scaling``.

    Returns:
        The scaled circuits using the scale_method.
    """
```

## Boundaries

**Ask before doing any of these** — they have consequences beyond the diff:

- Adding or changing a runtime dependency in `pyproject.toml` (it moves `uv.lock` and every
  downstream installer with it).
- Changing a public API signature. Mitiq follows semantic versioning; renames and
  parameter changes are breaking.
- Adding a new frontend integration — it touches `INTEGRATIONS.txt`, `mitiq/interface/`,
  the optional-dependency extras, and CI together.
- Anything large enough to need an RFC (see `CONTRIBUTING.md`).

**Never:**

- Bump `version` in `pyproject.toml`. That belongs to the release process
  (`docs/source/release.md`).
- Add a `# type: ignore` or loosen an annotation just to get `make check-types` to pass.
  The ~55 in the tree are deliberate; a new one needs a reason in the diff.
- Re-add the removed top-level `mitiq.pea` / `mitiq.shadows` / `mitiq.vd` modules — the
  `ImportError` shims are intentional.
- Commit build output: `docs/build/`, `.coverage`, `coverage.xml`, `mitiq.egg-info/`.

## Documentation

Docs are part of the deliverable, not a follow-up PR — a technique with no guide page is
not done. Everything lives in `docs/source` and builds with Sphinx + MyST.
Full detail is in [`docs/CONTRIBUTING_DOCS.md`](docs/CONTRIBUTING_DOCS.md).

### Building

| Command | Use it when |
| --- | --- |
| `make docs-lite` | **Default while writing.** Sets `DOCS_LITE=1`, skips all notebook execution. Seconds, not minutes. |
| `make docs` | Full build; executes every notebook. Run once before you call the work done. |
| `make docs-clean` | Fresh build, no cache. Needed after editing a toctree or `conf.py`. |
| `make linkcheck` | Checks external links. |

`SKIP_PYQUIL=1` skips only the pyQuil notebooks, which need a running QVM. Output goes to
`docs/build/` (gitignored). On a PR, the rendered preview is behind **Details** on the
`docs/readthedocs.org:mitiq` line of the merge box — that link works even when the build
fails, which is usually the fastest way to see what broke.

### Layout

- `docs/source/guide/` — the user guide. Each technique is a landing page (`zne.md`) plus
  five numbered pages answering a fixed set of questions:
  `zne-1-intro.md` (how do I use it), `-2-use-case.md` (when should I),
  `-3-options.md` (what else can I configure), `-4-low-level.md` (what actually happens),
  `-5-theory.md` (why it works). Follow that shape; don't invent a new page layout.
- `docs/source/examples/` — end-to-end tutorials. Start from `examples/template.md`.
- `docs/source/apidoc.md` — API reference, generated from docstrings.

### Rules that bite silently

These fail by producing a *successful* build with wrong or missing output.

- **A file not listed in a toctree still builds — it's just unreachable.** Add guide pages
  to `guide/guide.md`, examples to `examples/examples.md`, top-level pages to `index.md`.
- **Notebook cells execute during `make docs`.** Any new import must be added to the
  `docs` entry of `[dependency-groups]` in `pyproject.toml`, or the build breaks in CI
  while `make docs-lite` looked fine locally.
- **Write examples as MyST `.md`, not `.ipynb`** — they diff in review. Convert with
  `uv run jupytext notebook.ipynb --to myst`.
- **A new example needs three edits**, not one: the `.md` file, a listing in
  `examples/examples.md`, and a thumbnail in `_thumbnails/` registered in the
  `nbsphinx_thumbnails` dict in `conf.py`.
- **API docs need `eval-rst`**, because myst-parser can't parse Markdown docstrings yet:
  an `.. automodule:: mitiq.new_module` block with `:members:` inside an
  ` ```{eval-rst} ` fence in `apidoc.md`.
- **Citations go in `docs/source/refs.bib`** (kept alphabetical), cited as
  `` {cite}`key` ``. Not `mitiq.bib` — that's a symlink to `CITATION.bib` and isn't in
  `bibtex_bibfiles`.
- Use `` ```{code-block} python `` for a snippet that should be highlighted but not run.

### Writing style

Closest reference points are the [Google developer documentation style
guide](https://developers.google.com/style/) and the [Astro docs writing
guide](https://contribute.docs.astro.build/guides/writing-style/).

- **Address the reader as "you"; use the imperative for instructions.** "Pass an
  `Executor`", not "we can pass an executor" or "one may wish to pass". Avoid "we", "us",
  and "let's".
- **Active voice, short sentences, plain vocabulary.** A large share of readers are
  reading in a second language, and a larger share are physicists rather than software
  engineers — optimize for both.
- **"Mitiq" is capitalized in prose**; lowercase `mitiq` only as a code token
  (`from mitiq import zne`).
- **Expand a technique's name on first use per page**, then use the acronym: "zero-noise
  extrapolation (ZNE)". Acronyms stay uppercase in headings.
- **Headings**: sentence case (the prevailing style), no trailing punctuation, short —
  they become sidebar entries.
- **Every claim about a noise model or an error bound needs a citation.** Add it to
  `refs.bib` rather than describing the result loosely.
- **Link out for quantum-computing background instead of re-explaining it**, and document
  how to use a frontend *with Mitiq* rather than how that frontend works.
- **Show, then explain.** A runnable snippet with real output beats a paragraph
  describing what the function would return.

## Adding a new QEM technique

Substantial features require an accepted RFC first — see "Proposing a new feature" in
`CONTRIBUTING.md`. Once approved, the checklist is:

1. `mitiq/<acronym>/` with the code, and `mitiq/<acronym>/tests/`.
2. Export it from `mitiq/__init__.py` (or `mitiq/experimental/` if it isn't API-stable).
3. Add it to `qem_methods()` in `mitiq/utils.py`.
4. Add the module to `docs/source/apidoc.md`.
5. Add the user-guide pages under `docs/source/guide/` — landing page plus the five
   numbered pages, and list them in `guide/guide.md`. See [Documentation](#documentation).
6. Add a one-line entry to `docs/source/guide/glossary.md`.
7. Update the "Quick Tour" section of `README.md`.

## Pull requests

- Branch off `main`; never commit directly to it.
- Don't touch `CHANGELOG.md` in a feature PR — maintainers draft it at release time from
  the merged PRs (see `docs/source/release.md`).
- CI (`.github/workflows/build.yml`) runs format + mypy, the full suite on Python 3.11 and
  3.12 (with QVM/quilc containers), and a **"test without 3rd party packages"** job on a
  core-only install. That last one is the usual surprise: importing an optional frontend
  (qiskit, braket, pennylane, …) at module scope outside `mitiq/interface/mitiq_<x>/`
  will fail it. Keep such imports local to the function or the frontend package.
  `docs-build.yml` builds the docs separately.
- **Disclose AI use.** Add an `Assisted-by: <tool>` trailer to the commit and fill in
  the AI use section of the PR template. The PR description itself should be written by
  the human author — see "AI use policy" in `CONTRIBUTING.md`.
