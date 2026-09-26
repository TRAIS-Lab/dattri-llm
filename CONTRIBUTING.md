# Contributing to dattri-llm

Thanks for helping improve `dattri-llm`. This page covers the development
setup, the checks CI runs, and the conventions a pull request should follow.

## Development setup

```bash
git clone https://github.com/TRAIS-Lab/dattri-llm
cd dattri-llm
conda create -n dattri-llm python=3.10 && conda activate dattri-llm
pip install -e ".[dev]"     # the library plus test and integration dependencies
```

Python 3.10 or newer. `torch` and `tqdm` are the only hard dependencies; the
other integrations (`transformers`, `dattri`, `trl`, `ai2-olmo`) are optional
extras, listed in the README's installation section.

## Lint and format

CI (`.github/workflows/lint.yml`) runs ruff at a pinned version, because the
configuration (`select = ["ALL"]` with `preview = true`) is version-sensitive:

```bash
pip install ruff==0.15.20
ruff check .
ruff format --check .       # `ruff format .` applies the formatting
```

Run both before every commit. The rule set and its documented exceptions live
in `pyproject.toml` under `[tool.ruff]`.

## Tests

```bash
pytest -q -m "not gpu" tests                 # the CPU suite CI runs
pytest tests/gradient/test_hooks.py -q       # one file
make test                                    # the full suite on one GPU
```

- The `tests/` tree mirrors `dattri_llm/`. Tests build small models with
  synthetic data so the CPU suite runs in a few minutes.
- GPU-only tests are marked `@pytest.mark.gpu`. Multi-process tests spawn
  `gloo` workers so they run on CPU.
- A change to gradient capture or to an attribution method comes with a test
  that pins the guaranteed behaviour, checked against an unhooked or
  single-device reference where one exists.

## Examples

`examples/` holds runnable scripts, one directory per topic with a README.
CI (`.github/workflows/examples_test.yml`) runs the ones that need no large
download; keep a new example fast on CPU and add it there.

## Code style

- Google-style docstrings on public functions and classes; type hints on every
  function.
- `PascalCase` classes, `snake_case` functions and variables,
  `UPPER_SNAKE_CASE` constants. A `_`-prefixed name is private to its module.
- Names say what a thing governs: a residency argument names the cache it
  applies to, a constant names what it is for.

## Continuous integration

- `pytest.yml` (CPU tests), `lint.yml` (ruff) and `examples_test.yml` (fast
  examples) run on every push to `main`/`dev` and on pull requests that touch
  the library, tests, examples or `pyproject.toml`.
- Expensive checks run when a pull-request comment contains a trigger phrase:
  `run gpu test` (the GPU suite), `run expensive examples` (examples that
  download checkpoints or install `ai2-olmo`) and `run darglint` (docstring
  and signature agreement). `olmo_test.yml` runs the OLMo integration.

## Pull requests

- Open an issue before a large change and discuss the interface there.
- Keep a pull request to one change; mark it as a draft while it is in
  progress. A non-trivial change adds or updates tests.
- Pull requests are squash-merged; the commit message states what the change
  does.
- If an example or documentation check fails after a merge, fix it promptly in
  a follow-up pull request.
