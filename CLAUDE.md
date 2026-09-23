# CLAUDE.md

This file guides coding agents working in this repository. It states what the
library is, how it is laid out, how it is built and tested, and the policies
that apply to every change. Read the current source for any interface detail;
this file does not describe signatures.

## Project Overview

`dattri_llm` is a PyTorch library for training data attribution (TDA) at LLM
scale: given a trained or training model, it scores how much each training
example contributed to a model output. Every supported method works from
per-example gradients, and the library is the infrastructure that makes those
gradients cheap to capture, store and compare:

- **Efficiency** — gradients are kept in a factorized form (a layer's input
  activation and output gradient) and expanded to a dense per-example gradient
  only where a cost model finds that cheaper.
- **Compatibility** — capture attaches to an existing training loop through
  autograd hooks, without changes to the loop or its configuration, under
  single-device, DDP and FSDP training and inside frameworks such as Hugging
  Face Transformers, TRL and OLMo.
- **Extensibility** — attribution methods are attributors over one gradient
  stream, and applications that act during training (online data selection,
  optimizer-state recording) are callbacks on the capture path.

## Build & Development

```bash
conda create -n dattri-llm python=3.10 && conda activate dattri-llm
pip install -e ".[dev]"            # library + test and integration dependencies
pip install -e ".[transformers]"   # only the Hugging Face integration
```

Python 3.10 or newer; `torch` is the only hard dependency. The gradient layer
imports with torch alone; `transformers`, `dattri`, `trl` and `ai2-olmo` are
optional and imported lazily where they are used.

## Commands

```bash
make check                                  # ruff lint + ruff format --check (required)
make test                                   # the test suite on one GPU
pytest -q -m "not gpu"                      # CPU-only tests
pytest tests/gradient/test_hooks.py -q      # one file
make darglint                               # docstring/signature agreement (optional, slow)
```

Run `make check` before every commit; CI enforces it.

## Code Style

- Ruff, configured in `pyproject.toml` (`select = ["ALL"]` with the listed
  exceptions); `ruff format` is the formatter.
- Google-style docstrings on public functions and classes; type hints on every
  function.
- Naming: `PascalCase` classes, `snake_case` functions and variables,
  `UPPER_SNAKE_CASE` constants.
- Public functions are called across modules; a `_`-prefixed name is private to
  its module.
- Names say what a thing governs: a residency argument names the cache it
  applies to, a constant names what it is for.

## Testing

pytest; the `tests/` tree mirrors `dattri_llm/`. Tests build small models
with synthetic data so the CPU suite runs in a few minutes. GPU-only tests are
marked `@pytest.mark.gpu`. Multi-process tests spawn `gloo` workers so they
run on CPU. A change to gradient capture or to an attribution method comes
with a test that pins the guaranteed behaviour; correctness is checked
against an unhooked or single-device reference where one exists.

## Contributing

### Project Structure

- `dattri_llm/` — the library.
- `examples/` — runnable scripts, one directory per topic with a README.
- `tests/` — the test suite, mirroring `dattri_llm/`.
- `experiments/` — the code behind the reported measurements, ending at result
  files.

### CI/CD

- `pytest.yml` and `lint.yml` run on every push and pull request (CPU tests
  and `make check`); `examples_test.yml` runs the examples that need no large
  download.
- Expensive checks run on a pull-request comment: `run gpu test` (the GPU
  suite), `run expensive examples` (examples that download checkpoints),
  `run darglint`; `olmo_test.yml` runs the OLMo integration.
- A non-trivial change adds or updates tests.

### Pull Requests

- Open an issue before a large change; discuss the interface there.
- Keep a pull request to one change; mark it as a draft while it is in
  progress.
- Pull requests are squash-merged; the commit message states what the change
  does.
- If an example or documentation test fails after a merge, fix it in a
  follow-up pull request promptly.
