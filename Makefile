# Format/lint checks for dattri_llm (mirrors dattri's workflow).

PYTHON = python3

.PHONY: check ruff darglint test

# The full required check: lint + formatting.
check: ruff

ruff:
	$(PYTHON) -m ruff check dattri_llm tests examples
	$(PYTHON) -m ruff format --check dattri_llm tests examples

# Optional: docstring-signature checking.  darglint reads .darglint (google
# style, strictness=long).  Slow on the large modules; not part of `check`.
darglint:
	$(PYTHON) -m darglint $(shell find dattri_llm -name "*.py" -not -path "*__pycache__*")

test:
	CUDA_VISIBLE_DEVICES=0 $(PYTHON) -m pytest -q
