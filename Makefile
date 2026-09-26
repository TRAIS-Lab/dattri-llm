# Format/lint checks for dattri_llm (mirrors dattri's workflow).

PYTHON = python3

.PHONY: check ruff darglint test

# The full required check: lint + formatting, the same commands CI runs
# (.github/workflows/lint.yml) with the ruff it pins (ruff==0.15.20, in the
# dev extra; pyproject's required-version refuses any other).
check: ruff

ruff:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .

# Optional: docstring-signature checking.  darglint reads .darglint (google
# style, strictness=long).  Slow on the large modules; not part of `check`.
darglint:
	$(PYTHON) -m darglint $(shell find dattri_llm -name "*.py" -not -path "*__pycache__*")

test:
	CUDA_VISIBLE_DEVICES=0 $(PYTHON) -m pytest -q
