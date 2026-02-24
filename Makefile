PYTHON ?= python3

.PHONY: test build clean bench

test:
	$(PYTHON) -m pytest

build:
	$(PYTHON) -m build

bench:
	bash scripts/run_multiscale_bench.sh

clean:
	rm -rf build dist .pytest_cache .ruff_cache src/*.egg-info
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
	find . -name '*.pyc' -delete
