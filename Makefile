.PHONY: check fix freeze


init-dev:
	pip install -r requirements.txt ruff==0.5.0

freeze:
	@echo "Freezing dependencies..."
	poetry lock --no-update
	poetry export --without-hashes -o requirements.txt


# Default target
.DEFAULT_GOAL := check
