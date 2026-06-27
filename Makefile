.PHONY: format test verify check

# Formatting and linting via Ruff
format:
	@echo "Running ruff format..."
	python3 -m ruff format .
	@echo "Running ruff check and applying safe fixes..."
	python3 -m ruff check --fix .

# Testing
test:
	@echo "Running unit tests (Global)..."
	python3 -m unittest discover tests
	@echo "Running unit tests (Nodes)..."
	python3 -m unittest discover nodes/tests

# The Mega Loop for agents
verify: format test
	@echo "Pre-flight verification passed! Safe to submit."

# Alias for verify
check: verify
