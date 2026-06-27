.PHONY: format test verify check complexity

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

# Complexity Gate
complexity:
	@echo "Checking Cyclomatic Complexity (McCabe)..."
	xenon --max-absolute F --max-modules D --max-average B .
	@echo "Complexity gate passed!"
	@echo "Cyclomatic Complexity Analysis (C and below):"
	radon cc . -a -nc
	@echo "Maintainability Index (C and below):"
	radon mi . -nc

# The Mega Loop for agents
verify: format complexity test
	@echo "Pre-flight verification passed! Safe to submit."

# Alias for verify
check: verify
