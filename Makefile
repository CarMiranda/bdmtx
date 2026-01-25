.PHONY: clean test lint

clean:
	find . -name "*.pyi" -type d -exec rm -rf {} +
	find . -name "__pycache__" -type d -exec rm -rf {} +
	rm -rf build .coverage public coverage.xml

test:
	uv run --group=test -- coverage run -m pytest
	uv run --group=test -- coverage combine || true
	uv run --group=test -- coverage report -m
	uv run --group=test -- coverage xml

lint:
	uv run -- pre-commit run --all-files
