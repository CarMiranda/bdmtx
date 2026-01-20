.PHONY: install-dev test lint

install-dev:
	python -m pip install --upgrade pip
	pip install -r requirements-dev.txt

test:
	pytest -q

lint:
	flake8 src tests
