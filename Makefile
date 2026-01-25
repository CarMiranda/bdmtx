.PHONY: build clean test lint

build:
	# docker build -t ${REGION}-docker.pkg.dev/${PROJECT}/${REPO}/bdmtx-train:${IMAGE_TAG} -f Dockerfile.train .
	docker build -t bdmtx-train:latest -f Dockerfile.train .

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

requirements.txt:
	uv export --no-hashes --no-annotate --no-header --no-dev --no-emit-project --format=requirements.txt > requirements.txt
