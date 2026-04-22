.PHONY: help install dev test lint format serve bench bench-quick clean redis-up redis-down

help:
	@echo "AgentMesh — dev commands"
	@echo "  make install      Install runtime deps"
	@echo "  make dev          Install + pre-commit/editable mode"
	@echo "  make test         Run pytest"
	@echo "  make lint         Run ruff"
	@echo "  make format       Run ruff --fix"
	@echo "  make serve        Run the FastAPI server"
	@echo "  make bench        Run full benchmark (50 tasks)"
	@echo "  make bench-quick  Run a 10-task smoke benchmark"
	@echo "  make redis-up     Start local Redis via docker compose"
	@echo "  make redis-down   Stop local Redis"
	@echo "  make clean        Remove caches, build artefacts"

install:
	pip install -r requirements.txt

dev:
	pip install -e . -r requirements.txt

test:
	pytest -v --cov=agentmesh --cov-report=term-missing

lint:
	ruff check agentmesh tests benchmarks

format:
	ruff check --fix agentmesh tests benchmarks
	ruff format agentmesh tests benchmarks

serve:
	uvicorn agentmesh.api.app:app --host $${AGENTMESH_HOST:-0.0.0.0} --port $${AGENTMESH_PORT:-8000} --reload

bench:
	python -m benchmarks.run_benchmark --tasks all

bench-quick:
	python -m benchmarks.run_benchmark --tasks 10 --output benchmarks/results/quick.json

redis-up:
	docker compose up -d redis

redis-down:
	docker compose down

clean:
	rm -rf build dist *.egg-info
	rm -rf .pytest_cache .ruff_cache .mypy_cache .coverage htmlcov
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
