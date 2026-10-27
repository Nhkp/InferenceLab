.PHONY: install lint format typecheck test run mock up down build

install:
	uv sync --locked

lint:
	uv run --locked ruff check .
	uv run --locked ruff format --check .

format:
	uv run --locked ruff format .
	uv run --locked ruff check --fix .

typecheck:
	uv run --locked mypy

test:
	uv run --locked pytest -m "not gpu and not slow"

run:
	uv run --locked uvicorn inference_lab.gateway.app:app --host 127.0.0.1 --port 8000

mock:
	uv run --locked uvicorn inference_lab.benchmark.mock_server:app --host 127.0.0.1 --port 8001

up:
	docker compose up --build -d --wait

down:
	docker compose down

build:
	uv build
