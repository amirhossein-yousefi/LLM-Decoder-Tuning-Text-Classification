.PHONY: venv install test lint fmt

venv:
	python -m venv .venv

install:
	pip install -e .[dev]

test:
	pytest -q

lint:
	ruff check llm_cls tests

fmt:
	ruff format llm_cls tests

