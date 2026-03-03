.PHONY: setup test lint train translate help

setup:
	uv sync --dev

test:
	uv run pytest

lint:
	uv run ruff check .

train:
	uv run python examples/train_fr_en.py

translate:
	uv run python examples/translate_fr_en.py --help

help:
	@echo "Available targets:"
	@echo "  setup     Install dependencies with uv"
	@echo "  test      Run pytest"
	@echo "  lint      Run ruff"
	@echo "  train     Train the EN-FR model"
	@echo "  translate Show translation CLI help"
