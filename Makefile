.PHONY: setup test lint train translate download-data help

setup:
	uv sync --dev

test:
	uv run pytest

lint:
	uv run ruff check .

download-data:
	uv run python examples/download_data.py

train:
	uv run python examples/train_fr_en.py

translate:
	uv run python examples/translate_fr_en.py --help

help:
	@echo "Available targets:"
	@echo "  setup     Install dependencies with uv"
	@echo "  test      Run pytest"
	@echo "  lint      Run ruff"
	@echo "  download-data  Download and split the Europarl FR-EN corpus"
	@echo "  train     Train the EN-FR model"
	@echo "  translate Show translation CLI help"
