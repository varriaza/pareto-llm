#!/usr/bin/env bash
set -euo pipefail

echo "==> Installing dependencies..."
uv sync --extra dev

echo "==> Installing pre-commit hooks..."
uv run pre-commit install

echo "==> Detecting GPU backend and writing .env..."
uv run python scripts/init_env.py

echo ""
echo "Setup complete. Run 'pareto-llm --help' to get started."
