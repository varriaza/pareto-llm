#!/usr/bin/env bash
set -euo pipefail

echo "==> Installing dependencies..."
uv sync --extra dev

echo "==> Installing pre-commit hooks..."
uv run pre-commit install

echo "==> Detecting GPU backend and writing .env..."
uv run python scripts/init_env.py

# Install backend-specific extras
GPU_BACKEND=$(grep '^GPU_BACKEND=' .env 2>/dev/null | cut -d= -f2 | tr -d '"' || true)

if [ "${GPU_BACKEND}" = "cuda" ]; then
    echo "==> Installing CUDA extras..."
    uv sync --extra cuda
elif [ "${GPU_BACKEND}" = "mlx" ]; then
    echo "==> Installing MLX extras..."
    uv pip install "pareto-llm[mlx]"
fi

echo ""
echo "Optional: To install LiveBench extras (for the live_bench benchmark type):"
echo "  uv pip install \"pareto-llm[live-bench]\""

echo ""
echo "Setup complete. Run 'pareto-llm --help' to get started."
