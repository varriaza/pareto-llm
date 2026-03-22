#!/usr/bin/env bash
set -euo pipefail

# ─── GPU backend selection ────────────────────────────────────────────────────
echo "Which GPU backend are you using?"
echo "  1) CUDA  (NVIDIA GPU)"
echo "  2) MLX   (Apple Silicon)"
echo "  3) None"
read -rp "Enter 1, 2, or 3: " choice

case "${choice}" in
    1) GPU_BACKEND="cuda" ;;
    2) GPU_BACKEND="mlx" ;;
    3) GPU_BACKEND="none" ;;
    *) echo "Invalid choice. Exiting."; exit 1 ;;
esac

# ─── Install Python 3.12 ─────────────────────────────────────────────────────
echo "==> Installing Python 3.12..."
uv python install 3.12

# ─── Install base + dev dependencies ─────────────────────────────────────────
echo "==> Installing dependencies..."
uv sync --extra dev

# ─── Install backend-specific extras ─────────────────────────────────────────
if [ "${GPU_BACKEND}" = "cuda" ]; then
    echo "==> Installing CUDA extras (llama-cpp-python + uvicorn)..."
    uv sync --extra cuda --extra dev
elif [ "${GPU_BACKEND}" = "mlx" ]; then
    echo "==> Installing MLX extras..."
    uv sync --extra mlx --extra dev
fi

# ─── Write .env ───────────────────────────────────────────────────────────────
echo "==> Writing .env..."
uv run python scripts/init_env.py --backend "${GPU_BACKEND}"

# ─── Install pre-commit hooks ─────────────────────────────────────────────────
echo "==> Installing pre-commit hooks..."
uv run pre-commit install

echo ""
echo "Optional: To install LiveBench extras (for the live_bench benchmark type):"
echo "  uv pip install \"pareto-llm[live-bench]\""

echo ""
echo "Setup complete. Run 'pareto-llm --help' to get started."
