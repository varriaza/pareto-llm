#!/usr/bin/env bash
set -euo pipefail

# ─── GPU backend selection ────────────────────────────────────────────────────
echo "Which GPU backend are you using?"
echo "  1) CUDA  (NVIDIA GPU)"
echo "  2) MLX   (Apple Silicon)"
echo "  3) None"
printf "Enter 1, 2, or 3: "; read -r choice

case "${choice}" in
    1) GPU_BACKEND="cuda" ;;
    2) GPU_BACKEND="mlx" ;;
    3) GPU_BACKEND="none" ;;
    *) echo "Invalid choice. Exiting."; exit 1 ;;
esac

# ─── Install Python 3.12 ─────────────────────────────────────────────────────
echo "==> Installing Python 3.12..."
uv python install 3.12

# ─── Install dependencies ─────────────────────────────────────────────────────
echo "==> Installing dependencies..."
EXTRAS=(--extra dev --extra live-bench --extra terminal-bench)
if [ "${GPU_BACKEND}" = "cuda" ]; then
    echo "==> Installing CUDA extras (llama-cpp-python + uvicorn)..."
    EXTRAS+=(--extra cuda)
elif [ "${GPU_BACKEND}" = "mlx" ]; then
    echo "==> Installing MLX extras..."
    EXTRAS+=(--extra mlx)
fi
uv sync "${EXTRAS[@]}"

# ─── Write .env ───────────────────────────────────────────────────────────────
echo "==> Writing .env..."
uv run python scripts/init_env.py --backend "${GPU_BACKEND}"

# ─── Install pre-commit hooks ─────────────────────────────────────────────────
echo "==> Installing pre-commit hooks..."
uv run pre-commit install

echo ""
echo "Setup complete. Run 'pareto-llm --help' to get started."
