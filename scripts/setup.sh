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
    echo "==> Installing CUDA extras (llama-cpp-python pre-built wheel)..."
    if ! uv pip install "pareto-llm[cuda]" \
        --index "https://abetlen.github.io/llama-cpp-python/whl/cu124" \
        --index-strategy unsafe-best-match; then
        echo ""
        echo "ERROR: Failed to install llama-cpp-python."
        echo ""
        echo "The pre-built wheel was not available for your platform."
        echo "To build from source you need the CUDA Toolkit:"
        echo ""
        echo "  # Debian/Ubuntu (WSL2):"
        echo "  sudo apt-get install cuda-toolkit-12-4"
        echo "  export PATH=/usr/local/cuda-12.4/bin:\$PATH"
        echo "  export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:\$LD_LIBRARY_PATH"
        echo ""
        echo "  Then re-run:"
        echo "  CMAKE_ARGS=\"-DGGML_CUDA=on\" uv pip install \"pareto-llm[cuda]\""
        echo ""
        echo "  Full CUDA Toolkit downloads: https://developer.nvidia.com/cuda-downloads"
        exit 1
    fi
elif [ "${GPU_BACKEND}" = "mlx" ]; then
    echo "==> Installing MLX extras..."
    uv pip install "pareto-llm[mlx]"
fi

echo ""
echo "Setup complete. Run 'pareto-llm --help' to get started."
