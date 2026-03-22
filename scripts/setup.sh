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
    echo "==> Installing CUDA extras (building llama-cpp-python from source)..."
    echo "    Note: pre-built wheels are not available for Python 3.13 — source build required."
    echo "    This requires the CUDA Toolkit and build tools (cmake, gcc). This may take a few minutes."
    echo ""
    echo "    If you see a 'cospi/sinpi noexcept' compile error with CUDA 12.8 + glibc 2.41,"
    echo "    run: python3 scripts/patch_cuda_math.py  (then re-run setup)"
    echo ""
    if ! CMAKE_ARGS="-DGGML_CUDA=on" uv pip install "pareto-llm[cuda]" --no-cache-dir; then
        echo ""
        echo "ERROR: Failed to build llama-cpp-python from source."
        echo ""
        echo "Prerequisites:"
        echo "  sudo apt-get install cuda-toolkit cmake gcc g++"
        echo "  export PATH=/usr/local/cuda/bin:\$PATH"
        echo ""
        echo "  Full CUDA Toolkit downloads: https://developer.nvidia.com/cuda-downloads"
        exit 1
    fi
elif [ "${GPU_BACKEND}" = "mlx" ]; then
    echo "==> Installing MLX extras..."
    uv pip install "pareto-llm[mlx]"
fi

echo ""
echo "Optional: To install LiveBench extras (for the live_bench benchmark type):"
echo "  uv pip install \"pareto-llm[live-bench]\""

echo ""
echo "Setup complete. Run 'pareto-llm --help' to get started."
