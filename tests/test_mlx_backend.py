import os

import pytest

pytestmark = pytest.mark.mlx

if os.environ.get("GPU_BACKEND") != "mlx":
    pytest.skip("MLX backend requires Apple Silicon (GPU_BACKEND=mlx)", allow_module_level=True)


from pareto_llm.backend.mlx_backend import MLXBackend  # noqa: E402


def test_load_generate_unload():
    backend = MLXBackend()
    # Use the smallest available MLX model for a fast smoke test
    model_id = "mlx-community/Llama-3.2-1B-Instruct-4bit"
    backend.load(model_id)
    try:
        result = backend.generate("Say hello in one word.", max_tokens=10)
        assert result.text.strip() != ""
        assert result.prompt_tokens > 0
        assert result.gen_tokens > 0
        assert result.gen_tps > 0
        assert result.prompt_tps > 0
    finally:
        backend.unload()


def test_max_context_tokens():
    backend = MLXBackend()
    backend.load("mlx-community/Llama-3.2-1B-Instruct-4bit")
    try:
        ctx = backend.max_context_tokens()
        assert isinstance(ctx, int)
        assert ctx > 0
    finally:
        backend.unload()
