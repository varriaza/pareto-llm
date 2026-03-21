import os

import pytest

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(
        os.environ.get("GPU_BACKEND") != "cuda",
        reason="llama.cpp backend requires Linux + NVIDIA GPU (GPU_BACKEND=cuda)",
    ),
]

from pareto_llm.backend.llamacpp_backend import LlamaCppBackend  # noqa: E402


def test_load_generate_unload():
    backend = LlamaCppBackend()
    # Small Q4 GGUF for a fast smoke test
    model_id = "bartowski/Llama-3.2-1B-Instruct-GGUF:Q4_K_M"
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
    backend = LlamaCppBackend()
    backend.load("bartowski/Llama-3.2-1B-Instruct-GGUF:Q4_K_M")
    try:
        ctx = backend.max_context_tokens()
        assert isinstance(ctx, int)
        assert ctx > 0
    finally:
        backend.unload()


def test_all_layers_offloaded_to_gpu():
    """Verify GPU offloading is active (n_gpu_layers=-1 means 'all layers')."""
    backend = LlamaCppBackend()
    backend.load("bartowski/Llama-3.2-1B-Instruct-GGUF:Q4_K_M")
    try:
        # _n_gpu_layers is stored by load() from the constructor arg we passed
        assert backend._n_gpu_layers == -1, "Expected full GPU offloading (n_gpu_layers=-1)"
    finally:
        backend.unload()
