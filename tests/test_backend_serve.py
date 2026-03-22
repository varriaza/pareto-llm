"""Tests for LLMBackend.serve_openai() context manager.

Unit tests (no hardware) run always.
Integration tests require hardware and are marked:
  @pytest.mark.cuda  — NVIDIA GPU + llama-cpp-python
  @pytest.mark.mlx   — Apple Silicon + mlx-lm
"""

import time
import urllib.request

import pytest


def _v1_models_reachable(port: int) -> bool:
    try:
        urllib.request.urlopen(f"http://localhost:{port}/v1/models", timeout=2)
        return True
    except Exception:
        return False


# ─── Unit tests (no hardware) ────────────────────────────────────────────────


def test_llamacpp_serve_before_load_raises():
    from pareto_llm.backend.llamacpp_backend import LlamaCppBackend

    backend = LlamaCppBackend()
    with pytest.raises(RuntimeError, match="Model not loaded"):
        with backend.serve_openai(19876):
            pass


def test_mlx_serve_before_load_raises():
    from pareto_llm.backend.mlx_backend import MLXBackend

    backend = MLXBackend()
    with pytest.raises(RuntimeError, match="Model not loaded"):
        with backend.serve_openai(19877):
            pass


# ─── Integration tests ────────────────────────────────────────────────────────


@pytest.mark.cuda
def test_llamacpp_serve_exposes_v1_models():
    from pareto_llm.backend.llamacpp_backend import LlamaCppBackend

    backend = LlamaCppBackend()
    backend.load("bartowski/Llama-3.2-1B-Instruct-GGUF:Q4_K_M")
    try:
        port = 18765
        with backend.serve_openai(port):
            assert _v1_models_reachable(port), "/v1/models not reachable during context"
        time.sleep(0.5)
        assert not _v1_models_reachable(port), "/v1/models still reachable after context exit"
    finally:
        backend.unload()


@pytest.mark.mlx
def test_mlx_serve_exposes_v1_models():
    from pareto_llm.backend.mlx_backend import MLXBackend

    backend = MLXBackend()
    backend.load("mlx-community/Llama-3.2-1B-Instruct-4bit")
    try:
        port = 18766
        with backend.serve_openai(port):
            assert _v1_models_reachable(port), "/v1/models not reachable during context"
        time.sleep(0.5)
        assert not _v1_models_reachable(port), "/v1/models still reachable after context exit"
    finally:
        backend.unload()
