import contextlib
import os
import pathlib
import shutil
import subprocess

import pytest


def _detect_gpu_backend() -> str | None:
    if shutil.which("nvidia-smi"):
        try:
            subprocess.check_call(["nvidia-smi"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return "cuda"
        except subprocess.CalledProcessError:
            pass
    return None


if not os.environ.get("GPU_BACKEND"):
    from dotenv import load_dotenv

    load_dotenv(pathlib.Path(__file__).parent.parent / ".env")

if not os.environ.get("GPU_BACKEND"):
    detected = _detect_gpu_backend()
    if detected:
        os.environ["GPU_BACKEND"] = detected

from pareto_llm.backend.base import GenerationResult, LLMBackend  # noqa: E402


class MockBackend(LLMBackend):
    """Deterministic in-memory backend for use in tests."""

    def __init__(self) -> None:
        self.loaded_model: str | None = None
        self.unload_count: int = 0
        self._max_ctx: int = 4096

    def load(self, model_id: str) -> None:
        self.loaded_model = model_id

    def generate(self, prompt: str, max_tokens: int = 512) -> GenerationResult:
        return GenerationResult(
            text="mock output",
            prompt_tokens=len(prompt.split()),
            gen_tokens=10,
            prompt_tps=500.0,
            gen_tps=50.0,
        )

    def unload(self) -> None:
        self.unload_count += 1
        self.loaded_model = None

    def max_context_tokens(self) -> int:
        return self._max_ctx

    def serve_openai(self, port: int):
        return contextlib.nullcontext()


@pytest.fixture
def mock_backend() -> MockBackend:
    return MockBackend()


@pytest.fixture
def tmp_csv_path(tmp_path: pathlib.Path) -> pathlib.Path:
    return tmp_path / "results.csv"


@pytest.fixture
def sample_config_dict() -> dict:
    return {
        "run_label": "test_run",
        "defaults": {"runs_per_test": 2, "keep_model_files": False},
        "models": ["test-org/test-model"],
        "benchmarks": [
            {
                "type": "mock_bench",
                "name": "my_bench",
                "config": {"key": "value"},
            }
        ],
    }
