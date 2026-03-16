import pathlib

import pytest

from pareto_llm.backend.base import GenerationResult, LLMBackend


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
