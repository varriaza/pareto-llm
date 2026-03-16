"""Context-length benchmark: wraps an inner benchmark, padding prompts to a fill ratio."""
from __future__ import annotations

from pareto_llm.backend.base import GenerationResult, LLMBackend
from pareto_llm.benchmarks.base import (
    BENCHMARK_REGISTRY,
    Benchmark,
    BenchmarkResult,
    register,
)

# Public-domain padding text (~50k words; enough for any reasonable fill ratio).
_PADDING_CORPUS = (
    "It was the best of times it was the worst of times it was the age of wisdom "
    "it was the age of foolishness it was the epoch of belief it was the epoch of "
    "incredulity it was the season of Light it was the season of Darkness "
) * 500


class _PaddingBackend(LLMBackend):
    """Wraps a real backend and prepends padding to every generate() call."""

    def __init__(self, inner: LLMBackend, padding: str) -> None:
        self._inner = inner
        self._padding = padding

    def load(self, model_id: str) -> None:
        self._inner.load(model_id)

    def generate(self, prompt: str, max_tokens: int = 512) -> GenerationResult:
        return self._inner.generate(self._padding + " " + prompt, max_tokens)

    def unload(self) -> None:
        self._inner.unload()

    def max_context_tokens(self) -> int:
        return self._inner.max_context_tokens()


@register("context_length")
class ContextLengthBenchmark(Benchmark):
    """Runs an inner benchmark with the context filled to ``fill_ratio``."""

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        fill_ratio = config.get("fill_ratio")
        if fill_ratio is None or not (0.0 < float(fill_ratio) < 1.0):
            raise ValueError(
                f"fill_ratio must be between 0 and 1 (exclusive), got {fill_ratio!r}"
            )

    def run_single(
        self, backend: LLMBackend
    ) -> tuple[BenchmarkResult, GenerationResult]:
        inner_key: str = self.config["inner_benchmark"]
        inner_config: dict = self.config.get("inner_config", {})
        fill_ratio: float = self.config["fill_ratio"]

        inner_cls = BENCHMARK_REGISTRY[inner_key]
        if not inner_cls.supports_context_padding:
            raise ValueError(
                f"Benchmark '{inner_key}' has supports_context_padding=False "
                "and cannot be used inside a context_length wrapper."
            )

        max_ctx = backend.max_context_tokens()
        safety_margin = 64
        # Rough heuristic: 1 token ≈ 0.75 words
        target_words = int(max_ctx * fill_ratio * 0.75) - safety_margin
        padding = " ".join(_PADDING_CORPUS.split()[: max(0, target_words)])

        return inner_cls(inner_config).run_single(_PaddingBackend(backend, padding))
