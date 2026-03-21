import pytest

from pareto_llm.backend.base import GenerationResult
from pareto_llm.benchmarks.base import (
    BENCHMARK_REGISTRY,
    Benchmark,
    BenchmarkResult,
    register,
)
from pareto_llm.benchmarks.context_length import ContextLengthBenchmark

# ── Registry ──────────────────────────────────────────────────────────────────


def test_register_adds_class_to_registry():
    @register("_test_alpha")
    class AlphaBench(Benchmark):
        def run_single(self, backend):
            return BenchmarkResult(score=1.0, extra={}), backend.generate("x")

    assert "_test_alpha" in BENCHMARK_REGISTRY
    assert BENCHMARK_REGISTRY["_test_alpha"] is AlphaBench


def test_duplicate_register_raises():
    @register("_test_dup")
    class DupA(Benchmark):
        def run_single(self, backend):
            return BenchmarkResult(score=0.0, extra={}), backend.generate("x")

    with pytest.raises(KeyError, match="_test_dup"):

        @register("_test_dup")
        class DupB(Benchmark):
            def run_single(self, backend):
                return BenchmarkResult(score=0.0, extra={}), backend.generate("x")


def test_run_single_returns_tuple(mock_backend):
    @register("_test_beta")
    class BetaBench(Benchmark):
        def run_single(self, backend):
            gen = backend.generate("hello world")
            return BenchmarkResult(score=0.9, extra={"tps": gen.gen_tps}), gen

    bench = BENCHMARK_REGISTRY["_test_beta"](config={})
    bench_result, gen_result = bench.run_single(mock_backend)

    assert isinstance(bench_result, BenchmarkResult)
    assert bench_result.score == 0.9
    assert bench_result.extra["tps"] == 50.0
    assert isinstance(gen_result, GenerationResult)
    assert gen_result.gen_tps == 50.0


def test_benchmark_result_fields():
    r = BenchmarkResult(score=0.5, extra={"k": "v"})
    assert r.score == 0.5
    assert r.extra["k"] == "v"


def test_supports_context_padding_defaults_true():
    @register("_test_gamma")
    class GammaBench(Benchmark):
        def run_single(self, backend):
            return BenchmarkResult(score=0.0, extra={}), backend.generate("x")

    assert GammaBench.supports_context_padding is True


# ── Context-length wrapper ────────────────────────────────────────────────────


def test_context_length_is_registered():
    assert "context_length" in BENCHMARK_REGISTRY


def test_context_length_pads_prompt(mock_backend):
    """The wrapper must pass a longer prompt than the inner benchmark would alone."""
    captured: list[str] = []
    original_generate = mock_backend.generate

    def capturing_generate(prompt: str, max_tokens: int = 512):
        captured.append(prompt)
        return original_generate(prompt, max_tokens)

    mock_backend.generate = capturing_generate  # type: ignore[method-assign]
    mock_backend._max_ctx = 200  # small so padding is measurable

    @register("_inner_pad_test")
    class InnerBench(Benchmark):
        def run_single(self, backend):
            gen = backend.generate("short prompt")
            return BenchmarkResult(score=1.0, extra={}), gen

    bench = ContextLengthBenchmark(
        config={
            "fill_ratio": 0.5,
            "inner_benchmark": "_inner_pad_test",
            "inner_config": {},
        }
    )
    bench.run_single(mock_backend)

    assert captured, "generate was never called"
    assert len(captured[0].split()) > len("short prompt".split()), "prompt was not padded"


def test_context_length_rejects_unsupported_inner(mock_backend):
    @register("_no_pad_bench")
    class NoPadBench(Benchmark):
        supports_context_padding = False

        def run_single(self, backend):
            return BenchmarkResult(score=0.0, extra={}), backend.generate("x")

    bench = ContextLengthBenchmark(
        config={
            "fill_ratio": 0.25,
            "inner_benchmark": "_no_pad_bench",
            "inner_config": {},
        }
    )
    with pytest.raises(ValueError, match="supports_context_padding"):
        bench.run_single(mock_backend)


def test_context_length_invalid_fill_ratio():
    with pytest.raises(ValueError, match="fill_ratio"):
        ContextLengthBenchmark(config={"fill_ratio": 1.5, "inner_benchmark": "_inner_pad_test", "inner_config": {}})
