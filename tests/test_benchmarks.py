import pytest

from pareto_llm.backend.base import GenerationResult
from pareto_llm.benchmarks.base import (
    BENCHMARK_REGISTRY,
    Benchmark,
    BenchmarkResult,
    register,
)


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
