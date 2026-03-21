import csv
import pathlib
from unittest.mock import patch

import pytest

from pareto_llm.benchmarks.base import (
    BENCHMARK_REGISTRY,
    Benchmark,
    BenchmarkResult,
    register,
)
from pareto_llm.config import BenchmarkConfig, BenchmarkEntry, Defaults
from pareto_llm.runner import run

# ── Test-only benchmark fixtures ──────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _register_runner_mock():
    key = "_runner_mock"
    if key not in BENCHMARK_REGISTRY:

        @register(key)
        class RunnerMockBench(Benchmark):
            def run_single(self, backend):
                gen = backend.generate("test prompt")
                return BenchmarkResult(score=0.75, extra={"detail": "ok"}), gen

    yield


def _cfg(
    models: list[str],
    bench_names: list[str],
    runs: int = 2,
    keep: bool = False,
) -> BenchmarkConfig:
    return BenchmarkConfig(
        run_label="test_run",
        defaults=Defaults(runs_per_test=runs, keep_model_files=keep),
        models=models,
        benchmarks=[BenchmarkEntry(type="_runner_mock", name=n, config={}) for n in bench_names],
    )


def _rows(path: pathlib.Path) -> list[dict]:
    return list(csv.DictReader(path.open()))


# ── Row counts ────────────────────────────────────────────────────────────────


def test_2_models_2_benches_2_runs_gives_8_rows(tmp_csv_path, mock_backend):
    cfg = _cfg(["m/a", "m/b"], ["bench1", "bench2"], runs=2)
    with patch("pareto_llm.runner._create_backend", return_value=mock_backend):
        with patch("pareto_llm.runner._delete_hf_cache"):
            run(config=cfg, output_path=tmp_csv_path, gpu_backend="cuda")
    assert len(_rows(tmp_csv_path)) == 8


def test_run_num_increments_per_model_benchmark(tmp_csv_path, mock_backend):
    cfg = _cfg(["m/a"], ["bench1"], runs=3)
    with patch("pareto_llm.runner._create_backend", return_value=mock_backend):
        with patch("pareto_llm.runner._delete_hf_cache"):
            run(config=cfg, output_path=tmp_csv_path, gpu_backend="cuda")
    assert [r["run_num"] for r in _rows(tmp_csv_path)] == ["1", "2", "3"]


# ── Backend lifecycle ─────────────────────────────────────────────────────────


def test_backend_unload_called_once_per_model(tmp_csv_path, mock_backend):
    cfg = _cfg(["m/a", "m/b"], ["bench1"], runs=1)
    with patch("pareto_llm.runner._create_backend", return_value=mock_backend):
        with patch("pareto_llm.runner._delete_hf_cache"):
            run(config=cfg, output_path=tmp_csv_path, gpu_backend="cuda")
    assert mock_backend.unload_count == 2


def test_delete_cache_called_when_keep_false(tmp_csv_path, mock_backend):
    cfg = _cfg(["m/a"], ["bench1"], runs=1, keep=False)
    with patch("pareto_llm.runner._create_backend", return_value=mock_backend):
        with patch("pareto_llm.runner._delete_hf_cache") as mock_del:
            run(config=cfg, output_path=tmp_csv_path, gpu_backend="cuda")
    mock_del.assert_called_once_with("m/a")


def test_delete_cache_not_called_when_keep_true(tmp_csv_path, mock_backend):
    cfg = _cfg(["m/a"], ["bench1"], runs=1, keep=True)
    with patch("pareto_llm.runner._create_backend", return_value=mock_backend):
        with patch("pareto_llm.runner._delete_hf_cache") as mock_del:
            run(config=cfg, output_path=tmp_csv_path, gpu_backend="cuda")
    mock_del.assert_not_called()


# ── Error handling ────────────────────────────────────────────────────────────


def test_exception_writes_failure_row_and_continues(tmp_csv_path, mock_backend):
    call_count = 0

    if "_runner_flaky" not in BENCHMARK_REGISTRY:

        @register("_runner_flaky")
        class FlakyBench(Benchmark):
            def run_single(self, backend):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise RuntimeError("transient error")
                gen = backend.generate("x")
                return BenchmarkResult(score=1.0, extra={}), gen

    cfg = BenchmarkConfig(
        run_label="test_run",
        defaults=Defaults(runs_per_test=2, keep_model_files=False),
        models=["m/a"],
        benchmarks=[BenchmarkEntry(type="_runner_flaky", name="flaky", config={})],
    )
    with patch("pareto_llm.runner._create_backend", return_value=mock_backend):
        with patch("pareto_llm.runner._delete_hf_cache"):
            run(config=cfg, output_path=tmp_csv_path, gpu_backend="cuda")

    rows = _rows(tmp_csv_path)
    assert len(rows) == 2
    assert rows[0]["score"] == ""  # failure row
    assert "transient error" in rows[0]["extra_error"]
    assert rows[1]["score"] == "1.0"  # success row


def test_exception_does_not_skip_other_benchmarks(tmp_csv_path, mock_backend):
    """A failure in bench1/run1 must not skip bench2."""
    if "_runner_fail_once" not in BENCHMARK_REGISTRY:
        first_call = {"done": False}

        @register("_runner_fail_once")
        class FailOnceBench(Benchmark):
            def run_single(self, backend):
                if not first_call["done"]:
                    first_call["done"] = True
                    raise RuntimeError("once")
                gen = backend.generate("x")
                return BenchmarkResult(score=0.5, extra={}), gen

    cfg = _cfg(["m/a"], ["_runner_mock", "_runner_fail_once"], runs=1)
    with patch("pareto_llm.runner._create_backend", return_value=mock_backend):
        with patch("pareto_llm.runner._delete_hf_cache"):
            run(config=cfg, output_path=tmp_csv_path, gpu_backend="cuda")

    rows = _rows(tmp_csv_path)
    assert len(rows) == 2
    benchmark_names = {r["benchmark_name"] for r in rows}
    assert "_runner_mock" in benchmark_names
    assert "_runner_fail_once" in benchmark_names
