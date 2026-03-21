import pytest
from pydantic import ValidationError

from pareto_llm.config import BenchmarkConfig, BenchmarkEntry, Defaults


def test_parse_minimal_config():
    cfg = BenchmarkConfig(
        run_label="smoke",
        models=["test-org/model"],
        benchmarks=[BenchmarkEntry(type="coding", name="bench1", config={})],
    )
    assert cfg.run_label == "smoke"
    assert cfg.defaults.runs_per_test == 3
    assert cfg.defaults.keep_model_files is False


def test_parse_full_config():
    cfg = BenchmarkConfig(
        run_label="full",
        defaults=Defaults(runs_per_test=5, keep_model_files=True),
        models=["repo/a", "repo/b"],
        benchmarks=[
            BenchmarkEntry(
                type="tool_use",
                name="bfcl",
                config={"dataset": "bfcl", "sample_size": 10},
            ),
        ],
    )
    assert cfg.defaults.runs_per_test == 5
    assert len(cfg.models) == 2
    assert cfg.benchmarks[0].config["sample_size"] == 10


def test_missing_run_label_raises():
    with pytest.raises(ValidationError):
        BenchmarkConfig(
            models=["repo/model"],
            benchmarks=[BenchmarkEntry(type="x", name="y", config={})],
        )


def test_empty_models_raises():
    with pytest.raises(ValidationError):
        BenchmarkConfig(
            run_label="test",
            models=[],
            benchmarks=[BenchmarkEntry(type="x", name="y", config={})],
        )


def test_empty_benchmarks_raises():
    with pytest.raises(ValidationError):
        BenchmarkConfig(
            run_label="test",
            models=["repo/model"],
            benchmarks=[],
        )


def test_benchmark_entry_config_passthrough():
    entry = BenchmarkEntry(type="coding", name="bench", config={"arbitrary": "data", "n": 42})
    assert entry.config["n"] == 42


def test_from_dict_roundtrip(sample_config_dict):
    cfg = BenchmarkConfig.model_validate(sample_config_dict)
    assert cfg.run_label == "test_run"
    assert cfg.defaults.runs_per_test == 2
    assert cfg.benchmarks[0].type == "mock_bench"
