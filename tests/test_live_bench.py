import pytest

from pareto_llm.benchmarks.live_bench import LiveBenchBenchmark


def _valid_config(**overrides):
    cfg = {
        "port": 8766,
        "categories": ["coding"],
        "release_date": "latest",
        "sample_size": None,
        "sample_size_per_category": {},
        "seed": 42,
        "skip_agentic": True,
        "n_ctx": 8192,
        "parallel": 4,
        "jobs_dir": "./results/livebench",
    }
    cfg.update(overrides)
    return cfg


# ─── Config validation ────────────────────────────────────────────────────────


def test_valid_config_creates_instance():
    bench = LiveBenchBenchmark(_valid_config())
    assert bench.config["categories"] == ["coding"]
    assert bench.config["port"] == 8766


def test_supports_context_padding_is_false():
    assert LiveBenchBenchmark.supports_context_padding is False


def test_invalid_category_raises():
    with pytest.raises(ValueError, match="categories"):
        LiveBenchBenchmark(_valid_config(categories=["coding", "foobar"]))


def test_empty_categories_raises():
    with pytest.raises(ValueError, match="categories"):
        LiveBenchBenchmark(_valid_config(categories=[]))


def test_categories_all_is_valid():
    bench = LiveBenchBenchmark(_valid_config(categories="all"))
    assert bench.config["categories"] == "all"


def test_port_too_low_raises():
    with pytest.raises(ValueError, match="port"):
        LiveBenchBenchmark(_valid_config(port=80))


def test_port_too_high_raises():
    with pytest.raises(ValueError, match="port"):
        LiveBenchBenchmark(_valid_config(port=99999))


def test_sample_size_zero_raises():
    with pytest.raises(ValueError, match="sample_size"):
        LiveBenchBenchmark(_valid_config(sample_size=0))


def test_sample_size_per_category_zero_raises():
    with pytest.raises(ValueError, match="sample_size_per_category"):
        LiveBenchBenchmark(_valid_config(sample_size_per_category={"coding": 0}))


def test_invalid_release_date_raises():
    with pytest.raises(ValueError, match="release_date"):
        LiveBenchBenchmark(_valid_config(release_date="not-a-date"))


def test_valid_release_date():
    bench = LiveBenchBenchmark(_valid_config(release_date="2024-11-25"))
    assert bench.config["release_date"] == "2024-11-25"


def test_parallel_zero_raises():
    with pytest.raises(ValueError, match="parallel"):
        LiveBenchBenchmark(_valid_config(parallel=0))


def test_livebench_not_installed_raises(monkeypatch):
    import importlib.util as _ilu

    orig = _ilu.find_spec
    monkeypatch.setattr(_ilu, "find_spec", lambda name: None if name == "livebench" else orig(name))
    with pytest.raises(RuntimeError, match="livebench"):
        LiveBenchBenchmark(_valid_config())


# ─── Registration ─────────────────────────────────────────────────────────────


def test_live_bench_is_registered():
    from pareto_llm.benchmarks.base import BENCHMARK_REGISTRY

    assert "live_bench" in BENCHMARK_REGISTRY
