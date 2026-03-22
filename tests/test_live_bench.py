import logging
from unittest.mock import patch

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


# ─── _get_filtered_questions ──────────────────────────────────────────────────


def _make_questions(specs):
    """specs: list of (category, is_agentic) tuples"""
    questions = []
    for i, (cat, is_agentic) in enumerate(specs):
        q = {
            "question_id": f"q{i}",
            "category": cat,
            "turns": [f"question {i}"],
        }
        if is_agentic:
            # Agentic questions have category == "agentic_coding" (verified in Task 3 Step 1)
            q["category"] = "agentic_coding"
        questions.append(q)
    return questions


# Patch the instance method directly via patch.object
_PATCH_LOAD = "pareto_llm.benchmarks.live_bench.LiveBenchBenchmark._load_questions"


def test_category_filter_single(tmp_path):
    all_qs = _make_questions([("coding", False), ("coding", False), ("math", False)])
    bench = LiveBenchBenchmark(_valid_config(categories=["coding"], jobs_dir=str(tmp_path)))
    with patch.object(bench, "_load_questions", return_value=all_qs):
        result = bench._get_filtered_questions()
    assert len(result) == 2
    assert all(q["category"] == "coding" for q in result)


def test_category_all_returns_all(tmp_path):
    all_qs = _make_questions([("coding", False), ("math", False), ("reasoning", False)])
    bench = LiveBenchBenchmark(_valid_config(categories="all", jobs_dir=str(tmp_path)))
    with patch.object(bench, "_load_questions", return_value=all_qs):
        result = bench._get_filtered_questions()
    assert len(result) == 3


def test_skip_agentic_true(tmp_path):
    all_qs = _make_questions([("coding", False), ("coding", True)])
    bench = LiveBenchBenchmark(_valid_config(skip_agentic=True, jobs_dir=str(tmp_path)))
    with patch.object(bench, "_load_questions", return_value=all_qs):
        result = bench._get_filtered_questions()
    assert len(result) == 1
    assert result[0]["question_id"] == "q0"


def test_skip_agentic_false(tmp_path):
    all_qs = _make_questions([("coding", False), ("coding", True)])
    bench = LiveBenchBenchmark(_valid_config(categories="all", skip_agentic=False, jobs_dir=str(tmp_path)))
    with patch.object(bench, "_load_questions", return_value=all_qs):
        result = bench._get_filtered_questions()
    assert len(result) == 2
    assert any(q.get("category") == "agentic_coding" for q in result)


def test_sampling_global(tmp_path):
    all_qs = _make_questions([("coding", False)] * 10 + [("math", False)] * 10)
    bench = LiveBenchBenchmark(_valid_config(categories="all", sample_size=3, seed=42, jobs_dir=str(tmp_path)))
    with patch.object(bench, "_load_questions", return_value=all_qs):
        result = bench._get_filtered_questions()
    coding_count = sum(1 for q in result if q["category"] == "coding")
    math_count = sum(1 for q in result if q["category"] == "math")
    assert coding_count == 3
    assert math_count == 3


def test_sampling_per_category_override(tmp_path):
    all_qs = _make_questions([("coding", False)] * 10 + [("math", False)] * 10)
    bench = LiveBenchBenchmark(
        _valid_config(
            categories="all",
            sample_size=5,
            sample_size_per_category={"coding": 2},
            seed=42,
            jobs_dir=str(tmp_path),
        )
    )
    with patch.object(bench, "_load_questions", return_value=all_qs):
        result = bench._get_filtered_questions()
    coding_count = sum(1 for q in result if q["category"] == "coding")
    math_count = sum(1 for q in result if q["category"] == "math")
    assert coding_count == 2  # per-category override wins
    assert math_count == 5  # global sample_size applies


def test_sampling_seed_reproducible(tmp_path):
    all_qs = _make_questions([("coding", False)] * 20)

    def run():
        bench = LiveBenchBenchmark(_valid_config(categories=["coding"], sample_size=5, seed=42, jobs_dir=str(tmp_path)))
        with patch.object(bench, "_load_questions", return_value=all_qs):
            return bench._get_filtered_questions()

    assert [q["question_id"] for q in run()] == [q["question_id"] for q in run()]


def test_sampling_warns_when_exceeds_available(tmp_path, caplog):
    all_qs = _make_questions([("coding", False)] * 3)
    bench = LiveBenchBenchmark(_valid_config(categories=["coding"], sample_size=10, jobs_dir=str(tmp_path)))
    with patch.object(bench, "_load_questions", return_value=all_qs), caplog.at_level(logging.WARNING):
        result = bench._get_filtered_questions()
    assert len(result) == 3
    assert caplog.records


def test_no_questions_after_filter_raises(tmp_path):
    bench = LiveBenchBenchmark(_valid_config(categories=["coding"], jobs_dir=str(tmp_path)))
    with patch.object(bench, "_load_questions", return_value=[]):
        with pytest.raises(ValueError, match="No questions"):
            bench._get_filtered_questions()
