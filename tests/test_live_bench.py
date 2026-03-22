import contextlib
import importlib.util
import logging
import os
import socket
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Python 3.12 changed unittest.mock.patch to use pkgutil.resolve_name, which requires
# the target module to already exist in sys.modules. Pre-register lightweight stubs so
# patch() can find the modules without triggering livebench's heavyweight imports
# (e.g. spaCy model downloads at import time).
for _mod in ("livebench.gen_api_answer", "livebench.gen_ground_truth_judgment"):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

from pareto_llm.benchmarks.live_bench import LiveBenchBenchmark  # noqa: E402


@pytest.fixture(autouse=True)
def _mock_livebench_installed(monkeypatch):
    """Pretend livebench is installed for all tests in this module.

    - Patches importlib.util.find_spec so __init__ doesn't raise RuntimeError.
    - Pre-registers livebench submodules in sys.modules so lazy imports inside
      run_single resolve without livebench actually being installed.

    Tests that exercise the 'not installed' path override find_spec themselves;
    their monkeypatch call takes precedence within that test's scope.
    """
    _orig = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name: MagicMock() if name == "livebench" else _orig(name),
    )
    for _mod in (
        "livebench",
        "livebench.common",
        "livebench.gen_api_answer",
        "livebench.gen_ground_truth_judgment",
        "livebench.model",
        "livebench.model.api_model_config",
    ):
        if _mod not in sys.modules:
            monkeypatch.setitem(sys.modules, _mod, MagicMock())


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


# ─── run_single ───────────────────────────────────────────────────────────────


def _make_mock_backend():
    backend = MagicMock()
    backend.serve_openai.return_value = contextlib.nullcontext()
    return backend


def _write_judgment_files(run_dir: Path, judgments_by_category: dict):
    """Write mock ground_truth_judgment.jsonl files under run_dir/data/."""
    import json

    for category, scores in judgments_by_category.items():
        task_dir = run_dir / "data" / f"live_bench/{category}/test_task" / "model_judgment"
        task_dir.mkdir(parents=True, exist_ok=True)
        jfile = task_dir / "ground_truth_judgment.jsonl"
        with jfile.open("w") as f:
            for i, score in enumerate(scores):
                f.write(
                    json.dumps(
                        {
                            "question_id": f"q{i}",
                            "category": category,
                            "task": f"live_bench/{category}/test_task",
                            "model": "local",
                            "score": score,
                        }
                    )
                    + "\n"
                )


def test_run_single_score_computation(tmp_path):
    pre_filtered = [
        {"question_id": "q0", "category": "coding", "turns": ["q"]},
        {"question_id": "q1", "category": "coding", "turns": ["q"]},
        {"question_id": "q2", "category": "math", "turns": ["q"]},
        {"question_id": "q3", "category": "math", "turns": ["q"]},
    ]
    bench = LiveBenchBenchmark(_valid_config(jobs_dir=str(tmp_path)))

    def fake_gen_judgments(*args, **kwargs):
        _write_judgment_files(
            Path.cwd(),
            {
                "coding": [1.0, 0.0],  # mean = 0.5
                "math": [1.0, 1.0],  # mean = 1.0
            },
        )

    with (
        patch.object(bench, "_get_filtered_questions", return_value=pre_filtered),
        patch("livebench.gen_api_answer.run_questions", MagicMock()),
        patch("livebench.gen_ground_truth_judgment.gen_judgments", fake_gen_judgments),
        patch.object(bench, "_load_model_answers", return_value={}),
    ):
        result, gen = bench.run_single(_make_mock_backend())

    # overall = mean([0.5, 1.0]) = 0.75
    assert abs(result.score - 0.75) < 0.001
    assert result.extra["per_category_scores"]["coding"] == pytest.approx(0.5)
    assert result.extra["per_category_scores"]["math"] == pytest.approx(1.0)
    assert gen.text == ""
    assert gen.prompt_tokens == 0


def test_run_single_score_all_pass(tmp_path):
    pre_filtered = [{"question_id": "q0", "category": "coding", "turns": ["q"]}]
    bench = LiveBenchBenchmark(_valid_config(jobs_dir=str(tmp_path)))

    def fake_gen_judgments(*args, **kwargs):
        _write_judgment_files(Path.cwd(), {"coding": [1.0]})

    with (
        patch.object(bench, "_get_filtered_questions", return_value=pre_filtered),
        patch("livebench.gen_api_answer.run_questions", MagicMock()),
        patch("livebench.gen_ground_truth_judgment.gen_judgments", fake_gen_judgments),
        patch.object(bench, "_load_model_answers", return_value={}),
    ):
        result, _ = bench.run_single(_make_mock_backend())

    assert abs(result.score - 1.0) < 0.001


def test_run_single_score_none_pass(tmp_path):
    pre_filtered = [{"question_id": "q0", "category": "coding", "turns": ["q"]}]
    bench = LiveBenchBenchmark(_valid_config(jobs_dir=str(tmp_path)))

    def fake_gen_judgments(*args, **kwargs):
        _write_judgment_files(Path.cwd(), {"coding": [0.0]})

    with (
        patch.object(bench, "_get_filtered_questions", return_value=pre_filtered),
        patch("livebench.gen_api_answer.run_questions", MagicMock()),
        patch("livebench.gen_ground_truth_judgment.gen_judgments", fake_gen_judgments),
        patch.object(bench, "_load_model_answers", return_value={}),
    ):
        result, _ = bench.run_single(_make_mock_backend())

    assert abs(result.score - 0.0) < 0.001


def test_run_single_gen_tokens_and_tps(tmp_path):
    pre_filtered = [{"question_id": "q0", "category": "coding", "turns": ["q"]}]
    bench = LiveBenchBenchmark(_valid_config(jobs_dir=str(tmp_path)))

    fake_answers = {
        "local": {
            "q0": {"total_output_tokens": 42},
            "q1": {"total_output_tokens": 58},
        }
    }

    def fake_gen_judgments(*args, **kwargs):
        _write_judgment_files(Path.cwd(), {"coding": [1.0]})

    with (
        patch.object(bench, "_get_filtered_questions", return_value=pre_filtered),
        patch("livebench.gen_api_answer.run_questions", MagicMock()),
        patch("livebench.gen_ground_truth_judgment.gen_judgments", fake_gen_judgments),
        patch.object(bench, "_load_model_answers", return_value=fake_answers),
    ):
        _, gen = bench.run_single(_make_mock_backend())

    assert gen.gen_tokens == 100  # 42 + 58
    assert gen.gen_tps > 0  # wall-clock TPS: 100 tokens / elapsed seconds


def test_run_single_openai_key_unset_during_scoring(tmp_path, monkeypatch):
    """OPENAI_API_KEY must not be set during gen_judgments.

    AMPS_Hard math questions call is_equiv_llm() whenever OPENAI_API_KEY is present,
    which tries to reach OpenAI's real API. With a local/fake key this crashes the
    entire scoring pass.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key-for-test")

    pre_filtered = [{"question_id": "q0", "category": "coding", "turns": ["q"]}]
    bench = LiveBenchBenchmark(_valid_config(jobs_dir=str(tmp_path)))

    key_during_scoring: list[str | None] = []

    def fake_gen_judgments(*args, **kwargs):
        key_during_scoring.append(os.environ.get("OPENAI_API_KEY"))
        _write_judgment_files(Path.cwd(), {"coding": [1.0]})

    with (
        patch.object(bench, "_get_filtered_questions", return_value=pre_filtered),
        patch("livebench.gen_api_answer.run_questions", MagicMock()),
        patch("livebench.gen_ground_truth_judgment.gen_judgments", fake_gen_judgments),
        patch.object(bench, "_load_model_answers", return_value={}),
    ):
        bench.run_single(_make_mock_backend())

    assert key_during_scoring == [None], (
        "OPENAI_API_KEY must be unset during gen_judgments — if present, AMPS_Hard "
        "scorer calls the real OpenAI API with whatever key is set"
    )
    # Key must be restored after run_single completes
    assert os.environ.get("OPENAI_API_KEY") == "fake-key-for-test"


def test_run_single_port_in_use(tmp_path):
    bench = LiveBenchBenchmark(_valid_config(jobs_dir=str(tmp_path), port=19001))
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("127.0.0.1", 19001))
        with patch.object(bench, "_get_filtered_questions", return_value=[MagicMock()]):
            with pytest.raises(RuntimeError, match="already in use"):
                bench.run_single(_make_mock_backend())
