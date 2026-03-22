# LiveBench Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `LiveBenchBenchmark` class that runs LiveBench evaluations against a local OpenAI-compatible server and returns per-category scores.

**Architecture:** Follow `TerminalBenchmark`'s pattern — `run_single()` starts the model server via `backend.serve_openai()`, calls LiveBench's Python API for inference and offline scoring, reads judgment JSONL files, and returns a `BenchmarkResult`. LiveBench writes all files relative to CWD; we `os.chdir()` to a timestamped subdirectory of `jobs_dir` before calling LiveBench functions and restore CWD in a `finally` block.

**Tech Stack:** Python 3.13, livebench pinned to commit `18b524d2d3a32282300b3548ef69e7d904ece836`, pytest, unittest.mock

---

## File Map

| Action | Path | Responsibility |
|---|---|---|
| Create | `src/pareto_llm/benchmarks/live_bench.py` | `LiveBenchBenchmark` class |
| Create | `tests/test_live_bench.py` | All tests for the class |
| Modify | `src/pareto_llm/benchmarks/__init__.py` | Register `live_bench` benchmark |
| Modify | `pyproject.toml` | Add `live-bench` optional extra |
| Modify | `.github/workflows/ci.yml` | Install `live-bench` extra in CI |
| Modify | `scripts/setup.sh` | Opt-in install hint |
| Create | `configs/live_bench.yaml` | Example config |

---

## Task 1: Pin LiveBench dependency and wire CI

**Files:**
- Modify: `pyproject.toml`
- Modify: `.github/workflows/ci.yml`

- [ ] **Step 1: Add the `live-bench` extra to `pyproject.toml`**

In `pyproject.toml`, add after the `terminal-bench` block:

```toml
live-bench = [
    "livebench @ git+https://github.com/livebench/livebench.git@18b524d2d3a32282300b3548ef69e7d904ece836",
]
```

- [ ] **Step 2: Verify the dependency resolves**

Run:
```bash
uv pip install "pareto-llm[live-bench]"
```
Expected: Installs successfully. If livebench has missing transitive deps, add them to the `live-bench` extra.

- [ ] **Step 3: Add `--extra live-bench` to CI**

In `.github/workflows/ci.yml`, find the `uv sync` line and update it:
```yaml
run: uv sync --extra dev --extra terminal-bench --extra live-bench
```

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml .github/workflows/ci.yml
git commit -m "feat: add live-bench optional extra pinned to commit 18b524d2"
```

---

## Task 2: Scaffold `LiveBenchBenchmark` with config validation

**Files:**
- Create: `src/pareto_llm/benchmarks/live_bench.py`
- Create: `tests/test_live_bench.py`
- Modify: `src/pareto_llm/benchmarks/__init__.py`

- [ ] **Step 1: Write failing config validation tests**

Create `tests/test_live_bench.py`:

```python
import importlib.util
import re

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
```

- [ ] **Step 2: Run tests — expect ImportError (file doesn't exist yet)**

```bash
pytest tests/test_live_bench.py -v 2>&1 | head -20
```
Expected: `ModuleNotFoundError` or `ImportError`.

- [ ] **Step 3: Create `src/pareto_llm/benchmarks/live_bench.py`**

```python
"""LiveBench integration benchmark."""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import re
import socket
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from pareto_llm.backend.base import GenerationResult, LLMBackend
from pareto_llm.benchmarks.base import Benchmark, BenchmarkResult, register

_logger = logging.getLogger(__name__)

VALID_CATEGORIES = frozenset({
    "coding",
    "math",
    "reasoning",
    "language",
    "data_analysis",
    "instruction_following",
})

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


@register("live_bench")
class LiveBenchBenchmark(Benchmark):
    supports_context_padding = False

    DEFAULTS: dict = {
        "port": 8766,
        "categories": "all",
        "release_date": "latest",
        "sample_size": None,
        "sample_size_per_category": {},
        "seed": 42,
        "skip_agentic": True,
        "n_ctx": 8192,
        "parallel": 4,
        "jobs_dir": "./results/livebench",
    }

    def __init__(self, config: dict) -> None:
        merged = {**self.DEFAULTS, **config}
        super().__init__(merged)

        # Config validation first (before livebench import check)
        cats = self.config["categories"]
        if cats != "all":
            if not isinstance(cats, list) or len(cats) == 0:
                raise ValueError(
                    "categories must be 'all' or a non-empty list of valid category names"
                )
            invalid = set(cats) - VALID_CATEGORIES
            if invalid:
                raise ValueError(
                    f"categories contains invalid values: {invalid}. "
                    f"Must be subset of {sorted(VALID_CATEGORIES)}"
                )

        port = self.config["port"]
        if not (1024 <= port <= 65535):
            raise ValueError(f"port must be between 1024 and 65535, got {port}")

        sample_size = self.config["sample_size"]
        if sample_size is not None and sample_size <= 0:
            raise ValueError(f"sample_size must be > 0, got {sample_size}")

        for cat, sz in self.config["sample_size_per_category"].items():
            if sz <= 0:
                raise ValueError(
                    f"sample_size_per_category[{cat!r}] must be > 0, got {sz}"
                )

        release_date = self.config["release_date"]
        if release_date != "latest" and not _DATE_RE.match(release_date):
            raise ValueError(
                f"release_date must be 'latest' or YYYY-MM-DD, got {release_date!r}"
            )

        parallel = self.config["parallel"]
        if parallel < 1:
            raise ValueError(f"parallel must be >= 1, got {parallel}")

        # Livebench availability check last
        if importlib.util.find_spec("livebench") is None:
            raise RuntimeError(
                "livebench package not found. Install with: "
                "uv pip install 'pareto-llm[live-bench]'"
            )

    def _get_filtered_questions(self) -> list[dict]:
        raise NotImplementedError

    def run_single(self, backend: LLMBackend) -> tuple[BenchmarkResult, GenerationResult]:
        raise NotImplementedError
```

- [ ] **Step 4: Register in `__init__.py`**

In `src/pareto_llm/benchmarks/__init__.py`, add:
```python
from pareto_llm.benchmarks import live_bench as _live_bench  # noqa: F401
```

- [ ] **Step 5: Run validation tests — expect pass**

```bash
pytest tests/test_live_bench.py -v -k "not run_single and not filter and not skip and not sampling and not no_questions"
```
Expected: All pass.

- [ ] **Step 6: Commit**

```bash
git add src/pareto_llm/benchmarks/live_bench.py src/pareto_llm/benchmarks/__init__.py tests/test_live_bench.py
git commit -m "feat: scaffold LiveBenchBenchmark with config validation"
```

---

## Task 3: Implement `_get_filtered_questions()`

**Files:**
- Modify: `src/pareto_llm/benchmarks/live_bench.py`
- Modify: `tests/test_live_bench.py`

- [ ] **Step 1: Discover the LiveBench question-loading API**

With livebench installed, run this in a Python shell to find the question loader:
```python
from livebench import common
print([x for x in dir(common) if any(k in x.lower() for k in ("question", "load", "get"))])

import inspect
# Try the most likely candidates:
from livebench.common import load_questions  # or get_questions
print(inspect.signature(load_questions))
```

Also check how agentic questions are marked — look for a field like `"task_type"`, `"agentic"`, or a category value like `"agentic_coding"`:
```python
# Load a few questions and inspect their structure
qs = load_questions(bench_name="live_bench", livebench_release_option=None)
import json
print(json.dumps(qs[0], indent=2, default=str))
# Note: what field distinguishes agentic questions?
```

**Update `_AGENTIC_MARKER` and `_LOAD_FN` in the implementation below based on what you find.**

- [ ] **Step 2: Write failing filtering tests**

Add to `tests/test_live_bench.py`:

```python
import logging
from unittest.mock import patch


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
            # Field name verified in Task 3 Step 1 — update if needed
            q["task_type"] = "agentic_coding"
        questions.append(q)
    return questions


# Update this patch path to match the actual load function discovered in Step 1
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
    bench = LiveBenchBenchmark(_valid_config(skip_agentic=False, jobs_dir=str(tmp_path)))
    with patch.object(bench, "_load_questions", return_value=all_qs):
        result = bench._get_filtered_questions()
    assert len(result) == 2


def test_sampling_global(tmp_path):
    all_qs = _make_questions([("coding", False)] * 10 + [("math", False)] * 10)
    bench = LiveBenchBenchmark(_valid_config(categories="all", sample_size=3, seed=42, jobs_dir=str(tmp_path)))
    with patch.object(bench, "_load_questions", return_value=all_qs):
        result = bench._get_filtered_questions()
    coding_count = sum(1 for q in result if q["category"] == "coding")
    math_count = sum(1 for q in result if q["category"] == "math")
    assert coding_count <= 3
    assert math_count <= 3


def test_sampling_per_category_override(tmp_path):
    all_qs = _make_questions([("coding", False)] * 10 + [("math", False)] * 10)
    bench = LiveBenchBenchmark(_valid_config(
        categories="all",
        sample_size=5,
        sample_size_per_category={"coding": 2},
        seed=42,
        jobs_dir=str(tmp_path),
    ))
    with patch.object(bench, "_load_questions", return_value=all_qs):
        result = bench._get_filtered_questions()
    coding_count = sum(1 for q in result if q["category"] == "coding")
    math_count = sum(1 for q in result if q["category"] == "math")
    assert coding_count <= 2   # per-category override wins
    assert math_count <= 5     # global sample_size applies


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
```

- [ ] **Step 3: Run new tests — expect NotImplementedError**

```bash
pytest tests/test_live_bench.py -v -k "filter or skip or sampling or no_questions"
```
Expected: FAIL with `NotImplementedError` (method not implemented yet).

- [ ] **Step 4: Implement `_load_questions()` and `_get_filtered_questions()`**

Replace the `_get_filtered_questions` stub in `live_bench.py`:

```python
# ─────────────────────────────────────────────────────────────────────────────
# NOTE: Verify these two constants against the LiveBench version you installed:
#
#   _AGENTIC_FIELD — the question dict field that marks agentic tasks.
#   Common values: ("task_type", "agentic_coding") or ("agentic", True).
#   Inspect a loaded question dict to confirm (see Task 3 Step 1).
#
#   _LOAD_FN_PATH — dotted import path to the question-loading function.
#   Check livebench.common or livebench.gen_api_answer for load_questions /
#   get_questions, then confirm its signature accepts bench_name and
#   livebench_release_option kwargs.
# ─────────────────────────────────────────────────────────────────────────────
_AGENTIC_FIELD = ("task_type", "agentic_coding")   # (field_name, value_to_skip)
_LOAD_FN_PATH = "livebench.common.load_questions"  # verify at implementation time


def _load_questions(self) -> list[dict]:
    """Load all LiveBench questions from HuggingFace (internal, testable seam)."""
    import importlib

    module_path, fn_name = _LOAD_FN_PATH.rsplit(".", 1)
    mod = importlib.import_module(module_path)
    load_fn = getattr(mod, fn_name)

    release_date = self.config["release_date"]
    release_opt = None if release_date == "latest" else release_date
    return load_fn(bench_name="live_bench", livebench_release_option=release_opt)
```

Then replace the `_get_filtered_questions` stub:

```python
def _get_filtered_questions(self) -> list[dict]:
    import random

    all_questions = self._load_questions()

    # Filter by category
    cats = self.config["categories"]
    if cats != "all":
        cat_set = set(cats)
        all_questions = [q for q in all_questions if q.get("category") in cat_set]

    # Filter agentic questions
    if self.config["skip_agentic"]:
        field, skip_val = _AGENTIC_FIELD
        all_questions = [q for q in all_questions if q.get(field) != skip_val]

    # Subsample per category
    sample_size = self.config["sample_size"]
    per_cat = self.config["sample_size_per_category"]
    seed = self.config["seed"]

    if sample_size is not None or per_cat:
        by_cat: dict[str, list[dict]] = defaultdict(list)
        for q in all_questions:
            by_cat[q["category"]].append(q)

        sampled: list[dict] = []
        rng = random.Random(seed)
        for cat, qs in by_cat.items():
            limit = per_cat.get(cat, sample_size)
            if limit is not None:
                if limit > len(qs):
                    _logger.warning(
                        "sample_size=%d exceeds available questions (%d) for category %r; using all",
                        limit,
                        len(qs),
                        cat,
                    )
                else:
                    qs = rng.sample(qs, limit)
            sampled.extend(qs)
        all_questions = sampled

    if not all_questions:
        raise ValueError(
            f"No questions found for categories={self.config['categories']!r} "
            f"and release_date={self.config['release_date']!r}"
        )

    return all_questions
```

- [ ] **Step 5: Run the filtering tests — expect pass**

```bash
pytest tests/test_live_bench.py -v -k "filter or skip or sampling or no_questions"
```
Expected: All pass.

- [ ] **Step 6: Commit**

```bash
git add src/pareto_llm/benchmarks/live_bench.py tests/test_live_bench.py
git commit -m "feat: implement _get_filtered_questions() with category/agentic/sampling filters"
```

---

## Task 4: Implement `run_single()`

**Files:**
- Modify: `src/pareto_llm/benchmarks/live_bench.py`
- Modify: `tests/test_live_bench.py`

- [ ] **Step 1: Discover the `ModelConfig` and model-answer loader APIs**

With livebench installed, inspect:
```python
from livebench.model import get_model_config
import inspect
print(inspect.signature(get_model_config))
# Try building a local-server config:
cfg = get_model_config("local")
print(cfg)
```

Also find how to load the answer JSONL files back into memory (needed for `gen_judgments`):
```python
from livebench import common
print([x for x in dir(common) if "answer" in x.lower()])
# Then:
from livebench.common import load_model_answers  # adjust name if needed
print(inspect.signature(load_model_answers))
```

**Update `_build_model_config()` and the `load_model_answers` call in `run_single()` based on what you find.**

- [ ] **Step 2: Write failing `run_single` tests**

Add to `tests/test_live_bench.py`:

```python
import contextlib
import socket
from pathlib import Path
from unittest.mock import MagicMock, patch


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
                f.write(json.dumps({
                    "question_id": f"q{i}",
                    "category": category,
                    "task": f"live_bench/{category}/test_task",
                    "model": "local",
                    "score": score,
                }) + "\n")


def test_run_single_score_computation(tmp_path):
    pre_filtered = [
        {"question_id": "q0", "category": "coding", "turns": ["q"]},
        {"question_id": "q1", "category": "coding", "turns": ["q"]},
        {"question_id": "q2", "category": "math", "turns": ["q"]},
        {"question_id": "q3", "category": "math", "turns": ["q"]},
    ]
    bench = LiveBenchBenchmark(_valid_config(jobs_dir=str(tmp_path)))

    def fake_gen_judgments(*args, **kwargs):
        _write_judgment_files(Path.cwd(), {
            "coding": [1.0, 0.0],   # mean = 0.5
            "math": [1.0, 1.0],     # mean = 1.0
        })

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


def test_run_single_port_in_use(tmp_path):
    bench = LiveBenchBenchmark(_valid_config(jobs_dir=str(tmp_path), port=19001))
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("127.0.0.1", 19001))
        with patch.object(bench, "_get_filtered_questions", return_value=[MagicMock()]):
            with pytest.raises(RuntimeError, match="already in use"):
                bench.run_single(_make_mock_backend())
```

- [ ] **Step 3: Run the tests — expect NotImplementedError**

```bash
pytest tests/test_live_bench.py -v -k "run_single or port_in_use"
```
Expected: FAIL with `NotImplementedError`.

- [ ] **Step 4: Implement `run_single()` in `live_bench.py`**

Replace the `run_single` stub:

```python
def _load_model_answers(self, run_dir: Path) -> dict:
    """Load model answer JSONL files written by run_questions() (internal seam for testing).

    Verify the correct function and signature from livebench.common at implementation time.
    Common signature: load_model_answers(answer_dir, model_list) -> {model: {qid: answer}}
    """
    from livebench.common import load_model_answers  # verify import path

    # run_questions() writes to data/ relative to CWD (run_dir)
    return load_model_answers(
        answer_dir=str(run_dir / "data"),
        model_list=["local"],
    )


def run_single(self, backend: LLMBackend) -> tuple[BenchmarkResult, GenerationResult]:
    from livebench.gen_api_answer import run_questions
    from livebench.gen_ground_truth_judgment import gen_judgments
    from livebench.model import get_model_config  # verify import; adjust if needed

    port = self.config["port"]

    # Check port availability
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", port))
    except OSError:
        raise RuntimeError(f"Port {port} is already in use")

    questions = self._get_filtered_questions()

    # Timestamped run directory — LiveBench writes files relative to CWD,
    # so we chdir here and restore in the finally block.
    run_name = f"lbench_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    run_dir = Path(self.config["jobs_dir"]) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # LiteLLM requires OPENAI_API_KEY even for local servers
    os.environ.setdefault("OPENAI_API_KEY", "local")

    # Build model config for local server.
    # Inspect get_model_config() to confirm the correct call for a local endpoint.
    # The display name "local" becomes the answer file basename (local.jsonl).
    model_config = get_model_config("local")  # adjust if the API needs api_base here
    api_dict = {
        "api_key": os.environ.get("OPENAI_API_KEY", "local"),
        "api_base": f"http://localhost:{port}/v1",
    }

    orig_cwd = Path.cwd()
    try:
        os.chdir(run_dir)

        with backend.serve_openai(port, n_ctx=self.config["n_ctx"]):
            run_questions(
                parallel=self.config["parallel"],
                questions=questions,
                model_config=model_config,
                num_choices=1,
                max_tokens=4096,
                answer_file=None,
                stream=False,
                force_temperature=None,
                model_provider_override="openai",
                model_display_name_override="local",
                api_dict=api_dict,
            )

        # Scoring is offline — load answers and judge
        model_answers = self._load_model_answers(run_dir)

        gen_judgments(
            parallel=self.config["parallel"],
            questions=questions,
            output_file=None,
            model_answers=model_answers,
            model_list=["local"],
            remove_existing_file=False,
            bench_name="live_bench",
        )

    finally:
        os.chdir(orig_cwd)

    # Aggregate scores from judgment JSONL files
    per_category_scores: dict[str, float] = {}
    per_category_counts: dict[str, int] = {}
    all_scores: list[float] = []

    for jfile in run_dir.glob("**/ground_truth_judgment.jsonl"):
        with jfile.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                score = float(entry["score"])
                if score < 0:  # -1 marks invalid/missing scores
                    continue
                cat = entry.get("category", "unknown")
                all_scores.append(score)
                per_category_scores.setdefault(cat, 0.0)
                per_category_counts.setdefault(cat, 0)
                per_category_scores[cat] += score
                per_category_counts[cat] += 1

    for cat in per_category_scores:
        per_category_scores[cat] /= per_category_counts[cat]

    overall = sum(all_scores) / len(all_scores) if all_scores else 0.0

    return (
        BenchmarkResult(
            score=overall,
            extra={
                "per_category_scores": per_category_scores,
                "per_category_counts": per_category_counts,
                "run_name": run_name,
            },
        ),
        GenerationResult(text="", prompt_tokens=0, gen_tokens=0, prompt_tps=0.0, gen_tps=0.0),
    )
```

- [ ] **Step 5: Run the `run_single` tests — expect pass**

```bash
pytest tests/test_live_bench.py -v -k "run_single or port_in_use"
```
Expected: All pass.

- [ ] **Step 6: Run the full test suite**

```bash
pytest tests/test_live_bench.py -v
```
Expected: All pass.

- [ ] **Step 7: Commit**

```bash
git add src/pareto_llm/benchmarks/live_bench.py tests/test_live_bench.py
git commit -m "feat: implement LiveBenchBenchmark.run_single()"
```

---

## Task 5: Example config and setup.sh

**Files:**
- Create: `configs/live_bench.yaml`
- Modify: `scripts/setup.sh`

- [ ] **Step 1: Create `configs/live_bench.yaml`**

```yaml
run_label: live_bench_run

defaults:
  runs_per_test: 1        # each call triggers a full LiveBench job — don't repeat
  keep_model_files: true

# TODO: verify this model produces correctly-formatted responses for LiveBench.
# LiveBench uses standard chat completions; most instruction-following models work.
models:
  - bartowski/Qwen2.5-7B-Instruct-GGUF:Q4_K_M

benchmarks:
  - type: live_bench
    name: livebench_coding_sample
    config:
      port: 8766
      # Available categories: coding, math, reasoning, language, data_analysis, instruction_following
      # Use "all" to run every category, or a list to select specific ones.
      categories: ["coding"]
      release_date: "latest"
      sample_size: 5          # small sample for quick testing
      seed: 42
      skip_agentic: true      # set false to include Docker-based agentic coding tasks
      n_ctx: 8192
      parallel: 4
      jobs_dir: ./results/livebench
```

- [ ] **Step 2: Update `scripts/setup.sh`**

After the MLX install block and before the final `echo "Setup complete..."` line, add:

```bash
echo ""
echo "Optional: To install LiveBench extras (for the live_bench benchmark type):"
echo "  uv pip install \"pareto-llm[live-bench]\""
```

- [ ] **Step 3: Run full test suite one final time**

```bash
pytest -v
```
Expected: All tests pass.

- [ ] **Step 4: Run pre-commit**

```bash
export PATH="$HOME/.local/bin:$PATH"
pre-commit run --all-files
```
Fix any ruff/format issues, re-run until clean.

- [ ] **Step 5: Commit**

```bash
git add configs/live_bench.yaml scripts/setup.sh
git commit -m "feat: add live_bench example config and setup.sh install hint"
```
