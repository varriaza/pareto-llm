"""LiveBench integration benchmark."""

from __future__ import annotations

import importlib.util
import logging
import random
import re
from collections import defaultdict

from pareto_llm.backend.base import GenerationResult, LLMBackend
from pareto_llm.benchmarks.base import Benchmark, BenchmarkResult, register

_logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Verified against livebench installed in this repo:
#
#   _AGENTIC_FIELD — agentic questions have category == "agentic_coding"
#   (confirmed in livebench/gen_api_answer.py:
#    `[q for q in questions if q.get('category') == 'agentic_coding']`)
#
#   Question loading uses get_categories_tasks() + load_questions() from
#   livebench.common. load_questions() takes a HF Dataset as first arg, not
#   a bench_name string — so _load_questions() calls get_categories_tasks().
# ─────────────────────────────────────────────────────────────────────────────
_AGENTIC_FIELD = ("category", "agentic_coding")  # (field_name, value_to_skip)

VALID_CATEGORIES = frozenset(
    {
        "coding",
        "math",
        "reasoning",
        "language",
        "data_analysis",
        "instruction_following",
    }
)

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
        merged["sample_size_per_category"] = dict(merged["sample_size_per_category"])
        super().__init__(merged)

        # Config validation first (before livebench import check)
        cats = self.config["categories"]
        if cats != "all":
            if not isinstance(cats, list) or len(cats) == 0:
                raise ValueError("categories must be 'all' or a non-empty list of valid category names")
            invalid = set(cats) - VALID_CATEGORIES
            if invalid:
                raise ValueError(
                    f"categories contains invalid values: {invalid}. Must be subset of {sorted(VALID_CATEGORIES)}"
                )

        port = self.config["port"]
        if not (1024 <= port <= 65535):
            raise ValueError(f"port must be between 1024 and 65535, got {port}")

        sample_size = self.config["sample_size"]
        if sample_size is not None and sample_size <= 0:
            raise ValueError(f"sample_size must be > 0, got {sample_size}")

        for cat, sz in self.config["sample_size_per_category"].items():
            if sz <= 0:
                raise ValueError(f"sample_size_per_category[{cat!r}] must be > 0, got {sz}")

        release_date = self.config["release_date"]
        if release_date != "latest" and not _DATE_RE.match(release_date):
            raise ValueError(f"release_date must be 'latest' or YYYY-MM-DD, got {release_date!r}")

        parallel = self.config["parallel"]
        if parallel < 1:
            raise ValueError(f"parallel must be >= 1, got {parallel}")

        # Livebench availability check last
        if importlib.util.find_spec("livebench") is None:
            raise RuntimeError("livebench package not found. Install with: uv pip install 'pareto-llm[live-bench]'")

    def _load_questions(self) -> list[dict]:
        """Load all LiveBench questions from HuggingFace (internal, testable seam)."""
        from livebench.common import get_categories_tasks, load_questions

        release_date = self.config["release_date"]
        release_opt = None if release_date == "latest" else release_date

        categories, _tasks = get_categories_tasks("live_bench")
        all_questions: list[dict] = []
        for category_name, dataset in categories.items():
            qs = load_questions(
                dataset,
                livebench_release=release_opt,
            )
            # Attach category name so filtering works consistently
            for q in qs:
                if "category" not in q:
                    q["category"] = category_name
            all_questions.extend(qs)
        return all_questions

    def _get_filtered_questions(self) -> list[dict]:
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
                by_cat[q.get("category", "unknown")].append(q)

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

    def run_single(self, backend: LLMBackend) -> tuple[BenchmarkResult, GenerationResult]:
        raise NotImplementedError
