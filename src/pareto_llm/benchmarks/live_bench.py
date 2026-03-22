"""LiveBench integration benchmark."""

from __future__ import annotations

import importlib.util
import logging
import re

from pareto_llm.backend.base import GenerationResult, LLMBackend
from pareto_llm.benchmarks.base import Benchmark, BenchmarkResult, register

_logger = logging.getLogger(__name__)

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

    def _get_filtered_questions(self) -> list[dict]:
        raise NotImplementedError

    def run_single(self, backend: LLMBackend) -> tuple[BenchmarkResult, GenerationResult]:
        raise NotImplementedError
