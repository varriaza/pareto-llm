"""LiveBench integration benchmark."""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import random
import re
import socket
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

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

    # Oldest release whose instruction_following questions use the new IFBench format.
    # gen_judgments calls set() on question dicts for older questions, which crashes
    # because dicts are unhashable. Upstream bug: livebench/gen_ground_truth_judgment.py:439
    _MIN_IF_RELEASE = "2025-11-25"

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

        # Drop old-format instruction_following questions — they trigger an upstream bug
        # in gen_judgments (set() on unhashable dicts) for release_date < _MIN_IF_RELEASE.
        before = len(all_questions)
        all_questions = [
            q
            for q in all_questions
            if not (
                q.get("category") == "instruction_following"
                and q.get("livebench_release_date", "") < self._MIN_IF_RELEASE
            )
        ]
        skipped = before - len(all_questions)
        if skipped:
            _logger.warning(
                "Skipped %d old-format instruction_following questions "
                "(livebench_release_date < %s) due to upstream bug in gen_judgments.",
                skipped,
                self._MIN_IF_RELEASE,
            )

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

    def _load_model_answers(self, run_dir: Path) -> dict:
        """Load model answer JSONL files written by run_questions() (internal seam for testing).

        Reads JSONL files directly instead of calling livebench.common.load_model_answers()
        because that function requires a model_configs/ directory not present in the installed package.
        Returns {model_name: {question_id: answer_dict}} matching the format gen_judgments() expects.
        """
        answers: dict[str, dict] = {}
        for jsonl_file in (run_dir / "data").rglob("model_answer/*.jsonl"):
            model_name = jsonl_file.stem  # e.g. "local"
            if model_name not in answers:
                answers[model_name] = {}
            with jsonl_file.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    entry = json.loads(line)
                    qid = entry.get("question_id")
                    if qid is not None:
                        answers[model_name][qid] = entry
        return answers

    def run_single(self, backend: LLMBackend) -> tuple[BenchmarkResult, GenerationResult]:
        from livebench.gen_api_answer import run_questions
        from livebench.gen_ground_truth_judgment import gen_judgments
        from livebench.model.api_model_config import ModelConfig

        port = self.config["port"]

        # Check port availability BEFORE loading questions
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", port))
        except OSError:
            raise RuntimeError(f"Port {port} is already in use")

        questions = self._get_filtered_questions()

        # Timestamped run directory — LiveBench writes files relative to CWD,
        # so we chdir here and restore in the finally block.
        run_name = f"lbench_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        run_dir = (Path(self.config["jobs_dir"]) / run_name).resolve()
        run_dir.mkdir(parents=True, exist_ok=True)

        # LiteLLM requires OPENAI_API_KEY even for local servers
        os.environ.setdefault("OPENAI_API_KEY", "local")

        api_dict = {
            "api_key": os.environ.get("OPENAI_API_KEY", "local"),
            "api_base": f"http://localhost:{port}/v1",
        }

        # Build a minimal ModelConfig for the local server — avoids the need
        # for livebench's model_configs directory to be present.
        model_config = ModelConfig(
            display_name="local",
            api_name={"local": "local"},
            default_provider="local",
            aliases=[],
            api_kwargs={},
        )

        # Attach answer_file to each question so run_questions knows where to write
        answer_file = str(run_dir / "data" / "live_bench" / "model_answer" / "local.jsonl")

        orig_cwd = Path.cwd()
        _saved_key = None
        elapsed = 0.0
        try:
            os.chdir(run_dir)

            with backend.serve_openai(port, n_ctx=self.config["n_ctx"]):
                t0 = time.perf_counter()
                run_questions(
                    parallel=self.config["parallel"],
                    questions=questions,
                    model_config=model_config,
                    num_choices=1,
                    max_tokens=4096,
                    answer_file=answer_file,
                    stream=False,
                    force_temperature=None,
                    model_provider_override="local",
                    model_display_name_override="local",
                    api_dict=api_dict,
                )
                elapsed = time.perf_counter() - t0

            # Scoring is offline — load answers and judge.
            # Unset OPENAI_API_KEY so AMPS_Hard scorer doesn't try to call the real
            # OpenAI API (it was set to "local" just to satisfy LiteLLM above).
            _saved_key = os.environ.pop("OPENAI_API_KEY", None)
            model_answers = self._load_model_answers(run_dir)

            # Sum output tokens across all answers; prompt tokens are not available
            # from the OpenAI-compatible server responses.
            total_gen_tokens = sum(
                answer.get("total_output_tokens", 0)
                for answers_by_qid in model_answers.values()
                for answer in answers_by_qid.values()
            )

            # Write judgment JSONL relative to run_dir (our CWD at this point)
            judgment_file = "data/live_bench/model_judgment/ground_truth_judgment.jsonl"
            gen_judgments(
                parallel=self.config["parallel"],
                questions=questions,
                output_file=judgment_file,
                model_answers=model_answers,
                model_list=["local"],
                remove_existing_file=False,
                bench_name="live_bench",
            )

        finally:
            os.chdir(orig_cwd)
            if _saved_key is not None:
                os.environ["OPENAI_API_KEY"] = _saved_key

        # Aggregate scores from judgment JSONL files
        per_category_scores: dict[str, float] = {}
        per_category_counts: dict[str, int] = {}

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
                    per_category_scores.setdefault(cat, 0.0)
                    per_category_counts.setdefault(cat, 0)
                    per_category_scores[cat] += score
                    per_category_counts[cat] += 1

        for cat in per_category_scores:
            per_category_scores[cat] /= per_category_counts[cat]

        if not per_category_scores:
            raise RuntimeError(
                f"No judgment scores found under {run_dir}. "
                "Check that gen_judgments() wrote ground_truth_judgment.jsonl files."
            )
        overall = sum(per_category_scores.values()) / len(per_category_scores)

        return (
            BenchmarkResult(
                score=overall,
                extra={
                    "per_category_scores": per_category_scores,
                    "per_category_counts": per_category_counts,
                    "run_name": run_name,
                },
            ),
            GenerationResult(
                text="",
                prompt_tokens=0,
                gen_tokens=total_gen_tokens,
                prompt_tps=0.0,
                gen_tps=total_gen_tokens / elapsed if elapsed > 0 else 0.0,
            ),
        )
