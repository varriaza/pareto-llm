"""Orchestrates the model × benchmark × run_num test matrix."""
from __future__ import annotations

import logging
import pathlib
from typing import TYPE_CHECKING

import pareto_llm.benchmarks  # noqa: F401 — triggers @register side effects
from pareto_llm.benchmarks.base import BENCHMARK_REGISTRY
from pareto_llm.metrics.system import SystemMetricsCollector
from pareto_llm.storage.csv_writer import CsvWriter

if TYPE_CHECKING:
    from pareto_llm.backend.base import LLMBackend
    from pareto_llm.config import BenchmarkConfig

log = logging.getLogger(__name__)


def run(
    config: "BenchmarkConfig",
    output_path: pathlib.Path,
    gpu_backend: str,
) -> None:
    """Execute the full benchmark matrix and write results to *output_path*."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = CsvWriter(output_path)
    backend = _create_backend(gpu_backend)

    for model_id in config.models:
        log.info("Loading model: %s", model_id)
        backend.load(model_id)
        try:
            for bench_entry in config.benchmarks:
                bench_cls = BENCHMARK_REGISTRY[bench_entry.type]
                benchmark = bench_cls(bench_entry.config)
                for run_num in range(1, config.defaults.runs_per_test + 1):
                    log.info(
                        "[%s][%s] run %d/%d",
                        model_id, bench_entry.name,
                        run_num, config.defaults.runs_per_test,
                    )
                    try:
                        with SystemMetricsCollector(gpu_backend=gpu_backend) as collector:
                            bench_result, gen_result = benchmark.run_single(backend)
                        writer.append(
                            run_label=config.run_label,
                            model_id=model_id,
                            benchmark_name=bench_entry.name,
                            run_num=run_num,
                            result=bench_result,
                            gen_result=gen_result,
                            collector=collector,
                            max_ctx_tokens=backend.max_context_tokens(),
                        )
                    except Exception as exc:
                        log.error(
                            "[%s][%s] run %d failed: %s",
                            model_id, bench_entry.name, run_num, exc,
                        )
                        writer.append_failure(
                            run_label=config.run_label,
                            model_id=model_id,
                            benchmark_name=bench_entry.name,
                            run_num=run_num,
                            error=exc,
                        )
        finally:
            backend.unload()
        if not config.defaults.keep_model_files:
            _delete_hf_cache(model_id)


def _create_backend(gpu_backend: str) -> "LLMBackend":
    if gpu_backend == "mlx":
        from pareto_llm.backend.mlx_backend import MLXBackend
        return MLXBackend()
    if gpu_backend == "cuda":
        from pareto_llm.backend.llamacpp_backend import LlamaCppBackend
        return LlamaCppBackend()
    raise ValueError(f"Unknown GPU backend: {gpu_backend!r}. Expected 'mlx' or 'cuda'.")


def _delete_hf_cache(model_id: str) -> None:
    """Remove a model's Hugging Face cache files."""
    try:
        from huggingface_hub import scan_cache_dir
        cache_info = scan_cache_dir()
        for repo in cache_info.repos:
            if repo.repo_id == model_id or repo.repo_id == model_id.split(":")[0]:
                for revision in repo.revisions:
                    revision.refs  # ensure loaded
                # Delete via the strategy API
                delete_strategy = cache_info.delete_revisions(
                    *(rev.commit_hash for rev in repo.revisions)
                )
                delete_strategy.execute()
                log.info("Deleted HF cache for %s", model_id)
                return
        log.warning("No HF cache entry found for %s", model_id)
    except Exception as exc:
        log.warning("Failed to delete HF cache for %s: %s", model_id, exc)
