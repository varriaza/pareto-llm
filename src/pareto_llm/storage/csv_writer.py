"""Append-only CSV writer for benchmark results."""
from __future__ import annotations

import csv
import pathlib
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pareto_llm.backend.base import GenerationResult
    from pareto_llm.benchmarks.base import BenchmarkResult
    from pareto_llm.metrics.system import SystemMetricsCollector

_BASE_FIELDS = [
    "timestamp",
    "run_label",
    "model_id",
    "benchmark_name",
    "run_num",
    "score",
    "prompt_tokens",
    "gen_tokens",
    "gen_tps",
    "prompt_tps",
    "ram_max_gb",
    "ram_avg_gb",
    "gpu_ram_max_gb",
    "gpu_ram_avg_gb",
    "max_ctx_tokens",
]


class CsvWriter:
    def __init__(self, path: pathlib.Path) -> None:
        self._path = path
        self._extra_keys: list[str] = []  # extra_* column names seen so far

    def append(
        self,
        *,
        run_label: str,
        model_id: str,
        benchmark_name: str,
        run_num: int,
        result: "BenchmarkResult",
        gen_result: "GenerationResult",
        collector: "SystemMetricsCollector",
        max_ctx_tokens: int,
    ) -> None:
        extra_row = {f"extra_{k}": v for k, v in result.extra.items()}
        row = {
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "run_label": run_label,
            "model_id": model_id,
            "benchmark_name": benchmark_name,
            "run_num": run_num,
            "score": result.score,
            "prompt_tokens": gen_result.prompt_tokens,
            "gen_tokens": gen_result.gen_tokens,
            "gen_tps": gen_result.gen_tps,
            "prompt_tps": gen_result.prompt_tps,
            "ram_max_gb": collector.ram_max_gb,
            "ram_avg_gb": collector.ram_avg_gb,
            "gpu_ram_max_gb": collector.gpu_ram_max_gb
                if collector.gpu_ram_max_gb is not None
                else "",
            "gpu_ram_avg_gb": collector.gpu_ram_avg_gb
                if collector.gpu_ram_avg_gb is not None
                else "",
            "max_ctx_tokens": max_ctx_tokens,
            **extra_row,
        }
        self._write_row(row)

    def append_failure(
        self,
        *,
        run_label: str,
        model_id: str,
        benchmark_name: str,
        run_num: int,
        error: Exception,
    ) -> None:
        row: dict = {
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "run_label": run_label,
            "model_id": model_id,
            "benchmark_name": benchmark_name,
            "run_num": run_num,
            "score": "",
            "prompt_tokens": "",
            "gen_tokens": "",
            "gen_tps": "",
            "prompt_tps": "",
            "ram_max_gb": "",
            "ram_avg_gb": "",
            "gpu_ram_max_gb": "",
            "gpu_ram_avg_gb": "",
            "max_ctx_tokens": "",
            "extra_error": str(error),
        }
        self._write_row(row)

    # ── Internals ─────────────────────────────────────────────────────────────

    @property
    def _fieldnames(self) -> list[str]:
        return _BASE_FIELDS + self._extra_keys

    def _write_row(self, row: dict) -> None:
        # Detect any extra_* keys in this row that we haven't seen before
        new_keys = [
            k for k in row
            if k.startswith("extra_") and k not in self._extra_keys
        ]
        for k in new_keys:
            self._extra_keys.append(k)

        if new_keys and self._path.exists():
            # New column introduced mid-run: rewrite the entire file so the
            # header is updated and earlier rows get an empty cell for the new key.
            existing: list[dict] = []
            with self._path.open("r", newline="") as fh:
                existing = list(csv.DictReader(fh))
            with self._path.open("w", newline="") as fh:
                writer = csv.DictWriter(
                    fh, fieldnames=self._fieldnames, extrasaction="ignore", restval=""
                )
                writer.writeheader()
                for r in existing:
                    writer.writerow(r)
                writer.writerow(row)
        else:
            file_exists = self._path.exists()
            with self._path.open("a", newline="") as fh:
                writer = csv.DictWriter(
                    fh, fieldnames=self._fieldnames, extrasaction="ignore", restval=""
                )
                if not file_exists:
                    writer.writeheader()
                writer.writerow(row)
