"""Terminal Bench 2.0 integration benchmark."""

from __future__ import annotations

import asyncio
import concurrent.futures
import importlib.util
import logging
import math
import os
import random
import socket
import tomllib
import urllib.request
from datetime import datetime
from pathlib import Path

from pareto_llm.backend.base import GenerationResult, LLMBackend
from pareto_llm.benchmarks.base import Benchmark, BenchmarkResult, register

_logger = logging.getLogger(__name__)


@register("terminal_bench")
class TerminalBenchmark(Benchmark):
    supports_context_padding = False

    VALID_DIFFICULTIES = {"easy", "medium", "hard", "unknown"}

    DEFAULTS: dict = {
        "difficulties": ["medium", "hard"],
        "port": 8765,
        "harbor_agent": "terminus-2",
        "sample_size": None,
        "seed": 42,
        "dataset_version": "2.0",
        "jobs_dir": "./results/harbor",
        "n_concurrent": 4,
        "n_ctx": 8192,
    }

    def __init__(self, config: dict) -> None:
        merged = {**self.DEFAULTS, **config}
        super().__init__(merged)

        if importlib.util.find_spec("harbor") is None:
            raise RuntimeError("harbor package not found. Install with: uv tool install harbor or pip install harbor")

        invalid = set(self.config["difficulties"]) - self.VALID_DIFFICULTIES
        if invalid:
            raise ValueError(
                f"difficulties contains invalid values: {invalid}. Must be subset of {self.VALID_DIFFICULTIES}"
            )

        port = self.config["port"]
        if not (1024 <= port <= 65535):
            raise ValueError(f"port must be between 1024 and 65535, got {port}")

        sample_size = self.config["sample_size"]
        if sample_size is not None and sample_size <= 0:
            raise ValueError(f"sample_size must be > 0, got {sample_size}")

    @staticmethod
    def _fetch_difficulty(task_ref) -> str:
        repo = task_ref.git_url.replace("https://github.com/", "").replace(".git", "")
        url = f"https://raw.githubusercontent.com/{repo}/{task_ref.git_commit_id}/{task_ref.path}/task.toml"
        try:
            data = urllib.request.urlopen(url, timeout=10).read()
            cfg = tomllib.loads(data.decode())
            return cfg.get("metadata", {}).get("difficulty", "unknown")
        except Exception:
            return "unknown"

    def _get_filtered_tasks(self) -> list:
        from harbor.models.registry import RemoteRegistryInfo
        from harbor.models.trial.config import TaskConfig
        from harbor.registry.client import RegistryClientFactory

        registry = RemoteRegistryInfo()
        client = RegistryClientFactory.create(registry)
        dataset_spec = client.get_dataset_spec("terminal-bench", self.config["dataset_version"])

        _logger.info("Fetching task metadata for %d tasks…", len(dataset_spec.tasks))
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as pool:
            difficulties = list(pool.map(self._fetch_difficulty, dataset_spec.tasks))

        filtered: list[TaskConfig] = [
            TaskConfig(
                path=task_ref.path,
                git_url=task_ref.git_url,
                git_commit_id=task_ref.git_commit_id,
            )
            for task_ref, difficulty in zip(dataset_spec.tasks, difficulties)
            if difficulty in self.config["difficulties"]
        ]

        sample_size = self.config["sample_size"]
        if sample_size is not None:
            if sample_size > len(filtered):
                _logger.warning(
                    "sample_size=%d exceeds available tasks (%d); using all",
                    sample_size,
                    len(filtered),
                )
            else:
                filtered = random.Random(self.config["seed"]).sample(filtered, sample_size)

        if not filtered:
            raise ValueError(f"No tasks found with difficulties {self.config['difficulties']}")

        return filtered

    def run_single(self, backend: LLMBackend) -> tuple[BenchmarkResult, GenerationResult]:
        from harbor.job import Job
        from harbor.models.job.config import JobConfig, OrchestratorConfig
        from harbor.models.job.result import JobResult
        from harbor.models.trial.config import AgentConfig

        port = self.config["port"]

        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", port))
        except OSError:
            raise RuntimeError(f"Port {port} is already in use")

        filtered_tasks = self._get_filtered_tasks()

        job_config = JobConfig(
            jobs_dir=Path(self.config["jobs_dir"]),
            job_name=f"tbench_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
            orchestrator=OrchestratorConfig(n_concurrent_trials=self.config["n_concurrent"]),
            agents=[
                AgentConfig(
                    name=self.config["harbor_agent"],
                    model_name="openai/local",
                    kwargs={"api_base": f"http://localhost:{port}/v1"},
                )
            ],
            tasks=filtered_tasks,
        )

        # LiteLLM requires OPENAI_API_KEY to be set even for local servers.
        os.environ.setdefault("OPENAI_API_KEY", "local")

        with backend.serve_openai(port, n_ctx=self.config["n_ctx"]):
            job = Job(job_config)
            asyncio.run(job.run())

        result_path = Path(job_config.jobs_dir) / job_config.job_name / "result.json"
        job_result = JobResult.model_validate_json(result_path.read_text())

        total = 0
        passed = 0
        for evals_stats in job_result.stats.evals.values():
            for reward_val, trial_names in evals_stats.reward_stats.get("reward", {}).items():
                total += len(trial_names)
                if math.isclose(float(reward_val), 1.0):
                    passed += len(trial_names)

        accuracy = passed / total if total > 0 else 0.0

        return (
            BenchmarkResult(
                score=accuracy,
                extra={
                    "tasks_total": total,
                    "tasks_passed": passed,
                    "difficulties": self.config["difficulties"],
                },
            ),
            GenerationResult(text="", prompt_tokens=0, gen_tokens=0, prompt_tps=0.0, gen_tps=0.0),
        )
