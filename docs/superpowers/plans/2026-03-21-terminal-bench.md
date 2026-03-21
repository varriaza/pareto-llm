# Terminal Bench Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate Terminal Bench 2.0 (medium + hard tasks) into pareto-llm by adding a `serve_openai(port)` context manager to `LLMBackend` and implementing `TerminalBenchmark` using Harbor's Python API.

**Architecture:** `LLMBackend` gains an abstract `serve_openai(port)` method returning a context manager that starts an OpenAI-compatible HTTP server. `TerminalBenchmark.run_single()` filters tasks from Harbor's registry client, builds a `JobConfig` pointing to `localhost:{port}`, enters the serve context, runs the Harbor job via `asyncio.run(job.run())`, and parses `JobResult` for accuracy.

**Tech Stack:** Harbor 0.1.x, uvicorn, llama-cpp-python FastAPI app factory, mlx_lm.server (in-process or subprocess fallback), tomllib (stdlib), asyncio, contextlib, threading

---

### Task 1: Add `serve_openai` abstract method + update all concrete stubs

**Files:**
- Modify: `src/pareto_llm/backend/base.py`
- Modify: `tests/conftest.py`
- Modify: `src/pareto_llm/benchmarks/context_length.py`

- [ ] **Step 1: Run existing tests to establish baseline**

  Run: `uv run pytest tests/ -q`
  Expected: All pass.

- [ ] **Step 2: Add `serve_openai` abstract method to `LLMBackend`**

  Add `import contextlib` at top of `src/pareto_llm/backend/base.py`. Add to the class:

  ```python
  @abstractmethod
  def serve_openai(self, port: int) -> contextlib.AbstractContextManager[None]: ...
  ```

- [ ] **Step 3: Run tests — expect failure (MockBackend no longer concrete)**

  Run: `uv run pytest tests/ -q`
  Expected: `TypeError` — `MockBackend` can't be instantiated (missing abstract method).

- [ ] **Step 4: Update MockBackend in `tests/conftest.py`**

  Add `import contextlib` at top. Add to `MockBackend`:

  ```python
  def serve_openai(self, port: int):
      return contextlib.nullcontext()
  ```

- [ ] **Step 5: Add delegation to `_PaddingBackend` in `context_length.py`**

  Add to `_PaddingBackend` (after `max_context_tokens`):

  ```python
  def serve_openai(self, port: int):
      return self._inner.serve_openai(port)
  ```

- [ ] **Step 6: Run tests — all should pass**

  Run: `uv run pytest tests/ -q`
  Expected: All pass.

- [ ] **Step 7: Commit**

  ```bash
  git add src/pareto_llm/backend/base.py tests/conftest.py src/pareto_llm/benchmarks/context_length.py
  git commit -m "feat: add serve_openai abstract method to LLMBackend"
  ```

---

### Task 2: TerminalBenchmark — config validation

**Files:**
- Create: `tests/test_terminal_bench.py`
- Create: `src/pareto_llm/benchmarks/terminal_bench.py`

- [ ] **Step 1: Create `tests/test_terminal_bench.py` with config validation tests**

  ```python
  import importlib.util

  import pytest

  from pareto_llm.benchmarks.terminal_bench import TerminalBenchmark


  def _valid_config(**overrides):
      cfg = {
          "difficulties": ["medium", "hard"],
          "port": 8765,
          "harbor_agent": "terminus-2",
          "sample_size": None,
          "seed": 42,
          "dataset_version": "2.0",
          "jobs_dir": "./results/harbor",
          "n_concurrent": 4,
      }
      cfg.update(overrides)
      return cfg


  def test_invalid_difficulty_raises():
      with pytest.raises(ValueError, match="difficulties"):
          TerminalBenchmark(_valid_config(difficulties=["hard", "extreme"]))


  def test_port_too_low_raises():
      with pytest.raises(ValueError, match="port"):
          TerminalBenchmark(_valid_config(port=80))


  def test_port_too_high_raises():
      with pytest.raises(ValueError, match="port"):
          TerminalBenchmark(_valid_config(port=99999))


  def test_sample_size_zero_raises():
      with pytest.raises(ValueError, match="sample_size"):
          TerminalBenchmark(_valid_config(sample_size=0))


  def test_harbor_not_importable_raises(monkeypatch):
      import importlib.util as _ilu
      orig = _ilu.find_spec
      monkeypatch.setattr(_ilu, "find_spec", lambda name: None if name == "harbor" else orig(name))
      with pytest.raises(RuntimeError, match="harbor package not found"):
          TerminalBenchmark(_valid_config())


  def test_valid_config_creates_instance():
      bench = TerminalBenchmark(_valid_config())
      assert bench.config["difficulties"] == ["medium", "hard"]
      assert bench.config["port"] == 8765


  def test_supports_context_padding_is_false():
      assert TerminalBenchmark.supports_context_padding is False
  ```

- [ ] **Step 2: Run to verify ImportError (module doesn't exist yet)**

  Run: `uv run pytest tests/test_terminal_bench.py -q`
  Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Create `src/pareto_llm/benchmarks/terminal_bench.py` with `__init__` validation**

  ```python
  """Terminal Bench 2.0 integration benchmark."""

  from __future__ import annotations

  import asyncio
  import importlib.util
  import logging
  import math
  import random
  import socket
  import tomllib
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
      }

      def __init__(self, config: dict) -> None:
          merged = {**self.DEFAULTS, **config}
          super().__init__(merged)

          if importlib.util.find_spec("harbor") is None:
              raise RuntimeError(
                  "harbor package not found. Install with: uv tool install harbor or pip install harbor"
              )

          invalid = set(self.config["difficulties"]) - self.VALID_DIFFICULTIES
          if invalid:
              raise ValueError(
                  f"difficulties contains invalid values: {invalid}. "
                  f"Must be subset of {self.VALID_DIFFICULTIES}"
              )

          port = self.config["port"]
          if not (1024 <= port <= 65535):
              raise ValueError(f"port must be between 1024 and 65535, got {port}")

          sample_size = self.config["sample_size"]
          if sample_size is not None and sample_size <= 0:
              raise ValueError(f"sample_size must be > 0, got {sample_size}")

      def _get_filtered_tasks(self) -> list:
          raise NotImplementedError

      def run_single(self, backend: LLMBackend) -> tuple[BenchmarkResult, GenerationResult]:
          raise NotImplementedError
  ```

- [ ] **Step 4: Run config validation tests**

  Run: `uv run pytest tests/test_terminal_bench.py -v`
  Expected: All 7 tests pass.

- [ ] **Step 5: Commit**

  ```bash
  git add tests/test_terminal_bench.py src/pareto_llm/benchmarks/terminal_bench.py
  git commit -m "feat: add TerminalBenchmark skeleton with config validation"
  ```

---

### Task 3: TerminalBenchmark — task filtering

**Files:**
- Modify: `tests/test_terminal_bench.py`
- Modify: `src/pareto_llm/benchmarks/terminal_bench.py`

- [ ] **Step 1: Append task filtering tests to `tests/test_terminal_bench.py`**

  ```python
  import pathlib
  from unittest.mock import MagicMock, patch


  def _make_task_ref(path: pathlib.Path):
      """Build a mock task reference whose to_source_task_id().path is `path`."""
      source = MagicMock()
      source.path = path
      ref = MagicMock()
      ref.to_source_task_id.return_value = source
      return ref


  def _write_task_toml(tmp_path: pathlib.Path, name: str, difficulty: str) -> pathlib.Path:
      task_dir = tmp_path / name
      task_dir.mkdir()
      (task_dir / "task.toml").write_text(f'[metadata]\ndifficulty = "{difficulty}"\n')
      return task_dir


  def _mock_registry(tmp_path, dirs):
      mock_dataset_spec = MagicMock()
      mock_dataset_spec.tasks = [_make_task_ref(d) for d in dirs]
      mock_client = MagicMock()
      mock_client.get_dataset_spec.return_value = mock_dataset_spec
      return mock_client


  def test_filtering_selects_correct_difficulties(tmp_path):
      easy_dir = _write_task_toml(tmp_path, "easy", "easy")
      medium_dir = _write_task_toml(tmp_path, "medium", "medium")
      hard_dir = _write_task_toml(tmp_path, "hard", "hard")
      mock_client = _mock_registry(tmp_path, [easy_dir, medium_dir, hard_dir])

      with patch("harbor.registry.client.RegistryClientFactory") as mock_factory, \
           patch("harbor.models.registry.RemoteRegistryInfo"):
          mock_factory.create.return_value = mock_client
          bench = TerminalBenchmark(_valid_config(difficulties=["medium", "hard"]))
          tasks = bench._get_filtered_tasks()

      assert len(tasks) == 2
      paths = {t.path for t in tasks}
      assert medium_dir in paths
      assert hard_dir in paths
      assert easy_dir not in paths


  def test_sample_size_limits_tasks(tmp_path):
      dirs = [_write_task_toml(tmp_path, f"task_{i}", "hard") for i in range(10)]
      mock_client = _mock_registry(tmp_path, dirs)

      with patch("harbor.registry.client.RegistryClientFactory") as mock_factory, \
           patch("harbor.models.registry.RemoteRegistryInfo"):
          mock_factory.create.return_value = mock_client
          bench = TerminalBenchmark(_valid_config(difficulties=["hard"], sample_size=5, seed=42))
          tasks = bench._get_filtered_tasks()

      assert len(tasks) == 5


  def test_sample_size_is_deterministic(tmp_path):
      dirs = [_write_task_toml(tmp_path, f"task_{i}", "hard") for i in range(10)]

      def run():
          mock_client = _mock_registry(tmp_path, dirs)
          with patch("harbor.registry.client.RegistryClientFactory") as mock_factory, \
               patch("harbor.models.registry.RemoteRegistryInfo"):
              mock_factory.create.return_value = mock_client
              bench = TerminalBenchmark(_valid_config(difficulties=["hard"], sample_size=5, seed=42))
              return bench._get_filtered_tasks()

      assert [t.path for t in run()] == [t.path for t in run()]


  def test_sample_size_exceeds_available_uses_all(tmp_path, caplog):
      import logging
      dirs = [_write_task_toml(tmp_path, f"task_{i}", "hard") for i in range(3)]
      mock_client = _mock_registry(tmp_path, dirs)

      with patch("harbor.registry.client.RegistryClientFactory") as mock_factory, \
           patch("harbor.models.registry.RemoteRegistryInfo"), \
           caplog.at_level(logging.WARNING):
          mock_factory.create.return_value = mock_client
          bench = TerminalBenchmark(_valid_config(difficulties=["hard"], sample_size=10))
          tasks = bench._get_filtered_tasks()

      assert len(tasks) == 3
      assert caplog.records, "Expected a warning log"


  def test_zero_filtered_tasks_raises(tmp_path):
      easy_dir = _write_task_toml(tmp_path, "easy", "easy")
      mock_client = _mock_registry(tmp_path, [easy_dir])

      with patch("harbor.registry.client.RegistryClientFactory") as mock_factory, \
           patch("harbor.models.registry.RemoteRegistryInfo"):
          mock_factory.create.return_value = mock_client
          bench = TerminalBenchmark(_valid_config(difficulties=["hard"]))
          with pytest.raises(ValueError, match="No tasks found"):
              bench._get_filtered_tasks()
  ```

- [ ] **Step 2: Run to verify failure**

  Run: `uv run pytest tests/test_terminal_bench.py -k "filter or sample or zero" -q`
  Expected: `NotImplementedError`.

- [ ] **Step 3: Implement `_get_filtered_tasks` in `terminal_bench.py`**

  Replace the `_get_filtered_tasks` stub:

  ```python
  def _get_filtered_tasks(self) -> list:
      from harbor.models.registry import RemoteRegistryInfo
      from harbor.models.trial.config import TaskConfig
      from harbor.registry.client import RegistryClientFactory

      registry = RemoteRegistryInfo()
      client = RegistryClientFactory.create(registry)
      dataset_spec = client.get_dataset_spec("terminal-bench", self.config["dataset_version"])

      filtered: list[TaskConfig] = []
      for task_ref in dataset_spec.tasks:
          task_id = task_ref.to_source_task_id()
          config_path = task_id.path / "task.toml"
          if config_path.exists():
              cfg = tomllib.loads(config_path.read_text())
              difficulty = cfg.get("metadata", {}).get("difficulty", "unknown")
              if difficulty in self.config["difficulties"]:
                  filtered.append(TaskConfig(path=task_id.path))

      sample_size = self.config["sample_size"]
      if sample_size is not None:
          if sample_size > len(filtered):
              _logger.warning(
                  "sample_size=%d exceeds available tasks (%d); using all",
                  sample_size, len(filtered),
              )
          else:
              filtered = random.Random(self.config["seed"]).sample(filtered, sample_size)

      if not filtered:
          raise ValueError(f"No tasks found with difficulties {self.config['difficulties']}")

      return filtered
  ```

- [ ] **Step 4: Run all terminal bench tests so far**

  Run: `uv run pytest tests/test_terminal_bench.py -v`
  Expected: All pass (run_single tests not yet written).

- [ ] **Step 5: Commit**

  ```bash
  git add tests/test_terminal_bench.py src/pareto_llm/benchmarks/terminal_bench.py
  git commit -m "feat: implement TerminalBenchmark task filtering"
  ```

---

### Task 4: TerminalBenchmark — `run_single`

**Files:**
- Modify: `tests/test_terminal_bench.py`
- Modify: `src/pareto_llm/benchmarks/terminal_bench.py`

- [ ] **Step 1: Append `run_single` tests to `tests/test_terminal_bench.py`**

  ```python
  import contextlib
  import json
  import math
  import socket
  from unittest.mock import AsyncMock, MagicMock, patch


  def _make_mock_backend():
      backend = MagicMock()
      backend.serve_openai.return_value = contextlib.nullcontext()
      return backend


  def _setup_run_single_mocks(tmp_path, reward_stats):
      """
      Returns (bench, backend, patch context) ready for run_single.
      Writes a placeholder result.json when Job() is called.
      """
      bench = TerminalBenchmark(_valid_config(jobs_dir=str(tmp_path)))
      backend = _make_mock_backend()
      pre_filtered = [MagicMock()]

      mock_job_instance = MagicMock()
      mock_job_instance.run = AsyncMock()

      mock_evals = MagicMock()
      mock_evals.reward_stats = reward_stats
      mock_job_result = MagicMock()
      mock_job_result.stats.evals = {"key": mock_evals}

      def make_job(config):
          job_dir = tmp_path / config.job_name
          job_dir.mkdir(parents=True, exist_ok=True)
          (job_dir / "result.json").write_text("{}")
          return mock_job_instance

      return bench, backend, pre_filtered, make_job, mock_job_result


  def test_run_single_score_three_of_four(tmp_path):
      bench, backend, pre_filtered, make_job, mock_job_result = _setup_run_single_mocks(
          tmp_path, {"reward": {1.0: ["t1", "t2", "t3"], 0.0: ["t4"]}}
      )
      with patch.object(bench, "_get_filtered_tasks", return_value=pre_filtered), \
           patch("harbor.job.Job", side_effect=make_job), \
           patch("harbor.models.job.result.JobResult") as mock_jr:
          mock_jr.model_validate_json.return_value = mock_job_result
          result, gen = bench.run_single(backend)

      assert math.isclose(result.score, 0.75)
      assert result.extra["tasks_total"] == 4
      assert result.extra["tasks_passed"] == 3
      assert gen.text == ""
      assert gen.prompt_tokens == 0
      assert gen.gen_tokens == 0


  def test_run_single_score_all_pass(tmp_path):
      bench, backend, pre_filtered, make_job, mock_job_result = _setup_run_single_mocks(
          tmp_path, {"reward": {1.0: ["t1", "t2"]}}
      )
      with patch.object(bench, "_get_filtered_tasks", return_value=pre_filtered), \
           patch("harbor.job.Job", side_effect=make_job), \
           patch("harbor.models.job.result.JobResult") as mock_jr:
          mock_jr.model_validate_json.return_value = mock_job_result
          result, _ = bench.run_single(backend)

      assert math.isclose(result.score, 1.0)


  def test_run_single_score_none_pass(tmp_path):
      bench, backend, pre_filtered, make_job, mock_job_result = _setup_run_single_mocks(
          tmp_path, {"reward": {0.0: ["t1", "t2"]}}
      )
      with patch.object(bench, "_get_filtered_tasks", return_value=pre_filtered), \
           patch("harbor.job.Job", side_effect=make_job), \
           patch("harbor.models.job.result.JobResult") as mock_jr:
          mock_jr.model_validate_json.return_value = mock_job_result
          result, _ = bench.run_single(backend)

      assert math.isclose(result.score, 0.0)


  def test_port_in_use_raises(tmp_path):
      bench = TerminalBenchmark(_valid_config(jobs_dir=str(tmp_path), port=18999))
      with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
          s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
          s.bind(("127.0.0.1", 18999))
          with patch.object(bench, "_get_filtered_tasks", return_value=[MagicMock()]):
              with pytest.raises(RuntimeError, match="already in use"):
                  bench.run_single(_make_mock_backend())
  ```

- [ ] **Step 2: Run to verify failure**

  Run: `uv run pytest tests/test_terminal_bench.py -k "run_single or port_in_use" -q`
  Expected: `NotImplementedError`.

- [ ] **Step 3: Implement `run_single` in `terminal_bench.py`**

  Replace the `run_single` stub:

  ```python
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
          agents=[AgentConfig(
              name=self.config["harbor_agent"],
              model_name="openai/local",
              kwargs={"api_base": f"http://localhost:{port}"},
          )],
          tasks=filtered_tasks,
      )

      with backend.serve_openai(port):
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
  ```

- [ ] **Step 4: Run all terminal bench tests**

  Run: `uv run pytest tests/test_terminal_bench.py -v`
  Expected: All pass.

- [ ] **Step 5: Commit**

  ```bash
  git add tests/test_terminal_bench.py src/pareto_llm/benchmarks/terminal_bench.py
  git commit -m "feat: implement TerminalBenchmark.run_single"
  ```

---

### Task 5: Register terminal_bench + verify

**Files:**
- Modify: `src/pareto_llm/benchmarks/__init__.py`
- Modify: `tests/test_terminal_bench.py`

- [ ] **Step 1: Write registration test**

  Append to `tests/test_terminal_bench.py`:

  ```python
  def test_terminal_bench_is_registered():
      from pareto_llm.benchmarks.base import BENCHMARK_REGISTRY
      assert "terminal_bench" in BENCHMARK_REGISTRY
  ```

- [ ] **Step 2: Run to verify failure**

  Run: `uv run pytest tests/test_terminal_bench.py::test_terminal_bench_is_registered -v`
  Expected: `AssertionError` — key not found.

- [ ] **Step 3: Add import to `benchmarks/__init__.py`**

  Append to `src/pareto_llm/benchmarks/__init__.py`:

  ```python
  from pareto_llm.benchmarks import terminal_bench as _terminal_bench  # noqa: F401
  ```

- [ ] **Step 4: Run all tests**

  Run: `uv run pytest tests/ -q`
  Expected: All pass.

- [ ] **Step 5: Commit**

  ```bash
  git add src/pareto_llm/benchmarks/__init__.py tests/test_terminal_bench.py
  git commit -m "feat: register terminal_bench in benchmark registry"
  ```

---

### Task 6: LlamaCppBackend.serve_openai()

**Files:**
- Create: `tests/test_backend_serve.py`
- Modify: `src/pareto_llm/backend/llamacpp_backend.py`

- [ ] **Step 1: Create `tests/test_backend_serve.py` with unit + integration tests**

  ```python
  """Tests for LLMBackend.serve_openai().

  Unit tests (no hardware) run always.
  Integration tests require hardware and are marked:
    @pytest.mark.cuda  — NVIDIA GPU + llama-cpp-python
    @pytest.mark.mlx   — Apple Silicon + mlx-lm
  """

  import time
  import urllib.request

  import pytest


  def _v1_models_reachable(port: int) -> bool:
      try:
          urllib.request.urlopen(f"http://localhost:{port}/v1/models", timeout=2)
          return True
      except Exception:
          return False


  # ─── Unit tests ───────────────────────────────────────────────────────────────


  def test_llamacpp_serve_before_load_raises():
      from pareto_llm.backend.llamacpp_backend import LlamaCppBackend
      backend = LlamaCppBackend()
      with pytest.raises(RuntimeError, match="Model not loaded"):
          with backend.serve_openai(19876):
              pass


  def test_mlx_serve_before_load_raises():
      from pareto_llm.backend.mlx_backend import MLXBackend
      backend = MLXBackend()
      with pytest.raises(RuntimeError, match="Model not loaded"):
          with backend.serve_openai(19877):
              pass


  # ─── Integration tests ────────────────────────────────────────────────────────


  @pytest.mark.cuda
  def test_llamacpp_serve_exposes_v1_models():
      from pareto_llm.backend.llamacpp_backend import LlamaCppBackend
      backend = LlamaCppBackend()
      backend.load("bartowski/Llama-3.2-1B-Instruct-GGUF:Q4_K_M")
      try:
          port = 18765
          with backend.serve_openai(port):
              assert _v1_models_reachable(port), "/v1/models not reachable during context"
          time.sleep(0.5)
          assert not _v1_models_reachable(port), "/v1/models still reachable after context exit"
      finally:
          backend.unload()


  @pytest.mark.mlx
  def test_mlx_serve_exposes_v1_models():
      from pareto_llm.backend.mlx_backend import MLXBackend
      backend = MLXBackend()
      backend.load("mlx-community/Llama-3.2-1B-Instruct-4bit")
      try:
          port = 18766
          with backend.serve_openai(port):
              assert _v1_models_reachable(port), "/v1/models not reachable during context"
          time.sleep(0.5)
          assert not _v1_models_reachable(port), "/v1/models still reachable after context exit"
      finally:
          backend.unload()
  ```

- [ ] **Step 2: Run unit tests — expect failure (method not implemented)**

  Run: `uv run pytest tests/test_backend_serve.py::test_llamacpp_serve_before_load_raises -v`
  Expected: Error — `serve_openai` raises `NotImplementedError` (abstract method body is `...`).

  Actually it will raise `RuntimeError` from `...` — let's verify first: `uv run pytest tests/test_backend_serve.py -k "not cuda and not mlx" -v`

- [ ] **Step 3: Implement `LlamaCppBackend.serve_openai()`**

  Add imports at the top of `src/pareto_llm/backend/llamacpp_backend.py` (after existing):

  ```python
  import contextlib
  import threading
  import time
  import urllib.request
  ```

  Add method to `LlamaCppBackend`:

  ```python
  @contextlib.contextmanager
  def serve_openai(self, port: int):
      if self._llama is None:
          raise RuntimeError("Model not loaded. Call load() first.")

      from llama_cpp.server.app import create_app  # type: ignore[import]
      import uvicorn  # type: ignore[import]

      app = create_app(llama=self._llama)
      config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
      server = uvicorn.Server(config)
      thread = threading.Thread(target=server.run, daemon=True)
      thread.start()

      deadline = time.time() + 30
      while time.time() < deadline:
          try:
              urllib.request.urlopen(f"http://localhost:{port}/v1/models", timeout=1)
              break
          except Exception:
              time.sleep(0.5)
      else:
          server.should_exit = True
          thread.join(timeout=5)
          raise TimeoutError(f"Server on port {port} did not become healthy within 30s")

      try:
          yield
      finally:
          server.should_exit = True
          thread.join(timeout=10)
  ```

- [ ] **Step 4: Run unit tests — should pass**

  Run: `uv run pytest tests/test_backend_serve.py::test_llamacpp_serve_before_load_raises -v`
  Expected: PASS.

- [ ] **Step 5: Commit**

  ```bash
  git add tests/test_backend_serve.py src/pareto_llm/backend/llamacpp_backend.py
  git commit -m "feat: implement LlamaCppBackend.serve_openai()"
  ```

---

### Task 7: MLXBackend.serve_openai()

**Files:**
- Modify: `src/pareto_llm/backend/mlx_backend.py`

- [ ] **Step 1: Inspect mlx_lm.server to choose strategy**

  Run: `python -c "import mlx_lm.server; print(dir(mlx_lm.server))" 2>&1`

  Look for `create_app`. If present → in-process strategy. If absent → subprocess fallback.

- [ ] **Step 2a: (if `create_app` exists) Implement in-process strategy**

  Add imports to `src/pareto_llm/backend/mlx_backend.py`:

  ```python
  import contextlib
  import threading
  import time
  import urllib.request
  ```

  Add to `MLXBackend.__init__`: `self._model_id: str | None = None`
  Add to `MLXBackend.load()`: `self._model_id = model_id`

  Add method:

  ```python
  @contextlib.contextmanager
  def serve_openai(self, port: int):
      if self._model is None:
          raise RuntimeError("Model not loaded. Call load() first.")

      from mlx_lm.server import create_app  # type: ignore[import]
      import uvicorn  # type: ignore[import]

      app = create_app(self._model, self._tokenizer)
      config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
      server = uvicorn.Server(config)
      thread = threading.Thread(target=server.run, daemon=True)
      thread.start()

      deadline = time.time() + 30
      while time.time() < deadline:
          try:
              urllib.request.urlopen(f"http://localhost:{port}/v1/models", timeout=1)
              break
          except Exception:
              time.sleep(0.5)
      else:
          server.should_exit = True
          thread.join(timeout=5)
          raise TimeoutError(f"Server on port {port} did not become healthy within 30s")

      try:
          yield
      finally:
          server.should_exit = True
          thread.join(timeout=10)
  ```

- [ ] **Step 2b: (if `create_app` does NOT exist) Implement subprocess fallback**

  Add imports to `src/pareto_llm/backend/mlx_backend.py`:

  ```python
  import contextlib
  import subprocess
  import time
  import urllib.request
  ```

  Add to `MLXBackend.__init__`: `self._model_id: str | None = None`
  Add to `MLXBackend.load()`: `self._model_id = model_id`

  Add method:

  ```python
  @contextlib.contextmanager
  def serve_openai(self, port: int):
      if self._model is None:
          raise RuntimeError("Model not loaded. Call load() first.")

      self.unload()
      proc = subprocess.Popen(
          ["python", "-m", "mlx_lm.server", "--model", self._model_id, "--port", str(port)],
      )

      deadline = time.time() + 30
      while time.time() < deadline:
          try:
              urllib.request.urlopen(f"http://localhost:{port}/v1/models", timeout=1)
              break
          except Exception:
              time.sleep(0.5)
      else:
          proc.terminate()
          raise TimeoutError(f"Server on port {port} did not become healthy within 30s")

      try:
          yield
      finally:
          proc.terminate()
          proc.wait(timeout=10)
          self.load(self._model_id)
  ```

- [ ] **Step 3: Run MLX unit test — should pass**

  Run: `uv run pytest tests/test_backend_serve.py::test_mlx_serve_before_load_raises -v`
  Expected: PASS.

- [ ] **Step 4: Run all non-hardware backend serve tests**

  Run: `uv run pytest tests/test_backend_serve.py -k "not cuda and not mlx" -v`
  Expected: Both unit tests pass.

- [ ] **Step 5: Commit**

  ```bash
  git add src/pareto_llm/backend/mlx_backend.py
  git commit -m "feat: implement MLXBackend.serve_openai()"
  ```

---

### Task 8: `_env.py` — harbor + Docker checks + HARBOR_RESULTS_DIR

**Files:**
- Modify: `src/pareto_llm/_env.py`
- Modify: `tests/test_env.py`

- [ ] **Step 1: Write failing env tests**

  Append to `tests/test_env.py`:

  ```python
  import shutil
  from unittest.mock import MagicMock, patch


  def test_write_env_includes_harbor_results_dir(tmp_path):
      env_path = tmp_path / ".env"
      with patch.object(env_mod.shutil, "which", return_value="/usr/bin/harbor"), \
           patch.object(env_mod.subprocess, "run", return_value=MagicMock(returncode=0)):
          env_mod.write_env(env_path, "mlx")
      assert "HARBOR_RESULTS_DIR=./results/harbor" in env_path.read_text()


  def test_write_env_warns_harbor_missing(tmp_path, capsys):
      env_path = tmp_path / ".env"
      with patch.object(env_mod.shutil, "which", return_value=None), \
           patch.object(env_mod.subprocess, "run", return_value=MagicMock(returncode=0)):
          env_mod.write_env(env_path, "mlx")
      out = capsys.readouterr().out
      assert "[WARNING]" in out
      assert "harbor" in out.lower()


  def test_write_env_warns_docker_not_running(tmp_path, capsys):
      env_path = tmp_path / ".env"
      with patch.object(env_mod.shutil, "which", return_value="/usr/bin/harbor"), \
           patch.object(env_mod.subprocess, "run", return_value=MagicMock(returncode=1)):
          env_mod.write_env(env_path, "mlx")
      out = capsys.readouterr().out
      assert "[WARNING]" in out
      assert "docker" in out.lower()
  ```

- [ ] **Step 2: Run to verify failure**

  Run: `uv run pytest tests/test_env.py::test_write_env_includes_harbor_results_dir -v`
  Expected: `AssertionError` — key not in output.

- [ ] **Step 3: Update `write_env` in `src/pareto_llm/_env.py`**

  Replace the `write_env` function body:

  ```python
  def write_env(path: pathlib.Path, gpu_backend: str) -> None:
      if not shutil.which("harbor"):
          print("[WARNING] harbor not found. Install with: uv tool install harbor")

      try:
          result = subprocess.run(["docker", "info"], capture_output=True, timeout=10)
          if result.returncode != 0:
              print("[WARNING] Docker daemon not running. Terminal Bench requires Docker.")
      except (FileNotFoundError, subprocess.TimeoutExpired):
          print("[WARNING] Docker daemon not running. Terminal Bench requires Docker.")

      content = (
          "# Auto-generated — do not commit\n"
          f"GPU_BACKEND={gpu_backend}\n"
          "RESULTS_DIR=./results\n"
          "HARBOR_RESULTS_DIR=./results/harbor\n"
          "KEEP_MODEL_FILES=false\n"
      )
      path.write_text(content)
      print(f"Wrote {path}  (GPU_BACKEND={gpu_backend})")
  ```

- [ ] **Step 4: Run env tests**

  Run: `uv run pytest tests/test_env.py -v`
  Expected: All pass.

- [ ] **Step 5: Commit**

  ```bash
  git add src/pareto_llm/_env.py tests/test_env.py
  git commit -m "feat: add harbor/docker checks and HARBOR_RESULTS_DIR to write_env"
  ```

---

### Task 9: pyproject.toml deps + configs/example.yaml

**Files:**
- Modify: `pyproject.toml`
- Modify: `configs/example.yaml`

- [ ] **Step 1: Update `pyproject.toml` optional dependencies**

  Replace the `[project.optional-dependencies]` section:

  ```toml
  [project.optional-dependencies]
  terminal-bench = ["harbor>=0.1"]
  mlx = [
      "mlx-lm>=0.20",
      "uvicorn>=0.20",
  ]
  cuda = [
      "llama-cpp-python>=0.2",
      "nvidia-ml-py>=12.0",
      "uvicorn>=0.20",
  ]
  dev = [
      "pytest>=8.0",
      "pytest-cov>=5.0",
      "ruff>=0.4",
      "mypy>=1.10",
  ]
  ```

- [ ] **Step 2: Add terminal_bench example to `configs/example.yaml`**

  Append after the existing `context_length` entry:

  ```yaml
    # Terminal Bench 2.0 — agentic tasks (medium + hard difficulty)
    # Requires: pip install harbor[terminal-bench]  AND  Docker running
    # Note: set runs_per_test: 1 — each call triggers a full Harbor job
    - type: terminal_bench
      name: tbench_medium_hard
      config:
        difficulties: ["medium", "hard"]
        port: 8765
        harbor_agent: terminus-2
        sample_size: 20        # optional: random 20-task subset
        seed: 42               # seed for sample_size reproducibility
        dataset_version: "2.0"
        jobs_dir: ./results/harbor
        n_concurrent: 4
  ```

- [ ] **Step 3: Run full test suite**

  Run: `uv run pytest tests/ -q -k "not cuda and not mlx"`
  Expected: All pass.

- [ ] **Step 4: Commit**

  ```bash
  git add pyproject.toml configs/example.yaml
  git commit -m "feat: add terminal-bench dep extra, uvicorn to mlx/cuda, and example config"
  ```

---

### Final check

- [ ] **Run full non-hardware test suite one last time**

  Run: `uv run pytest tests/ -q -k "not cuda and not mlx"`
  Expected: All pass, no errors.

- [ ] **Verify linting**

  Run: `uv run ruff check src/ tests/`
  Expected: No errors.
