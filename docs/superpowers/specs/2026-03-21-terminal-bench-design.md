# Terminal Bench Integration — Design Spec

**Date:** 2026-03-21
**Status:** Approved

---

## Goal

Integrate Terminal Bench 2.0 (medium + hard difficulty tasks) into the `pareto-llm` benchmark suite as a first-class benchmark type. The LLM under test is served via an OpenAI-compatible HTTP server backed by the already-loaded `LLMBackend`, and Harbor's Python API orchestrates the agentic task evaluation.

---

## Architecture

```
runner.py
  └─ backend.load(model_id)
  └─ TerminalBenchmark.run_single(backend)
       ├─ use Harbor Python API to get medium+hard task paths from registry
       ├─ build JobConfig (AgentConfig with api_base=http://localhost:{port})
       ├─ with backend.serve_openai(port):  ← context manager, starts HTTP server
       │    └─ [polls /v1/models until healthy]
       │    ├─ job = Job(config)
       │    └─ asyncio.run(job.run())
       ├─ parse JobResult → accuracy score
       └─ return BenchmarkResult, zeroed GenerationResult
  └─ backend.unload()
```

**Key design decision:** Harbor's Python API (`from harbor.job import Job`) is used directly rather than a CLI subprocess. This avoids subprocess overhead, provides direct access to `JobResult`, and lets `AgentConfig.kwargs["api_base"]` carry the local server URL without environment variable juggling.

---

## Components

### 1. `LLMBackend.serve_openai(port: int)` — new abstract method

Added to `src/pareto_llm/backend/base.py`.

**Signature:**
```python
@abstractmethod
def serve_openai(self, port: int) -> contextlib.AbstractContextManager[None]: ...
```

**Usage:** `with backend.serve_openai(port):` — on `__enter__`, starts an OpenAI-compatible HTTP server using the already-loaded model; on `__exit__`, shuts it down.

**Contract:** `load()` must be called before `serve_openai()`. Raises `RuntimeError("Model not loaded. Call load() first.")` if called before `load()`.

**Health check:** Both implementations poll `GET http://localhost:{port}/v1/models` (0.5s interval, 30s timeout, using `urllib.request` to avoid adding an http client dependency). Raises `TimeoutError` if the server does not respond healthy within 30s.

**MLXBackend implementation:**

`mlx_lm` (≥0.20) exposes `mlx_lm.server.run_server()` — its exact signature and whether it accepts pre-loaded model objects must be verified against the installed version during implementation. Two strategies, in preference order:

1. **In-process (preferred):** If `mlx_lm.server` exposes an app factory that accepts loaded `(model, tokenizer)` objects (e.g., `create_app(model, tokenizer)`), run it via `uvicorn.Server` in a daemon thread. A `threading.Event` signals shutdown; `__exit__` sets the event and joins the thread.

2. **Subprocess fallback:** If no in-process API exists for passing pre-loaded objects: call `self.unload()` first (freeing VRAM), launch `subprocess.Popen(["python", "-m", "mlx_lm.server", "--model", self._model_id, "--port", str(port)])`. On `__exit__`, terminate the subprocess and call `self.load(self._model_id)` to restore the loaded state. The backend must store `self._model_id` during `load()` for this to work.

The implementer should check `mlx_lm.server` source at implementation time and choose accordingly.

**LlamaCppBackend implementation:**

`llama-cpp-python` ships a FastAPI app factory: `llama_cpp.server.app.create_app(llama=self._llama)`. Use the existing `Llama` instance (no second load). Run via `uvicorn.Server` in a daemon `threading.Thread`. On `__exit__`, call `server.should_exit = True` and join the thread. `uvicorn` is added to the `cuda` optional dependency group.

**`_PaddingBackend` (benchmarks/context_length.py):** Must add a delegation — `def serve_openai(self, port): return self._inner.serve_openai(port)` — because `serve_openai` is `@abstractmethod` on `LLMBackend`. Without this, `_PaddingBackend` cannot be instantiated after the abstract method is added.

**`MockBackend` (tests/conftest.py):** Must add `def serve_openai(self, port): return contextlib.nullcontext()`. Requires `import contextlib` at top of `conftest.py`.

---

### 2. `benchmarks/terminal_bench.py`

Registered as `@register("terminal_bench")` with `supports_context_padding = False`.

**Config keys:**

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `difficulties` | `list[str]` | `["medium", "hard"]` | Task difficulty levels to include |
| `port` | `int` | `8765` | Local port for the OpenAI-compatible server |
| `harbor_agent` | `str` | `"terminus-2"` | Harbor agent name |
| `sample_size` | `int \| None` | `None` | If set, randomly sample this many tasks from the filtered set |
| `seed` | `int` | `42` | Random seed for `sample_size` sampling |
| `dataset_version` | `str` | `"2.0"` | Terminal Bench dataset version |
| `jobs_dir` | `str` | `"./results/harbor"` | Directory where Harbor writes job results |
| `n_concurrent` | `int` | `4` | Number of parallel Harbor trials |

**Validation at `__init__`:**
- `difficulties` values must be a subset of `{"easy", "medium", "hard", "unknown"}`
- `port` must be 1024–65535
- `sample_size`, if set, must be > 0
- If `harbor` package is not importable, raise `RuntimeError` with: `"harbor package not found. Install with: uv tool install harbor or pip install harbor"`

**`run_single()` steps:**

**Step 1 — Enumerate and filter tasks:**

Use Harbor's Python API:

```python
from harbor.registry.client import RegistryClientFactory
from harbor.models.registry import RemoteRegistryInfo

registry = RemoteRegistryInfo()   # default Harbor registry URL
client = RegistryClientFactory.create(registry)
dataset_spec = client.get_dataset_spec("terminal-bench", self.config["dataset_version"])

# Each task in dataset_spec has a path; task.toml has metadata.difficulty
filtered_tasks = []
for task_id in dataset_spec.tasks:
    task_id = task_id.to_source_task_id()
    # task.toml is at task_id.path / "task.toml"
    config_path = task_id.path / "task.toml"
    if config_path.exists():
        import tomllib
        cfg = tomllib.loads(config_path.read_text())
        difficulty = cfg.get("metadata", {}).get("difficulty", "unknown")
        if difficulty in self.config["difficulties"]:
            filtered_tasks.append(TaskConfig(path=task_id.path))
```

If `sample_size` is set and `sample_size <= len(filtered_tasks)`, randomly sample using `random.Random(seed).sample(filtered_tasks, sample_size)`. If `sample_size > len(filtered_tasks)`, log a warning and use all filtered tasks.

If zero tasks remain after filtering, raise `ValueError(f"No tasks found with difficulties {difficulties}")`.

**Step 2 — Build `JobConfig`:**

```python
from harbor.models.job.config import JobConfig, OrchestratorConfig
from harbor.models.trial.config import AgentConfig

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
```

**Step 3 — Serve and run:**

```python
with backend.serve_openai(port):
    job = Job(job_config)
    asyncio.run(job.run())
job_result = JobResult.model_validate_json(
    (Path(job_config.jobs_dir) / job_config.job_name / "result.json").read_text()
)
```

(Harbor writes `result.json` to `{jobs_dir}/{job_name}/result.json`.)

**Step 4 — Extract accuracy:**

```python
# JobStats.evals is keyed by format_agent_evals_key(agent, model, dataset)
# reward_stats["reward"] maps reward value → list[trial_name]
total = 0
passed = 0
for evals_stats in job_result.stats.evals.values():
    for reward_val, trial_names in evals_stats.reward_stats.get("reward", {}).items():
        total += len(trial_names)
        if math.isclose(float(reward_val), 1.0):
            passed += len(trial_names)
accuracy = passed / total if total > 0 else 0.0
```

**Step 5 — Return:**

```python
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

The zeroed `GenerationResult` is intentional: Terminal Bench is agentic (multi-turn), so per-generation stats are not meaningful. The CSV columns `gen_tps`, `prompt_tps` etc. will be `0` for Terminal Bench rows — acceptable because downstream analysis can filter by benchmark type.

**`runs_per_test` note:** Harbor runs each task once per `job.run()` call. If `runs_per_test > 1` in the runner config, `run_single()` is called multiple times — each call triggers a full Harbor job. This is intentional (repeated evaluation for variance measurement) but expensive. Users should set `runs_per_test: 1` for Terminal Bench unless they need variance data. No enforcement in code; note in the config example.

**Port conflict:** Before starting the server, bind a test socket to the port and close it. If binding fails with `OSError`, raise `RuntimeError(f"Port {port} is already in use")` before entering the context manager.

---

### 3. `benchmarks/__init__.py`

Add:
```python
import pareto_llm.benchmarks.terminal_bench  # noqa: F401
```

---

### 4. `src/pareto_llm/_env.py` changes

*(Note: changes go to `_env.py`, NOT `scripts/init_env.py` which is a thin wrapper.)*

In `write_env()` or equivalent function:
- Check `shutil.which("harbor")` → print `[WARNING] harbor not found. Install with: uv tool install harbor` if missing
- Check Docker: `subprocess.run(["docker", "info"], capture_output=True, timeout=10)` → print `[WARNING] Docker daemon not running. Terminal Bench requires Docker.` if exit code non-zero
- Add `HARBOR_RESULTS_DIR=./results/harbor` to the generated `.env` content (always written, regardless of whether harbor is installed)

---

### 5. `pyproject.toml`

```toml
[project.optional-dependencies]
terminal-bench = ["harbor>=0.1"]
mlx = [
    "mlx-lm>=0.20",
    "uvicorn>=0.20",   # ← for MLXBackend.serve_openai() in-process strategy
]
cuda = [
    "llama-cpp-python>=0.2",
    "nvidia-ml-py>=12.0",
    "uvicorn>=0.20",   # ← for LlamaCppBackend.serve_openai()
]
```

`harbor` version 0.1.45 is confirmed on PyPI. `uvicorn` added to `cuda` because `serve_openai` uses it.

---

### 6. Example config entry (`configs/example.yaml`)

```yaml
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
    # Note: set runs_per_test: 1 in defaults for Terminal Bench
    # to avoid running the full Harbor job multiple times.
```

---

## Testing

**`tests/test_terminal_bench.py`:**

- Config validation: invalid difficulty value → `ValueError` at `__init__`; port out of range → `ValueError`; `sample_size=0` → `ValueError`
- `harbor` not importable → `RuntimeError` with install hint at `__init__`
- Task filtering: given a mock dataset with easy/medium/hard tasks, only medium+hard are selected
- `sample_size` within range: correct number of tasks; `sample_size > available` → uses all, emits warning
- `sample_size` sampling is deterministic given same `seed`
- `run_single()` with mocked `backend.serve_openai()` (nullcontext), mocked `Job.run()` that writes a known `result.json` → returns correct `BenchmarkResult(score=0.75, ...)`
- Score calculation: `reward_stats={"reward": {1.0: ["t1","t2","t3"], 0.0: ["t4"]}}` → score=0.75
- Port already in use → `RuntimeError` before server starts
- Zero filtered tasks → `ValueError`

**`tests/test_backend_serve.py`** (new file):

- `@pytest.mark.mlx`: `MLXBackend.serve_openai(port)` — after `load()`, context enters successfully, `GET /v1/models` returns 200 while inside context, returns non-200 or connection refused after exit
- `@pytest.mark.cuda`: same test for `LlamaCppBackend.serve_openai(port)`
- Both backends: calling `serve_openai()` before `load()` raises `RuntimeError`
- `conftest.py` `MockBackend` updated with `serve_openai()` stub returning `contextlib.nullcontext()`

---

## File Map Changes

| File | Change |
|------|--------|
| `src/pareto_llm/backend/base.py` | Add `serve_openai(port)` abstract method |
| `src/pareto_llm/backend/mlx_backend.py` | Implement `serve_openai()` |
| `src/pareto_llm/backend/llamacpp_backend.py` | Implement `serve_openai()` |
| `src/pareto_llm/benchmarks/terminal_bench.py` | New file: `@register("terminal_bench")` |
| `src/pareto_llm/benchmarks/__init__.py` | Add import for `terminal_bench` |
| `src/pareto_llm/benchmarks/context_length.py` | Add `serve_openai()` delegation to `_PaddingBackend` |
| `src/pareto_llm/_env.py` | Add harbor + Docker checks, HARBOR_RESULTS_DIR |
| `pyproject.toml` | Add `terminal-bench` extra, `uvicorn` to `cuda` extra |
| `configs/example.yaml` | Add `terminal_bench` example entry |
| `tests/test_terminal_bench.py` | New file |
| `tests/test_backend_serve.py` | New file |
| `tests/conftest.py` | Add `serve_openai` stub to `MockBackend` |

---

## Out of Scope

- Terminal Bench 3.0 (can be added by changing `dataset_version`)
- Parallel Harbor runs via cloud (Daytona, E2B, etc.)
- Custom Harbor agents beyond `terminus-2` and other named agents
- ROCm / CPU-only backends
- Aggregating token usage across agentic turns into `GenerationResult`
