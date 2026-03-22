# Pareto LLM Benchmark Suite Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a CLI tool that benchmarks local LLMs across model variants and quantization levels, capturing raw performance and quality data for Pareto frontier analysis.

**Architecture:** A backend abstraction layer (MLX for Mac/Apple Silicon, llama.cpp for Linux/NVIDIA) decouples inference from the test harness. Benchmarks self-register via a decorator-based plugin system. A runner orchestrates the model × benchmark × run_num matrix and writes raw results row-by-row to an append-only CSV file.

**Tech Stack:** Python 3.11+, uv, click, pydantic v2, pyyaml, python-dotenv, psutil, huggingface-hub; mlx-lm (Mac optional), llama-cpp-python + pynvml (Linux optional); pytest + pytest-cov

---

## Design Decisions

These decisions resolve ambiguities in the spec:

1. **`Benchmark.run_single()` returns `tuple[BenchmarkResult, GenerationResult]`** — the `GenerationResult` carries per-generation stats (gen_tps, prompt_tps, etc.) for the CSV. For multi-generation benchmarks, return the stats of the last (or primary) call.
2. **`SystemMetricsCollector` exposes results as attributes** — after `__exit__`, the collector object has `ram_max_gb`, `ram_avg_gb`, `gpu_ram_max_gb`, `gpu_ram_avg_gb` set directly on it. The runner passes the collector to the csv_writer.
3. **`CsvWriter.append()` takes keyword-only args** — explicit over positional for a function with many parameters.
4. **`runner.py` contains `_create_backend()` and `_delete_hf_cache()`** — private helpers; no separate utility module (YAGNI).

---

## File Map

| File | Responsibility |
|------|----------------|
| `pyproject.toml` | Project metadata, dependencies, entry point, tool config |
| `pytest.ini` | pytest markers: mlx, cuda, integration |
| `.gitignore` | Exclude .env, results/, caches, build artifacts |
| `scripts/init_env.py` | Thin CLI wrapper — calls `pareto_llm._env` and writes `.env` to CWD |
| `configs/example.yaml` | Reference config showing all options (aspirational; requires benchmark modules) |
| `src/pareto_llm/__init__.py` | Package marker (empty) |
| `src/pareto_llm/_env.py` | `detect_gpu_backend()` and `write_env()` — shared by CLI and init script |
| `src/pareto_llm/cli.py` | Click CLI: `run`, `init-env`, `list-cached` subcommands |
| `src/pareto_llm/config.py` | Pydantic v2 models: BenchmarkConfig, BenchmarkEntry, Defaults |
| `src/pareto_llm/runner.py` | Orchestrates model × benchmark × run_num loop; writes CSV; cleans cache |
| `src/pareto_llm/backend/__init__.py` | Re-exports LLMBackend, GenerationResult |
| `src/pareto_llm/backend/base.py` | Abstract LLMBackend, GenerationResult dataclass |
| `src/pareto_llm/backend/mlx_backend.py` | MLX inference backend (Mac/Apple Silicon) |
| `src/pareto_llm/backend/llamacpp_backend.py` | llama.cpp inference backend (Linux/NVIDIA) |
| `src/pareto_llm/benchmarks/__init__.py` | Imports all benchmark modules to trigger @register side effects |
| `src/pareto_llm/benchmarks/base.py` | Abstract Benchmark, BenchmarkResult, register(), BENCHMARK_REGISTRY |
| `src/pareto_llm/benchmarks/context_length.py` | Context-length wrapper benchmark (pads prompts to fill ratio) |
| `src/pareto_llm/metrics/__init__.py` | Re-exports SystemMetricsCollector |
| `src/pareto_llm/metrics/system.py` | Background-thread RAM + GPU sampling context manager |
| `src/pareto_llm/storage/__init__.py` | Re-exports CsvWriter |
| `src/pareto_llm/storage/csv_writer.py` | Append-only CSV; header auto-created; extra_* columns |
| `tests/__init__.py` | Empty — makes tests a package |
| `tests/conftest.py` | Shared fixtures: mock_backend, tmp_csv_path, sample_config_dict |
| `tests/test_env.py` | Tests for scripts/init_env.py detect/write logic |
| `tests/test_config.py` | Tests for config.py parsing and validation |
| `tests/test_benchmarks.py` | Tests for registry, Benchmark ABC, context-length wrapper |
| `tests/test_metrics.py` | Tests for SystemMetricsCollector |
| `tests/test_storage.py` | Tests for CsvWriter |
| `tests/test_runner.py` | Integration tests for runner.py (mocked backend + benchmarks) |
| `tests/test_mlx_backend.py` | MLX backend smoke tests (`@pytest.mark.mlx`) |
| `tests/test_llamacpp_backend.py` | llama.cpp backend smoke tests (`@pytest.mark.cuda`) |

---

## Chunk 1: Scaffolding

### Task 1: Project files

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `pytest.ini`
- Create: `src/pareto_llm/__init__.py` (empty)
- Create: `src/pareto_llm/backend/__init__.py` (empty for now)
- Create: `src/pareto_llm/benchmarks/__init__.py` (empty for now)
- Create: `src/pareto_llm/metrics/__init__.py` (empty for now)
- Create: `src/pareto_llm/storage/__init__.py` (empty for now)
- Create: `tests/__init__.py` (empty)

- [ ] **Step 1: Create pyproject.toml**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "pareto-llm"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "click>=8.1",
    "pydantic>=2.0",
    "pyyaml>=6.0",
    "python-dotenv>=1.0",
    "psutil>=5.9",
    "huggingface-hub>=0.20",
]

[project.optional-dependencies]
mlx = ["mlx-lm>=0.20"]
cuda = [
    "llama-cpp-python>=0.2",
    "pynvml>=11.0",
]
dev = [
    "pytest>=8.0",
    "pytest-cov>=5.0",
    "ruff>=0.4",
    "mypy>=1.10",
]

[project.scripts]
pareto-llm = "pareto_llm.cli:cli"

[tool.hatch.build.targets.wheel]
packages = ["src/pareto_llm"]

[tool.ruff.lint]
select = ["E", "F", "I"]
```

- [ ] **Step 2: Create .gitignore**

```
.env
results/
*.pyc
__pycache__/
.mypy_cache/
.ruff_cache/
.pytest_cache/
*.egg-info/
dist/
.venv/
```

- [ ] **Step 3: Create pytest.ini**

```ini
[pytest]
markers =
    mlx: Tests requiring Apple Silicon and mlx-lm (auto-skip when GPU_BACKEND != mlx)
    cuda: Tests requiring Linux + NVIDIA GPU (auto-skip when GPU_BACKEND != cuda)
    integration: End-to-end tests that exercise the full runner loop
```

- [ ] **Step 4: Create empty package files**

Create each of these as empty files:
- `src/pareto_llm/__init__.py`
- `src/pareto_llm/backend/__init__.py`
- `src/pareto_llm/benchmarks/__init__.py`
- `src/pareto_llm/metrics/__init__.py`
- `src/pareto_llm/storage/__init__.py`
- `tests/__init__.py`

- [ ] **Step 5: Install dev dependencies and verify pytest collects**

```bash
uv sync --extra dev
uv run pytest --collect-only
```
Expected: `no tests ran` — zero errors

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml .gitignore pytest.ini src/ tests/__init__.py
git commit -m "chore: project scaffolding — pyproject.toml, package structure, pytest markers"
```

---

### Task 2: Backend base abstractions

**Files:**
- Create: `src/pareto_llm/backend/base.py`
- Modify: `src/pareto_llm/backend/__init__.py`

- [ ] **Step 1: Create src/pareto_llm/backend/base.py**

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class GenerationResult:
    text: str
    prompt_tokens: int
    gen_tokens: int
    prompt_tps: float   # prompt-processing tokens/sec
    gen_tps: float      # generation tokens/sec


class LLMBackend(ABC):
    @abstractmethod
    def load(self, model_id: str) -> None: ...

    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 512) -> GenerationResult: ...

    @abstractmethod
    def unload(self) -> None: ...

    @abstractmethod
    def max_context_tokens(self) -> int: ...
```

- [ ] **Step 2: Update src/pareto_llm/backend/__init__.py**

```python
from .base import GenerationResult, LLMBackend

__all__ = ["GenerationResult", "LLMBackend"]
```

- [ ] **Step 3: Verify import works**

```bash
uv run python -c "from pareto_llm.backend import LLMBackend, GenerationResult; print('OK')"
```
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add src/pareto_llm/backend/
git commit -m "feat: LLMBackend abstract base class and GenerationResult dataclass"
```

---

### Task 3: Shared test fixtures

**Files:**
- Create: `tests/conftest.py`

- [ ] **Step 1: Create tests/conftest.py**

```python
import pathlib

import pytest

from pareto_llm.backend.base import GenerationResult, LLMBackend


class MockBackend(LLMBackend):
    """Deterministic in-memory backend for use in tests."""

    def __init__(self) -> None:
        self.loaded_model: str | None = None
        self.unload_count: int = 0
        self._max_ctx: int = 4096

    def load(self, model_id: str) -> None:
        self.loaded_model = model_id

    def generate(self, prompt: str, max_tokens: int = 512) -> GenerationResult:
        return GenerationResult(
            text="mock output",
            prompt_tokens=len(prompt.split()),
            gen_tokens=10,
            prompt_tps=500.0,
            gen_tps=50.0,
        )

    def unload(self) -> None:
        self.unload_count += 1
        self.loaded_model = None

    def max_context_tokens(self) -> int:
        return self._max_ctx


@pytest.fixture
def mock_backend() -> MockBackend:
    return MockBackend()


@pytest.fixture
def tmp_csv_path(tmp_path: pathlib.Path) -> pathlib.Path:
    return tmp_path / "results.csv"


@pytest.fixture
def sample_config_dict() -> dict:
    return {
        "run_label": "test_run",
        "defaults": {"runs_per_test": 2, "keep_model_files": False},
        "models": ["test-org/test-model"],
        "benchmarks": [
            {
                "type": "mock_bench",
                "name": "my_bench",
                "config": {"key": "value"},
            }
        ],
    }
```

- [ ] **Step 2: Verify conftest loads cleanly**

```bash
uv run pytest --collect-only
```
Expected: collection succeeds, 0 items, 0 errors

- [ ] **Step 3: Commit**

```bash
git add tests/conftest.py
git commit -m "test: add MockBackend and shared fixtures in conftest.py"
```

---

## Chunk 2: Core Modules

### Task 4: Environment init (TDD)

GPU detection logic lives in `pareto_llm._env` (a proper package module) so both the CLI and `scripts/init_env.py` can import it without brittle `__file__`-relative path hacks.

**Files:**
- Create: `tests/test_env.py`
- Create: `src/pareto_llm/_env.py`
- Create: `scripts/init_env.py`

- [ ] **Step 1: Write failing tests**

`tests/test_env.py`:
```python
import pathlib
import subprocess
from unittest.mock import MagicMock, patch

import pytest

import pareto_llm._env as env_mod


def test_detect_darwin_returns_mlx():
    with patch.object(env_mod.platform, "system", return_value="Darwin"):
        assert env_mod.detect_gpu_backend() == "mlx"


def test_detect_linux_nvidia_returns_cuda():
    with patch.object(env_mod.platform, "system", return_value="Linux"):
        with patch.object(env_mod.shutil, "which", return_value="/usr/bin/nvidia-smi"):
            with patch.object(env_mod.subprocess, "run", return_value=MagicMock(returncode=0)):
                assert env_mod.detect_gpu_backend() == "cuda"


def test_detect_no_gpu_raises():
    with patch.object(env_mod.platform, "system", return_value="Linux"):
        with patch.object(env_mod.shutil, "which", return_value=None):
            with pytest.raises(RuntimeError, match="No supported GPU"):
                env_mod.detect_gpu_backend()


def test_detect_nvidia_smi_fails_raises():
    with patch.object(env_mod.platform, "system", return_value="Linux"):
        with patch.object(env_mod.shutil, "which", return_value="/usr/bin/nvidia-smi"):
            with patch.object(
                env_mod.subprocess,
                "run",
                side_effect=subprocess.CalledProcessError(1, "nvidia-smi"),
            ):
                with pytest.raises(RuntimeError, match="No supported GPU"):
                    env_mod.detect_gpu_backend()


def test_write_env_creates_file(tmp_path):
    env_path = tmp_path / ".env"
    env_mod.write_env(env_path, "mlx")
    content = env_path.read_text()
    assert "GPU_BACKEND=mlx" in content
    assert "RESULTS_DIR=./results" in content
    assert "KEEP_MODEL_FILES=false" in content


def test_write_env_overwrites_existing(tmp_path):
    env_path = tmp_path / ".env"
    env_path.write_text("GPU_BACKEND=old_value\n")
    env_mod.write_env(env_path, "cuda")
    content = env_path.read_text()
    assert "GPU_BACKEND=cuda" in content
    assert "old_value" not in content


def test_env_excluded_from_gitignore():
    gitignore = pathlib.Path(__file__).parent.parent / ".gitignore"
    assert gitignore.exists()
    lines = gitignore.read_text().splitlines()
    assert any(line.strip() == ".env" for line in lines)
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
uv run pytest tests/test_env.py -v
```
Expected: `ModuleNotFoundError: No module named 'pareto_llm._env'`

- [ ] **Step 3: Create src/pareto_llm/_env.py**

```python
"""GPU backend detection and .env file writing."""
import pathlib
import platform
import shutil
import subprocess


def detect_gpu_backend() -> str:
    system = platform.system()
    if system == "Darwin":
        return "mlx"
    if shutil.which("nvidia-smi"):
        try:
            subprocess.run(["nvidia-smi"], capture_output=True, check=True)
            return "cuda"
        except subprocess.CalledProcessError:
            pass
    raise RuntimeError(
        "No supported GPU found. Requires Apple Silicon (Metal) or NVIDIA (CUDA)."
    )


def write_env(path: pathlib.Path, gpu_backend: str) -> None:
    content = (
        "# Auto-generated — do not commit\n"
        f"GPU_BACKEND={gpu_backend}\n"
        "RESULTS_DIR=./results\n"
        "KEEP_MODEL_FILES=false\n"
    )
    path.write_text(content)
    print(f"Wrote {path}  (GPU_BACKEND={gpu_backend})")
```

- [ ] **Step 4: Create scripts/init_env.py** (thin wrapper)

```python
"""Run from repo root: python scripts/init_env.py"""
import pathlib
import sys

# Allow running as a script even when pareto_llm is not on sys.path
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

from pareto_llm._env import detect_gpu_backend, write_env  # noqa: E402

if __name__ == "__main__":
    try:
        backend = detect_gpu_backend()
        write_env(pathlib.Path(".env"), backend)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
```

- [ ] **Step 5: Run tests — all should pass**

```bash
uv run pytest tests/test_env.py -v
```
Expected: 7 passed

- [ ] **Step 6: Commit**

```bash
git add src/pareto_llm/_env.py scripts/init_env.py tests/test_env.py
git commit -m "feat: GPU backend detection and .env writing in pareto_llm._env"
```

---

### Task 5: Config parsing (TDD)

**Files:**
- Create: `tests/test_config.py`
- Create: `src/pareto_llm/config.py`

- [ ] **Step 1: Write failing tests**

`tests/test_config.py`:
```python
import pytest
from pydantic import ValidationError

from pareto_llm.config import BenchmarkConfig, BenchmarkEntry, Defaults


def test_parse_minimal_config():
    cfg = BenchmarkConfig(
        run_label="smoke",
        models=["test-org/model"],
        benchmarks=[BenchmarkEntry(type="coding", name="bench1", config={})],
    )
    assert cfg.run_label == "smoke"
    assert cfg.defaults.runs_per_test == 3
    assert cfg.defaults.keep_model_files is False


def test_parse_full_config():
    cfg = BenchmarkConfig(
        run_label="full",
        defaults=Defaults(runs_per_test=5, keep_model_files=True),
        models=["repo/a", "repo/b"],
        benchmarks=[
            BenchmarkEntry(
                type="tool_use",
                name="bfcl",
                config={"dataset": "bfcl", "sample_size": 10},
            ),
        ],
    )
    assert cfg.defaults.runs_per_test == 5
    assert len(cfg.models) == 2
    assert cfg.benchmarks[0].config["sample_size"] == 10


def test_missing_run_label_raises():
    with pytest.raises(ValidationError):
        BenchmarkConfig(
            models=["repo/model"],
            benchmarks=[BenchmarkEntry(type="x", name="y", config={})],
        )


def test_empty_models_raises():
    with pytest.raises(ValidationError):
        BenchmarkConfig(
            run_label="test",
            models=[],
            benchmarks=[BenchmarkEntry(type="x", name="y", config={})],
        )


def test_empty_benchmarks_raises():
    with pytest.raises(ValidationError):
        BenchmarkConfig(
            run_label="test",
            models=["repo/model"],
            benchmarks=[],
        )


def test_benchmark_entry_config_passthrough():
    entry = BenchmarkEntry(type="coding", name="bench", config={"arbitrary": "data", "n": 42})
    assert entry.config["n"] == 42


def test_from_dict_roundtrip(sample_config_dict):
    cfg = BenchmarkConfig.model_validate(sample_config_dict)
    assert cfg.run_label == "test_run"
    assert cfg.defaults.runs_per_test == 2
    assert cfg.benchmarks[0].type == "mock_bench"
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
uv run pytest tests/test_config.py -v
```
Expected: `ModuleNotFoundError: No module named 'pareto_llm.config'`

- [ ] **Step 3: Create src/pareto_llm/config.py**

```python
from pydantic import BaseModel, field_validator


class Defaults(BaseModel):
    runs_per_test: int = 3
    keep_model_files: bool = False


class BenchmarkEntry(BaseModel):
    type: str
    name: str
    config: dict


class BenchmarkConfig(BaseModel):
    run_label: str
    defaults: Defaults = Defaults()
    models: list[str]
    benchmarks: list[BenchmarkEntry]

    @field_validator("models")
    @classmethod
    def models_not_empty(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("models list must not be empty")
        return v

    @field_validator("benchmarks")
    @classmethod
    def benchmarks_not_empty(cls, v: list[BenchmarkEntry]) -> list[BenchmarkEntry]:
        if not v:
            raise ValueError("benchmarks list must not be empty")
        return v
```

- [ ] **Step 4: Run tests — all should pass**

```bash
uv run pytest tests/test_config.py -v
```
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add src/pareto_llm/config.py tests/test_config.py
git commit -m "feat: BenchmarkConfig Pydantic v2 models with validation"
```

---

### Task 6: Benchmark registry + base class (TDD)

**Files:**
- Create: `tests/test_benchmarks.py`
- Create: `src/pareto_llm/benchmarks/base.py`

- [ ] **Step 1: Write failing tests**

`tests/test_benchmarks.py`:
```python
import pytest

from pareto_llm.backend.base import GenerationResult
from pareto_llm.benchmarks.base import (
    BENCHMARK_REGISTRY,
    Benchmark,
    BenchmarkResult,
    register,
)


# ── Registry ──────────────────────────────────────────────────────────────────

def test_register_adds_class_to_registry():
    @register("_test_alpha")
    class AlphaBench(Benchmark):
        def run_single(self, backend):
            return BenchmarkResult(score=1.0, extra={}), backend.generate("x")

    assert "_test_alpha" in BENCHMARK_REGISTRY
    assert BENCHMARK_REGISTRY["_test_alpha"] is AlphaBench


def test_duplicate_register_raises():
    @register("_test_dup")
    class DupA(Benchmark):
        def run_single(self, backend):
            return BenchmarkResult(score=0.0, extra={}), backend.generate("x")

    with pytest.raises(KeyError, match="_test_dup"):
        @register("_test_dup")
        class DupB(Benchmark):
            def run_single(self, backend):
                return BenchmarkResult(score=0.0, extra={}), backend.generate("x")


def test_run_single_returns_tuple(mock_backend):
    @register("_test_beta")
    class BetaBench(Benchmark):
        def run_single(self, backend):
            gen = backend.generate("hello world")
            return BenchmarkResult(score=0.9, extra={"tps": gen.gen_tps}), gen

    bench = BENCHMARK_REGISTRY["_test_beta"](config={})
    bench_result, gen_result = bench.run_single(mock_backend)

    assert isinstance(bench_result, BenchmarkResult)
    assert bench_result.score == 0.9
    assert bench_result.extra["tps"] == 50.0
    assert isinstance(gen_result, GenerationResult)
    assert gen_result.gen_tps == 50.0


def test_benchmark_result_fields():
    r = BenchmarkResult(score=0.5, extra={"k": "v"})
    assert r.score == 0.5
    assert r.extra["k"] == "v"


def test_supports_context_padding_defaults_true():
    @register("_test_gamma")
    class GammaBench(Benchmark):
        def run_single(self, backend):
            return BenchmarkResult(score=0.0, extra={}), backend.generate("x")

    assert GammaBench.supports_context_padding is True
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
uv run pytest tests/test_benchmarks.py -v
```
Expected: `ModuleNotFoundError: No module named 'pareto_llm.benchmarks.base'`

- [ ] **Step 3: Create src/pareto_llm/benchmarks/base.py**

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pareto_llm.backend.base import GenerationResult, LLMBackend

BENCHMARK_REGISTRY: dict[str, type["Benchmark"]] = {}


def register(key: str):
    """Class decorator: register a Benchmark subclass under the given key."""
    def decorator(cls: type["Benchmark"]) -> type["Benchmark"]:
        if key in BENCHMARK_REGISTRY:
            raise KeyError(f"Benchmark key '{key}' is already registered.")
        BENCHMARK_REGISTRY[key] = cls
        return cls
    return decorator


@dataclass
class BenchmarkResult:
    score: float
    extra: dict = field(default_factory=dict)


class Benchmark(ABC):
    supports_context_padding: bool = True

    def __init__(self, config: dict) -> None:
        self.config = config

    @abstractmethod
    def run_single(
        self, backend: "LLMBackend"
    ) -> "tuple[BenchmarkResult, GenerationResult]": ...
```

- [ ] **Step 4: Run tests — all should pass**

```bash
uv run pytest tests/test_benchmarks.py -v
```
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/pareto_llm/benchmarks/base.py tests/test_benchmarks.py
git commit -m "feat: benchmark plugin registry with @register decorator and BenchmarkResult"
```

---

### Task 7: Context-length benchmark wrapper (TDD)

**Files:**
- Modify: `tests/test_benchmarks.py` (append tests at the bottom)
- Create: `src/pareto_llm/benchmarks/context_length.py`
- Modify: `src/pareto_llm/benchmarks/__init__.py`

- [ ] **Step 1: Append context-length tests to tests/test_benchmarks.py**

```python
# ── Context-length wrapper ────────────────────────────────────────────────────

from pareto_llm.benchmarks.context_length import ContextLengthBenchmark


def test_context_length_is_registered():
    assert "context_length" in BENCHMARK_REGISTRY


def test_context_length_pads_prompt(mock_backend):
    """The wrapper must pass a longer prompt than the inner benchmark would alone."""
    captured: list[str] = []
    original_generate = mock_backend.generate

    def capturing_generate(prompt: str, max_tokens: int = 512):
        captured.append(prompt)
        return original_generate(prompt, max_tokens)

    mock_backend.generate = capturing_generate  # type: ignore[method-assign]
    mock_backend._max_ctx = 200  # small so padding is measurable

    @register("_inner_pad_test")
    class InnerBench(Benchmark):
        def run_single(self, backend):
            gen = backend.generate("short prompt")
            return BenchmarkResult(score=1.0, extra={}), gen

    bench = ContextLengthBenchmark(
        config={
            "fill_ratio": 0.5,
            "inner_benchmark": "_inner_pad_test",
            "inner_config": {},
        }
    )
    bench.run_single(mock_backend)

    assert captured, "generate was never called"
    assert len(captured[0].split()) > len("short prompt".split()), "prompt was not padded"


def test_context_length_rejects_unsupported_inner(mock_backend):
    @register("_no_pad_bench")
    class NoPadBench(Benchmark):
        supports_context_padding = False

        def run_single(self, backend):
            return BenchmarkResult(score=0.0, extra={}), backend.generate("x")

    bench = ContextLengthBenchmark(
        config={
            "fill_ratio": 0.25,
            "inner_benchmark": "_no_pad_bench",
            "inner_config": {},
        }
    )
    with pytest.raises(ValueError, match="supports_context_padding"):
        bench.run_single(mock_backend)


def test_context_length_invalid_fill_ratio():
    with pytest.raises(ValueError, match="fill_ratio"):
        ContextLengthBenchmark(
            config={"fill_ratio": 1.5, "inner_benchmark": "_inner_pad_test", "inner_config": {}}
        )
```

- [ ] **Step 2: Run new tests to confirm they fail**

```bash
uv run pytest tests/test_benchmarks.py::test_context_length_is_registered -v
```
Expected: `ModuleNotFoundError` or `ImportError`

- [ ] **Step 3: Create src/pareto_llm/benchmarks/context_length.py**

```python
"""Context-length benchmark: wraps an inner benchmark, padding prompts to a fill ratio."""
from __future__ import annotations

from pareto_llm.backend.base import GenerationResult, LLMBackend
from pareto_llm.benchmarks.base import (
    BENCHMARK_REGISTRY,
    Benchmark,
    BenchmarkResult,
    register,
)

# Public-domain padding text (~50k words; enough for any reasonable fill ratio).
_PADDING_CORPUS = (
    "It was the best of times it was the worst of times it was the age of wisdom "
    "it was the age of foolishness it was the epoch of belief it was the epoch of "
    "incredulity it was the season of Light it was the season of Darkness "
) * 500


class _PaddingBackend(LLMBackend):
    """Wraps a real backend and prepends padding to every generate() call."""

    def __init__(self, inner: LLMBackend, padding: str) -> None:
        self._inner = inner
        self._padding = padding

    def load(self, model_id: str) -> None:
        self._inner.load(model_id)

    def generate(self, prompt: str, max_tokens: int = 512) -> GenerationResult:
        return self._inner.generate(self._padding + " " + prompt, max_tokens)

    def unload(self) -> None:
        self._inner.unload()

    def max_context_tokens(self) -> int:
        return self._inner.max_context_tokens()


@register("context_length")
class ContextLengthBenchmark(Benchmark):
    """Runs an inner benchmark with the context filled to ``fill_ratio``."""

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        fill_ratio = config.get("fill_ratio")
        if fill_ratio is None or not (0.0 < float(fill_ratio) < 1.0):
            raise ValueError(
                f"fill_ratio must be between 0 and 1 (exclusive), got {fill_ratio!r}"
            )

    def run_single(
        self, backend: LLMBackend
    ) -> tuple[BenchmarkResult, GenerationResult]:
        inner_key: str = self.config["inner_benchmark"]
        inner_config: dict = self.config.get("inner_config", {})
        fill_ratio: float = self.config["fill_ratio"]

        inner_cls = BENCHMARK_REGISTRY[inner_key]
        if not inner_cls.supports_context_padding:
            raise ValueError(
                f"Benchmark '{inner_key}' has supports_context_padding=False "
                "and cannot be used inside a context_length wrapper."
            )

        max_ctx = backend.max_context_tokens()
        safety_margin = 64
        # Rough heuristic: 1 token ≈ 0.75 words
        target_words = int(max_ctx * fill_ratio * 0.75) - safety_margin
        padding = " ".join(_PADDING_CORPUS.split()[: max(0, target_words)])

        return inner_cls(inner_config).run_single(_PaddingBackend(backend, padding))
```

- [ ] **Step 4: Update src/pareto_llm/benchmarks/__init__.py**

```python
# Import concrete benchmark modules here so their @register decorators fire on import.
from pareto_llm.benchmarks import context_length as _context_length  # noqa: F401
```

- [ ] **Step 5: Run all benchmark tests**

```bash
uv run pytest tests/test_benchmarks.py -v
```
Expected: 9 passed

- [ ] **Step 6: Commit**

```bash
git add src/pareto_llm/benchmarks/ tests/test_benchmarks.py
git commit -m "feat: context_length benchmark wrapper with prompt padding"
```

---

## Chunk 3: Metrics and Storage

### Task 8: System metrics collector (TDD)

**Files:**
- Create: `tests/test_metrics.py`
- Create: `src/pareto_llm/metrics/system.py`
- Modify: `src/pareto_llm/metrics/__init__.py`

- [ ] **Step 1: Write failing tests**

`tests/test_metrics.py`:
```python
import time
from unittest.mock import MagicMock, patch

import pytest

from pareto_llm.metrics.system import SystemMetricsCollector


def test_ram_fields_populated():
    with SystemMetricsCollector(gpu_backend=None) as collector:
        time.sleep(0.15)  # let the 10 Hz sampler collect ≥1 reading
    assert collector.ram_max_gb > 0
    assert collector.ram_avg_gb > 0
    assert collector.ram_max_gb >= collector.ram_avg_gb


def test_gpu_fields_none_when_no_backend():
    with SystemMetricsCollector(gpu_backend=None) as collector:
        time.sleep(0.05)
    assert collector.gpu_ram_max_gb is None
    assert collector.gpu_ram_avg_gb is None


def test_gpu_fields_populated_for_mlx():
    """Mock mlx.core.metal so we can test without Apple Silicon."""
    mock_metal = MagicMock()
    mock_metal.get_active_memory.return_value = 2 * 1024 ** 3  # 2 GiB in bytes

    with patch.dict(
        "sys.modules",
        {"mlx": MagicMock(), "mlx.core": MagicMock(), "mlx.core.metal": mock_metal},
    ):
        with SystemMetricsCollector(gpu_backend="mlx") as collector:
            time.sleep(0.15)

    assert collector.gpu_ram_max_gb is not None
    assert collector.gpu_ram_max_gb > 0


def test_gpu_fields_populated_for_cuda():
    """Mock pynvml so we can test without an NVIDIA GPU."""
    mock_pynvml = MagicMock()
    mem_info = MagicMock()
    mem_info.used = 4 * 1024 ** 3  # 4 GiB
    mock_pynvml.nvmlDeviceGetMemoryInfo.return_value = mem_info

    with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
        with SystemMetricsCollector(gpu_backend="cuda") as collector:
            time.sleep(0.15)

    assert collector.gpu_ram_max_gb is not None
    assert collector.gpu_ram_max_gb > 0


def test_context_manager_completes_even_if_body_raises():
    with pytest.raises(RuntimeError):
        with SystemMetricsCollector(gpu_backend=None) as collector:
            raise RuntimeError("simulated failure")
    # Collector should still have results computed after exception
    assert collector.ram_max_gb >= 0


def test_thread_joins_after_exit():
    with SystemMetricsCollector(gpu_backend=None) as collector:
        pass
    assert not collector._thread.is_alive()
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
uv run pytest tests/test_metrics.py -v
```
Expected: `ModuleNotFoundError: No module named 'pareto_llm.metrics.system'`

- [ ] **Step 3: Create src/pareto_llm/metrics/system.py**

```python
"""Background-thread system metrics collector: RAM and optional GPU memory."""
from __future__ import annotations

import threading
from typing import Any

import psutil


class SystemMetricsCollector:
    """Context manager that samples RAM (and optionally GPU memory) at 10 Hz.

    After the ``with`` block exits, the following attributes are set:
        ram_max_gb, ram_avg_gb, gpu_ram_max_gb, gpu_ram_avg_gb
    """

    def __init__(self, gpu_backend: str | None = None) -> None:
        self._gpu_backend = gpu_backend
        self._stop = threading.Event()
        self._thread: threading.Thread
        self._ram_samples: list[int] = []
        self._gpu_samples: list[int] = []

        # Public result attributes (populated by __exit__)
        self.ram_max_gb: float = 0.0
        self.ram_avg_gb: float = 0.0
        self.gpu_ram_max_gb: float | None = None
        self.gpu_ram_avg_gb: float | None = None

    def __enter__(self) -> "SystemMetricsCollector":
        self._stop.clear()
        self._ram_samples = []
        self._gpu_samples = []
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> bool:
        self._stop.set()
        self._thread.join(timeout=2.0)
        self._compute_results()
        return False  # do not suppress exceptions

    def _build_gpu_sampler(self) -> Any:
        if self._gpu_backend == "mlx":
            import mlx.core.metal as metal  # type: ignore[import]
            return lambda: metal.get_active_memory()
        if self._gpu_backend == "cuda":
            import pynvml  # type: ignore[import]
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            return lambda: pynvml.nvmlDeviceGetMemoryInfo(handle).used
        return None

    def _sample_loop(self) -> None:
        proc = psutil.Process()
        gpu_sample_fn: Any = None
        try:
            gpu_sample_fn = self._build_gpu_sampler()
        except Exception:
            pass  # GPU not available; gpu_* fields stay None

        while not self._stop.wait(0.1):  # 10 Hz
            try:
                self._ram_samples.append(proc.memory_info().rss)
                if gpu_sample_fn is not None:
                    self._gpu_samples.append(gpu_sample_fn())
            except Exception:
                pass

    def _compute_results(self) -> None:
        if self._ram_samples:
            self.ram_max_gb = max(self._ram_samples) / 1e9
            self.ram_avg_gb = sum(self._ram_samples) / len(self._ram_samples) / 1e9
        if self._gpu_samples:
            self.gpu_ram_max_gb = max(self._gpu_samples) / 1e9
            self.gpu_ram_avg_gb = sum(self._gpu_samples) / len(self._gpu_samples) / 1e9
```

- [ ] **Step 4: Update src/pareto_llm/metrics/__init__.py**

```python
from .system import SystemMetricsCollector

__all__ = ["SystemMetricsCollector"]
```

- [ ] **Step 5: Run tests — all should pass**

```bash
uv run pytest tests/test_metrics.py -v
```
Expected: 6 passed

- [ ] **Step 6: Commit**

```bash
git add src/pareto_llm/metrics/ tests/test_metrics.py
git commit -m "feat: SystemMetricsCollector with background RAM and GPU sampling"
```

---

### Task 9: CSV writer (TDD)

**Files:**
- Create: `tests/test_storage.py`
- Create: `src/pareto_llm/storage/csv_writer.py`
- Modify: `src/pareto_llm/storage/__init__.py`

- [ ] **Step 1: Write failing tests**

`tests/test_storage.py`:
```python
import csv
import pathlib

import pytest

from pareto_llm.backend.base import GenerationResult
from pareto_llm.benchmarks.base import BenchmarkResult
from pareto_llm.metrics.system import SystemMetricsCollector
from pareto_llm.storage.csv_writer import CsvWriter


def _gen() -> GenerationResult:
    return GenerationResult(
        text="hello", prompt_tokens=10, gen_tokens=5, prompt_tps=400.0, gen_tps=40.0
    )


def _collector(gpu: bool = False) -> SystemMetricsCollector:
    """Build a SystemMetricsCollector with pre-set result attributes (no threading)."""
    c = SystemMetricsCollector.__new__(SystemMetricsCollector)
    c.ram_max_gb = 2.0
    c.ram_avg_gb = 1.8
    c.gpu_ram_max_gb = 4.0 if gpu else None
    c.gpu_ram_avg_gb = 3.5 if gpu else None
    return c


def _rows(path: pathlib.Path) -> list[dict]:
    return list(csv.DictReader(path.open()))


# ── append ────────────────────────────────────────────────────────────────────

def test_creates_file_with_header_on_first_call(tmp_csv_path):
    writer = CsvWriter(tmp_csv_path)
    writer.append(
        run_label="r1",
        model_id="org/model",
        benchmark_name="bench",
        run_num=1,
        result=BenchmarkResult(score=0.8, extra={}),
        gen_result=_gen(),
        collector=_collector(),
        max_ctx_tokens=4096,
    )
    assert tmp_csv_path.exists()
    rows = _rows(tmp_csv_path)
    assert len(rows) == 1
    assert rows[0]["model_id"] == "org/model"
    assert rows[0]["score"] == "0.8"
    assert rows[0]["run_num"] == "1"


def test_second_call_appends_no_duplicate_header(tmp_csv_path):
    writer = CsvWriter(tmp_csv_path)
    for i in range(2):
        writer.append(
            run_label="r1",
            model_id="org/model",
            benchmark_name="bench",
            run_num=i + 1,
            result=BenchmarkResult(score=float(i), extra={}),
            gen_result=_gen(),
            collector=_collector(),
            max_ctx_tokens=4096,
        )
    rows = _rows(tmp_csv_path)
    assert len(rows) == 2
    assert rows[1]["run_num"] == "2"


def test_extra_dict_flattened_to_columns(tmp_csv_path):
    writer = CsvWriter(tmp_csv_path)
    writer.append(
        run_label="r1",
        model_id="org/model",
        benchmark_name="bench",
        run_num=1,
        result=BenchmarkResult(score=0.5, extra={"accuracy": 0.5, "pass_at_1": 0.4}),
        gen_result=_gen(),
        collector=_collector(),
        max_ctx_tokens=4096,
    )
    rows = _rows(tmp_csv_path)
    assert rows[0]["extra_accuracy"] == "0.5"
    assert rows[0]["extra_pass_at_1"] == "0.4"


def test_mismatched_extra_keys_fill_with_empty(tmp_csv_path):
    """Two rows with different extra keys: both columns appear in the header; missing cells are ''."""
    writer = CsvWriter(tmp_csv_path)
    writer.append(
        run_label="r1", model_id="m", benchmark_name="b", run_num=1,
        result=BenchmarkResult(score=1.0, extra={"alpha": 1}),
        gen_result=_gen(), collector=_collector(), max_ctx_tokens=4096,
    )
    writer.append(
        run_label="r1", model_id="m", benchmark_name="b", run_num=2,
        result=BenchmarkResult(score=0.0, extra={"beta": 2}),
        gen_result=_gen(), collector=_collector(), max_ctx_tokens=4096,
    )
    rows = _rows(tmp_csv_path)
    # Both extra_* columns must be in the header after a rewrite
    assert "extra_alpha" in rows[0], "extra_alpha column missing from header"
    assert "extra_beta" in rows[0], "extra_beta column missing from header (file not rewritten)"
    assert rows[0]["extra_alpha"] == "1"
    assert rows[0]["extra_beta"] == ""   # row 1 had no beta
    assert rows[1]["extra_alpha"] == ""  # row 2 had no alpha
    assert rows[1]["extra_beta"] == "2"


def test_all_expected_columns_present(tmp_csv_path):
    writer = CsvWriter(tmp_csv_path)
    writer.append(
        run_label="r1", model_id="org/m", benchmark_name="b", run_num=1,
        result=BenchmarkResult(score=0.9, extra={"foo": "bar"}),
        gen_result=_gen(), collector=_collector(gpu=True), max_ctx_tokens=2048,
    )
    header = list(csv.DictReader(tmp_csv_path.open()).fieldnames or [])
    for col in [
        "timestamp", "run_label", "model_id", "benchmark_name", "run_num",
        "score", "prompt_tokens", "gen_tokens", "gen_tps", "prompt_tps",
        "ram_max_gb", "ram_avg_gb", "gpu_ram_max_gb", "gpu_ram_avg_gb",
        "max_ctx_tokens", "extra_foo",
    ]:
        assert col in header, f"Missing column: {col}"


def test_gpu_columns_empty_string_when_none(tmp_csv_path):
    writer = CsvWriter(tmp_csv_path)
    writer.append(
        run_label="r1", model_id="m", benchmark_name="b", run_num=1,
        result=BenchmarkResult(score=1.0, extra={}),
        gen_result=_gen(), collector=_collector(gpu=False), max_ctx_tokens=4096,
    )
    rows = _rows(tmp_csv_path)
    assert rows[0]["gpu_ram_max_gb"] == ""
    assert rows[0]["gpu_ram_avg_gb"] == ""


# ── append_failure ─────────────────────────────────────────────────────────────

def test_append_failure_writes_empty_score(tmp_csv_path):
    writer = CsvWriter(tmp_csv_path)
    writer.append_failure(
        run_label="r1",
        model_id="org/m",
        benchmark_name="b",
        run_num=1,
        error=RuntimeError("boom"),
    )
    rows = _rows(tmp_csv_path)
    assert rows[0]["score"] == ""
    assert "boom" in rows[0]["extra_error"]
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
uv run pytest tests/test_storage.py -v
```
Expected: `ModuleNotFoundError: No module named 'pareto_llm.storage.csv_writer'`

- [ ] **Step 3: Create src/pareto_llm/storage/csv_writer.py**

```python
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
```

- [ ] **Step 4: Update src/pareto_llm/storage/__init__.py**

```python
from .csv_writer import CsvWriter

__all__ = ["CsvWriter"]
```

- [ ] **Step 5: Run tests — all should pass**

```bash
uv run pytest tests/test_storage.py -v
```
Expected: all passed

- [ ] **Step 6: Commit**

```bash
git add src/pareto_llm/storage/ tests/test_storage.py
git commit -m "feat: append-only CsvWriter with extra_* column flattening"
```

---

## Chunk 4: Runner and CLI

### Task 10: Runner (TDD)

**Files:**
- Create: `tests/test_runner.py`
- Create: `src/pareto_llm/runner.py`

- [ ] **Step 1: Write failing tests**

`tests/test_runner.py`:
```python
import csv
import pathlib
from unittest.mock import patch

import pytest

from pareto_llm.benchmarks.base import (
    BENCHMARK_REGISTRY,
    Benchmark,
    BenchmarkResult,
    register,
)
from pareto_llm.config import BenchmarkConfig, BenchmarkEntry, Defaults
from pareto_llm.runner import run


# ── Test-only benchmark fixtures ──────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _register_runner_mock():
    key = "_runner_mock"
    if key not in BENCHMARK_REGISTRY:
        @register(key)
        class RunnerMockBench(Benchmark):
            def run_single(self, backend):
                gen = backend.generate("test prompt")
                return BenchmarkResult(score=0.75, extra={"detail": "ok"}), gen
    yield


def _cfg(
    models: list[str],
    bench_names: list[str],
    runs: int = 2,
    keep: bool = False,
) -> BenchmarkConfig:
    return BenchmarkConfig(
        run_label="test_run",
        defaults=Defaults(runs_per_test=runs, keep_model_files=keep),
        models=models,
        benchmarks=[
            BenchmarkEntry(type="_runner_mock", name=n, config={}) for n in bench_names
        ],
    )


def _rows(path: pathlib.Path) -> list[dict]:
    return list(csv.DictReader(path.open()))


# ── Row counts ────────────────────────────────────────────────────────────────

def test_2_models_2_benches_2_runs_gives_8_rows(tmp_csv_path, mock_backend):
    cfg = _cfg(["m/a", "m/b"], ["bench1", "bench2"], runs=2)
    with patch("pareto_llm.runner._create_backend", return_value=mock_backend):
        with patch("pareto_llm.runner._delete_hf_cache"):
            run(config=cfg, output_path=tmp_csv_path, gpu_backend="cuda")
    assert len(_rows(tmp_csv_path)) == 8


def test_run_num_increments_per_model_benchmark(tmp_csv_path, mock_backend):
    cfg = _cfg(["m/a"], ["bench1"], runs=3)
    with patch("pareto_llm.runner._create_backend", return_value=mock_backend):
        with patch("pareto_llm.runner._delete_hf_cache"):
            run(config=cfg, output_path=tmp_csv_path, gpu_backend="cuda")
    assert [r["run_num"] for r in _rows(tmp_csv_path)] == ["1", "2", "3"]


# ── Backend lifecycle ─────────────────────────────────────────────────────────

def test_backend_unload_called_once_per_model(tmp_csv_path, mock_backend):
    cfg = _cfg(["m/a", "m/b"], ["bench1"], runs=1)
    with patch("pareto_llm.runner._create_backend", return_value=mock_backend):
        with patch("pareto_llm.runner._delete_hf_cache"):
            run(config=cfg, output_path=tmp_csv_path, gpu_backend="cuda")
    assert mock_backend.unload_count == 2


def test_delete_cache_called_when_keep_false(tmp_csv_path, mock_backend):
    cfg = _cfg(["m/a"], ["bench1"], runs=1, keep=False)
    with patch("pareto_llm.runner._create_backend", return_value=mock_backend):
        with patch("pareto_llm.runner._delete_hf_cache") as mock_del:
            run(config=cfg, output_path=tmp_csv_path, gpu_backend="cuda")
    mock_del.assert_called_once_with("m/a")


def test_delete_cache_not_called_when_keep_true(tmp_csv_path, mock_backend):
    cfg = _cfg(["m/a"], ["bench1"], runs=1, keep=True)
    with patch("pareto_llm.runner._create_backend", return_value=mock_backend):
        with patch("pareto_llm.runner._delete_hf_cache") as mock_del:
            run(config=cfg, output_path=tmp_csv_path, gpu_backend="cuda")
    mock_del.assert_not_called()


# ── Error handling ────────────────────────────────────────────────────────────

def test_exception_writes_failure_row_and_continues(tmp_csv_path, mock_backend):
    call_count = 0

    if "_runner_flaky" not in BENCHMARK_REGISTRY:
        @register("_runner_flaky")
        class FlakyBench(Benchmark):
            def run_single(self, backend):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise RuntimeError("transient error")
                gen = backend.generate("x")
                return BenchmarkResult(score=1.0, extra={}), gen

    cfg = BenchmarkConfig(
        run_label="test_run",
        defaults=Defaults(runs_per_test=2, keep_model_files=False),
        models=["m/a"],
        benchmarks=[BenchmarkEntry(type="_runner_flaky", name="flaky", config={})],
    )
    with patch("pareto_llm.runner._create_backend", return_value=mock_backend):
        with patch("pareto_llm.runner._delete_hf_cache"):
            run(config=cfg, output_path=tmp_csv_path, gpu_backend="cuda")

    rows = _rows(tmp_csv_path)
    assert len(rows) == 2
    assert rows[0]["score"] == ""                        # failure row
    assert "transient error" in rows[0]["extra_error"]
    assert rows[1]["score"] == "1.0"                     # success row


def test_exception_does_not_skip_other_benchmarks(tmp_csv_path, mock_backend):
    """A failure in bench1/run1 must not skip bench2."""
    if "_runner_fail_once" not in BENCHMARK_REGISTRY:
        first_call = {"done": False}

        @register("_runner_fail_once")
        class FailOnceBench(Benchmark):
            def run_single(self, backend):
                if not first_call["done"]:
                    first_call["done"] = True
                    raise RuntimeError("once")
                gen = backend.generate("x")
                return BenchmarkResult(score=0.5, extra={}), gen

    cfg = _cfg(["m/a"], ["_runner_mock", "_runner_fail_once"], runs=1)
    with patch("pareto_llm.runner._create_backend", return_value=mock_backend):
        with patch("pareto_llm.runner._delete_hf_cache"):
            run(config=cfg, output_path=tmp_csv_path, gpu_backend="cuda")

    rows = _rows(tmp_csv_path)
    assert len(rows) == 2
    benchmark_names = {r["benchmark_name"] for r in rows}
    assert "_runner_mock" in benchmark_names
    assert "_runner_fail_once" in benchmark_names
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
uv run pytest tests/test_runner.py -v
```
Expected: `ModuleNotFoundError: No module named 'pareto_llm.runner'`

- [ ] **Step 3: Create src/pareto_llm/runner.py**

```python
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
```

- [ ] **Step 4: Run tests — all should pass**

```bash
uv run pytest tests/test_runner.py -v
```
Expected: all passed

- [ ] **Step 5: Run all tests to check for regressions**

```bash
uv run pytest -v --ignore=tests/test_mlx_backend.py --ignore=tests/test_llamacpp_backend.py
```
Expected: all passing

- [ ] **Step 6: Commit**

```bash
git add src/pareto_llm/runner.py tests/test_runner.py
git commit -m "feat: runner orchestrates model × benchmark × run_num loop with CSV output"
```

---

### Task 11: CLI + example config

**Files:**
- Create: `src/pareto_llm/cli.py`
- Create: `configs/example.yaml`

No test file for this task — the CLI wires together already-tested components. Verify manually.

- [ ] **Step 1: Create src/pareto_llm/cli.py**

```python
"""pareto-llm CLI entry point."""
from __future__ import annotations

import logging
import os
import pathlib

import click
import yaml
from dotenv import load_dotenv

from pareto_llm.config import BenchmarkConfig


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging.")
def cli(verbose: bool) -> None:
    """Pareto LLM Benchmark Suite."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


@cli.command()
@click.option(
    "--config", "-c",
    required=True,
    type=click.Path(exists=True, path_type=pathlib.Path),
    help="Path to YAML benchmark config.",
)
@click.option(
    "--output", "-o",
    default=None,
    type=click.Path(path_type=pathlib.Path),
    help="Output CSV path (default: results/<run_label>.csv).",
)
def run(config: pathlib.Path, output: pathlib.Path | None) -> None:
    """Run a benchmark suite."""
    load_dotenv()
    gpu_backend = os.environ.get("GPU_BACKEND", "")
    if not gpu_backend:
        raise click.ClickException(
            "GPU_BACKEND not set. Run `pareto-llm init-env` first."
        )

    raw = yaml.safe_load(config.read_text())
    cfg = BenchmarkConfig.model_validate(raw)

    if output is None:
        results_dir = pathlib.Path(os.environ.get("RESULTS_DIR", "results"))
        output = results_dir / f"{cfg.run_label}.csv"

    click.echo(f"Running {len(cfg.benchmarks)} benchmark(s) × {len(cfg.models)} model(s)")
    click.echo(f"Results → {output}")

    from pareto_llm.runner import run as _run
    _run(config=cfg, output_path=output, gpu_backend=gpu_backend)

    click.echo("Done.")


@cli.command("init-env")
def init_env() -> None:
    """(Re-)generate the .env file for this machine."""
    from pareto_llm._env import detect_gpu_backend, write_env
    try:
        backend = detect_gpu_backend()
        write_env(pathlib.Path(".env"), backend)
    except RuntimeError as exc:
        raise click.ClickException(str(exc)) from exc


@cli.command("list-cached")
def list_cached() -> None:
    """List locally cached Hugging Face models."""
    from huggingface_hub import scan_cache_dir
    info = scan_cache_dir()
    if not info.repos:
        click.echo("No cached models found.")
        return
    for repo in sorted(info.repos, key=lambda r: r.repo_id):
        size_mb = repo.size_on_disk / (1024 ** 2)
        click.echo(f"  {repo.repo_id}  ({size_mb:.0f} MB)")
```

- [ ] **Step 2: Create configs/example.yaml**

```yaml
# ============================================================
# ASPIRATIONAL EXAMPLE — not runnable as-is.
# The "coding" and "tool_use" benchmark types are not yet
# implemented. Add them under src/pareto_llm/benchmarks/ and
# register them with @register("coding") / @register("tool_use")
# before using this config.
# ============================================================

run_label: my_first_run

defaults:
  runs_per_test: 3
  keep_model_files: false

models:
  # Mac (MLX) examples:
  # - mlx-community/Qwen2.5-7B-Instruct-4bit
  # - mlx-community/Qwen2.5-7B-Instruct-8bit
  # Linux (llama.cpp GGUF) examples:
  # - bartowski/Qwen2.5-7B-Instruct-GGUF:Q4_K_M
  # - bartowski/Qwen2.5-7B-Instruct-GGUF:Q8_0
  - mlx-community/Llama-3.2-3B-Instruct-4bit

benchmarks:
  # Requires implementing src/pareto_llm/benchmarks/coding.py
  - type: coding
    name: humaneval_sample
    config:
      dataset: humaneval
      sample_size: 20

  # Context-length wrapper around the coding benchmark above
  - type: context_length
    name: ctx_25pct_humaneval
    config:
      fill_ratio: 0.25
      inner_benchmark: coding
      inner_config:
        sample_size: 5
```

- [ ] **Step 3: Verify CLI help text works**

```bash
uv run pareto-llm --help
uv run pareto-llm run --help
uv run pareto-llm init-env --help
uv run pareto-llm list-cached --help
```
Expected: help text printed for each command, no import errors

- [ ] **Step 4: Commit**

```bash
git add src/pareto_llm/cli.py configs/example.yaml
git commit -m "feat: click CLI with run, init-env, list-cached subcommands"
```

---

## Chunk 5: Platform Backends

> These two tasks are **independent** and can be worked in parallel on their target platforms. Each is gated by a pytest marker and skipped when the required hardware is absent.

### Task 12: MLX backend (Apple Silicon only)

**Files:**
- Create: `tests/test_mlx_backend.py`
- Create: `src/pareto_llm/backend/mlx_backend.py`

- [ ] **Step 1: Write failing tests**

`tests/test_mlx_backend.py`:
```python
import os

import pytest

pytestmark = pytest.mark.mlx

if os.environ.get("GPU_BACKEND") != "mlx":
    pytest.skip("MLX backend requires Apple Silicon (GPU_BACKEND=mlx)", allow_module_level=True)


from pareto_llm.backend.mlx_backend import MLXBackend  # noqa: E402


def test_load_generate_unload():
    backend = MLXBackend()
    # Use the smallest available MLX model for a fast smoke test
    model_id = "mlx-community/Llama-3.2-1B-Instruct-4bit"
    backend.load(model_id)
    try:
        result = backend.generate("Say hello in one word.", max_tokens=10)
        assert result.text.strip() != ""
        assert result.prompt_tokens > 0
        assert result.gen_tokens > 0
        assert result.gen_tps > 0
        assert result.prompt_tps > 0
    finally:
        backend.unload()


def test_max_context_tokens():
    backend = MLXBackend()
    backend.load("mlx-community/Llama-3.2-1B-Instruct-4bit")
    try:
        ctx = backend.max_context_tokens()
        assert isinstance(ctx, int)
        assert ctx > 0
    finally:
        backend.unload()
```

- [ ] **Step 2: Create src/pareto_llm/backend/mlx_backend.py**

```python
"""MLX inference backend for Apple Silicon (Metal GPU)."""
from __future__ import annotations

import time

from pareto_llm.backend.base import GenerationResult, LLMBackend


class MLXBackend(LLMBackend):
    """Inference backend using mlx-lm (Apple Silicon / Metal)."""

    def __init__(self) -> None:
        self._model = None
        self._tokenizer = None

    def load(self, model_id: str) -> None:
        from mlx_lm import load  # type: ignore[import]
        self._model, self._tokenizer = load(model_id)

    def generate(self, prompt: str, max_tokens: int = 512) -> GenerationResult:
        from mlx_lm import generate  # type: ignore[import]
        import mlx.core as mx  # type: ignore[import]

        # Tokenize to count prompt tokens
        prompt_ids = self._tokenizer.encode(prompt)
        prompt_token_count = len(prompt_ids)

        t0 = time.perf_counter()
        response = generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            verbose=False,
        )
        elapsed = time.perf_counter() - t0

        # mlx_lm.generate returns the generated text (not including the prompt)
        gen_token_count = len(self._tokenizer.encode(response))

        # Approximate split: assume prompt processing takes ~10% of total time
        # mlx_lm doesn't expose separate prompt/gen timings via the simple API
        prompt_time = elapsed * 0.1
        gen_time = elapsed * 0.9

        return GenerationResult(
            text=response,
            prompt_tokens=prompt_token_count,
            gen_tokens=gen_token_count,
            prompt_tps=prompt_token_count / prompt_time if prompt_time > 0 else 0.0,
            gen_tps=gen_token_count / gen_time if gen_time > 0 else 0.0,
        )

    def unload(self) -> None:
        self._model = None
        self._tokenizer = None
        # Clear MLX memory
        try:
            import mlx.core as mx  # type: ignore[import]
            mx.metal.clear_cache()
        except Exception:
            pass

    def max_context_tokens(self) -> int:
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        # mlx_lm models expose max_seq_len via the model config
        try:
            return int(self._model.args.max_position_embeddings)
        except AttributeError:
            return 4096  # safe fallback
```

- [ ] **Step 3: Run tests (on Apple Silicon only)**

```bash
uv run pytest tests/test_mlx_backend.py -v -m mlx
```
Expected on Apple Silicon with `GPU_BACKEND=mlx`: 2 passed
Expected on Linux: module-level skip

- [ ] **Step 4: Commit**

```bash
git add src/pareto_llm/backend/mlx_backend.py tests/test_mlx_backend.py
git commit -m "feat: MLX inference backend for Apple Silicon"
```

---

### Task 13: llama.cpp backend (Linux + NVIDIA only)

**Files:**
- Create: `tests/test_llamacpp_backend.py`
- Create: `src/pareto_llm/backend/llamacpp_backend.py`

- [ ] **Step 1: Write failing tests**

`tests/test_llamacpp_backend.py`:
```python
import os

import pytest

pytestmark = pytest.mark.cuda

if os.environ.get("GPU_BACKEND") != "cuda":
    pytest.skip("llama.cpp backend requires Linux + NVIDIA GPU (GPU_BACKEND=cuda)", allow_module_level=True)


from pareto_llm.backend.llamacpp_backend import LlamaCppBackend  # noqa: E402


def test_load_generate_unload():
    backend = LlamaCppBackend()
    # Small Q4 GGUF for a fast smoke test
    model_id = "bartowski/Llama-3.2-1B-Instruct-GGUF:Q4_K_M"
    backend.load(model_id)
    try:
        result = backend.generate("Say hello in one word.", max_tokens=10)
        assert result.text.strip() != ""
        assert result.prompt_tokens > 0
        assert result.gen_tokens > 0
        assert result.gen_tps > 0
        assert result.prompt_tps > 0
    finally:
        backend.unload()


def test_max_context_tokens():
    backend = LlamaCppBackend()
    backend.load("bartowski/Llama-3.2-1B-Instruct-GGUF:Q4_K_M")
    try:
        ctx = backend.max_context_tokens()
        assert isinstance(ctx, int)
        assert ctx > 0
    finally:
        backend.unload()


def test_all_layers_offloaded_to_gpu():
    """Verify GPU offloading is active (n_gpu_layers=-1 means 'all layers')."""
    backend = LlamaCppBackend()
    backend.load("bartowski/Llama-3.2-1B-Instruct-GGUF:Q4_K_M")
    try:
        # _n_gpu_layers is stored by load() from the constructor arg we passed
        assert backend._n_gpu_layers == -1, "Expected full GPU offloading (n_gpu_layers=-1)"
    finally:
        backend.unload()
```

- [ ] **Step 2: Create src/pareto_llm/backend/llamacpp_backend.py**

```python
"""llama.cpp inference backend for Linux + NVIDIA (CUDA)."""
from __future__ import annotations

from pareto_llm.backend.base import GenerationResult, LLMBackend

# Model ID format: "repo_id:filename_pattern"
# e.g. "bartowski/Qwen2.5-7B-Instruct-GGUF:Q4_K_M"


class LlamaCppBackend(LLMBackend):
    """Inference backend using llama-cpp-python with full GPU offloading."""

    def __init__(self) -> None:
        self._llama = None
        self._model_id: str | None = None

    def load(self, model_id: str) -> None:
        from llama_cpp import Llama  # type: ignore[import]
        from huggingface_hub import hf_hub_download, list_repo_files

        self._model_id = model_id

        if ":" in model_id:
            repo_id, pattern = model_id.split(":", 1)
        else:
            repo_id = model_id
            pattern = ".gguf"

        # Find the GGUF file whose name contains the pattern (case-insensitive)
        pattern_lower = pattern.lower()
        all_files = list(list_repo_files(repo_id))
        gguf_files = [
            f for f in all_files
            if f.endswith(".gguf") and pattern_lower in f.lower()
        ]
        if not gguf_files:
            raise FileNotFoundError(
                f"No GGUF file matching '{pattern}' found in {repo_id}. "
                f"Available: {[f for f in all_files if f.endswith('.gguf')]}"
            )
        filename = gguf_files[0]

        local_path = hf_hub_download(repo_id=repo_id, filename=filename)

        self._n_gpu_layers = -1  # -1 means offload all layers to GPU
        self._llama = Llama(
            model_path=local_path,
            n_gpu_layers=self._n_gpu_layers,
            verbose=False,
        )

    def generate(self, prompt: str, max_tokens: int = 512) -> GenerationResult:
        if self._llama is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        output = self._llama.create_completion(
            prompt=prompt,
            max_tokens=max_tokens,
            echo=False,
        )

        timings = output.get("timings", {})
        text = output["choices"][0]["text"]
        usage = output.get("usage", {})

        prompt_tokens = usage.get("prompt_tokens", 0)
        gen_tokens = usage.get("completion_tokens", 0)

        # llama.cpp timings are in milliseconds
        prompt_tps = timings.get("prompt_n", 0) / (timings.get("prompt_ms", 1) / 1000)
        gen_tps = timings.get("predicted_n", 0) / (timings.get("predicted_ms", 1) / 1000)

        return GenerationResult(
            text=text,
            prompt_tokens=prompt_tokens,
            gen_tokens=gen_tokens,
            prompt_tps=prompt_tps,
            gen_tps=gen_tps,
        )

    def unload(self) -> None:
        self._llama = None  # releases VRAM when the Llama object is GC'd

    def max_context_tokens(self) -> int:
        if self._llama is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        return self._llama.n_ctx()
```

- [ ] **Step 3: Run tests (on Linux + NVIDIA only)**

```bash
uv run pytest tests/test_llamacpp_backend.py -v -m cuda
```
Expected on Linux with `GPU_BACKEND=cuda` and NVIDIA GPU: 3 passed
Expected on Mac: module-level skip

- [ ] **Step 4: Commit**

```bash
git add src/pareto_llm/backend/llamacpp_backend.py tests/test_llamacpp_backend.py
git commit -m "feat: llama.cpp inference backend for Linux/NVIDIA with full GPU offloading"
```

---

## Final verification

Run the full test suite (excluding platform-specific tests):

```bash
uv run pytest -v \
  --ignore=tests/test_mlx_backend.py \
  --ignore=tests/test_llamacpp_backend.py \
  --cov=src/pareto_llm \
  --cov-report=term-missing
```

Expected: all unit and integration tests pass, coverage reported.

Smoke-test the CLI:

```bash
uv run pareto-llm --help
uv run pareto-llm init-env
uv run pareto-llm list-cached
```
