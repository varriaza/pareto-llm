# LiveBench Integration Design

**Date:** 2026-03-21
**Branch:** to be implemented on a new feature branch
**Status:** Approved

---

## Overview

Add a `LiveBenchBenchmark` class that integrates [LiveBench](https://github.com/livebench/livebench) into the pareto-llm benchmark runner. LiveBench is a contamination-resistant LLM evaluation suite with 18 tasks across 6 categories, updated monthly. The integration follows the same architectural pattern as `TerminalBenchmark`: start a local OpenAI-compatible server via `backend.serve_openai()`, run the external evaluation tool against it, parse result files, and return a `BenchmarkResult`.

---

## Architecture & Flow

`LiveBenchBenchmark` subclasses `Benchmark` and implements `run_single(backend)`. Execution has three phases:

```
1. [Inference]   backend.serve_openai(port, n_ctx=...) starts the local OpenAI server
                 livebench.gen_api_answer.run_questions() sends prompts to http://localhost:{port}/v1
                 → writes answer JSONL files into jobs_dir

2. [Scoring]     (server already shut down — scoring is offline)
                 livebench.gen_ground_truth_judgment() reads answer files, writes judgment JSONL files

3. [Aggregation] Read judgment JSONL files
                 Compute mean score per category and overall mean
                 Return BenchmarkResult(score=overall_mean, extra={per-category scores, counts})
```

`GenerationResult` is a zero-filled stub (same as `TerminalBenchmark`) since LiveBench does not expose per-request timing data.

The `supports_context_padding` class attribute is set to `False` (same as `TerminalBenchmark`) because LiveBench manages its own prompt construction.

---

## Configuration Schema

```yaml
benchmarks:
  - type: live_bench
    name: livebench_all
    config:
      port: 8766
      # Available categories: coding, math, reasoning, language, data_analysis, instruction_following
      # Use "all" to run every category, or a list to select specific ones.
      categories: "all"
      release_date: "latest"       # pin to a specific release, e.g. "2024-11-25"
      sample_size: 20              # optional: max questions per category (null = run all)
      sample_size_per_category:    # optional: per-category overrides (take precedence over sample_size)
        coding: 10
        math: 5
      seed: 42                     # seed for sample reproducibility
      skip_agentic: true           # skip Docker-dependent agentic coding tasks
      n_ctx: 8192
      parallel: 4                  # concurrent API requests to local server (named "parallel" to match LiveBench's own --parallel CLI arg)
      jobs_dir: ./results/livebench
```

### Defaults

| Field | Default |
|---|---|
| `categories` | `"all"` |
| `release_date` | `"latest"` |
| `sample_size` | `null` |
| `sample_size_per_category` | `{}` |
| `seed` | `42` |
| `skip_agentic` | `true` |
| `n_ctx` | `8192` |
| `parallel` | `4` |
| `port` | `8766` |
| `jobs_dir` | `./results/livebench` |

### Validation (in `__init__`, config checks before livebench availability check)

- `categories` must be `"all"` or a **non-empty** list of valid category names; an empty list raises `ValueError` immediately in `__init__` (not deferred to `_get_filtered_questions()`)
- `port` must be between 1024 and 65535
- `sample_size` must be `null` or > 0
- `sample_size_per_category` values must be > 0
- `release_date` must be `"latest"` or match `YYYY-MM-DD` format
- `parallel` must be >= 1
- livebench package availability checked last (same lesson as Terminal Bench)

---

## Components

### `src/pareto_llm/benchmarks/live_bench.py`

New file. Key class: `LiveBenchBenchmark` decorated with `@register("live_bench")`.

**Import discipline:** No livebench symbol is imported at module scope. All `import livebench.*` statements are inside method bodies (same pattern as `terminal_bench.py` with `harbor`). This ensures the module is importable — and `@register("live_bench")` fires — even when the `live-bench` extra is not installed.

**`__init__(config)`**
- Merges config with `DEFAULTS`
- Validates all fields (config checks first, livebench import check last)

**`_get_filtered_questions()`**
- Imports livebench question-loading utilities inside the method body
- Loads questions for selected categories and release date from HuggingFace
- Filters out agentic questions if `skip_agentic=True`
- Applies `sample_size` and `sample_size_per_category` subsampling with seed
- Returns filtered question list

**`run_single(backend)`**
- Checks port availability
- Calls `_get_filtered_questions()`
- Sets `OPENAI_API_KEY` env var (LiteLLM requirement for local servers)
- Generates a timestamped `run_name` (e.g. `lbench_20260321_153042_123456`) to scope output files; same pattern as `TerminalBenchmark`'s `tbench_{datetime}`
- Uses `backend.serve_openai(port, n_ctx=n_ctx)` context manager for inference phase
- Calls `run_questions()` with `api_base=http://localhost:{port}/v1` and `answer_file` rooted under `jobs_dir/run_name/`
- After server shuts down: calls judgment/scoring functions
- Reads judgment JSONL files from known paths under `jobs_dir/run_name/`; computes per-category and overall mean scores
- Returns `(BenchmarkResult, GenerationResult)`

**Note on LiveBench file paths:** The exact paths where `run_questions()` and `gen_ground_truth_judgment()` write files must be verified against the pinned LiveBench version at implementation time. If LiveBench's API does not accept an `answer_file` or `bench_dir` override that scopes output under `jobs_dir/run_name/`, use a `tempfile.mkdtemp()` working directory instead and copy/read results from there. The aggregation step must glob for `**/ground_truth_judgment.jsonl` under the run directory rather than assuming a hardcoded path.

### `src/pareto_llm/benchmarks/__init__.py`

Add import:
```python
from pareto_llm.benchmarks import live_bench as _live_bench  # noqa: F401
```

### `pyproject.toml`

New optional extra:
```toml
[project.optional-dependencies]
live-bench = [
    "livebench @ git+https://github.com/livebench/livebench.git@<pinned-tag>",
]
```

Tag/commit to be pinned at implementation time to latest stable release. **Risk:** `uv sync` hard-fails if the git ref does not exist. The implementer must verify the pinned ref resolves successfully before merging, including in CI. Add `--extra live-bench` to CI's `uv sync` line only after the ref is confirmed valid.

### `configs/live_bench.yaml`

Example config file demonstrating common usage patterns (all categories, coding-only, with sample_size).

### `scripts/setup.sh`

New opt-in install block (not tied to GPU backend):
```bash
echo "==> To install LiveBench extras (optional):"
echo "    uv pip install \"pareto-llm[live-bench]\""
```

Or as an interactive prompt — to be decided at implementation time.

### `.github/workflows/ci.yml`

Add `--extra live-bench` to the `uv sync` install step so CI has livebench available for tests. This must only be done after the pinned git ref in `pyproject.toml` is confirmed resolvable, otherwise `uv sync` will fail the CI pipeline entirely.

---

## Testing (`tests/test_live_bench.py`)

All tests mock livebench imports so they run in CI without network access.

| Test | What it covers |
|---|---|
| `test_valid_config_creates_instance` | Happy-path instantiation with defaults |
| `test_invalid_category_raises` | Unknown category name raises `ValueError` |
| `test_invalid_port_raises` | Port out of range raises `ValueError` |
| `test_invalid_sample_size_raises` | `sample_size <= 0` raises `ValueError` |
| `test_invalid_release_date_raises` | Bad date format raises `ValueError` |
| `test_livebench_not_installed_raises` | Missing livebench raises `RuntimeError` |
| `test_sampling_global` | `sample_size` limits questions per category |
| `test_sampling_per_category_override` | `sample_size_per_category` overrides global |
| `test_sampling_seed_reproducible` | Same seed → same sample |
| `test_sampling_warns_when_exceeds_available` | `sample_size > available` logs warning |
| `test_category_filter_single` | Only selected category's questions passed to inference |
| `test_skip_agentic_true` | Agentic questions excluded when `skip_agentic=True` |
| `test_skip_agentic_false` | Agentic questions included when `skip_agentic=False` |
| `test_run_single_score_computation` | Mock judgment files → correct score and extra dict |
| `test_run_single_port_in_use` | Port already bound raises `RuntimeError` |

---

## Data Flow Detail

```
HuggingFace dataset
       ↓
_get_filtered_questions()
  - filter by category
  - filter out agentic (if skip_agentic)
  - subsample per category with seed
       ↓
backend.serve_openai(port, n_ctx)   ← server starts
       ↓
run_questions(questions, api_base=http://localhost:{port}/v1, parallel=N)
  → jobs_dir/{run_name}/**/model_answer/{model}.jsonl  (exact structure verified at impl time)
       ↓
server shuts down
       ↓
gen_ground_truth_judgment(...)
  → jobs_dir/{run_name}/**/model_judgment/ground_truth_judgment.jsonl  (exact structure verified at impl time)
       ↓
glob for ground_truth_judgment.jsonl under jobs_dir/run_name/
  → per_category_scores: {coding: 0.65, math: 0.42, ...}
  → overall_score: mean of all question scores
       ↓
BenchmarkResult(score=overall_score, extra={per_category_scores, question_counts})
```

---

## Error Handling

- Port already in use → raise `RuntimeError` before starting (same as Terminal Bench)
- No questions after filtering → raise `ValueError`
- `run_questions()` failure → propagate exception (runner logs and records failure)
- Missing judgment files after scoring → raise `RuntimeError` with path info
- livebench not installed → raise `RuntimeError` with install instructions

---

## Out of Scope

- Displaying LiveBench leaderboard output (`show_livebench_result.py`) — scores are captured directly from judgment files
- Supporting non-OpenAI-compatible backends (e.g. Anthropic direct) — all inference goes through `serve_openai()`
- Caching downloaded questions between runs — LiveBench handles its own HuggingFace cache
