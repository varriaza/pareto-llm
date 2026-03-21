# Repo Mermaid Diagram Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a Mermaid diagram in `docs/architecture.md` showing every file in the repo, what it does, and how it connects to other files.

**Architecture:** Single Markdown file with an embedded `graph TD` Mermaid diagram using subgraphs to group files by subsystem (Entry, Config, Orchestration, Backends, Benchmarks, Metrics, Storage, Tests). Edges represent import/call relationships with brief labels.

**Tech Stack:** Mermaid graph syntax, Markdown

---

## Chunk 1: Create the diagram file

### Task 1: Write `docs/architecture.md`

**Files:**
- Create: `docs/architecture.md`

- [ ] **Step 1: Write the diagram**

  Create `docs/architecture.md` with the full Mermaid diagram below.

- [ ] **Step 2: Verify the file renders (manual)**

  Open the file in a Mermaid-capable viewer (GitHub, VS Code Mermaid Preview extension, or https://mermaid.live) and confirm layout is legible.

- [ ] **Step 3: Commit**

```bash
git add docs/architecture.md
git commit -m "docs: add Mermaid architecture diagram of all repo files"
```

---

## Diagram content

```mermaid
graph TD
    subgraph Entry["Entry Points"]
        CLI["<b>cli.py</b>\nClick CLI\nrun / init-env / list-cached"]
        SCRIPT["<b>scripts/init_env.py</b>\nStandalone env-setup script"]
    end

    subgraph Config["Configuration"]
        ENV["<b>_env.py</b>\nDetects GPU backend\nwrites .env file"]
        CFG["<b>config.py</b>\nPydantic schema:\nBenchmarkConfig / BenchmarkEntry"]
        YAML["<b>configs/example.yaml</b>\nExample YAML config"]
    end

    subgraph Orchestration["Orchestration"]
        RUNNER["<b>runner.py</b>\nTriple loop: models × benchmarks × runs\nloads/unloads backend per model\ndeletes HF cache on request"]
    end

    subgraph Backends["LLM Backends"]
        BBASE["<b>backend/base.py</b>\nLLMBackend (abstract)\nGenerationResult dataclass"]
        MLX["<b>backend/mlx_backend.py</b>\nMLXBackend\nApple Silicon / Metal\n(lazy import: mlx_lm)"]
        LLAMA["<b>backend/llamacpp_backend.py</b>\nLlamaCppBackend\nNVIDIA CUDA / llama.cpp\n(lazy import: llama_cpp)"]
    end

    subgraph Benchmarks["Benchmark System"]
        BKBASE["<b>benchmarks/base.py</b>\nBenchmark (abstract)\nBENCHMARK_REGISTRY\n@register decorator"]
        CTX["<b>benchmarks/context_length.py</b>\nContextLengthBenchmark\nPads prompt to fill_ratio\nwraps real backend"]
        BKINIT["<b>benchmarks/__init__.py</b>\nImports context_length\nto trigger @register side-effect"]
    end

    subgraph Metrics["Metrics"]
        SYS["<b>metrics/system.py</b>\nSystemMetricsCollector\nBackground thread, 10 Hz sampling\nRAM (psutil) + GPU memory"]
    end

    subgraph Storage["Storage"]
        CSV["<b>storage/csv_writer.py</b>\nCsvWriter\nAppend-only CSV output\nDynamic extra_* columns"]
    end

    subgraph Tests["Tests"]
        CONF["<b>tests/conftest.py</b>\nMockBackend fixture\ntmp_csv_path / sample_config_dict"]
        TENV["<b>tests/test_env.py</b>\nGPU detection logic"]
        TCFG["<b>tests/test_config.py</b>\nYAML parsing & validation"]
        TBENCH["<b>tests/test_benchmarks.py</b>\nRegistry + context-length wrapper"]
        TMET["<b>tests/test_metrics.py</b>\nRAM & GPU sampling"]
        TSTORE["<b>tests/test_storage.py</b>\nCSV writing"]
        TRUN["<b>tests/test_runner.py</b>\nFull orchestration loop"]
        TMLX["<b>tests/test_mlx_backend.py</b>\nReal MLX tests\n(@pytest.mark.mlx)"]
        TCUDA["<b>tests/test_llamacpp_backend.py</b>\nReal llama.cpp tests\n(@pytest.mark.cuda)"]
    end

    %% Entry → core modules
    CLI -->|"loads .env via"| ENV
    CLI -->|"parses YAML via"| CFG
    CLI -->|"calls"| RUNNER
    SCRIPT -->|"calls"| ENV
    YAML -.->|"parsed by"| CFG

    %% Config → Runner
    CFG -->|"BenchmarkConfig passed to"| RUNNER

    %% Runner → subsystems
    RUNNER -->|"imports (triggers @register)"| BKINIT
    RUNNER -->|"reads BENCHMARK_REGISTRY from"| BKBASE
    RUNNER -->|"wraps each run with"| SYS
    RUNNER -->|"writes rows to"| CSV
    RUNNER -->|"lazy import on mlx"| MLX
    RUNNER -->|"lazy import on cuda"| LLAMA

    %% Backend hierarchy
    BBASE -->|"implemented by"| MLX
    BBASE -->|"implemented by"| LLAMA

    %% Benchmark hierarchy
    BKBASE -->|"implemented by"| CTX
    BKINIT -->|"imports"| CTX
    CTX -->|"reads BENCHMARK_REGISTRY from"| BKBASE
    CTX -->|"wraps _PaddingBackend around"| BBASE

    %% Metrics → GPU backends (conditional)
    SYS -->|"gpu_backend=mlx → mlx.core.metal"| MLX
    SYS -->|"gpu_backend=cuda → pynvml"| LLAMA

    %% Tests → subjects
    CONF -.->|"fixture used by"| TRUN
    CONF -.->|"fixture used by"| TBENCH
    TENV -.->|"tests"| ENV
    TCFG -.->|"tests"| CFG
    TBENCH -.->|"tests"| BKBASE
    TBENCH -.->|"tests"| CTX
    TMET -.->|"tests"| SYS
    TSTORE -.->|"tests"| CSV
    TRUN -.->|"tests"| RUNNER
    TMLX -.->|"tests"| MLX
    TCUDA -.->|"tests"| LLAMA
```
