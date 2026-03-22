"""llama.cpp inference backend for Linux + NVIDIA (CUDA)."""

from __future__ import annotations

import contextlib
import subprocess
import time
import urllib.request

from pareto_llm.backend.base import GenerationResult, LLMBackend

# Model ID format: "repo_id:filename_pattern"
# e.g. "bartowski/Qwen2.5-7B-Instruct-GGUF:Q4_K_M"


class LlamaCppBackend(LLMBackend):
    """Inference backend using llama-cpp-python with full GPU offloading."""

    def __init__(self) -> None:
        self._llama = None
        self._model_id: str | None = None
        self._model_path: str | None = None

    def load(self, model_id: str) -> None:
        from huggingface_hub import hf_hub_download, list_repo_files
        from llama_cpp import Llama  # type: ignore[import]

        self._model_id = model_id

        if ":" in model_id:
            repo_id, pattern = model_id.split(":", 1)
        else:
            repo_id = model_id
            pattern = ".gguf"

        # Find the GGUF file whose name contains the pattern (case-insensitive)
        pattern_lower = pattern.lower()
        all_files = list(list_repo_files(repo_id))
        gguf_files = [f for f in all_files if f.endswith(".gguf") and pattern_lower in f.lower()]
        if not gguf_files:
            raise FileNotFoundError(
                f"No GGUF file matching '{pattern}' found in {repo_id}. "
                f"Available: {[f for f in all_files if f.endswith('.gguf')]}"
            )
        filename = gguf_files[0]

        local_path = hf_hub_download(repo_id=repo_id, filename=filename)
        self._model_path = local_path

        self._n_gpu_layers = -1  # -1 means offload all layers to GPU
        self._llama = Llama(
            model_path=local_path,
            n_gpu_layers=self._n_gpu_layers,
            verbose=False,
        )

    def generate(self, prompt: str, max_tokens: int = 512) -> GenerationResult:
        if self._llama is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        t0 = time.perf_counter()
        output = self._llama.create_completion(
            prompt=prompt,
            max_tokens=max_tokens,
            echo=False,
        )
        elapsed = time.perf_counter() - t0

        timings = output.get("timings") or {}
        text = output["choices"][0]["text"]
        usage = output.get("usage", {})

        prompt_tokens = usage.get("prompt_tokens", 0)
        gen_tokens = usage.get("completion_tokens", 0)

        # Prefer llama.cpp's own timing data; fall back to wall-clock if unavailable
        prompt_tps = (
            timings.get("prompt_per_second")
            or (timings.get("prompt_n", 0) / (timings.get("prompt_ms", 1) / 1000))
            or (prompt_tokens / elapsed if elapsed > 0 else 0.0)
        )
        gen_tps = (
            timings.get("predicted_per_second")
            or (timings.get("predicted_n", 0) / (timings.get("predicted_ms", 1) / 1000))
            or (gen_tokens / elapsed if elapsed > 0 else 0.0)
        )

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

    @contextlib.contextmanager
    def serve_openai(self, port: int, n_ctx: int = 8192):
        if self._llama is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        model_path = self._model_path
        model_id = self._model_id
        self.unload()
        proc = subprocess.Popen(
            [
                "python3",
                "-m",
                "llama_cpp.server",
                "--model",
                model_path,
                "--n_gpu_layers",
                "-1",
                "--n_ctx",
                str(n_ctx),
                "--port",
                str(port),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
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
            proc.wait(timeout=10)
            raise TimeoutError(f"Server on port {port} did not become healthy within 30s")

        try:
            yield
        finally:
            proc.terminate()
            proc.wait(timeout=10)
            self.load(model_id)
