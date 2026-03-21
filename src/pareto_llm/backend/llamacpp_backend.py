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
