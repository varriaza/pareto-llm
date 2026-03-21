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
