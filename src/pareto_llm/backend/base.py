import contextlib
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class GenerationResult:
    text: str
    prompt_tokens: int
    gen_tokens: int
    prompt_tps: float  # prompt-processing tokens/sec
    gen_tps: float  # generation tokens/sec


class LLMBackend(ABC):
    @abstractmethod
    def load(self, model_id: str) -> None: ...

    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 512) -> GenerationResult: ...

    @abstractmethod
    def unload(self) -> None: ...

    @abstractmethod
    def max_context_tokens(self) -> int: ...

    @abstractmethod
    def serve_openai(self, port: int, n_ctx: int = 8192) -> contextlib.AbstractContextManager[None]: ...
