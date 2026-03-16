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
