# Import concrete benchmark modules here so their @register decorators fire on import.
from pareto_llm.benchmarks import context_length as _context_length  # noqa: F401
from pareto_llm.benchmarks import terminal_bench as _terminal_bench  # noqa: F401
