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
            import importlib

            metal = importlib.import_module("mlx.core.metal")  # type: ignore[import]
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
