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
