import contextlib
import logging
import math
import pathlib
import socket
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pareto_llm.benchmarks.terminal_bench import TerminalBenchmark


def _valid_config(**overrides):
    cfg = {
        "difficulties": ["medium", "hard"],
        "port": 8765,
        "harbor_agent": "terminus-2",
        "sample_size": None,
        "seed": 42,
        "dataset_version": "2.0",
        "jobs_dir": "./results/harbor",
        "n_concurrent": 4,
    }
    cfg.update(overrides)
    return cfg


# ─── Config validation ────────────────────────────────────────────────────────


def test_invalid_difficulty_raises():
    with pytest.raises(ValueError, match="difficulties"):
        TerminalBenchmark(_valid_config(difficulties=["hard", "extreme"]))


def test_port_too_low_raises():
    with pytest.raises(ValueError, match="port"):
        TerminalBenchmark(_valid_config(port=80))


def test_port_too_high_raises():
    with pytest.raises(ValueError, match="port"):
        TerminalBenchmark(_valid_config(port=99999))


def test_sample_size_zero_raises():
    with pytest.raises(ValueError, match="sample_size"):
        TerminalBenchmark(_valid_config(sample_size=0))


def test_harbor_not_importable_raises(monkeypatch):
    import importlib.util as _ilu

    orig = _ilu.find_spec
    monkeypatch.setattr(_ilu, "find_spec", lambda name: None if name == "harbor" else orig(name))
    with pytest.raises(RuntimeError, match="harbor package not found"):
        TerminalBenchmark(_valid_config())


def test_valid_config_creates_instance():
    bench = TerminalBenchmark(_valid_config())
    assert bench.config["difficulties"] == ["medium", "hard"]
    assert bench.config["port"] == 8765


def test_supports_context_padding_is_false():
    assert TerminalBenchmark.supports_context_padding is False


# ─── Task filtering ───────────────────────────────────────────────────────────


def _make_task_ref(path: pathlib.Path, difficulty: str):
    ref = MagicMock()
    ref.path = path
    ref.git_url = "https://github.com/fake/repo.git"
    ref.git_commit_id = "abc123"
    ref._difficulty = difficulty
    return ref


def _write_task_toml(tmp_path: pathlib.Path, name: str, difficulty: str) -> pathlib.Path:
    task_dir = tmp_path / name
    task_dir.mkdir()
    (task_dir / "task.toml").write_text(f'[metadata]\ndifficulty = "{difficulty}"\n')
    return task_dir


def _mock_registry(dirs_and_difficulties):
    mock_dataset_spec = MagicMock()
    mock_dataset_spec.tasks = [_make_task_ref(d, diff) for d, diff in dirs_and_difficulties]
    mock_client = MagicMock()
    mock_client.get_dataset_spec.return_value = mock_dataset_spec
    return mock_client


def _fetch_difficulty_from_ref(task_ref):
    """Test-only stand-in for _fetch_difficulty that reads the mock _difficulty attribute."""
    return task_ref._difficulty


_PATCH_FETCH = patch(
    "pareto_llm.benchmarks.terminal_bench.TerminalBenchmark._fetch_difficulty",
    staticmethod(_fetch_difficulty_from_ref),
)


def test_filtering_selects_correct_difficulties(tmp_path):
    easy_dir = _write_task_toml(tmp_path, "easy", "easy")
    medium_dir = _write_task_toml(tmp_path, "medium", "medium")
    hard_dir = _write_task_toml(tmp_path, "hard", "hard")
    mock_client = _mock_registry([(easy_dir, "easy"), (medium_dir, "medium"), (hard_dir, "hard")])

    with (
        patch("harbor.registry.client.RegistryClientFactory") as mock_factory,
        patch("harbor.models.registry.RemoteRegistryInfo"),
        _PATCH_FETCH,
    ):
        mock_factory.create.return_value = mock_client
        bench = TerminalBenchmark(_valid_config(difficulties=["medium", "hard"]))
        tasks = bench._get_filtered_tasks()

    assert len(tasks) == 2
    paths = {t.path for t in tasks}
    assert medium_dir in paths
    assert hard_dir in paths
    assert easy_dir not in paths


def test_sample_size_limits_tasks(tmp_path):
    dirs = [(d, "hard") for d in [_write_task_toml(tmp_path, f"task_{i}", "hard") for i in range(10)]]
    mock_client = _mock_registry(dirs)

    with (
        patch("harbor.registry.client.RegistryClientFactory") as mock_factory,
        patch("harbor.models.registry.RemoteRegistryInfo"),
        _PATCH_FETCH,
    ):
        mock_factory.create.return_value = mock_client
        bench = TerminalBenchmark(_valid_config(difficulties=["hard"], sample_size=5, seed=42))
        tasks = bench._get_filtered_tasks()

    assert len(tasks) == 5


def test_sample_size_is_deterministic(tmp_path):
    dirs = [(d, "hard") for d in [_write_task_toml(tmp_path, f"task_{i}", "hard") for i in range(10)]]

    def run():
        mock_client = _mock_registry(dirs)
        with (
            patch("harbor.registry.client.RegistryClientFactory") as mock_factory,
            patch("harbor.models.registry.RemoteRegistryInfo"),
            _PATCH_FETCH,
        ):
            mock_factory.create.return_value = mock_client
            bench = TerminalBenchmark(_valid_config(difficulties=["hard"], sample_size=5, seed=42))
            return bench._get_filtered_tasks()

    assert [t.path for t in run()] == [t.path for t in run()]


def test_sample_size_exceeds_available_uses_all(tmp_path, caplog):
    dirs = [(d, "hard") for d in [_write_task_toml(tmp_path, f"task_{i}", "hard") for i in range(3)]]
    mock_client = _mock_registry(dirs)

    with (
        patch("harbor.registry.client.RegistryClientFactory") as mock_factory,
        patch("harbor.models.registry.RemoteRegistryInfo"),
        caplog.at_level(logging.WARNING),
        _PATCH_FETCH,
    ):
        mock_factory.create.return_value = mock_client
        bench = TerminalBenchmark(_valid_config(difficulties=["hard"], sample_size=10))
        tasks = bench._get_filtered_tasks()

    assert len(tasks) == 3
    assert caplog.records, "Expected a warning log"


def test_zero_filtered_tasks_raises(tmp_path):
    easy_dir = _write_task_toml(tmp_path, "easy", "easy")
    mock_client = _mock_registry([(easy_dir, "easy")])

    with (
        patch("harbor.registry.client.RegistryClientFactory") as mock_factory,
        patch("harbor.models.registry.RemoteRegistryInfo"),
        _PATCH_FETCH,
    ):
        mock_factory.create.return_value = mock_client
        bench = TerminalBenchmark(_valid_config(difficulties=["hard"]))
        with pytest.raises(ValueError, match="No tasks found"):
            bench._get_filtered_tasks()


# ─── run_single ───────────────────────────────────────────────────────────────


def _make_mock_backend():
    backend = MagicMock()
    backend.serve_openai.return_value = contextlib.nullcontext()
    return backend


def _setup_run_mocks(tmp_path, reward_stats):
    from harbor.models.trial.config import TaskConfig

    bench = TerminalBenchmark(_valid_config(jobs_dir=str(tmp_path)))
    backend = _make_mock_backend()
    task_dir = tmp_path / "fake_task"
    task_dir.mkdir()
    pre_filtered = [TaskConfig(path=task_dir)]

    mock_job_instance = MagicMock()
    mock_job_instance.run = AsyncMock()

    mock_evals = MagicMock()
    mock_evals.reward_stats = reward_stats
    mock_job_result = MagicMock()
    mock_job_result.stats.evals = {"key": mock_evals}

    def make_job(config):
        job_dir = tmp_path / config.job_name
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / "result.json").write_text("{}")
        return mock_job_instance

    return bench, backend, pre_filtered, make_job, mock_job_result


def test_run_single_score_three_of_four(tmp_path):
    bench, backend, pre_filtered, make_job, mock_job_result = _setup_run_mocks(
        tmp_path, {"reward": {1.0: ["t1", "t2", "t3"], 0.0: ["t4"]}}
    )
    with (
        patch.object(bench, "_get_filtered_tasks", return_value=pre_filtered),
        patch("harbor.job.Job", side_effect=make_job),
        patch("harbor.models.job.result.JobResult") as mock_jr,
    ):
        mock_jr.model_validate_json.return_value = mock_job_result
        result, gen = bench.run_single(backend)

    assert math.isclose(result.score, 0.75)
    assert result.extra["tasks_total"] == 4
    assert result.extra["tasks_passed"] == 3
    assert gen.text == ""
    assert gen.prompt_tokens == 0
    assert gen.gen_tokens == 0


def test_run_single_score_all_pass(tmp_path):
    bench, backend, pre_filtered, make_job, mock_job_result = _setup_run_mocks(
        tmp_path, {"reward": {1.0: ["t1", "t2"]}}
    )
    with (
        patch.object(bench, "_get_filtered_tasks", return_value=pre_filtered),
        patch("harbor.job.Job", side_effect=make_job),
        patch("harbor.models.job.result.JobResult") as mock_jr,
    ):
        mock_jr.model_validate_json.return_value = mock_job_result
        result, _ = bench.run_single(backend)

    assert math.isclose(result.score, 1.0)


def test_run_single_score_none_pass(tmp_path):
    bench, backend, pre_filtered, make_job, mock_job_result = _setup_run_mocks(
        tmp_path, {"reward": {0.0: ["t1", "t2"]}}
    )
    with (
        patch.object(bench, "_get_filtered_tasks", return_value=pre_filtered),
        patch("harbor.job.Job", side_effect=make_job),
        patch("harbor.models.job.result.JobResult") as mock_jr,
    ):
        mock_jr.model_validate_json.return_value = mock_job_result
        result, _ = bench.run_single(backend)

    assert math.isclose(result.score, 0.0)


def test_port_in_use_raises(tmp_path):
    bench = TerminalBenchmark(_valid_config(jobs_dir=str(tmp_path), port=18999))
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("127.0.0.1", 18999))
        with patch.object(bench, "_get_filtered_tasks", return_value=[MagicMock()]):
            with pytest.raises(RuntimeError, match="already in use"):
                bench.run_single(_make_mock_backend())


# ─── Registration ─────────────────────────────────────────────────────────────


def test_terminal_bench_is_registered():
    from pareto_llm.benchmarks.base import BENCHMARK_REGISTRY

    assert "terminal_bench" in BENCHMARK_REGISTRY
