import pathlib
import subprocess
from unittest.mock import MagicMock, patch

import pytest

import pareto_llm._env as env_mod


def test_detect_darwin_returns_mlx():
    with patch.object(env_mod.platform, "system", return_value="Darwin"):
        assert env_mod.detect_gpu_backend() == "mlx"


def test_detect_linux_nvidia_returns_cuda():
    with patch.object(env_mod.platform, "system", return_value="Linux"):
        with patch.object(env_mod.shutil, "which", return_value="/usr/bin/nvidia-smi"):
            with patch.object(env_mod.subprocess, "run", return_value=MagicMock(returncode=0)):
                assert env_mod.detect_gpu_backend() == "cuda"


def test_detect_no_gpu_raises():
    with patch.object(env_mod.platform, "system", return_value="Linux"):
        with patch.object(env_mod.shutil, "which", return_value=None):
            with pytest.raises(RuntimeError, match="No supported GPU"):
                env_mod.detect_gpu_backend()


def test_detect_nvidia_smi_fails_raises():
    with patch.object(env_mod.platform, "system", return_value="Linux"):
        with patch.object(env_mod.shutil, "which", return_value="/usr/bin/nvidia-smi"):
            with patch.object(
                env_mod.subprocess,
                "run",
                side_effect=subprocess.CalledProcessError(1, "nvidia-smi"),
            ):
                with pytest.raises(RuntimeError, match="No supported GPU"):
                    env_mod.detect_gpu_backend()


def test_write_env_creates_file(tmp_path):
    env_path = tmp_path / ".env"
    env_mod.write_env(env_path, "mlx")
    content = env_path.read_text()
    assert "GPU_BACKEND=mlx" in content
    assert "RESULTS_DIR=./results" in content
    assert "KEEP_MODEL_FILES=false" in content


def test_write_env_overwrites_existing(tmp_path):
    env_path = tmp_path / ".env"
    env_path.write_text("GPU_BACKEND=old_value\n")
    env_mod.write_env(env_path, "cuda")
    content = env_path.read_text()
    assert "GPU_BACKEND=cuda" in content
    assert "old_value" not in content


def test_env_excluded_from_gitignore():
    gitignore = pathlib.Path(__file__).parent.parent / ".gitignore"
    assert gitignore.exists()
    lines = gitignore.read_text().splitlines()
    assert any(line.strip() == ".env" for line in lines)
