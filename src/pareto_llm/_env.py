"""GPU backend detection and .env file writing."""
import pathlib
import platform
import shutil
import subprocess


def detect_gpu_backend() -> str:
    system = platform.system()
    if system == "Darwin":
        return "mlx"
    if shutil.which("nvidia-smi"):
        try:
            subprocess.run(["nvidia-smi"], capture_output=True, check=True)
            return "cuda"
        except subprocess.CalledProcessError:
            pass
    raise RuntimeError(
        "No supported GPU found. Requires Apple Silicon (Metal) or NVIDIA (CUDA)."
    )


def write_env(path: pathlib.Path, gpu_backend: str) -> None:
    content = (
        "# Auto-generated — do not commit\n"
        f"GPU_BACKEND={gpu_backend}\n"
        "RESULTS_DIR=./results\n"
        "KEEP_MODEL_FILES=false\n"
    )
    path.write_text(content)
    print(f"Wrote {path}  (GPU_BACKEND={gpu_backend})")
