"""Run from repo root: python scripts/init_env.py [--backend cuda|mlx|none]"""

import argparse
import pathlib
import sys

# Allow running as a script even when pareto_llm is not on sys.path
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

from pareto_llm._env import detect_gpu_backend, write_env  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["cuda", "mlx", "none"], default=None)
    args = parser.parse_args()

    try:
        backend = args.backend if args.backend is not None else detect_gpu_backend()
        write_env(pathlib.Path(".env"), backend)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
