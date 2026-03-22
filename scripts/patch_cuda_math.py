#!/usr/bin/env python3
"""Patch CUDA math_functions.h to fix noexcept conflict with glibc 2.41.

CUDA 12.8 headers declare cospi/sinpi/cospif/sinpif without noexcept, but
glibc 2.41 declares them with noexcept(true). This causes a compile error
when building llama-cpp-python from source:

    error: declaration of 'cospi' has a different exception specifier

Run this script once (with sudo) before building llama-cpp-python:

    sudo python3 scripts/patch_cuda_math.py
    CMAKE_ARGS="-DGGML_CUDA=on" uv pip install "pareto-llm[cuda]" --no-cache-dir
"""

import pathlib
import re
import sys

HEADER = pathlib.Path("/usr/local/cuda/targets/x86_64-linux/include/crt/math_functions.h")

FUNCTIONS = ["cospi", "cospif", "sinpi", "sinpif"]


def main() -> None:
    if not HEADER.exists():
        print(f"Header not found: {HEADER}")
        print("Is the CUDA Toolkit installed?")
        sys.exit(1)

    text = HEADER.read_text()
    original = text

    for fn in FUNCTIONS:
        pattern = rf"(extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__[^;]*\b{fn}\([^)]*\));"
        replacement = r"\1 noexcept(true);"
        text = re.sub(pattern, replacement, text)

    if text == original:
        print("Header already patched (or pattern not found) — no changes made.")
        return

    HEADER.write_text(text)
    print(f"Patched {HEADER}")
    print("You can now build llama-cpp-python:")
    print('  CMAKE_ARGS="-DGGML_CUDA=on" uv pip install "pareto-llm[cuda]" --no-cache-dir')


if __name__ == "__main__":
    main()
