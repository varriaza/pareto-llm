"""pareto-llm CLI entry point."""
from __future__ import annotations

import logging
import os
import pathlib

import click
import yaml
from dotenv import load_dotenv

from pareto_llm.config import BenchmarkConfig


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging.")
def cli(verbose: bool) -> None:
    """Pareto LLM Benchmark Suite."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


@cli.command()
@click.option(
    "--config", "-c",
    required=True,
    type=click.Path(exists=True, path_type=pathlib.Path),
    help="Path to YAML benchmark config.",
)
@click.option(
    "--output", "-o",
    default=None,
    type=click.Path(path_type=pathlib.Path),
    help="Output CSV path (default: results/<run_label>.csv).",
)
def run(config: pathlib.Path, output: pathlib.Path | None) -> None:
    """Run a benchmark suite."""
    load_dotenv()
    gpu_backend = os.environ.get("GPU_BACKEND", "")
    if not gpu_backend:
        raise click.ClickException(
            "GPU_BACKEND not set. Run `pareto-llm init-env` first."
        )

    raw = yaml.safe_load(config.read_text())
    cfg = BenchmarkConfig.model_validate(raw)

    if output is None:
        results_dir = pathlib.Path(os.environ.get("RESULTS_DIR", "results"))
        output = results_dir / f"{cfg.run_label}.csv"

    click.echo(f"Running {len(cfg.benchmarks)} benchmark(s) × {len(cfg.models)} model(s)")
    click.echo(f"Results → {output}")

    from pareto_llm.runner import run as _run
    _run(config=cfg, output_path=output, gpu_backend=gpu_backend)

    click.echo("Done.")


@cli.command("init-env")
def init_env() -> None:
    """(Re-)generate the .env file for this machine."""
    from pareto_llm._env import detect_gpu_backend, write_env
    try:
        backend = detect_gpu_backend()
        write_env(pathlib.Path(".env"), backend)
    except RuntimeError as exc:
        raise click.ClickException(str(exc)) from exc


@cli.command("list-cached")
def list_cached() -> None:
    """List locally cached Hugging Face models."""
    from huggingface_hub import scan_cache_dir
    info = scan_cache_dir()
    if not info.repos:
        click.echo("No cached models found.")
        return
    for repo in sorted(info.repos, key=lambda r: r.repo_id):
        size_mb = repo.size_on_disk / (1024 ** 2)
        click.echo(f"  {repo.repo_id}  ({size_mb:.0f} MB)")
