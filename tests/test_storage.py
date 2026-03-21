import csv
import pathlib

from pareto_llm.backend.base import GenerationResult
from pareto_llm.benchmarks.base import BenchmarkResult
from pareto_llm.metrics.system import SystemMetricsCollector
from pareto_llm.storage.csv_writer import CsvWriter


def _gen() -> GenerationResult:
    return GenerationResult(text="hello", prompt_tokens=10, gen_tokens=5, prompt_tps=400.0, gen_tps=40.0)


def _collector(gpu: bool = False) -> SystemMetricsCollector:
    """Build a SystemMetricsCollector with pre-set result attributes (no threading)."""
    c = SystemMetricsCollector.__new__(SystemMetricsCollector)
    c.ram_max_gb = 2.0
    c.ram_avg_gb = 1.8
    c.gpu_ram_max_gb = 4.0 if gpu else None
    c.gpu_ram_avg_gb = 3.5 if gpu else None
    return c


def _rows(path: pathlib.Path) -> list[dict]:
    return list(csv.DictReader(path.open()))


# ── append ────────────────────────────────────────────────────────────────────


def test_creates_file_with_header_on_first_call(tmp_csv_path):
    writer = CsvWriter(tmp_csv_path)
    writer.append(
        run_label="r1",
        model_id="org/model",
        benchmark_name="bench",
        run_num=1,
        result=BenchmarkResult(score=0.8, extra={}),
        gen_result=_gen(),
        collector=_collector(),
        max_ctx_tokens=4096,
    )
    assert tmp_csv_path.exists()
    rows = _rows(tmp_csv_path)
    assert len(rows) == 1
    assert rows[0]["model_id"] == "org/model"
    assert rows[0]["score"] == "0.8"
    assert rows[0]["run_num"] == "1"


def test_second_call_appends_no_duplicate_header(tmp_csv_path):
    writer = CsvWriter(tmp_csv_path)
    for i in range(2):
        writer.append(
            run_label="r1",
            model_id="org/model",
            benchmark_name="bench",
            run_num=i + 1,
            result=BenchmarkResult(score=float(i), extra={}),
            gen_result=_gen(),
            collector=_collector(),
            max_ctx_tokens=4096,
        )
    rows = _rows(tmp_csv_path)
    assert len(rows) == 2
    assert rows[1]["run_num"] == "2"


def test_extra_dict_flattened_to_columns(tmp_csv_path):
    writer = CsvWriter(tmp_csv_path)
    writer.append(
        run_label="r1",
        model_id="org/model",
        benchmark_name="bench",
        run_num=1,
        result=BenchmarkResult(score=0.5, extra={"accuracy": 0.5, "pass_at_1": 0.4}),
        gen_result=_gen(),
        collector=_collector(),
        max_ctx_tokens=4096,
    )
    rows = _rows(tmp_csv_path)
    assert rows[0]["extra_accuracy"] == "0.5"
    assert rows[0]["extra_pass_at_1"] == "0.4"


def test_mismatched_extra_keys_fill_with_empty(tmp_csv_path):
    """Two rows with different extra keys: both columns appear in the header; missing cells are ''."""
    writer = CsvWriter(tmp_csv_path)
    writer.append(
        run_label="r1",
        model_id="m",
        benchmark_name="b",
        run_num=1,
        result=BenchmarkResult(score=1.0, extra={"alpha": 1}),
        gen_result=_gen(),
        collector=_collector(),
        max_ctx_tokens=4096,
    )
    writer.append(
        run_label="r1",
        model_id="m",
        benchmark_name="b",
        run_num=2,
        result=BenchmarkResult(score=0.0, extra={"beta": 2}),
        gen_result=_gen(),
        collector=_collector(),
        max_ctx_tokens=4096,
    )
    rows = _rows(tmp_csv_path)
    # Both extra_* columns must be in the header after a rewrite
    assert "extra_alpha" in rows[0], "extra_alpha column missing from header"
    assert "extra_beta" in rows[0], "extra_beta column missing from header (file not rewritten)"
    assert rows[0]["extra_alpha"] == "1"
    assert rows[0]["extra_beta"] == ""  # row 1 had no beta
    assert rows[1]["extra_alpha"] == ""  # row 2 had no alpha
    assert rows[1]["extra_beta"] == "2"


def test_all_expected_columns_present(tmp_csv_path):
    writer = CsvWriter(tmp_csv_path)
    writer.append(
        run_label="r1",
        model_id="org/m",
        benchmark_name="b",
        run_num=1,
        result=BenchmarkResult(score=0.9, extra={"foo": "bar"}),
        gen_result=_gen(),
        collector=_collector(gpu=True),
        max_ctx_tokens=2048,
    )
    header = list(csv.DictReader(tmp_csv_path.open()).fieldnames or [])
    for col in [
        "timestamp",
        "run_label",
        "model_id",
        "benchmark_name",
        "run_num",
        "score",
        "prompt_tokens",
        "gen_tokens",
        "gen_tps",
        "prompt_tps",
        "ram_max_gb",
        "ram_avg_gb",
        "gpu_ram_max_gb",
        "gpu_ram_avg_gb",
        "max_ctx_tokens",
        "extra_foo",
    ]:
        assert col in header, f"Missing column: {col}"


def test_gpu_columns_empty_string_when_none(tmp_csv_path):
    writer = CsvWriter(tmp_csv_path)
    writer.append(
        run_label="r1",
        model_id="m",
        benchmark_name="b",
        run_num=1,
        result=BenchmarkResult(score=1.0, extra={}),
        gen_result=_gen(),
        collector=_collector(gpu=False),
        max_ctx_tokens=4096,
    )
    rows = _rows(tmp_csv_path)
    assert rows[0]["gpu_ram_max_gb"] == ""
    assert rows[0]["gpu_ram_avg_gb"] == ""


# ── append_failure ─────────────────────────────────────────────────────────────


def test_append_failure_writes_empty_score(tmp_csv_path):
    writer = CsvWriter(tmp_csv_path)
    writer.append_failure(
        run_label="r1",
        model_id="org/m",
        benchmark_name="b",
        run_num=1,
        error=RuntimeError("boom"),
    )
    rows = _rows(tmp_csv_path)
    assert rows[0]["score"] == ""
    assert "boom" in rows[0]["extra_error"]
