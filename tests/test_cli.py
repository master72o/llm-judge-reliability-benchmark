"""
Integration test for judge benchmark CLI.
"""

import os
import pytest
from judge_benchmark.cli import load_benchmark_dataset
from judge_benchmark.schema import JudgePromptType
from judge_benchmark.judge_engine import LLMJudgeEngine
from judge_benchmark.metrics import JudgeMetricsCalculator
from judge_benchmark.report_generator import ReportGenerator


def test_cli_load_dataset(tmp_path):
    jsonl = tmp_path / "bm_test.jsonl"
    jsonl.write_text(
        '{"id": "bm1", "prompt": "Hi", "response_good": "Hello", "response_flawed": "Bye"}\n',
        encoding="utf-8"
    )

    items = load_benchmark_dataset(str(jsonl))
    assert len(items) == 1
    assert items[0].id == "bm1"


def test_full_benchmark_pipeline_execution(tmp_path):
    dataset_file = tmp_path / "sample.jsonl"
    dataset_file.write_text(
        '{"id": "bm1", "prompt": "Hi", "response_good": "Hello", "response_flawed": "Bye"}\n',
        encoding="utf-8"
    )

    items = load_benchmark_dataset(str(dataset_file))
    engine = LLMJudgeEngine()
    records = [engine.evaluate_item(items[0], pt, True) for pt in [JudgePromptType.BASELINE, JudgePromptType.CHAIN_OF_THOUGHT]]

    metrics = JudgeMetricsCalculator.evaluate_benchmark_suite(records)
    out_dir = tmp_path / "results"
    report_path = tmp_path / "report.md"

    ReportGenerator.export_results(records, str(out_dir))
    ReportGenerator.generate_markdown_report(metrics, {}, str(report_path))

    assert os.path.exists(out_dir / "judge_benchmark_results.json")
    assert os.path.exists(out_dir / "judge_benchmark_results.csv")
    assert os.path.exists(report_path)
