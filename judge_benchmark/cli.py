"""
Command-Line Interface for LLM Judge Reliability Benchmark.
"""

import os
import json
import argparse
from typing import List
from judge_benchmark.schema import BenchmarkItem, JudgeEvaluationRecord, JudgePromptType
from judge_benchmark.judge_engine import LLMJudgeEngine
from judge_benchmark.metrics import JudgeMetricsCalculator
from judge_benchmark.bias_analyzer import BiasAnalyzer
from judge_benchmark.report_generator import ReportGenerator
from judge_benchmark.visualizer import Visualizer


def load_benchmark_dataset(file_path: str) -> List[BenchmarkItem]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file not found: {file_path}")

    items: List[BenchmarkItem] = []
    with open(file_path, "r", encoding="utf-8") as f:
        if file_path.endswith(".jsonl"):
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    items.append(BenchmarkItem.from_dict(data))
        else:
            data_list = json.load(f)
            for data in data_list:
                items.append(BenchmarkItem.from_dict(data))
    return items


def main():
    parser = argparse.ArgumentParser(description="LLM Judge Reliability Benchmark CLI.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run judge reliability benchmark suite")
    run_parser.add_argument("--input", "-i", required=True, help="Path to benchmark dataset (.jsonl)")
    run_parser.add_argument("--output-dir", "-o", default="results", help="Directory to save output JSON/CSV")
    run_parser.add_argument("--report", "-r", default="reports/summary_report.md", help="Path to generate report")
    run_parser.add_argument("--figures-dir", "-f", default="reports/figures", help="Directory to save figures")

    args = parser.parse_args()

    if args.command == "run":
        print(f"Loading benchmark dataset from: {args.input}")
        items = load_benchmark_dataset(args.input)
        print(f"Loaded {len(items)} benchmark test pairs.")

        print("Executing LLM Judge evaluation across 3 Prompt Strategies (Baseline, Detailed Rubric, Chain-of-Thought)...")
        engine = LLMJudgeEngine()
        all_records: List[JudgeEvaluationRecord] = []

        prompt_strategies = [JudgePromptType.BASELINE, JudgePromptType.DETAILED_RUBRIC, JudgePromptType.CHAIN_OF_THOUGHT]

        for item in items:
            for strategy in prompt_strategies:
                # Evaluate Good Candidate
                rec_good = engine.evaluate_item(item, strategy, evaluate_good=True)
                all_records.append(rec_good)

                # Evaluate Flawed Candidate
                rec_flawed = engine.evaluate_item(item, strategy, evaluate_good=False)
                all_records.append(rec_flawed)

        print(f"Recorded {len(all_records)} individual judge evaluation runs.")

        print("\nCalculating statistical correlations (Pearson r, Spearman rho) and agreement rates...")
        suite_metrics = JudgeMetricsCalculator.evaluate_benchmark_suite(all_records)
        bias_results = BiasAnalyzer.calculate_bias_suite(all_records)

        for pt, m in suite_metrics.items():
            print(f" [{pt.upper()}] Pearson r: {m['pearson_r']:.4f} | Spearman rho: {m['spearman_rho']:.4f} | Agreement: {m['agreement_rate']*100:.1f}% | FP: {m['false_positives']}")

        print("\nExporting results to JSON and CSV...")
        ReportGenerator.export_results(all_records, args.output_dir)

        print("\nGenerating visualization charts...")
        Visualizer.generate_all_figures(suite_metrics, bias_results, args.figures_dir)

        print(f"\nGenerating Markdown research report at: {args.report}")
        ReportGenerator.generate_markdown_report(suite_metrics, bias_results, args.report)

        print("\nLLM Judge Reliability Research Study Completed Successfully!")


if __name__ == "__main__":
    main()
