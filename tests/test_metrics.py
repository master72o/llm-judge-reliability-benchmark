"""
Unit tests for correlation and reliability statistics.
"""

import pytest
from judge_benchmark.schema import JudgeEvaluationRecord, JudgePromptType
from judge_benchmark.metrics import JudgeMetricsCalculator


def test_perfect_correlation_metrics():
    records = [
        JudgeEvaluationRecord(
            item_id="1", prompt="p", response="r1", human_score=1.0, judge_score=1.0,
            prompt_type=JudgePromptType.CHAIN_OF_THOUGHT, passed_human=True, passed_judge=True
        ),
        JudgeEvaluationRecord(
            item_id="2", prompt="p", response="r2", human_score=0.0, judge_score=0.0,
            prompt_type=JudgePromptType.CHAIN_OF_THOUGHT, passed_human=False, passed_judge=False
        ),
    ]

    metrics = JudgeMetricsCalculator.calculate_prompt_strategy_metrics(records)
    assert metrics["pearson_r"] == 1.0
    assert metrics["spearman_rho"] == 1.0
    assert metrics["agreement_rate"] == 1.0
    assert metrics["false_positives"] == 0
    assert metrics["false_negatives"] == 0


def test_verbosity_and_formatting_bias_analyzer():
    from judge_benchmark.bias_analyzer import BiasAnalyzer

    records = [
        JudgeEvaluationRecord(
            item_id="1",
            prompt="p",
            response="short bad",
            human_score=0.2,
            judge_score=0.2,
            prompt_type=JudgePromptType.CHAIN_OF_THOUGHT,
            passed_human=False,
            passed_judge=False,
        ),
        JudgeEvaluationRecord(
            item_id="2",
            prompt="p",
            response="```python\nprint('verbose formatted bad response with many words')\n```",
            human_score=0.2,
            judge_score=0.85,
            prompt_type=JudgePromptType.CHAIN_OF_THOUGHT,
            passed_human=False,
            passed_judge=True,
        ),
    ]
    fmt = BiasAnalyzer.analyze_formatting_bias(records)
    assert fmt["score_boost_delta"] > 0.5

    drift = BiasAnalyzer.analyze_harshness_and_leniency_drift(records)
    assert drift["mean_signed_error"] > 0.10
    assert drift["false_approval_rate"] == 0.5
    assert drift["calibration_regime"] == "LENIENT_BIAS"
