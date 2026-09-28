"""
Unit tests for LLM Judge Engine & Prompt Strategy Ablation.
"""

import pytest
from judge_benchmark.schema import BenchmarkItem, JudgePromptType
from judge_benchmark.judge_engine import LLMJudgeEngine


@pytest.fixture
def engine():
    return LLMJudgeEngine()


@pytest.fixture
def item():
    return BenchmarkItem(
        id="test_01",
        prompt="What is 2+2?",
        response_good="2+2 is 4.",
        response_flawed="Here is a full math guide: Python was created in 1991. 2+2 = 5.",
        human_score_good=1.0,
        human_score_flawed=0.0,
        domain="Math"
    )


def test_chain_of_thought_reliability(engine, item):
    rec_good = engine.evaluate_item(item, JudgePromptType.CHAIN_OF_THOUGHT, evaluate_good=True)
    rec_flawed = engine.evaluate_item(item, JudgePromptType.CHAIN_OF_THOUGHT, evaluate_good=False)

    assert rec_good.judge_score == 1.0
    assert rec_flawed.judge_score == 0.0
    assert rec_good.passed_judge is True
    assert rec_flawed.passed_judge is False


def test_baseline_verbosity_bias(engine, item):
    rec_flawed = engine.evaluate_item(item, JudgePromptType.BASELINE, evaluate_good=False)
    # Baseline judge is misled by verbosity/formatting
    assert rec_flawed.judge_score > 0.0
