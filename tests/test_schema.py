"""
Unit tests for judge benchmark data schemas.
"""

import pytest
from judge_benchmark.schema import JudgePromptType, BiasType, JudgeEvaluationRecord, BenchmarkItem


def test_judge_prompt_type_enum():
    assert JudgePromptType.BASELINE.value == "baseline"
    assert JudgePromptType.DETAILED_RUBRIC.value == "detailed_rubric"
    assert JudgePromptType.CHAIN_OF_THOUGHT.value == "chain_of_thought"


def test_judge_record_serialization():
    rec = JudgeEvaluationRecord(
        item_id="bm_01_good",
        prompt="What is 2+2?",
        response="4",
        human_score=1.0,
        judge_score=1.0,
        prompt_type=JudgePromptType.CHAIN_OF_THOUGHT,
        passed_human=True,
        passed_judge=True,
        reasoning="Correct"
    )
    d = rec.to_dict()
    assert d["item_id"] == "bm_01_good"
    assert d["prompt_type"] == "chain_of_thought"
    assert d["passed_human"] is True


def test_benchmark_item_deserialization():
    raw = {
        "id": "bm_1",
        "prompt": "Capital of France?",
        "response_good": "Paris",
        "response_flawed": "London",
        "domain": "QA"
    }
    item = BenchmarkItem.from_dict(raw)
    assert item.id == "bm_1"
    assert item.response_good == "Paris"
    assert item.response_flawed == "London"
