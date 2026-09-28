"""
Data Schemas for LLM Judge Reliability Benchmark.
"""

from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List


class JudgePromptType(str, Enum):
    BASELINE = "baseline"
    DETAILED_RUBRIC = "detailed_rubric"
    CHAIN_OF_THOUGHT = "chain_of_thought"


class BiasType(str, Enum):
    VERBOSITY = "verbosity_bias"
    FORMATTING = "formatting_bias"
    POSITION = "position_bias"
    STYLE = "style_bias"


@dataclass
class JudgeEvaluationRecord:
    item_id: str
    prompt: str
    response: str
    human_score: float  # 0.0 to 1.0
    judge_score: float  # 0.0 to 1.0
    prompt_type: JudgePromptType
    passed_human: bool
    passed_judge: bool
    reasoning: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "item_id": self.item_id,
            "prompt": self.prompt,
            "response": self.response,
            "human_score": round(self.human_score, 4),
            "judge_score": round(self.judge_score, 4),
            "prompt_type": self.prompt_type.value if isinstance(self.prompt_type, JudgePromptType) else self.prompt_type,
            "passed_human": self.passed_human,
            "passed_judge": self.passed_judge,
            "reasoning": self.reasoning,
        }


@dataclass
class BenchmarkItem:
    id: str
    prompt: str
    response_good: str
    response_flawed: str
    human_score_good: float = 1.0
    human_score_flawed: float = 0.2
    domain: str = "general"
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BenchmarkItem":
        return cls(
            id=data["id"],
            prompt=data["prompt"],
            response_good=data["response_good"],
            response_flawed=data["response_flawed"],
            human_score_good=data.get("human_score_good", 1.0),
            human_score_flawed=data.get("human_score_flawed", 0.2),
            domain=data.get("domain", "general"),
            metadata=data.get("metadata", {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "prompt": self.prompt,
            "response_good": self.response_good,
            "response_flawed": self.response_flawed,
            "human_score_good": self.human_score_good,
            "human_score_flawed": self.human_score_flawed,
            "domain": self.domain,
            "metadata": self.metadata,
        }
