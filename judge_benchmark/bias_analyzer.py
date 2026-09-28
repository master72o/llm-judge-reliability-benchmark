"""
Bias Analysis Engine investigating Verbosity, Formatting, Position, and Style Biases in LLM Judges.
"""

import scipy.stats as stats
import numpy as np
from typing import List, Dict, Any
from judge_benchmark.schema import JudgeEvaluationRecord, JudgePromptType


class BiasAnalyzer:
    """Analyzes systematic biases in LLM judges across prompt strategies."""

    @staticmethod
    def analyze_verbosity_bias(records: List[JudgeEvaluationRecord]) -> Dict[str, Any]:
        """Measure correlation between word count and judge score for flawed responses."""
        flawed_records = [r for r in records if r.human_score < 0.5]
        if not flawed_records:
            return {"correlation": 0.0, "p_value": 1.0}

        word_counts = [len(r.response.split()) for r in flawed_records]
        judge_scores = [r.judge_score for r in flawed_records]

        r_val, p_val = stats.pearsonr(word_counts, judge_scores) if len(word_counts) > 1 else (0.0, 1.0)
        return {
            "pearson_r": round(float(r_val), 4),
            "p_value": round(float(p_val), 4),
            "bias_severity": "High" if r_val > 0.4 else "Moderate" if r_val > 0.2 else "Low"
        }

    @staticmethod
    def analyze_formatting_bias(records: List[JudgeEvaluationRecord]) -> Dict[str, Any]:
        """Compare mean score boost for responses containing markdown code blocks."""
        formatted = [r.judge_score for r in records if "```" in r.response]
        unformatted = [r.judge_score for r in records if "```" not in r.response]

        mean_fmt = float(np.mean(formatted)) if formatted else 0.0
        mean_unfmt = float(np.mean(unformatted)) if unformatted else 0.0
        delta = mean_fmt - mean_unfmt

        return {
            "mean_formatted_score": round(mean_fmt, 4),
            "mean_unformatted_score": round(mean_unfmt, 4),
            "score_boost_delta": round(delta, 4),
        }

    @staticmethod
    def calculate_bias_suite(records: List[JudgeEvaluationRecord]) -> Dict[str, Any]:
        by_prompt: Dict[str, List[JudgeEvaluationRecord]] = {}
        for r in records:
            pt = r.prompt_type.value if hasattr(r.prompt_type, "value") else str(r.prompt_type)
            if pt not in by_prompt:
                by_prompt[pt] = []
            by_prompt[pt].append(r)

        suite_results = {}
        for pt, recs in by_prompt.items():
            suite_results[pt] = {
                "verbosity_bias": BiasAnalyzer.analyze_verbosity_bias(recs),
                "formatting_bias": BiasAnalyzer.analyze_formatting_bias(recs),
            }

        return suite_results
