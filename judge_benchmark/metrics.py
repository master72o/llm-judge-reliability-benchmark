"""
Statistical Metrics Module: Pearson r, Spearman rho, Kendall tau, Confusion Matrix.
"""

import numpy as np
import scipy.stats as stats
from typing import List, Dict, Any
from judge_benchmark.schema import JudgeEvaluationRecord, JudgePromptType


class JudgeMetricsCalculator:
    """Calculates correlation and reliability statistics for LLM judge benchmarks."""

    @staticmethod
    def calculate_prompt_strategy_metrics(records: List[JudgeEvaluationRecord]) -> Dict[str, Any]:
        if not records:
            return {"count": 0, "agreement_rate": 0.0}

        human_scores = [r.human_score for r in records]
        judge_scores = [r.judge_score for r in records]

        # Correlations
        r_val, _ = stats.pearsonr(human_scores, judge_scores) if len(human_scores) > 1 else (1.0, 0.0)
        rho_val, _ = stats.spearmanr(human_scores, judge_scores) if len(human_scores) > 1 else (1.0, 0.0)
        tau_val, _ = stats.kendalltau(human_scores, judge_scores) if len(human_scores) > 1 else (1.0, 0.0)

        # Classification metrics
        tp = sum(1 for r in records if r.passed_human and r.passed_judge)
        tn = sum(1 for r in records if not r.passed_human and not r.passed_judge)
        fp = sum(1 for r in records if not r.passed_human and r.passed_judge)  # False Positive (over-scoring bad response)
        fn = sum(1 for r in records if r.passed_human and not r.passed_judge)  # False Negative (under-scoring good response)

        total = len(records)
        agree_count = tp + tn
        agreement_rate = agree_count / total if total > 0 else 1.0

        fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fn_rate = fn / (fn + tp) if (fn + tp) > 0 else 0.0

        return {
            "total_evaluated": total,
            "pearson_r": round(float(r_val), 4),
            "spearman_rho": round(float(rho_val), 4),
            "kendall_tau": round(float(tau_val), 4),
            "agreement_rate": round(agreement_rate, 4),
            "true_positives": tp,
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
            "false_positive_rate": round(fp_rate, 4),
            "false_negative_rate": round(fn_rate, 4),
        }

    @classmethod
    def evaluate_benchmark_suite(cls, records: List[JudgeEvaluationRecord]) -> Dict[str, Any]:
        by_prompt: Dict[str, List[JudgeEvaluationRecord]] = {}
        for r in records:
            pt = r.prompt_type.value if hasattr(r.prompt_type, "value") else str(r.prompt_type)
            if pt not in by_prompt:
                by_prompt[pt] = []
            by_prompt[pt].append(r)

        suite_metrics = {}
        for pt, recs in by_prompt.items():
            suite_metrics[pt] = cls.calculate_prompt_strategy_metrics(recs)

        return suite_metrics
