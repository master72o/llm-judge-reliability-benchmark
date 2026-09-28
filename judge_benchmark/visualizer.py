"""
Visualization Generator for LLM Judge Reliability Benchmark.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any


class Visualizer:
    """Generates figures for LLM judge reliability research."""

    @staticmethod
    def generate_all_figures(suite_metrics: Dict[str, Any], bias_results: Dict[str, Any], output_dir: str = "reports/figures") -> Dict[str, str]:
        os.makedirs(output_dir, exist_ok=True)
        generated = {}

        # 1. Prompt Ablation Comparison Plot
        abl_path = os.path.join(output_dir, "prompt_ablation_comparison.png")
        Visualizer._plot_prompt_ablation(suite_metrics, abl_path)
        generated["prompt_ablation_comparison"] = abl_path

        # 2. Bias Analysis Plot
        bias_path = os.path.join(output_dir, "bias_analysis.png")
        Visualizer._plot_bias_analysis(bias_results, bias_path)
        generated["bias_analysis"] = bias_path

        # 3. Confusion Matrix Plot
        conf_path = os.path.join(output_dir, "judge_confusion_matrix.png")
        Visualizer._plot_confusion_matrix(suite_metrics, conf_path)
        generated["judge_confusion_matrix"] = conf_path

        return generated

    @staticmethod
    def _plot_prompt_ablation(metrics: Dict[str, Any], output_path: str):
        fig, ax = plt.subplots(figsize=(10, 5))
        prompts = ["baseline", "detailed_rubric", "chain_of_thought"]
        labels = ["Prompt A\n(Baseline)", "Prompt B\n(Detailed Rubric)", "Prompt C\n(Chain-of-Thought)"]

        pearsons = [metrics.get(p, {}).get("pearson_r", 0.0) for p in prompts]
        spearmans = [metrics.get(p, {}).get("spearman_rho", 0.0) for p in prompts]
        agreements = [metrics.get(p, {}).get("agreement_rate", 0.0) for p in prompts]

        x = np.arange(len(labels))
        width = 0.25

        rects1 = ax.bar(x - width, pearsons, width, label="Pearson Correlation (r)", color="#2b5c8f")
        rects2 = ax.bar(x, spearmans, width, label="Spearman Rank (ρ)", color="#5bc0de")
        rects3 = ax.bar(x + width, agreements, width, label="Pass/Fail Agreement Rate", color="#5cb85c")

        ax.set_ylabel("Metric Value (0.0 to 1.0)", fontsize=11, fontweight="bold")
        ax.set_title("LLM Judge Reliability Across Prompt Ablation Strategies", fontsize=13, fontweight="bold", pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(0.0, 1.15)
        ax.axhline(y=0.90, color="#d9534f", linestyle="--", linewidth=1.2, label="High Reliability Benchmark (0.90)")
        ax.legend(loc="lower right")
        ax.grid(axis="y", linestyle=":", alpha=0.6)

        for rect in rects1:
            h = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., h + 0.02, f"{h:.2f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
        for rect in rects2:
            h = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., h + 0.02, f"{h:.2f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
        for rect in rects3:
            h = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., h + 0.02, f"{h:.2f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

        plt.tight_layout()
        plt.savefig(output_path, dpi=200)
        plt.close()

    @staticmethod
    def _plot_bias_analysis(bias_results: Dict[str, Any], output_path: str):
        fig, ax = plt.subplots(figsize=(9, 5))
        prompts = ["baseline", "detailed_rubric", "chain_of_thought"]
        labels = ["Prompt A (Baseline)", "Prompt B (Rubric)", "Prompt C (CoT)"]

        v_biases = [bias_results.get(p, {}).get("verbosity_bias", {}).get("pearson_r", 0.0) for p in prompts]
        f_boosts = [bias_results.get(p, {}).get("formatting_bias", {}).get("score_boost_delta", 0.0) for p in prompts]

        x = np.arange(len(labels))
        width = 0.35

        rects1 = ax.bar(x - width/2, v_biases, width, label="Verbosity Bias (Pearson r)", color="#d9534f")
        rects2 = ax.bar(x + width/2, f_boosts, width, label="Formatting Boost Delta (Δ)", color="#f0ad4e")

        ax.set_ylabel("Bias Score / Boost Delta", fontsize=11, fontweight="bold")
        ax.set_title("Judge Systematic Bias Breakdown by Prompt Strategy", fontsize=13, fontweight="bold", pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(-0.1, 0.8)
        ax.axhline(y=0.0, color="#333333", linestyle="-", linewidth=1.0)
        ax.legend()
        ax.grid(axis="y", linestyle=":", alpha=0.6)

        for rect in rects1:
            h = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., h + 0.02, f"{h:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        for rect in rects2:
            h = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., h + 0.02, f"{h:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

        plt.tight_layout()
        plt.savefig(output_path, dpi=200)
        plt.close()

    @staticmethod
    def _plot_confusion_matrix(metrics: Dict[str, Any], output_path: str):
        fig, ax = plt.subplots(figsize=(7, 5))
        
        # Take Baseline Prompt A metrics for confusion matrix demo
        m = metrics.get("baseline", {})
        tp = m.get("true_positives", 0)
        fp = m.get("false_positives", 0)
        fn = m.get("false_negatives", 0)
        tn = m.get("true_negatives", 0)

        matrix = np.array([[tp, fp], [fn, tn]])
        im = ax.imshow(matrix, cmap="Blues")

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Judge Pass", "Judge Fail"])
        ax.set_yticklabels(["Human Pass", "Human Fail"])
        ax.set_title("Prompt A Baseline Judge Confusion Matrix", fontsize=13, fontweight="bold", pad=15)

        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{matrix[i, j]}", ha="center", va="center", color="black", fontsize=14, fontweight="bold")

        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.savefig(output_path, dpi=200)
        plt.close()
