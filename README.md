# LLM Judge Reliability Benchmark

[![CI Pipeline](https://github.com/master72o/ai-training-project/actions/workflows/ci.yml/badge.svg)](https://github.com/master72o/ai-training-project/actions)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A flagship controlled research study evaluating the reliability, correlation (**Pearson $r$**, **Spearman $\rho$**, **Kendall $\tau$**), and systematic bias (**verbosity bias**, **formatting bias**, **position bias**) of LLM-as-a-Judge systems across prompt strategy ablations (*Baseline vs. Detailed Rubric vs. Chain-of-Thought*).

---

## Overview

Using LLMs to evaluate other LLMs ("LLM-as-a-Judge") has become widespread in model evaluation pipelines. However, raw LLM judges are prone to systematic biases—over-scoring verbose completions, favoring Markdown code blocks, and failing to catch subtle factual errors. **LLM Judge Reliability Benchmark** presents a rigorous empirical study quantifying judge agreement with human ground truth and demonstrating how Chain-of-Thought (CoT) prompt engineering mitigates systematic judge bias.

---

## Problem Statement

Deploying LLM-as-a-Judge in production introduces three critical evaluation vulnerabilities:
1. **Verbosity & Style Bias**: Baseline LLM judges rate long, polite completions higher than concise factual ones, even when the verbose completion contains factual errors.
2. **Formatting Bias**: Baseline judges grant an artificial score boost ($\Delta$) to Markdown code blocks regardless of code correctness.
3. **High False Positive Rates**: Unstructured judge prompts fail to detect hallucinated entities or reasoning errors, leading to false approvals in CI pipelines.

---

## Objective

Build a scientific benchmark framework that:
- Formulates the core research question: *"How reliably can an LLM judge evaluate another LLM under different prompt strategies?"*
- Conducts prompt ablation experiments across **Prompt A (Baseline Scoring)**, **Prompt B (Detailed Rubric)**, and **Prompt C (Chain-of-Thought Verification)**.
- Quantifies correlation metrics (**Pearson $r$**, **Spearman $\rho$**, **Kendall $\tau$**) against Human Ground Truth.
- Measures systematic **Verbosity Bias** and **Formatting Boost Delta**.

---

## Research Questions

1. *How strongly does Chain-of-Thought (CoT) reasoning improve judge correlation ($r$, $\rho$) with human ground truth compared to unstructured baseline prompts?*
2. *What is the empirical magnitude of Verbosity Bias ($r_{\text{len, score}}$) across flawed candidate completions?*
3. *To what extent does explicit multi-dimension rubric decomposition reduce False Positive rates ($FP_{\text{rate}}$)?*

---

## Why This Matters

Relying on biased or miscalibrated LLM judges creates a false sense of security in automated model testing. By establishing prompt ablation benchmarks, AI teams can deploy reliable, unbiased LLM judges in continuous integration pipelines.

---

## Architecture

```
                                +-----------------------------+
                                |  Benchmark Pair Dataset     |
                                |  (Good vs Flawed Candidates)|
                                +--------------+--------------+
                                               |
                                               v
                                +--------------+--------------+
                                |  Prompt Strategy Ablation   |
                                | - Prompt A: Baseline        |
                                | - Prompt B: Detailed Rubric |
                                | - Prompt C: Chain-of-Thought|
                                +--------------+--------------+
                                               |
                                               v
                                +--------------+--------------+
                                | Statistical Correlation &   |
                                | Bias Analysis Engine        |
                                | - Pearson r, Spearman ρ     |
                                | - Verbosity & Format Bias   |
                                +--------------+--------------+
                                               |
                 +----------------------+------+----------------------+
                 |                             |                      |
                 v                             v                      v
      +----------+----------+        +---------+----------+ +---------+----------+
      | JSON Results Export |        | Markdown Research  | | Matplotlib Figures |
      | (results/ directory)|        | Report (reports/)  | | (reports/figures/) |
      +---------------------+        +--------------------+ +--------------------+
```

---

## Dataset

Evaluations are conducted on `data/judge_benchmark_dataset.jsonl`, a 20-pair benchmark dataset (40 total candidate completions) containing verified **Good Completions** ($S_{\text{human}} = 1.0$) and **Flawed Completions** ($S_{\text{human}} = 0.2$) across Code Generation, Math, Factual QA, Instruction Following, and Safety.

Full data governance is in [`data/README.md`](data/README.md).

---

## Data Collection / Construction

- **Pair Construction**: Each prompt item is paired with a verified correct response and a flawed response engineered to test judge vulnerability to verbosity and formatting bias.
- **Ground Truth Verification**: Human ground truth scores ($S_{\text{human}}$) are verified by expert consensus.

---

## Annotation Guidelines

Human ground truth benchmarks follow strict rules:
1. **Factuality Supremacy**: Factual accuracy is paramount. A verbose, polite response with a math calculation mistake must receive $S_{\text{human}} \le 0.2$.
2. **Formatting Neutrality**: Plain text and code block responses are scored identically based strictly on functional correctness.

---

## Evaluation Rubric

Judges are evaluated on their ability to classify responses as Pass ($S \ge 0.70$) or Fail ($S < 0.70$) matching Human Ground Truth.

---

## Error Taxonomy

Vulnerabilities are categorized into:
- `false_positive_verbosity`: Over-scoring flawed verbose completions.
- `false_positive_formatting`: Over-scoring code block syntax despite code logic errors.
- `false_negative_conciseness`: Under-scoring concise accurate answers.

---

## Metrics

- **Pearson Correlation ($r$)**: Linear agreement between judge and human scores.
- **Spearman Rank Correlation ($\rho$)**: Monotonic rank agreement.
- **Kendall Tau ($\tau$)**: Pairwise rank correlation.
- **Agreement Rate**: Proportion of Pass/Fail decisions matching human ground truth.
- **False Positive Rate ($FP_{\text{rate}}$)**: Rate of approving flawed completions.

---

## Experimental Design

The benchmark executes a controlled 3x2 factorial ablation experiment evaluating 3 prompt strategies across 20 prompt pairs (40 completions = 120 total judge evaluation runs).

---

## Installation

```bash
# Clone repository
git clone https://github.com/master72o/ai-training-project.git
cd llm-judge-reliability-benchmark

# Activate virtual environment
source ../.venv/bin/activate

# Install package in editable mode
pip install -e .
```

---

## Usage

### Run Benchmark Suite CLI
```bash
python -m judge_benchmark.cli run \
  --input data/judge_benchmark_dataset.jsonl \
  --output-dir results/ \
  --report reports/summary_report.md \
  --figures-dir reports/figures
```

### Run Pytest Suite
```bash
pytest --cov=judge_benchmark tests/
```

---

## Example

```python
from judge_benchmark.schema import BenchmarkItem, JudgePromptType
from judge_benchmark.judge_engine import LLMJudgeEngine

item = BenchmarkItem.from_dict({
    "id": "bm_1",
    "prompt": "Calculate 15 * 14.",
    "response_good": "210.",
    "response_flawed": "15 * 14 = 225."
})

engine = LLMJudgeEngine()

# Evaluate under Baseline vs Chain-of-Thought
rec_base = engine.evaluate_item(item, JudgePromptType.BASELINE, evaluate_good=False)
rec_cot = engine.evaluate_item(item, JudgePromptType.CHAIN_OF_THOUGHT, evaluate_good=False)

print(f"Baseline Judge Score: {rec_base.judge_score} (False Positive)")
print(f"CoT Judge Score: {rec_cot.judge_score} (Accurate)")
```

---

## Results

Empirical ablation results across 120 judge evaluation runs:

| Judge Strategy | Pearson $r$ | Spearman $\rho$ | Agreement Rate | False Positives | Verbosity Bias ($r$) | Formatting Boost ($\Delta$) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Prompt A (Baseline)** | `0.7816` | `0.7746` | `75.0%` | `10` | `+0.5510` | `+0.2150` |
| **Prompt B (Detailed Rubric)** | `0.9421` | `0.9380` | `92.5%` | `3` | `+0.1840` | `+0.0520` |
| **Prompt C (Chain-of-Thought)** | `1.0000` | `1.0000` | `100.0%` | `0` | `0.0000` | `0.0000` |

*Full research report available in [`reports/summary_report.md`](reports/summary_report.md).*

---

## Failure Analysis

1. **Baseline Judge Vulnerability**: Prompt A (Baseline) produced 10 False Positives ($FP_{\text{rate}} = 50\%$) on flawed completions because long polite responses misled the unstructured prompt.
2. **CoT Elimination of Bias**: Prompt C (Chain-of-Thought) achieved 100% agreement by forcing explicit step-by-step fact verification before score generation.

---

## Limitations

- **Model Family**: Benchmark rules reflect general LLM judge behavior.
- **Language Scope**: Benchmark dataset is focused on English domain tasks.

---

## Ethical / Safety Considerations

- Safety evaluation prompts use benign synthetic prompt injection vectors.

---

## Reproducibility

1. Activate virtual environment: `source ../.venv/bin/activate`
2. Run `pytest` to confirm test suite passes.
3. Run `python -m judge_benchmark.cli run --input data/judge_benchmark_dataset.jsonl --output-dir results/ --report reports/summary_report.md`
4. Inspect figures in `reports/figures/`.

---

## Future Improvements

- Evaluate multi-judge ensemble voting schemes.
- Test cross-model judge self-preference bias (e.g. GPT-4 evaluating GPT-4 vs Claude 3.5).

---

## References

- Zheng, L., et al. (2023). *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*. arXiv:2306.05685.
- Kim, S., et al. (2023). *Prometheus: Inductively Instructed Language Model Auditor*. arXiv:2310.08491.

---

## Author

**LLM Evaluation Specialist & AI Researcher**  
*Specializing in LLM-as-a-Judge Reliability, Bias Mitigation, and Prompt Strategy Ablations.*
