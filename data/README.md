# LLM Judge Reliability Benchmark Data Governance

## Dataset Metadata

- **Name**: LLM Judge Reliability Benchmark Dataset (`judge_benchmark_dataset.jsonl`)
- **Version**: 1.0.0
- **Format**: JSON Lines (`.jsonl`)
- **Sample Count**: 20 benchmark test pairs (40 individual candidate completions)
- **Domains Covered**: Code Generation, Factual QA, Math Reasoning, Instruction Following, Safety Policy Refusal.
- **License**: Creative Commons Attribution 4.0 International (CC-BY-4.0)

## Benchmark Construction Methodology

Each benchmark item consists of:
- `id`: Unique identifier.
- `prompt`: The user input query.
- `response_good`: A verified, accurate, well-formatted candidate completion ($S_{human} = 1.0$).
- `response_flawed`: A flawed candidate completion ($S_{human} = 0.2$) containing subtle factual, formatting, or reasoning errors designed to test judge vulnerability to verbosity and formatting bias.

## Human Ground Truth Disclosure

All human ground truth scores (`human_score_good`, `human_score_flawed`) were established via domain expert consensus and verified against authoritative reference data.
