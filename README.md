A public repository containing datasets and code for the paper [HateXScore: A Metric Suite for Evaluating Reasoning Quality in Hate Speech Explanations](https://arxiv.org/pdf/2601.13547) (EACL 2026).

HateXScore is a metric suite for evaluating the quality of model-generated explanations in hate speech detection. It focuses on whether an explanation clearly states a conclusion, quotes relevant evidence faithfully, identifies the targeted protected group, and remains logically consistent with the model prediction.

This repository provides a modular implementation of HateXScore with the following components:

- `HTC`: Hate-Type Check
- `QF`: Quotation Faithfulness
- `TGI`: Target-Group Identification
- `CC`: Consistency Check

The final `HateXScore` is computed from these four sub-metrics. By default, all metrics have equal weight, and the repository also supports configurable metric weights.

## Repository Structure

```text
hatexscore/
├── __init__.py
├── htc.py
├── qf.py
├── tgi.py
├── cc.py
└── utils.py
```

## What Each Module Does

- `hatexscore/htc.py`: conclusion detection logic and label extraction utilities
- `hatexscore/qf.py`: quotation overlap extraction, masking, probability estimation, and quotation faithfulness scoring
- `hatexscore/tgi.py`: protected-group matching with language-aware tokenization and lemmatization
- `hatexscore/cc.py`: consistency rule between prediction, QF, and TGI
- `hatexscore/utils.py`: evaluator class, CLI entrypoint, dataset loading, protected-group lists, and final score aggregation

## Features

- Supports English, Chinese, and Korean
- Supports multiple protected-group inventories
- Works on JSONL reasoning outputs paired with CSV datasets
- Keeps the original evaluation logic while exposing configurable final metric weights
- Designed to align with the HateXScore paper workflow

## Installation

Create and activate a Python environment first, then install the dependencies used in the current implementation.

```bash
pip install numpy pandas spacy jieba fuzzysearch openai konlpy
```

You may also need:

- a spaCy English model such as `en_core_web_sm`
- Java installed for `konlpy` in Korean settings

Example:

```bash
python -m spacy download en_core_web_sm
```

## Input Format

The current implementation expects:

1. A JSONL file containing generated reasoning outputs
2. A CSV file containing the original input text and gold labels

Each JSONL line should contain fields used by the script such as:

```json
{"ID": .., "text": "...", "raw": "...model explanation...", "label": "hateful", "flag": true}
```

The CSV schema depends on the dataset selected with `--dataset`. The script already contains dataset-specific column mappings for:

- `implicit`
- `hatexplain`
- `hatecheck`
- `toxicn`
- `hasoc`
- `kold`

## Usage

Because the code uses package-relative imports, run it from the parent directory of `hatexscore` with module mode:

```bash
python -m hatexscore.utils \
  --dataset hatexplain \
  --data_path /path/to/reason_output.json \
  --input_csv /path/to/input.csv \
  --output_dir /path/to/output_dir \
  --model gpt \
  --protected_group un_en \
  --lang en
```

## Configurable Metric Weights

The original code used a simple average across the four metrics. This repository keeps the same default behavior by setting all weights to `1.0`, but also lets you adjust the final aggregation.

Available arguments:

- `--weight_htc`
- `--weight_qf`
- `--weight_tgi`
- `--weight_cc`

Example:

```bash
python -m hatexscore.utils \
  --dataset hatexplain \
  --data_path /path/to/reason_output.json \
  --input_csv /path/to/input.csv \
  --output_dir /path/to/output_dir \
  --model gpt \
  --protected_group un_en \
  --lang en \
  --weight_htc 1.0 \
  --weight_qf 2.0 \
  --weight_tgi 1.0 \
  --weight_cc 1.0
```

The final score is computed as a weighted average:

```text
HateXScore =
  (HTC * w_htc + QF * w_qf + TGI * w_tgi + CC * w_cc)
  / (w_htc + w_qf + w_tgi + w_cc)
```

## Protected Group Inventories

The current implementation includes several built-in inventories:

- `facebook`
- `youtube`
- `un_en`
- `un_zh`
- `un_kr`

These are selected through `--protected_group`.

## Output

For each sample, the script writes a JSON object containing:

- input text
- reasoning
- gold label
- predicted label
- per-metric scores
- final `HateXScore`

The output file is written to:

```text
{output_dir}/{dataset}_metric_{model}.json
```

## Programmatic Use

You can also import the evaluator directly:

```python
from hatexscore import ReasoningMetricsEvaluator

weights = {
    "HTC": 1.0,
    "Quotation Faithfulness": 1.0,
    "TGI": 1.0,
    "Consistency Check": 1.0,
}

evaluator = ReasoningMetricsEvaluator(
    language="en",
    metric_weights=weights,
    runtime_args=args,
)
```

Then evaluate a sample:

```python
sample = {
    "text": "example text",
    "reasoning": "example explanation",
    "gold_label": "hateful",
    "prediction": "hateful",
}

result = evaluator.evaluate_sample(sample, target_group_list)
```

## Notes

- The current implementation preserves the original logic of the provided codebase as closely as possible.
- `QF` uses the OpenRouter-compatible `OpenAI` client configured in the source code.
- If you run `python utils.py` directly inside the `hatexscore/` directory, relative imports may fail. Use `python -m hatexscore.utils` instead.
- Some dependencies in the original larger script are not needed in this simplified modular repo.

## Citation

If you use this repository in academic work, please cite the HateXScore paper.

```bibtex
@article{hatexscore,
  title={HateXScore: A Metric Suite for Evaluating Reasoning Quality in Hate Speech Explanations},
  author={Hu, Yujia and Lee, Roy Ka-Wei}
}
```

## Disclaimer

This repository is intended for research use on hate speech detection and explanation evaluation. It may process sensitive or offensive text as part of the evaluation pipeline.
