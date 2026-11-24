# Quantized Retail Recommender

A reproducible retail recommendation demo: generate synthetic shopping baskets, train a compact next-item predictor, probe frequent item pairs, and compare FP32, dynamic INT8, and static INT8 post-training quantization.

## Features
- Synthetic transactions with reproducible co-purchase patterns.
- Compact PyTorch next-item prediction from basket prefixes.
- Top-k recommendations that exclude items already in the basket.
- Frequent item-pair probing using transaction support.
- Baseline model size and held-out next-item accuracy.
- Dynamic and static post-training INT8 quantization.
- Streamlit dashboard, pytest suite, Dockerfile, and GitHub Actions CI.

## Quick start
Python 3.12 is recommended.
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
streamlit run app/main.py
```
Run automated tests with `pytest -q`.

## Method
1. Generate synthetic baskets from common shopping themes.
2. Convert each basket into prefix-context / next-item training examples.
3. Train a small feed-forward PyTorch predictor.
4. Evaluate the baseline and apply dynamic and calibrated static INT8 quantization.
5. Compare serialized state-dict sizes and next-item accuracy.
6. Rank frequent item pairs by support.

## Project layout
```text
app/core.py       data generation, training, evaluation, quantization
app/main.py       Streamlit dashboard
tests/            automated tests
.github/workflows/ci.yml
Dockerfile
requirements.txt
```

## Limitations
The data is synthetic and intended as an educational baseline, not production demand forecasting. The model is a compact next-item predictor over basket-prefix features, not a pretrained general-purpose language model. For real use, replace synthetic transactions with properly licensed data and evaluate on a chronological holdout. Quantized inference support varies by PyTorch version and CPU backend.

## Security
No API keys are needed. Never commit secrets, customer transaction data, or personally identifiable information.
