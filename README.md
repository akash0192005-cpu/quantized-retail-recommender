# Quantized Retail Recommender

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/sr9f4ajnrd/badge.svg)](https://ask.thili.ai/b/sr9f4ajnrd)

This project develops a smart retail recommender that learns from shopping basket data to suggest frequently bought items. You will then apply quantization techniques to significantly reduce the model's size, making it suitable for deployment on resource-constrained devices or for faster loading times.

## What you are building

**Goal.** You will build a smart retail recommender that identifies frequently bought items, then shrink its model using quantization to reduce its footprint for edge device deployment or faster loading.

**Done when.** You'll know you're done when you have trained a language model on shopping baskets for next-item prediction, applied post-training quantization to the model's weights, and demonstrated a smaller, functional recommendation model.

## Milestones

Tick these off as you go, and commit after each one.

- [ ] **1. Prepare synthetic shopping basket data for training**  
      Hint: Use data generation techniques to create your synthetic shopping basket dataset.
- [ ] **2. Continue pretraining a small LM on the basket data**  
      Hint: Apply language model pretraining techniques from 'Teach a Language Model What Sells Together' to your basket data.
- [ ] **3. Implement next-item prediction and itemset probing**  
      Hint: Utilize the trained LM for next-item prediction and itemset recovery as shown in 'Teach a Language Model What Sells Together'.
- [ ] **4. Measure the baseline size and accuracy of the model**  
      Hint: Calculate the model's memory footprint and evaluate its recommendation accuracy before quantization.
- [ ] **5. Apply dynamic and static post-training quantization to the model**  
      Hint: Implement post-training quantization methods, dynamic and static, as covered in 'Shrink a Neural Network with Quantisation'.
- [ ] **6. Demonstrate size reduction and evaluate accuracy impact**  
      Hint: Compare the size and accuracy of your quantized models against the baseline.

## Run it

You need Python 3.12 and git.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then choose a model; see the comments inside
streamlit run app/main.py
pytest
```

Your work goes in `app/core.py` (one stub per milestone) and `tests/test_core.py`. The first tests pass already; they check the plumbing.

## Settings and secrets

Model settings live in `.env`, which is ignored by git. **Never commit `.env` or paste a key into code.** If a key ever reaches GitHub, treat it as leaked and make a new one.

## Stretch ideas

- Explore further quantization methods beyond dynamic and static, such as quantization-aware training.
- Analyze the performance impact of the quantized model in terms of inference speed or memory usage on a simulated edge device.
