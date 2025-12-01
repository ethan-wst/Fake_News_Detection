# LIAR: Fake News Detection

This repository contains a demo Jupyter notebook and Python code used for a fake news detection project in TXST CS4371 Computer System Security. The purpose of this project is to improve the performance and accuracy of the legacy code's fake news detection using the LIAR dataset and RoBERTa transformer models.

This work is based on the original LIAR dataset [paper](https://arxiv.org/abs/1705.00648) and code by William Yang Wang and focuses on extending the datatset's utility beyond non-machine learning baselines utilizing a transformer neural network. 

Guidance and inspiration were drawn from Hemang Thakar and Brijesh Bhatt [paper](https://www.researchsquare.com/article/rs-4721974/v1) discussing a binary classification approach using roBERTa model

Contributors: Andrew Sikes, Elham Islam, Ethan West, Trinity Boston


## Contents

- `demo.ipynb` — main demo notebook, showcasing data processing, inference, and ingress gate features
- `training.py` — script for training the RoBERTa transformer model with ordinal classification
- `liar_dataset/` — expected location for LIAR dataset TSV files (`train.tsv`, `valid.tsv`, `test.tsv`)
- `legacy_code/` — older scripts from original repo for reference
  - `classifier.py` — example classifier script
  - `prediction.py` — example prediction script
  - `front.py` — front end code
  - `final-fnd.py` — JSON

## Prerequisites

### Required packages
- pandas
- torch
- scikit-learn
- transformers
- ipywidgets
- numpy
- matplotlib
- seaborn
- plotly
- kaleido

### Installation
```bash
pip install -r requirements.txt
```

## Model Architecture

### RoBERTa Ordinal Classifier
The model uses a fine-tuned RoBERTa-base transformer with a custom classification head designed for ordinal classification across 6 truthfulness categories to accurately categorize news articles:

1. **pants-fire** (0) - Completely false
2. **false** (1) - Mostly false
3. **barely-true** (2) - Some truth, mostly false
4. **half-true** (3) - Mix of true and false
5. **mostly-true** (4) - Mostly accurate
6. **true** (5) - Completely accurate

### Key Features
- **Structured Input**: Statements are enriched with metadata using special separator tokens:
  ```
  [CLS] statement [SEP] subject [SEP] context [SEP]
  ```
- **Ordinal Classification**: Treats truthfulness as ordered categories rather than independent classes
- **Class Weighting**: Handles imbalanced dataset with weighted cross-entropy loss
- **Truthfulness Scoring**: Converts class probabilities to continuous scores (0.0 to 1.0) using expected value approach

### Truthfulness Score Calculation
The model outputs a continuous truthfulness score using:
```python
class_values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
score = sum(probability[i] * class_values[i] for all classes)
```

This provides a smooth, interpretable score that reflects both the prediction and model confidence.

## Running Inference

Use the provided `demo.ipynb` notebook for batch inference, pretrained [model](https://drive.google.com/drive/folders/1TZmK_MFVFX90dVSA_1U4gZ258-ll-z7p?usp=sharing) is available

### Inference Notebook Features
The `demo.ipynb` notebook provides:
- Batch processing of test dataset
- Detailed probability distributions for each prediction
- Statistical analysis of truthfulness scores by true label
- Multiple accuracy metrics (exact, ±1 class, ±2 classes)
- CSV export of predictions with scores

## Training the Model

### Quick Start
```bash
python training.py
```

### Training Configuration
- **Model**: `roberta-base` (124M parameters)
- **Training epochs**: 10 (with early stopping)
- **Batch size**: 16
- **Learning rate**: 2e-5 with linear warmup (10% of training)
- **Optimizer**: AdamW with weight decay (0.01)
- **Loss function**: Weighted cross-entropy
- **Dropout**: 0.3 for regularization

### Expected Performance
- **Exact accuracy**: 25-30% (6-way classification)
- **Accuracy ±1 class**: 60-70%
- **Accuracy ±2 classes**: 85-90%
- **Macro F1**: 0.25-0.35

### Model Output
The trained model saves to `best_liar_model.pt` and can generate:
1. **Class predictions**: Discrete label (0-5)
2. **Class probabilities**: Confidence across all 6 classes
3. **Truthfulness score**: Continuous value from 0.0 (false) to 1.0 (true)

## Running Inference

Use the provided `demo.ipynb` notebook for batch inference, pretrained [model](https://drive.google.com/drive/folders/1TZmK_MFVFX90dVSA_1U4gZ258-ll-z7p?usp=sharing) is available

### Inference Notebook Features
The `demo.ipynb` notebook provides:
- Batch processing of test dataset
- Detailed probability distributions for each prediction
- Statistical analysis of truthfulness scores by true label
- Multiple accuracy metrics (exact, ±1 class, ±2 classes)
- CSV export of predictions with scores

## Model Interpretability

### Understanding Predictions
Each prediction includes:
- **Truthfulness Score**: Overall confidence in truthfulness (0.0-1.0)
- **Class Probabilities**: Distribution showing model uncertainty
- **Predicted Label**: Most likely category

Example:
```
Statement: "Unemployment is at an all-time low"
Truthfulness Score: 0.68
Predicted Label: mostly-true

Class Probabilities:
  pants-fire    0.02  ▌
  false         0.05  █
  barely-true   0.12  ██
  half-true     0.18  ███
  mostly-true   0.45  █████████
  true          0.18  ███
```

This shows the model is fairly confident it's mostly true (0.68 score), with highest probability on "mostly-true" class but some uncertainty.

### Accuracy Metrics
The model provides multiple evaluation perspectives:
1. **Exact Match**: Did the model get the precise category?
2. **Within ±1 Class**: Is the prediction within one category? (e.g., predicting "false" instead of "pants-fire")
3. **Within ±2 Classes**: Is it reasonably close?
4. **Binary Accuracy**: Treating as true/false (score >= 0.5 threshold)

## Dataset Information

### LIAR Dataset
- **Training**: 10,269 statements
- **Validation**: 1,284 statements  
- **Test**: 1,267 statements
- **Source**: POLITIFACT fact-checking platform
- **Attributes used**: statement (text), subject (topic), context (where/when)

### Why These Attributes?
We focus on **statement**, **subject**, and **context** because:
- Consistent and available across all samples
- Reproducible in real-world scenarios
- Don't require external knowledge about specific speakers
- Excluded: Speaker name, job, state, party (too sparse/unique for meaningful patterns)


## Citation


```bibtex
@inproceedings{wang2017liar,
  title={"liar, liar pants on fire": A new benchmark dataset for fake news detection},
  author={Wang, William Yang},
  booktitle={Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)},
  pages={422--426},
  year={2017}
}
```
