# HDFS Log Anomaly Detection

## Overview

This project implements machine learning models for anomaly detection in distributed system logs using the HDFS dataset.

Each HDFS block generates an ordered sequence of system events. The goal is to predict whether a block execution is **Normal/Success** or **Anomalous/Fail**.

The project compares two approaches:

1. **Logistic Regression baseline** using a bag-of-events representation
2. **LSTM sequence model** using ordered event traces

The baseline uses event counts only, while the LSTM model learns from the order of events in each block execution.

---

## Dataset

Source: LogPAI HDFS Log Anomaly Dataset

This project uses the preprocessed files:

- `event_occurrence_matrix.csv`
- `anomaly_label.csv`
- `event_traces.csv`

Each row corresponds to one `BlockId`.

### Baseline Features

From `event_occurrence_matrix.csv`:

- `E1–E29`: Count of each event template within a block

This representation ignores event order and treats each block as a bag of event counts.

### LSTM Features

From `event_traces.csv`:

- `Features`: Ordered event sequence for each block

Example:

```text
[E5, E22, E5, E5, E11, E9, E26, ...]
```

This representation preserves the order of events and enables sequence modeling.

### Labels

Depending on the file:

- `Normal` / `Success` → 0
- `Anomaly` / `Fail` → 1

---

## Models

### 1. Logistic Regression Baseline

The baseline model uses a bag-of-events representation.

Pipeline:

```text
event counts per BlockId → Logistic Regression → Normal/Anomaly
```

Model details:

- Logistic Regression
- Balanced class weights
- Stratified train/test split

This model answers:

> Can event frequency alone detect anomalous HDFS blocks?

---

### 2. LSTM Sequence Model

The LSTM model uses ordered event sequences.

Pipeline:

```text
event sequence → embedding layer → bidirectional LSTM → linear classifier → Success/Fail
```

Model details:

- Event embedding layer
- Bidirectional LSTM
- Dropout
- Linear classification layer
- Class-weighted cross entropy loss

This model answers:

> Does event order provide useful signal beyond event counts?

---

## Evaluation Metrics

The models are evaluated using:

- Precision
- Recall
- F1 Score
- ROC-AUC
- PR-AUC
- Confusion Matrix

Because anomaly detection is an imbalanced classification problem, **precision, recall, F1, and PR-AUC** are especially important.

---

## Results

### Baseline Model

| Metric | Score |
|---|---:|
| Precision | 0.9012 |
| Recall | 0.9988 |
| F1 Score | 0.9475 |
| ROC-AUC | 0.9987 |
| PR-AUC | 0.9079 |

### LSTM Model

| Metric | Score |
|---|---:|
| Precision | 0.9615 |
| Recall | 1.0000 |
| F1 Score | 0.9804 |
| ROC-AUC | 0.9999 |
| PR-AUC | 0.9963 |

### Interpretation

The baseline model already performs strongly, showing that event counts contain useful anomaly-detection signal.

The LSTM model improves performance, especially:

- Precision: 0.9012 → 0.9615
- F1 Score: 0.9475 → 0.9804
- PR-AUC: 0.9079 → 0.9963

This suggests that event ordering provides additional useful information beyond simple event counts.

In practical anomaly detection terms, the LSTM model catches nearly all failures while producing fewer false alarms.

There is a big gap between PR-AUC between baseline and LSTM model, meaning that the LSTM's probability scores are much better calibrated on the imbalanced classes.

---

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Reproduce Baseline

Run:

```bash
python -m src.models.train_baseline
```

This trains the logistic regression baseline using the event occurrence matrix.

---

## Train LSTM Model

Run:

```bash
python -m src.models.train_lstm
```

This trains the LSTM model using ordered event traces.

---

## Outputs

After running the baseline:

- `outputs/baseline_model.joblib` — Trained baseline model
- `outputs/metrics.json` — Baseline evaluation metrics
- `outputs/confusion_matrix.png` — Baseline confusion matrix

After running the LSTM model:

- `outputs/lstm_model.pt` — Trained LSTM checkpoint
- `outputs/lstm_metrics.json` — LSTM evaluation metrics
- `outputs/confusion_matrix_lstm.png` — LSTM confusion matrix

---

## Project Structure

```text
hdfs-anomaly-detector/
│
├── src/
│   ├── data/
│   │   └── parse.py
│   │
│   └── models/
│       ├── train_baseline.py
│       └── train_lstm.py
│
├── data/
│   └── raw/              # not tracked in git
│
├── outputs/              # not tracked in git
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Notes

- Raw datasets are not included in this repository.
- Download the HDFS dataset separately from LogPAI.
- Large files such as raw logs, CSV datasets, model artifacts, and output plots should not be committed to GitHub.
- The baseline is useful as a simple comparison point.
- The LSTM model is the main sequence-modeling component of this project.

---

## Next Steps

Possible extensions:

- Add Spark-based preprocessing from raw `HDFS.log`
- Track experiments with MLflow
- Add FastAPI inference endpoint
- Add Docker support
- Simulate streaming log ingestion with Kafka
