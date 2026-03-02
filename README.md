# HDFS Log Anomaly Detection – Baseline Model

## Overview

This project implements a baseline anomaly detection model for distributed system logs using the HDFS dataset.

Each block in HDFS generates a sequence of system events.  
The goal is to predict whether a block execution is **Normal** or **Anomalous** based on event occurrence features.

This baseline uses a **bag-of-events representation** (event counts per block) and a **logistic regression classifier**.

---

## Dataset

Source: LogPAI HDFS Log Anomaly Dataset

This project uses the preprocessed files:

- `event_occurrence_matrix.csv`
- `anomaly_label.csv`

Each row corresponds to one BlockId.

### Features
- E1–E29: Count of each event template within a block

### Label
- Normal → 0
- Anomaly → 1

---

## Model

Baseline model:
- Logistic Regression
- Balanced class weights
- Train/test split with stratification

Evaluation metrics:
- Precision
- Recall
- F1 Score
- Confusion Matrix

---


## Setup

Create virtual environment:

``` bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---



## Reproduce Baseline

Run:

``` bash
python -m src.models.train_baseline
```

---

## Outputs

After running the baseline:

-   `outputs/baseline_model.joblib` -- Trained model artifact
-   `outputs/metrics.json` -- Evaluation metrics
-   `outputs/confusion_matrix.png` -- Visualization of predictions


---

## Notes

-   Raw datasets are not included in this repository.
-   Download HDFS dataset separately from LogPAI.