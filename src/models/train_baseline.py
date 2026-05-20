import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
import seaborn as sns

import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline

def train_baseline_model(X, y, block_ids, test_size=0.2, random_state=42):
    """
    Train a baseline logistic regression model on event occurrence matrix.
    
    Args:
        X: Event occurrence matrix (bag of events)
        y: Labels (binary anomaly indicator)
        block_ids: Block identifiers for stratified split
        test_size: Proportion of data for testing
        random_state: Random seed
    
    Returns:
        Dictionary with model, metrics, and predictions
    """
    
    # Split by block_id to avoid data leakage
    unique_blocks = np.unique(block_ids)
    np.random.seed(random_state)
    np.random.shuffle(unique_blocks)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)  # keeps anomaly ratio similar

    # # Train logistic regression model
    # model = LogisticRegression(max_iter=1000, random_state=random_state)
    model = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
    ("clf", LogisticRegression(
        solver="saga",
        max_iter=200,
        class_weight="balanced",
        random_state=random_state
    ))
])
    
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Metrics
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    roc_auc = roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else float("nan")
    pr_auc = (
        average_precision_score(y_test, y_proba)
        if len(np.unique(y_test)) > 1
        else float("nan")
    )

    # Print results
    print("=" * 50)
    print("BASELINE LINEAR REGRESSION MODEL RESULTS")
    print("=" * 50)
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print(f"PR-AUC:    {pr_auc:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Anomaly"]))
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix - Baseline Model')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('outputs/confusion_matrix.png')
    plt.close()
    
    return {
        'model': model,
        'metrics': {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
        },
        'confusion_matrix': cm,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_proba': y_proba,
    }

import os
import json
import joblib
from pathlib import Path

def main():
    os.makedirs("outputs", exist_ok=True)

    # Load data
    anomaly_labels = pd.read_csv(
        "data/raw/HDFS_v1/preprocessed/anomaly_label.csv"
    )

    event_occurrence = pd.read_csv(
        "data/raw/HDFS_v1/preprocessed/event_occurrence_matrix.csv"
    )

    # Keep only event columns
    feature_cols = [c for c in event_occurrence.columns if c.startswith("E")]
    X = (
        event_occurrence[feature_cols]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
        .to_numpy(dtype=np.float32)
    )

    # Merge labels
    merged = pd.merge(
        event_occurrence[["BlockId"]],
        anomaly_labels,
        on="BlockId",
        how="inner"
    )

    y = (
        merged["Label"]
        .astype(str)
        .map({"Normal": 0, "Anomaly": 1})
        .to_numpy(dtype=np.int8)
    )

    results = train_baseline_model(X, y, None)

    # Save model
    joblib.dump(results["model"], "outputs/baseline_model.joblib")

    # Save metrics
    with open("outputs/metrics.json", "w") as f:
        json.dump(results["metrics"], f, indent=4)

    print("\nSaved:")
    print(" - outputs/baseline_model.joblib")
    print(" - outputs/metrics.json")
    print(" - outputs/confusion_matrix.png")

if __name__ == "__main__":
    main()

# if __name__ == '__main__':
#     # Load data
#     anomaly_labels = pd.read_csv('data/raw/HDFS_v1/preprocessed/anomaly_label.csv')
#     # event_occurrence also contains a "Label" column (success/fail) that is unrelated to the
#     # anomaly label in anomaly_labels.csv.  Drop it now to avoid duplicate column names after
#     # merging.  The only Label we want to keep is the one from anomaly_labels.
#     event_occurrence = pd.read_csv('data/raw/HDFS_v1/preprocessed/event_occurrence_matrix.csv')
#     if 'Label' in event_occurrence.columns:
#         event_occurrence = event_occurrence.drop(columns=['Label'])

#     # Merge datasets by BlockId to ensure alignment
#     merged = pd.merge(event_occurrence, anomaly_labels, on='BlockId', how='inner')

#     # After merging we should have a single "Label" column (anomaly indicator).  If pandas
#     # added suffixes for any reason, normalize the name here.
#     if 'Label_y' in merged.columns:
#         merged.rename(columns={'Label_y': 'Label'}, inplace=True)
#     if 'Label_x' in merged.columns:
#         merged.drop(columns=['Label_x'], inplace=True)

#     # Extract features and labels
#     X = merged.drop(columns=['BlockId', 'Label']).values
#     # y is currently strings like "Normal" / "Anomaly", convert to binary 0/1
#     y = merged["Label"].astype(str).map({"Normal": 0, "Anomaly": 1}).to_numpy(dtype=np.int8)
#     block_ids = merged['BlockId'].values

#     # Train baseline model
#     results = train_baseline_model(X, y, block_ids)
    