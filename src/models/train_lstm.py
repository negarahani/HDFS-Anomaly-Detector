"""
Train an LSTM classifier on HDFS event traces.

Each block is represented as a sequence of event template ids (E1, E2, ...).
The model learns whether the block execution ended in Success or Fail.
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from src.data.parse import extract_features_from_traces


# ---------------------------------------------------------------------------
# PyTorch dataset: wraps padded integer sequences and binary labels
# ---------------------------------------------------------------------------
class EventSequenceDataset(Dataset):
    """One sample = one block's event sequence, its label, and precomputed length."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        # X: (n_samples, seq_len) int array; 0 = padding, 1..vocab_size = events
        self.X = torch.from_numpy(X).long()
        self.y = torch.from_numpy(y).long()
        self.lengths = torch.from_numpy(sequence_lengths(X)).long()

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx], self.lengths[idx]


# ---------------------------------------------------------------------------
# Model: embedding -> bidirectional LSTM -> linear classifier
# ---------------------------------------------------------------------------
class HDFS_LSTM(nn.Module):
    """
    Sequence classifier for HDFS block execution traces.

    Input:  batch of padded event-id sequences
    Output: logits over {Success, Fail}
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 1,
        num_classes: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()

        # Map integer event ids to dense vectors; index 0 is reserved for padding
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size + 1,
            embedding_dim=embed_dim,
            padding_idx=0,
        )

        # Bidirectional LSTM reads the sequence in both directions
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(dropout)

        # * 2 because forward and backward final hidden states are concatenated
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len)
        embedded = self.embedding(x)

        # pack_padded_sequence skips padding tokens so the LSTM only
        # processes real events, which is more efficient and correct
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (hidden, _) = self.lstm(packed)

        # hidden shape: (num_layers * 2, batch, hidden_dim)
        # take the last layer's forward and backward states and concat them
        hidden = hidden.view(self.lstm.num_layers, 2, x.size(0), self.lstm.hidden_size)
        hidden = torch.cat((hidden[-1, 0], hidden[-1, 1]), dim=1)

        return self.fc(self.dropout(hidden))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def sequence_lengths(X: np.ndarray) -> np.ndarray:
    """
    Count non-padding events per sequence.

    Sequences are right-padded with 0, so the number of non-zero entries
    equals the true length of each trace.
    """
    return (X > 0).sum(axis=1).astype(np.int64)


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------
def train_lstm_model(
    X: np.ndarray,
    y: np.ndarray,
    block_ids: np.ndarray | None = None,
    vocab_size: int | None = None,
    max_len: int | None = 100,
    test_size: float = 0.2,
    batch_size: int = 256,
    epochs: int = 5,
    lr: float = 1e-3,
    random_state: int = 42,
):
    """
    Train and evaluate an LSTM on padded event sequences.

    Args:
        X:          (n_samples, seq_len) integer sequences
        y:          binary labels — 0 = Success, 1 = Fail
        block_ids:  reserved for future block-level train/test splits
        vocab_size: number of distinct event templates (inferred from X if None)
        max_len:    truncate sequences to this length (None keeps full length)
        test_size:  fraction of data held out for evaluation
        batch_size: mini-batch size for DataLoader
        epochs:     number of full passes over the training set
        lr:         Adam learning rate
        random_state: seed for reproducible train/test split

    Returns:
        dict with trained model, metrics, predictions, and metadata
    """
    del block_ids  # not used yet; kept for block-level splits later

    # --- Optional truncation ------------------------------------------------
    # Long traces are cut to max_len to reduce memory and training time.
    # Most event information tends to appear early in the sequence.
    if max_len is not None:
        X = X[:, :max_len]

    if vocab_size is None:
        vocab_size = int(X.max())

    # --- Device selection ---------------------------------------------------
    # Prefer Apple GPU (MPS) on Mac, then NVIDIA CUDA, otherwise CPU
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    # --- Train / test split -------------------------------------------------
    # stratify=y keeps the Success/Fail ratio similar in both splits,
    # which matters because Fail blocks are much rarer than Success
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # --- DataLoaders --------------------------------------------------------
    train_ds = EventSequenceDataset(X_train, y_train)
    test_ds = EventSequenceDataset(X_test, y_test)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # --- Loss with class weights --------------------------------------------
    # Fail blocks are underrepresented; up-weight them so the model
    # does not collapse to always predicting Success
    class_counts = np.bincount(y_train, minlength=2).astype(np.float32)
    class_weights = class_counts.sum() / (2.0 * np.maximum(class_counts, 1.0))
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=device)
    )

    # --- Model and optimizer ------------------------------------------------
    model = HDFS_LSTM(vocab_size=vocab_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # --- Training loop ------------------------------------------------------
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for batch_x, batch_y, batch_lengths in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            lengths = batch_lengths.to(device)

            optimizer.zero_grad()
            logits = model(batch_x, lengths)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(batch_y)

        avg_loss = total_loss / len(train_ds)
        print(f"Epoch {epoch}/{epochs} — train loss: {avg_loss:.4f}")

    # --- Evaluation on held-out test set ------------------------------------
    model.eval()
    all_preds, all_probs, all_labels = [], [], []

    with torch.no_grad():
        for batch_x, batch_y, batch_lengths in test_loader:
            batch_x = batch_x.to(device)
            lengths = batch_lengths.to(device)
            logits = model(batch_x, lengths)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_probs.append(probs)
            all_labels.append(batch_y.numpy())

    y_pred = np.concatenate(all_preds)
    y_proba = np.concatenate(all_probs)
    y_test_np = np.concatenate(all_labels)

    # --- Metrics ------------------------------------------------------------
    precision = precision_score(y_test_np, y_pred, zero_division=0)
    recall = recall_score(y_test_np, y_pred, zero_division=0)
    f1 = f1_score(y_test_np, y_pred, zero_division=0)
    cm = confusion_matrix(y_test_np, y_pred)

    # Threshold-free metrics using P(Fail); undefined if test set has one class only
    roc_auc = roc_auc_score(y_test_np, y_proba) if len(np.unique(y_test_np)) > 1 else float("nan")
    pr_auc = (
        average_precision_score(y_test_np, y_proba)
        if len(np.unique(y_test_np)) > 1
        else float("nan")
    )

    print("=" * 50)
    print("LSTM MODEL RESULTS")
    print("=" * 50)
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print(f"PR-AUC:    {pr_auc:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_test_np, y_pred, target_names=["Success", "Fail"]))

    # --- Save confusion matrix plot -----------------------------------------
    os.makedirs("outputs", exist_ok=True)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title("Confusion Matrix - LSTM Model")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig("outputs/confusion_matrix_lstm.png")
    plt.close()

    return {
        "model": model,
        "vocab_size": vocab_size,
        "max_len": max_len,
        "metrics": {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
        },
        "confusion_matrix": cm,
        "y_test": y_test_np,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "device": str(device),
    }


# ---------------------------------------------------------------------------
# Entry point: load data, train, persist artifacts
# ---------------------------------------------------------------------------
def main():
    os.makedirs("outputs", exist_ok=True)

    # --- Load preprocessed event traces -------------------------------------
    # Each row is one HDFS block with its event sequence and outcome label
    event_traces = pd.read_csv(
        "data/raw/HDFS_v1/preprocessed/event_traces.csv"
    )
    event_traces = event_traces[["BlockId", "Label", "Features"]]

    # --- Feature extraction -------------------------------------------------
    # Parse Features strings like "[E5,E22,E5]" into padded integer matrices
    X, event_vocab = extract_features_from_traces(event_traces)

    # Map string labels to binary targets for classification
    y = event_traces["Label"].map({"Success": 0, "Fail": 1}).to_numpy(dtype=np.int8)
    block_ids = event_traces["BlockId"].to_numpy()

    print(f"X shape: {X.shape}, vocab size: {len(event_vocab)}")
    print(f"labels: Success={int((y == 0).sum())}, Fail={int((y == 1).sum())}")

    # --- Train and evaluate -------------------------------------------------
    results = train_lstm_model(X, y, block_ids, vocab_size=len(event_vocab))

    # --- Persist model checkpoint and metrics -------------------------------
    # Checkpoint stores weights plus metadata needed to reload the model
    checkpoint = {
        "state_dict": results["model"].state_dict(),
        "vocab_size": results["vocab_size"],
        "max_len": results["max_len"],
        "event_vocab": event_vocab,
    }
    torch.save(checkpoint, "outputs/lstm_model.pt")

    with open("outputs/lstm_metrics.json", "w") as f:
        json.dump(results["metrics"], f, indent=4)

    print("\nSaved:")
    print(" - outputs/lstm_model.pt")
    print(" - outputs/lstm_metrics.json")
    print(" - outputs/confusion_matrix_lstm.png")


if __name__ == "__main__":
    main()
