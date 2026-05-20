import re

import numpy as np
import pandas as pd

_EVENT_PATTERN = re.compile(r"E(\d+)")


def parse_feature_string(feature_str: str) -> list[str]:
    """Parse a Features cell like '[E5,E22,E5]' into a list of event ids."""
    s = feature_str.strip()
    if s.startswith("["):
        s = s[1:]
    if s.endswith("]"):
        s = s[:-1]
    return [part.strip() for part in s.split(",") if part.strip()]


def build_event_vocab(event_ids: list[str]) -> dict[str, int]:
    """Map event ids (E1, E2, ...) to contiguous integer indices starting at 1."""
    unique = sorted(set(event_ids), key=lambda e: int(_EVENT_PATTERN.match(e).group(1)))
    return {event_id: idx + 1 for idx, event_id in enumerate(unique)}


def encode_sequences(sequences: list[list[str]], vocab: dict[str, int]) -> list[list[int]]:
    return [[vocab[event_id] for event_id in seq] for seq in sequences]


def pad_sequences(sequences: list[list[int]], max_len: int | None = None) -> np.ndarray:
    if max_len is None:
        max_len = max(len(seq) for seq in sequences)
    padded = np.zeros((len(sequences), max_len), dtype=np.int32)
    for i, seq in enumerate(sequences):
        padded[i, : len(seq)] = seq
    return padded


def extract_features_from_traces(
    event_traces: pd.DataFrame,
    feature_col: str = "Features",
) -> tuple[np.ndarray, dict[str, int]]:
    """
    Extract and encode event sequences from an event_traces dataframe.

    Returns:
        X: padded integer array of shape (n_samples, max_seq_len)
        vocab: mapping from event id strings to integer indices
    """
    sequences = event_traces[feature_col].map(parse_feature_string).tolist()
    all_events = [event_id for seq in sequences for event_id in seq]
    vocab = build_event_vocab(all_events)
    encoded = encode_sequences(sequences, vocab)
    return pad_sequences(encoded), vocab
