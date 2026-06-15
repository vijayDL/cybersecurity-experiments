"""Shared dataset loading and splitting.

Centralizes row loading, the safe/unsafe ground-truth read, and the balanced
train/val split so `optimize_gepa.py` and `run_eval.py` agree on exactly which
rows GEPA trains on. This lets evaluation hold those rows out and avoid the
train-on-test leakage that would otherwise inflate the reported metrics.
"""

from __future__ import annotations

import csv
import random

csv.field_size_limit(10**9)

DATA = "data/RedTeam_FalsePositive_Analysis.csv"
DETECTOR = "malwaregen.AnyCode"


def load_rows(detector=DETECTOR):
    rows = []
    with open(DATA) as f:
        for row in csv.DictReader(f):
            if detector is None or row["Detector"] == detector:
                rows.append(row)
    return rows


def truth_is_safe(row):
    # "Is it actually safe?" TRUE == safe, FALSE == unsafe
    return row["Is it actually safe?"].strip().upper() == "TRUE"


def row_key(row):
    """Stable content-based identity for a row, used for set membership."""
    return (row["Prompt"], row["Output"])


def split_balanced(rows, train_size, val_size, seed):
    """Deterministically split rows into balanced train/val sets.

    train and val are disjoint (val is drawn from the post-train remainder).
    Given the same (rows, train_size, val_size, seed) this is reproducible, so
    callers can reconstruct the exact train/val rows to exclude from eval.
    """
    rng = random.Random(seed)
    safe = [r for r in rows if truth_is_safe(r)]
    unsafe = [r for r in rows if not truth_is_safe(r)]
    rng.shuffle(safe)
    rng.shuffle(unsafe)

    train_safe_n = min(len(safe), train_size // 2)
    train_unsafe_n = min(len(unsafe), train_size - train_safe_n)
    train = safe[:train_safe_n] + unsafe[:train_unsafe_n]

    safe_rem = safe[train_safe_n:]
    unsafe_rem = unsafe[train_unsafe_n:]
    val_safe_n = min(len(safe_rem), val_size // 2)
    val_unsafe_n = min(len(unsafe_rem), val_size - val_safe_n)
    val = safe_rem[:val_safe_n] + unsafe_rem[:val_unsafe_n]

    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def train_val_keys(rows, train_size, val_size, seed):
    """Return the set of row_key()s used by split_balanced for train+val."""
    train, val = split_balanced(rows, train_size, val_size, seed)
    return {row_key(r) for r in train + val}
