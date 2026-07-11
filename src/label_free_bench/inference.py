"""Dataset-aware uncertainty and paired publication inference."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def stable_seed(*parts: str, base_seed: int = 2026) -> int:
    payload = "\x1f".join((str(base_seed), *parts)).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def hierarchical_mean_ci(
    rows: Sequence[Mapping[str, Any]],
    value_key: str,
    *,
    rounds: int = 10_000,
    seed: int = 2026,
) -> tuple[float, float, float]:
    """Estimate a dataset-equal mean and hierarchical percentile interval."""
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = _numeric(row[value_key])
        if np.isfinite(value):
            grouped[str(row["dataset"])].append(value)
    groups = [np.asarray(values, dtype=float) for values in grouped.values() if values]
    if not groups:
        return float("nan"), float("nan"), float("nan")

    point = float(np.mean([values.mean() for values in groups]))
    rng = np.random.default_rng(seed)
    selected = rng.integers(0, len(groups), size=(rounds, len(groups)))
    sampled_means = np.empty(selected.shape, dtype=float)
    for position in range(len(groups)):
        for group_index, values in enumerate(groups):
            mask = selected[:, position] == group_index
            selected_count = int(mask.sum())
            if selected_count == 0:
                continue
            draws = rng.choice(
                values,
                size=(selected_count, len(values)),
                replace=True,
            )
            sampled_means[mask, position] = draws.mean(axis=1)
    bootstrap = sampled_means.mean(axis=1)
    low, high = np.quantile(bootstrap, [0.025, 0.975])
    return point, float(low), float(high)


def exact_sign_flip_pvalue(dataset_effects: Sequence[float]) -> float:
    """Return an exact two-sided paired sign-flip p-value."""
    effects = np.asarray(dataset_effects, dtype=float)
    effects = effects[np.isfinite(effects)]
    if effects.size == 0:
        return float("nan")
    if effects.size > 20:
        raise ValueError("exact sign-flip test supports at most 20 datasets")

    observed = abs(float(effects.mean()))
    assignment_count = 1 << effects.size
    extreme = 0
    for mask in range(assignment_count):
        signs = np.array(
            [1.0 if mask & (1 << bit) else -1.0 for bit in range(effects.size)]
        )
        statistic = abs(float(np.mean(signs * effects)))
        extreme += statistic >= observed - 1e-15
    return extreme / assignment_count


def holm_adjust(pvalues: Sequence[float]) -> np.ndarray:
    """Apply Holm family-wise error correction while preserving input order."""
    values = np.asarray(pvalues, dtype=float)
    adjusted = np.full(values.shape, np.nan, dtype=float)
    finite_indices = np.flatnonzero(np.isfinite(values))
    if finite_indices.size == 0:
        return adjusted

    order = finite_indices[np.argsort(values[finite_indices])]
    running = 0.0
    count = len(order)
    for rank, index in enumerate(order):
        corrected = min(1.0, (count - rank) * values[index])
        running = max(running, corrected)
        adjusted[index] = running
    return adjusted


def _numeric(value: Any) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, str):
        lowered = value.lower()
        if lowered == "true":
            return 1.0
        if lowered == "false":
            return 0.0
    return float(value)
