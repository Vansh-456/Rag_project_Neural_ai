"""Uncertainty: percentile bootstrap CIs and paired bootstrap significance tests.

With ~30-50 queries per split, differences of a few NDCG points are usually *not*
significant — reporting CIs is what keeps the ablation honest.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def bootstrap_ci(
    values: Sequence[float], n_samples: int = 1000, confidence: float = 0.95, seed: int = 42
) -> tuple[float, float, float]:
    """(mean, lower, upper) percentile bootstrap CI of the mean."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return (0.0, 0.0, 0.0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_samples, arr.size))
    means = arr[idx].mean(axis=1)
    alpha = (1 - confidence) / 2
    return (
        float(arr.mean()),
        float(np.quantile(means, alpha)),
        float(np.quantile(means, 1 - alpha)),
    )


def paired_bootstrap(
    a: Sequence[float], b: Sequence[float], n_samples: int = 2000, seed: int = 42
) -> dict[str, float]:
    """Test mean(b - a) != 0 on paired per-query scores.

    Returns the observed difference, its 95% CI, and a two-sided p-value estimated as
    2 * min(P(diff* <= 0), P(diff* >= 0)) over bootstrap resamples.
    """
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    if x.shape != y.shape:
        raise ValueError("paired samples must have equal length")
    if x.size == 0:
        return {"diff": 0.0, "ci_low": 0.0, "ci_high": 0.0, "p_value": 1.0, "n": 0}
    d = y - x
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(n_samples, d.size))
    means = d[idx].mean(axis=1)
    p = 2 * min(float((means <= 0).mean()), float((means >= 0).mean()))
    return {
        "diff": float(d.mean()),
        "ci_low": float(np.quantile(means, 0.025)),
        "ci_high": float(np.quantile(means, 0.975)),
        "p_value": min(1.0, p),
        "n": int(d.size),
    }
