"""
DeLong test for comparing two correlated ROC AUCs.

WHAT THIS IS
------------
The DeLong test asks whether two models, scored on the SAME patients, have
significantly different AUCs. Because both scores come from the same patients,
their AUCs are correlated; DeLong's method estimates that correlation and gives
a valid significance test for the difference.

IMPLEMENTATION
--------------
This is the "fast DeLong" algorithm (Sun & Xu, 2014), the standard O(n log n)
formulation widely used in the literature. `delong_roc_variance` returns the
AUC and its variance for one model; `delong_test` returns both AUCs and a
two-sided p-value for their difference.

ASSUMPTIONS / LIMITATIONS (documented, as requested)
----------------------------------------------------
- Binary ground truth (0/1) with both classes present.
- The AUC computed via mid-ranks equals the Mann-Whitney AUC (ties handled by
  mid-ranks); it matches sklearn.roc_auc_score.
- The p-value uses the asymptotic normal approximation of the AUC difference;
  it is most reliable away from the boundaries and with a non-trivial number of
  cases per class (so treat p-values from this 30-patient pilot as indicative).
- Both score vectors must be paired (same patients, same order). If the two
  scores are identical the difference variance is ~0 and the p-value is ~1.
- No scipy dependency: the normal survival function is computed with math.erfc.
"""

from __future__ import annotations

import math

import numpy as np


def _compute_midrank(x: np.ndarray) -> np.ndarray:
    """Mid-ranks of x (ties get the average rank). 1-based, as DeLong expects."""
    order = np.argsort(x)
    sorted_x = x[order]
    n = len(x)
    ranks = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n, dtype=float)
    out[order] = ranks
    return out


def _fast_delong(preds_sorted: np.ndarray, n_pos: int):
    """
    Fast DeLong for k score vectors on the same cases.

    Parameters
    ----------
    preds_sorted : ndarray, shape (k, n)
        Scores with the n_pos positive cases in the FIRST columns.
    n_pos : int
        Number of positive cases.

    Returns
    -------
    aucs : ndarray (k,)
    cov  : ndarray (k, k)  delong covariance of the AUCs
    """
    m = n_pos
    n = preds_sorted.shape[1] - m
    k = preds_sorted.shape[0]
    pos = preds_sorted[:, :m]
    neg = preds_sorted[:, m:]

    tx = np.empty([k, m])
    ty = np.empty([k, n])
    tz = np.empty([k, m + n])
    for r in range(k):
        tx[r, :] = _compute_midrank(pos[r, :])
        ty[r, :] = _compute_midrank(neg[r, :])
        tz[r, :] = _compute_midrank(preds_sorted[r, :])

    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    cov = sx / m + sy / n
    return aucs, np.atleast_2d(cov)


def _ground_truth_stats(y: np.ndarray):
    y = np.asarray(y)
    uniq = set(np.unique(y).tolist())
    if not uniq <= {0, 1}:
        raise ValueError(f"delong: ground truth must be binary 0/1, got {sorted(uniq)}")
    if len(uniq) < 2:
        raise ValueError("delong: ground truth has only one class; AUC undefined.")
    order = (-y).argsort(kind="mergesort")   # positives (1) first, stable
    n_pos = int(np.sum(y == 1))
    return order, n_pos


def _norm_sf(z: float) -> float:
    """Survival function of the standard normal, via erfc (no scipy)."""
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def delong_roc_variance(y_true, y_score):
    """Return (auc, variance) for a single model's ROC AUC."""
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    order, n_pos = _ground_truth_stats(y_true)
    preds_sorted = y_score[order][np.newaxis, :]
    aucs, cov = _fast_delong(preds_sorted, n_pos)
    return float(aucs[0]), float(cov[0, 0])


def delong_test(y_true, y_score_1, y_score_2) -> dict:
    """
    Compare two paired ROC AUCs with the DeLong test.

    Returns dict: auc_1, auc_2, auc_diff, z, p_value (two-sided).
    """
    y_true = np.asarray(y_true, dtype=float)
    s1 = np.asarray(y_score_1, dtype=float)
    s2 = np.asarray(y_score_2, dtype=float)
    if not (len(y_true) == len(s1) == len(s2)):
        raise ValueError("delong_test: y_true, y_score_1, y_score_2 must be same length.")

    order, n_pos = _ground_truth_stats(y_true)
    preds_sorted = np.vstack((s1, s2))[:, order]
    aucs, cov = _fast_delong(preds_sorted, n_pos)

    var_diff = cov[0, 0] + cov[1, 1] - 2.0 * cov[0, 1]
    auc_diff = float(aucs[0] - aucs[1])
    z = auc_diff / math.sqrt(var_diff) if var_diff > 0 else 0.0
    p = 2.0 * _norm_sf(abs(z))
    p = float(min(1.0, max(0.0, p)))
    return {"auc_1": float(aucs[0]), "auc_2": float(aucs[1]),
            "auc_diff": auc_diff, "z": float(z), "p_value": p}
