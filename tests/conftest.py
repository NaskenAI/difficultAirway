"""
Shared test setup.

macOS OpenMP guard
------------------
PyTorch and XGBoost each ship their own OpenMP runtime. On macOS, if torch
initialises OpenMP first and XGBoost then tries to fit a model in the same
process, the process segfaults. Whichever library initialises OpenMP FIRST
wins, so here — before any test module imports torch — we do a one-row XGBoost
fit to claim the OpenMP runtime. After this, torch and XGBoost coexist safely
for the rest of the test session.

In production this conflict never arises: embeddings (torch) and model training
(XGBoost) run in separate processes (the Makefile targets / the subprocess in
face_model._ensure_features).
"""

from __future__ import annotations

try:
    import numpy as _np
    from xgboost import XGBClassifier as _XGB

    _XGB(n_estimators=1, n_jobs=1).fit(_np.zeros((4, 2)), _np.array([0, 1, 0, 1]))
except Exception:  # xgboost missing or unbuildable -> tests that need it will skip/fail clearly
    pass
