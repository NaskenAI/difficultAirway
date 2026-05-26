"""
Baseline models and cross-validated evaluation.

WHAT THIS DOES
--------------
Given a per-patient feature table and the labels, this module:
  1. builds a simple, leakage-safe model (a scikit-learn Pipeline)
  2. evaluates it with the patient-level cross-validation folds from splits.py
  3. returns honest performance metrics (AUC, sensitivity, specificity, ...)

WHY A PIPELINE
--------------
A scikit-learn "Pipeline" chains preprocessing + model into one object. The
key benefit: when you call .fit() on a training fold, EVERY step -- imputation,
scaling, the classifier -- is fitted on that training fold ONLY. The test fold
is then transformed using those fitted values. This is what prevents the test
data from leaking into preprocessing. Doing scaling by hand outside CV is the
classic subtle leak; the Pipeline removes that whole class of mistake.

THE BASELINE MODEL
------------------
Logistic regression with:
  - median imputation (fills missing values)
  - standardisation (mean 0, variance 1)
  - balanced class weights (difficult airway is rare; this stops the model
    from ignoring the minority class)

Logistic regression is the right BASELINE: simple, fast, hard to overfit, and
a fair yardstick. Stronger models (XGBoost) come in Block B; you want to know
they actually beat this baseline.

METRICS
-------
For each fold the held-out patients are scored, then pooled across folds:
  - AUC          : ranking quality, threshold-free (primary endpoint)
  - sensitivity  : of truly difficult airways, the fraction caught
  - specificity  : of truly easy airways, the fraction correctly cleared
  - accuracy, PPV, NPV
All at the 0.5 probability threshold (threshold tuning is a later step).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from airway import config, splits


@dataclass
class CVResult:
    """Holds the outcome of one cross-validated evaluation."""

    modality: str                       # 'face', 'ultrasound', ...
    n_patients: int
    n_features: int
    auc_mean: float
    auc_std: float
    sensitivity: float
    specificity: float
    accuracy: float
    ppv: float
    npv: float
    per_fold_auc: list = field(default_factory=list)
    # pooled out-of-fold predictions, useful for later fusion / plots
    oof_true: list = field(default_factory=list)
    oof_prob: list = field(default_factory=list)

    def summary_row(self) -> dict:
        """Return a flat dict suitable for a results CSV."""
        return {
            "modality": self.modality,
            "n_patients": self.n_patients,
            "n_features": self.n_features,
            "auc_mean": round(self.auc_mean, 4),
            "auc_std": round(self.auc_std, 4),
            "sensitivity": round(self.sensitivity, 4),
            "specificity": round(self.specificity, 4),
            "accuracy": round(self.accuracy, 4),
            "ppv": round(self.ppv, 4),
            "npv": round(self.npv, 4),
        }


def make_baseline_pipeline() -> Pipeline:
    """
    Build the leakage-safe baseline pipeline:
        impute missing -> standardise -> logistic regression.

    Every step is re-fitted on each training fold by cross-validation.
    """
    return Pipeline(steps=[
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(
            class_weight="balanced",       # handle rare difficult-airway class
            max_iter=1000,
            random_state=config.RANDOM_SEED,
        )),
    ])


def _classification_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """Compute threshold-0.5 metrics from true labels and predicted probs."""
    y_pred = (y_prob >= 0.5).astype(int)

    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    # guard every denominator against division by zero
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    acc = (tp + tn) / len(y_true) if len(y_true) else 0.0
    ppv = tp / (tp + fp) if (tp + fp) else 0.0
    npv = tn / (tn + fn) if (tn + fn) else 0.0
    return {"sensitivity": sens, "specificity": spec, "accuracy": acc,
            "ppv": ppv, "npv": npv}


def evaluate_modality(
    feature_table: pd.DataFrame,
    patient_table: pd.DataFrame,
    feature_cols: list[str],
    modality: str,
) -> CVResult:
    """
    Cross-validate the baseline model for ONE modality.

    Parameters
    ----------
    feature_table : DataFrame
        One row per patient: study_id + feature columns.
    patient_table : DataFrame
        One row per patient: study_id + label (from loaders.build_patient_table).
    feature_cols : list[str]
        Which columns of feature_table to use as model inputs.
    modality : str
        Label for reporting, e.g. 'face' or 'ultrasound'.

    Returns
    -------
    CVResult
    """
    # join features to labels on study_id; inner join keeps only patients
    # who have BOTH features and a label
    data = feature_table.merge(
        patient_table[[config.ID_COL, config.LABEL_COL]],
        on=config.ID_COL, how="inner",
    )
    if data.empty:
        raise ValueError(
            f"evaluate_modality({modality}): no patients have both features "
            f"and a label after the join."
        )

    # build the patient-level CV folds from the joined patients only
    folds = splits.patient_level_folds(data)
    splits.assert_no_leakage(folds)

    per_fold_auc: list[float] = []
    oof_true: list[int] = []
    oof_prob: list[float] = []

    for fold in folds:
        train = splits.select_rows(data, fold.train_ids)
        test = splits.select_rows(data, fold.test_ids)

        x_train = train[feature_cols].to_numpy()
        y_train = train[config.LABEL_COL].to_numpy()
        x_test = test[feature_cols].to_numpy()
        y_test = test[config.LABEL_COL].to_numpy()

        pipe = make_baseline_pipeline()
        pipe.fit(x_train, y_train)
        # predict_proba returns probabilities for [class_0, class_1];
        # column 1 is the probability of 'difficult airway'
        prob = pipe.predict_proba(x_test)[:, 1]

        # AUC is only defined if the test fold has both classes present
        if len(np.unique(y_test)) == 2:
            per_fold_auc.append(roc_auc_score(y_test, prob))

        oof_true.extend(y_test.tolist())
        oof_prob.extend(prob.tolist())

    oof_true_arr = np.array(oof_true)
    oof_prob_arr = np.array(oof_prob)

    # pooled AUC across all out-of-fold predictions
    pooled_auc = (
        roc_auc_score(oof_true_arr, oof_prob_arr)
        if len(np.unique(oof_true_arr)) == 2 else float("nan")
    )
    metrics = _classification_metrics(oof_true_arr, oof_prob_arr)

    return CVResult(
        modality=modality,
        n_patients=len(data),
        n_features=len(feature_cols),
        auc_mean=float(np.mean(per_fold_auc)) if per_fold_auc else pooled_auc,
        auc_std=float(np.std(per_fold_auc)) if per_fold_auc else 0.0,
        sensitivity=metrics["sensitivity"],
        specificity=metrics["specificity"],
        accuracy=metrics["accuracy"],
        ppv=metrics["ppv"],
        npv=metrics["npv"],
        per_fold_auc=per_fold_auc,
        oof_true=oof_true,
        oof_prob=oof_prob,
    )
