# Difficult Airway Prediction — Manuscript (working draft)

> Working draft for the bi-modal (facial image + point-of-care ultrasound)
> difficult-airway prediction pilot. The Results tables are populated with the
> pipeline's current output, which is computed on **synthetic/dummy data** and
> is clearly labelled as such — these are not clinical results. Methods describe
> what the code does; they deliberately do **not** claim performance.

## Methods

### Outcome definition

The primary outcome was a difficult laryngeal view, defined as a Cormack–Lehane
(CL) grade of 3 or 4 at direct laryngoscopy. CL grades 1–2 were treated as
non-difficult. The same binary outcome was used for both the facial-image model
and the ultrasound model.

### Ultrasound feature cleaning

Point-of-care ultrasound measurements (anterior neck soft-tissue thickness at
the vocal cords, hyomental distance in neutral and extended head positions, and
the skin-to-epiglottis distance) were assembled into a per-patient table using a
fixed column schema. Column names from the source export were mapped to the
schema through an explicit alias table; any schema column absent from the export
was retained as an all-missing placeholder and reported. All measurement columns
were coerced to numeric values, with non-numeric or blank entries set to missing.
Features that were entirely missing across the cohort were dropped with a logged
warning. No scaling or imputation was performed at this stage; values were left
in their natural units (millimetres).

### Derived feature: hyomental distance ratio

The hyomental distance ratio was computed as the ratio of the hyomental distance
in the extended head position to that in the neutral position
(extended ÷ neutral). The ratio was defined as missing when the neutral distance
was zero, negative, or missing, or when the extended distance was missing, to
avoid division by zero and non-physical values. The ratio was persisted as an
engineered feature in the cleaned ultrasound feature table.

### Within-fold mean imputation

Missing ultrasound values were imputed using the mean of each feature. To
prevent information leakage from held-out patients, imputation was performed
inside the cross-validation pipeline: the feature means were estimated on the
training fold only and then applied to both the training and held-out folds. No
imputation was performed on the full dataset prior to cross-validation.

### Classifiers

For each modality, two classifiers were trained: L2-regularised logistic
regression and gradient-boosted decision trees (XGBoost). For the ultrasound
model, both classifiers were preceded in a single pipeline by within-fold mean
imputation; logistic regression additionally standardised features (zero mean,
unit variance) using statistics estimated on the training fold only.

### Class imbalance

Because difficult airways (CL 3–4) are uncommon, class imbalance was addressed
without resampling: logistic regression used balanced class weights, and XGBoost
used a positive-class scaling factor (`scale_pos_weight`) set to the ratio of
negative to positive cases in each training fold.

### Cross-validation

Models were evaluated with repeated stratified cross-validation performed at the
patient level (5 folds × 2 repeats). Splitting was by unique patient identifier
so that all images and measurements belonging to a patient fell entirely within
either the training or the held-out fold, never both. Stratification preserved
the difficult-airway proportion across folds. Discrimination was summarised with
the area under the receiver-operating-characteristic curve (AUC); sensitivity,
specificity, accuracy, and predictive values were computed from pooled
out-of-fold predictions at a 0.5 probability threshold.

### Feature importance

Two complementary measures of feature importance were computed for the
ultrasound model. Permutation importance was estimated leakage-safely on each
held-out fold by measuring the decrease in AUC when each feature's values were
randomly permuted, and was averaged across folds. XGBoost gain importance —
the average improvement in the split criterion attributable to each feature —
was computed from the XGBoost model refit on all patients.

### Probability calibration

Each single-modality model (the primary classifier was L2-regularised logistic
regression) was recalibrated using isotonic regression. Calibration was
performed within the cross-validation: in every training fold the base model and
the isotonic calibrator were fitted using an inner cross-validation on the
training patients only, and calibrated probabilities were then produced for the
held-out patients out-of-sample. Calibration quality was summarised with the
Brier score (per fold and pooled) and with reliability diagrams comparing
predicted probabilities to observed event frequencies.

### Late fusion

The two modalities were combined by late fusion. A logistic-regression
meta-learner took the two calibrated out-of-fold probabilities (face and
ultrasound) as its only inputs. The meta-learner was trained and evaluated with
the same patient-level stratified 5×2 cross-validation: within each fold it was
fitted on the training patients' calibrated probabilities and applied to the
held-out patients' calibrated probabilities, so validation labels never entered
meta-training. As a no-learning reference, an unweighted average of the two
calibrated probabilities was computed for the same patients. The learned fusion
was compared against this average baseline; failure to exceed it was flagged as
a warning indicating possibly non-complementary modalities rather than treated
as an error.

### Clinical comparators

Three bedside scores — Mallampati class, the LEMON score, and the Wilson risk
sum — were computed from the pre-operative examination and evaluated on the same
cross-validation folds as the fusion model. Discrimination was summarised by AUC
(using the raw/ordinal score); operating-point metrics (sensitivity,
specificity, PPV, NPV, accuracy, balanced accuracy, and F1) were computed at
fixed, pre-specified thresholds (Mallampati class ≥ 3, LEMON ≥ 2, Wilson ≥ 2;
probability models thresholded at 0.5).

### Statistical comparison

The fused model was compared against each comparator (Mallampati, LEMON, Wilson,
the face model, the ultrasound model, and the average-probability baseline)
using the DeLong test for two correlated ROC AUCs, evaluated on one
probability/score per patient. To control the family-wise error rate across the
six comparisons, significance was assessed against a Bonferroni-adjusted
threshold of α = 0.05 / 6 ≈ 0.0083.

### Reproducibility

All random seeds were fixed. Frozen image embeddings were computed once outside
cross-validation; every step that learns from data (imputation, scaling, the
classifiers, and the XGBoost class-weighting factor) was fitted inside each
cross-validation fold.

## Results

> ⚠️ **SYNTHETIC / DUMMY DATA.** The numbers below were generated by the
> repository pipeline on **synthetic plumbing data** (30 fake patients,
> randomly-initialised image embeddings because pretrained weights were
> unavailable offline). They demonstrate that the pipeline runs end-to-end and
> are **not clinical results**. Replace `data/raw/` with real data and re-run to
> obtain reportable values. The face model is trivially separable on this
> synthetic data (AUC = 1.00), which is an artefact of the dummy generator, not
> evidence of performance.

### Table 1. Cohort (synthetic)

| Characteristic | Value |
| --- | --- |
| Patients (face + ultrasound + outcome) | 30 |
| Difficult airway, CL 3–4 — n (%) | 8 (26.7%) |
| CL grade 1 / 2 / 3 / 4 — n | 11 / 11 / 5 / 3 |
| Age, years — mean ± SD | 52.4 ± 19.7 |
| Sex — F / M | 19 / 11 |
| BMI — mean ± SD | 27.6 ± 3.8 |
| Inter-observer agreement (CL), quadratic-weighted κ | 0.90 |

_Source: `reports/data_audit_report.md`. Cross-validation: patient-level
stratified 5×2._

### Table 2. Per-model performance on the common cohort (synthetic; patient-level 5×2 CV)

AUC is mean ± SD across folds; operating-point metrics are pooled out-of-fold at
the thresholds in Methods.

| Model | AUC (mean ± SD) | Sens. | Spec. | PPV | NPV | Acc. | Bal. acc. | F1 | Brier |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Face (calibrated LR) | 1.00 ± 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 |
| Ultrasound (calibrated LR) | 0.62 ± 0.26 | 0.44 | 0.91 | 0.64 | 0.82 | 0.78 | 0.67 | 0.52 | 0.18 |
| Fusion — logistic meta-learner | 1.00 ± 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | — |
| Fusion — average baseline | 1.00 ± 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | — |
| Mallampati (≥ 3) | 0.92 ± 0.11 | 0.75 | 0.68 | 0.46 | 0.88 | 0.70 | 0.72 | 0.57 | — |
| LEMON (≥ 2) | 0.97 ± 0.05 | 1.00 | 0.80 | 0.67 | 1.00 | 0.86 | 0.90 | 0.80 | — |
| Wilson (≥ 2) | 0.90 ± 0.13 | 1.00 | 0.32 | 0.35 | 1.00 | 0.50 | 0.66 | 0.52 | — |

_Source: `reports/per_model_metrics.csv`, `reports/fusion_cv_metrics.csv`,
`reports/calibration_metrics.csv`. On this synthetic data the learned fusion
does not exceed the average baseline (both AUC = 1.00); the pipeline logs this
sanity-check warning._

### Table 3. DeLong comparisons — fused model vs. comparators (synthetic)

Two-sided DeLong test, one value per patient; Bonferroni threshold α = 0.0083
(6 comparisons).

| Comparison | AUC (fused) | AUC (comparator) | ΔAUC | z | p | Significant (α = 0.0083) |
| --- | --- | --- | --- | --- | --- | --- |
| Fused vs Mallampati | 1.00 | 0.85 | 0.15 | 1.91 | 0.056 | No |
| Fused vs LEMON | 1.00 | 0.95 | 0.05 | 1.44 | 0.149 | No |
| Fused vs Wilson | 1.00 | 0.90 | 0.10 | 1.81 | 0.071 | No |
| Fused vs Face | 1.00 | 1.00 | 0.00 | 0.00 | 1.000 | No |
| Fused vs Ultrasound | 1.00 | 0.67 | 0.33 | 2.20 | 0.028 | No |
| Fused vs Average baseline | 1.00 | 1.00 | 0.00 | 0.00 | 1.000 | No |

_Source: `reports/delong_comparisons.csv`. No comparison survives Bonferroni
correction at this sample size; do not interpret these synthetic results
clinically._

### Table 4. Ultrasound feature importance (synthetic)

| Feature | Permutation importance (mean ± SD) | XGBoost gain |
| --- | --- | --- |
| Hyomental distance, neutral | 0.27 ± 0.14 | 0.35 |
| Anterior neck soft tissue (DSTVC) | 0.07 ± 0.09 | 0.18 |
| Skin-to-epiglottis distance | 0.06 ± 0.13 | 0.19 |
| Hyomental distance ratio | 0.04 ± 0.09 | 0.16 |
| Hyomental distance, extended | −0.05 ± 0.10 | 0.13 |

_Source: `reports/us_feature_importance.csv`, `reports/us_feature_importance.png`._
