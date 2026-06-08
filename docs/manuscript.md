# Difficult Airway Prediction — Manuscript (working draft)

> Working draft for the bi-modal (facial image + point-of-care ultrasound)
> difficult-airway prediction pilot, framed as a prospective single-centre
> feasibility study. The Results tables are populated with the pipeline's
> current output, which is computed on **synthetic/dummy data** and is clearly
> labelled as such — these are not clinical results. Methods describe what the
> code actually does; they deliberately do **not** claim performance.

<details>
<summary><strong>AUTHOR FILL-IN KEY</strong> — every <code>&lt;PLACEHOLDER&gt;</code> and where to get it (click to expand)</summary>

Before submission, find-and-replace each `<PLACEHOLDER>` in the narrative with
the real value from the cited `reports/` file. Values are intentionally left
blank here — the current Results tables are synthetic and must not be quoted.
Items in `[CITE: …]` are references the author must locate; items in `[FILL: …]`
are administrative details the author must supply.

| Placeholder | Source file | Column / field | Value |
| --- | --- | --- | --- |
| `<N_PATIENTS>` | `reports/data_audit_report.md` (or `per_model_metrics.csv` `n`) | cohort size (face + US + outcome) | |
| `<N_DIFFICULT>` | `reports/data_audit_report.md` | n with CL 3–4 | |
| `<PREVALENCE>` | `reports/data_audit_report.md` | difficult-airway rate (%) | |
| `<AUC_FUSED>` | `reports/per_model_metrics.csv` | model=`fusion:logreg`, threshold_type=`fixed_0.5`, `auc_mean` | |
| `<CI_FUSED>` | `reports/bootstrap_ci.csv` | model=`fused_prob`, metric=`auc`, `ci_lower`–`ci_upper` | |
| `<SENS_FUSED>` | `reports/per_model_metrics.csv` | `fusion:logreg`, `sensitivity` | |
| `<SPEC_FUSED>` | `reports/per_model_metrics.csv` | `fusion:logreg`, `specificity` | |
| `<SENS_FUSED_HS>` | `reports/per_model_metrics.csv` | `fusion:logreg`, threshold_type=`high_sensitivity`, `sensitivity` | |
| `<SPEC_FUSED_HS>` | `reports/per_model_metrics.csv` | `fusion:logreg`, threshold_type=`high_sensitivity`, `specificity` | |
| `<BRIER_FUSED>` | `reports/calibration_metrics.csv` (per-modality), or compute from `reports/fusion_fold_predictions.csv` | Brier score | |
| `<AUC_FACE>` | `reports/per_model_metrics.csv` | model=`face`, `auc_mean` | |
| `<AUC_US>` | `reports/per_model_metrics.csv` | model=`ultrasound`, `auc_mean` | |
| `<AUC_AVG>` | `reports/per_model_metrics.csv` | model=`fusion:average`, `auc_mean` | |
| `<AUC_MALLAMPATI>` | `reports/per_model_metrics.csv` | model=`mallampati`, `auc_mean` | |
| `<AUC_LEMON>` | `reports/per_model_metrics.csv` | model=`lemon`, `auc_mean` | |
| `<AUC_WILSON>` | `reports/per_model_metrics.csv` | model=`wilson`, `auc_mean` | |
| `<AUC_BEST_BEDSIDE>` | `reports/per_model_metrics.csv` | highest of mallampati/lemon/wilson `auc_mean` | |
| `<AUC_CLINICAL>` | `reports/clinical_baseline_metrics.csv` | clinical baseline, threshold_type=`fixed_0.5`, `auc_mean` | |
| `<CI_CLINICAL>` | `reports/bootstrap_ci.csv` (after adding `clinical_prob`) or `clinical_baseline_metrics.csv` | clinical baseline, AUC 95% CI | |
| `<COLUMNS_CLINICAL>` | `reports/clinical_baseline_metrics.csv` | `columns_used` field | |
| `<P_FUSED_VS_BEST_BEDSIDE>` | `reports/delong_comparisons.csv` | `p_value` for fused vs the best bedside score | |
| `<P_FUSED_VS_FACE>` | `reports/delong_comparisons.csv` | `fused_vs_face`, `p_value` | |
| `<P_FUSED_VS_US>` | `reports/delong_comparisons.csv` | `fused_vs_ultrasound`, `p_value` | |
| `<P_FUSED_VS_AVG>` | `reports/delong_comparisons.csv` | `fused_vs_average`, `p_value` | |
| `<P_FUSED_VS_CLINICAL>` | `reports/delong_comparisons.csv` | `fused_vs_clinical`, `p_value` | |
| `<BONFERRONI_ALPHA>` | computed: 0.05 / number of DeLong comparisons | now 7 comparisons → ≈ 0.0071 | |
| `<TOP_US_FEATURE>` | `reports/shap_ultrasound_importance.csv` (or `us_feature_importance.csv`) | feature ranked #1 | |
| `<TOP_US_FEATURE_2>` | `reports/shap_ultrasound_importance.csv` | feature ranked #2 | |
| `<ICC_TOP_US>` | `reports/data_audit_report.md` | highest ultrasound ICC(2,1) + n_pairs | |
| `<ICC_RANGE>` | `reports/data_audit_report.md` | min–max ICC across US features | |
| `<NB_FUSED_AT_PT>` | `reports/decision_curve.csv` | nb_fused at the clinically relevant threshold | |
| `<DCA_THRESHOLD_RANGE>` | `reports/decision_curve.csv` | threshold range where fused net benefit exceeds treat-all/none and best bedside | |

</details>

## Abstract

### Abstract (≈250 words)

**Background.** Unanticipated difficult laryngoscopy contributes to
airway-related morbidity, yet routine bedside predictors — the Mallampati
classification, the LEMON score, and the Wilson risk sum — discriminate only
modestly and reproduce inconsistently between observers
[CITE: limited accuracy of Mallampati/LEMON/Wilson]. Whether combining
complementary, objectively measured signals improves prediction beyond routine
clinical assessment is unclear.

**Methods.** We conducted a single-site prospective pilot feasibility study in
`<N_PATIENTS>` adults undergoing general anaesthesia requiring direct
laryngoscopy. Facial images were summarised by frozen ImageNet-pretrained
ResNet-18 embeddings and point-of-care ultrasound by standardised anterior-neck
measurements. The outcome was a difficult laryngeal view (Cormack–Lehane grade
3–4). Single-modality logistic-regression models were calibrated with isotonic
regression and combined by a logistic-regression late-fusion meta-learner, all
within patient-level 5×2 cross-validation with every preprocessing step fitted
inside folds. The fused model was benchmarked against the bedside scores and
against a clinical baseline model (routine bedside variables) using the DeLong
test with Bonferroni correction; uncertainty was summarised with 1000-sample
patient-level bootstrap 95% confidence intervals, and clinical utility with
decision-curve analysis.

**Results.** Of `<N_PATIENTS>` patients, `<N_DIFFICULT>` (`<PREVALENCE>`) had a
difficult view. The fused model achieved an AUC of `<AUC_FUSED>` (95% CI
`<CI_FUSED>`) versus `<AUC_BEST_BEDSIDE>` for the best bedside score (DeLong
p = `<P_FUSED_VS_BEST_BEDSIDE>`) and `<AUC_CLINICAL>` for a clinical baseline
model (p = `<P_FUSED_VS_CLINICAL>`).

**Conclusions.** This pilot demonstrates a reproducible, leakage-controlled
bi-modal pipeline for difficult-airway prediction and a workflow that can be
delivered in routine peri-operative practice. Whether bi-modal machine learning
adds clinically meaningful discrimination beyond bedside assessment cannot be
established at this sample size and requires a multicentre, externally validated
study.

### Abstract — short version (≈150 words)

Unanticipated difficult laryngoscopy contributes to airway morbidity, and bedside
predictors discriminate only modestly [CITE: limited accuracy of Mallampati/LEMON/Wilson].
In a single-site prospective pilot of `<N_PATIENTS>` adults, we predicted a
difficult laryngeal view (Cormack–Lehane grade 3–4) from facial-image embeddings
(frozen ResNet-18) and point-of-care ultrasound. Calibrated single-modality
logistic-regression models were combined by a logistic-regression late-fusion
meta-learner within patient-level 5×2 cross-validation, with all preprocessing
fitted inside folds, and benchmarked against the Mallampati, LEMON, and Wilson
scores and a routine clinical baseline model (DeLong test, Bonferroni-corrected;
bootstrap 95% CIs; decision-curve analysis). The fused model achieved an AUC of
`<AUC_FUSED>` (95% CI `<CI_FUSED>`) versus `<AUC_BEST_BEDSIDE>` for the best
bedside score (p = `<P_FUSED_VS_BEST_BEDSIDE>`). This pilot establishes a
reproducible, deliverable bi-modal pipeline; whether it improves on bedside
assessment requires multicentre, externally validated confirmation.

## Introduction

Failure to anticipate a difficult airway remains a source of preventable
peri-operative harm: difficulty that is recognised only after induction of
anaesthesia narrows the margin for safe rescue and is associated with hypoxaemia,
airway trauma, and, rarely, catastrophic outcomes
[CITE: difficult-airway complications; incidence]. Reliable preoperative
identification of patients at elevated risk would allow anaesthetists to prepare
equipment, expertise, and a considered plan in advance
[CITE: prevalence of unanticipated difficult intubation, large cohort].

The predictors used at the bedside today — the Mallampati classification, the
LEMON assessment, the Wilson risk sum, and component measures such as the
thyromental distance — are quick and inexpensive, but individually they offer
limited discrimination and are sensitive to how and by whom they are elicited
[CITE: systematic review of bedside airway tests]. Their modest and variable
performance motivates the search for objective, reproducible signals that could
complement or sharpen clinical judgement.

Two such signals have each shown association with difficult laryngoscopy. Point-
of-care ultrasound of the anterior neck — for example, soft-tissue thickness at
the level of the vocal cords and the hyomental distance and its change with neck
extension — captures anatomy relevant to glottic exposure
[CITE: ultrasound airway markers]. Independently, facial morphology summarised by
machine-learning representations has been linked to airway difficulty
[CITE: facial-image/ML airway prediction]. Each modality, however, has typically
been studied in isolation.

What is comparatively under-explored is whether **combining** modalities adds
information beyond either alone — and beyond the bedside examination a clinician
already performs — and whether such a combination can be built and evaluated in a
way that resists the optimism that plagues small predictive-modelling studies, in
particular information leakage from preprocessing or from splitting images rather
than patients. A reproducible, leakage-controlled, patient-level pipeline that
fuses modalities and benchmarks them against the incumbent bedside scores would
help establish whether the multimodal direction is worth pursuing at scale.

We therefore undertook a single-site pilot feasibility study with three aims:
(i) to develop and internally validate a bi-modal model that predicts a difficult
laryngeal view (Cormack–Lehane grade 3–4) from facial-image embeddings and point-
of-care ultrasound; (ii) to benchmark this model against the Mallampati, LEMON,
and Wilson scores and against a clinical baseline model built from routine
variables, under identical patient-level cross-validation; and (iii) to establish
the feasibility, data pipeline, workflow burden, and effect-size signals needed
to design an adequately powered multicentre study. A voice/acoustic modality was
deliberately deferred to a future version of the model and is not evaluated here.
This report presents methods and feasibility; it is not powered to establish
clinical performance.

## Methods

### Study design and setting

This was a prospective, observational, single-centre pilot feasibility study
evaluating a leakage-controlled multimodal machine-learning pipeline for the
prediction of difficult laryngoscopy. The study was conducted at
[FILL: institution] between [FILL: start month/year] and [FILL: end month/year].
Institutional Ethics Committee approval was obtained before recruitment
([FILL: protocol/IEC number]) and written informed consent, including a clause
covering the research use of facial images, was obtained from every participant
before any data were collected. As a pilot, the study was designed to estimate
recruitment feasibility, data completeness, the practicality of the acquisition
workflow, and preliminary effect sizes to inform a future multicentre study; it
was not powered to demonstrate the statistical superiority of any model.
Reporting follows the TRIPOD+AI guidance for prediction-model studies
[CITE: TRIPOD+AI statement] and, for the diagnostic-accuracy elements, the STARD
guidance [CITE: STARD 2015 statement].

### Participants

We prospectively screened adults scheduled for elective surgery under general
anaesthesia requiring direct laryngoscopy and tracheal intubation.

- **Inclusion criteria:** age ≥ 18 years; ASA physical status I–III; planned
  direct laryngoscopy and intubation.
- **Exclusion criteria:** anatomically distorted upper airway (e.g. prior major
  head-and-neck surgery, radiotherapy, or maxillofacial trauma); emergency or
  rapid-sequence inductions that precluded protocolised data collection;
  inability to be positioned for standardised imaging or ultrasound; and refusal
  of consent.

### Outcome (reference standard)

The primary outcome was a difficult laryngeal view, defined strictly as a
Cormack–Lehane (CL) grade of 3 or 4 at the first direct laryngoscopy; CL grades
1–2 were classified as non-difficult. The CL grade was assigned by the attending
anaesthesiologist immediately after laryngoscopy. To reduce the risk of
incorporation bias, the outcome assessor was blinded to the point-of-care
ultrasound measurements, the facial images, and any model output; the assessor
was necessarily aware of routine bedside examination findings as part of normal
care [FILL: confirm blinding arrangements as actually implemented]. The same
binary outcome was used for all models. Difficult mask ventilation and the number
of intubation attempts were [FILL: recorded as secondary descriptors / not
recorded] and were not used to define the primary outcome.

### Bedside clinical predictors

During the preoperative assessment a trained investigator recorded the modified
Mallampati class, the components of the LEMON assessment, and the Wilson risk
sum, together with age, sex, body-mass index, and — where available —
thyromental distance and neck circumference. These were used both as comparator
scores and, in a pre-specified secondary analysis, as the inputs to a clinical
baseline model (below).

### Facial image acquisition

[FILL: number] standardised digital facial photographs were captured per patient
([FILL: e.g. frontal at rest, frontal with mouth open, left and right profile])
using [FILL: camera/phone model] at a fixed subject-to-camera distance of
[FILL: distance] under [FILL: lighting]. Patients were seated upright in a
neutral pose with glasses and masks removed; the presence of a beard or other
feature obscuring landmarks was recorded as a usability note. Images were stored
de-identified under a study identifier.

### Point-of-care airway ultrasound

Anterior-neck ultrasound was performed in the preoperative area by an
investigator trained in airway sonography, using a [FILL: machine] with a
[FILL: high-frequency linear, e.g. 5–12 MHz] probe, with the patient supine and
the neck neutral. Four pre-defined measurements were obtained: anterior neck
soft-tissue thickness at the level of the vocal cords; hyomental distance in the
neutral position; hyomental distance in the extended position; and the
skin-to-epiglottis distance. The hyomental distance ratio (extended ÷ neutral)
was derived from these. The acquisition protocol, landmarks, and number of
repeated captures per measurement are specified in [FILL: supplementary protocol].

To assess measurement reproducibility, a random [FILL: e.g. 10–15]% of patients
underwent independent re-measurement of all ultrasound features by a second
sonographer blinded to the first set of values and to the outcome; inter-observer
agreement was summarised with the intraclass correlation coefficient (ICC) for
each continuous measurement. Inter-observer agreement for the CL grade, where a
second assessor was available, was summarised with Cohen's κ (unweighted and
quadratic-weighted) as reported by the data-audit step.

### Ultrasound feature cleaning

Point-of-care ultrasound measurements were assembled into a per-patient table
using a fixed column schema. Column names from the source export were mapped to
the schema through an explicit alias table; any schema column absent from the
export was retained as an all-missing placeholder and reported. All measurement
columns were coerced to numeric values, with non-numeric or blank entries set to
missing. Features that were entirely missing across the cohort were dropped with
a logged warning. No scaling or imputation was performed at this stage; values
were left in their natural units (millimetres).

### Derived feature: hyomental distance ratio

The hyomental distance ratio was computed as the ratio of the hyomental distance
in the extended head position to that in the neutral position
(extended ÷ neutral). The ratio was defined as missing when the neutral distance
was zero, negative, or missing, or when the extended distance was missing, to
avoid division by zero and non-physical values. The ratio was persisted as an
engineered feature in the cleaned ultrasound feature table.

### Feature representation

**Facial images.** Each image was face-detected, cropped to an eye-centred
square, and resized to 224 × 224 pixels. A frozen ImageNet-pretrained ResNet-18,
with its classification head removed, produced a 512-dimensional embedding per
image; per-patient features were formed by concatenating the element-wise mean
and maximum of a patient's image embeddings (1024 dimensions). Given the pilot
cohort size, the network was used purely as a fixed feature extractor with no
end-to-end fine-tuning, which is more defensible against overfitting than
training a convolutional network on a small sample.

**Ultrasound.** The cleaned four measurements and the derived ratio formed the
ultrasound feature set. Within the cross-validation pipeline, missing values were
imputed using the mean of each feature estimated on the training fold only and
then applied to both the training and held-out folds; no imputation was performed
on the full dataset before cross-validation.

### Models

For each modality, an L2-regularised logistic regression (the pre-specified
primary classifier) and a gradient-boosted tree model (XGBoost) were trained.
Within the cross-validation pipeline, logistic-regression features were
standardised (zero mean, unit variance) using training-fold statistics only.
Class imbalance was handled without resampling: logistic regression used balanced
class weights, and XGBoost used a positive-class scaling factor
(`scale_pos_weight`) set to the training-fold negative-to-positive ratio.

In a pre-specified secondary analysis, a **clinical baseline model** (L2 logistic
regression on age, sex, BMI, Mallampati class, and — where available —
thyromental distance) was trained and evaluated under the identical
cross-validation scheme, to test whether the imaging and ultrasound modalities
add discrimination beyond routinely available bedside information.

### Probability calibration

Each single-modality model (the primary classifier was L2-regularised logistic
regression) was recalibrated using isotonic regression. Calibration was performed
within the cross-validation: in every training fold the base model and the
isotonic calibrator were fitted using an inner cross-validation on the training
patients only, and calibrated probabilities were then produced for the held-out
patients out-of-sample. Where the minority-class count in a training fold was too
small to support the inner calibration split, that fold's uncalibrated
probabilities were carried forward and the event was logged. Calibration quality
was summarised with the Brier score (per fold and pooled) and with reliability
diagrams comparing predicted probabilities to observed event frequencies.

### Late fusion

The two modalities were combined by late fusion. A logistic-regression
meta-learner took the two calibrated out-of-fold probabilities (face and
ultrasound) as its only inputs. The meta-learner was trained and evaluated with
the same patient-level stratified 5×2 cross-validation: within each fold it was
fitted on the training patients' calibrated probabilities and applied to the
held-out patients' calibrated probabilities, so validation labels never entered
meta-training. As a no-learning reference, an unweighted average of the two
calibrated probabilities was computed for the same patients. The learned fusion
was compared against this average baseline; failure to exceed it was flagged as a
warning indicating possibly non-complementary modalities rather than treated as
an error.

### Clinical comparators

Three bedside scores — Mallampati class, the LEMON score, and the Wilson risk
sum — were computed from the pre-operative examination and evaluated on the same
cross-validation folds as the fusion model. Discrimination was summarised by AUC
(using the raw/ordinal score); operating-point metrics (sensitivity, specificity,
PPV, NPV, accuracy, balanced accuracy, and F1) were computed at fixed,
pre-specified thresholds (Mallampati class ≥ 3, LEMON ≥ 2, Wilson ≥ 2).

### Cross-validation

All models were evaluated with repeated stratified cross-validation at the
patient level (5 folds × 2 repeats). Splitting was by unique patient identifier,
so that all images and measurements belonging to a patient fell entirely within
either the training or the held-out fold, never both. Stratification preserved
the difficult-airway proportion across folds. A single common cohort (patients
with facial, ultrasound, and outcome data) and a single set of fold assignments
were used for every model and for all downstream calibration, fusion, and
comparison steps, so that no model was advantaged by a different split.

### Performance metrics and decision thresholds

Discrimination was summarised by the AUC. Operating-point metrics (sensitivity,
specificity, PPV, NPV, accuracy, balanced accuracy, and F1) were computed from
pooled out-of-fold predictions at two thresholds: a fixed 0.5 threshold and the
Youden-optimal threshold. Because airway tools prioritise avoiding missed
difficult airways, we additionally report the operating point at a pre-specified
high-sensitivity target of [FILL: e.g. ≥ 0.90 sensitivity], with the
corresponding specificity read from the ROC. Bedside scores were evaluated at the
conventional cut-points above.

### Uncertainty and class imbalance

Because difficult laryngoscopy is uncommon (expected prevalence approximately
5–11% in elective populations [CITE: incidence of difficult laryngoscopy]), small
positive-case counts make analytic confidence intervals unreliable. All 95%
confidence intervals for AUC and operating-point metrics were therefore estimated
by bootstrap resampling of patients (1000 resamples); resamples containing a
single outcome class were excluded from the AUC interval and the number of valid
resamples was recorded.

### Feature importance

Two complementary measures of feature importance were computed for the ultrasound
model. Permutation importance was estimated leakage-safely on each held-out fold
by measuring the decrease in AUC when each feature's values were randomly
permuted, and was averaged across folds. SHAP values (TreeExplainer) were
computed for the ultrasound gradient-boosted model to summarise each feature's
contribution; for the facial-image model the magnitude of the logistic-regression
coefficients was reported instead, the embedding dimensions being individually
uninterpretable.

### Statistical comparison

The fused model was compared against each comparator (Mallampati, LEMON, Wilson,
the face model, the ultrasound model, the clinical baseline model, and the
average-probability reference) using the DeLong test for two correlated ROC AUCs,
evaluated on one probability or score per patient. Family-wise error across the
comparison set was controlled with a Bonferroni-adjusted threshold
(α = `<BONFERRONI_ALPHA>`; 0.05 divided by the number of comparisons). Consistent
with the pilot design, these tests are interpreted as descriptive estimates of
effect size rather than as confirmatory hypothesis tests.

### Clinical utility

To assess whether the model would improve decisions rather than only
discrimination, decision-curve analysis was performed, plotting net benefit
across a range of threshold probabilities for the fused model, the best bedside
score, and the default strategies of treating all and treating no patients as
difficult [CITE: Vickers decision-curve analysis].

### Feasibility outcomes

Pre-specified feasibility outcomes were the proportions of screened patients
enrolled and analysed; the per-modality data-completeness and usability rates
(usable facial images, completed ultrasound measurements, and per-feature
ultrasound missingness); and the mean time required to collect the full
multimodal dataset per patient [FILL: record acquisition time prospectively].
These are reported as a study-flow and data-feasibility table.

### Reproducibility

All random seeds were fixed. Frozen image embeddings were computed once outside
cross-validation; every step that learns from data (imputation, scaling,
calibration, the classifiers, and the XGBoost class-weighting factor) was fitted
inside each cross-validation fold. Analysis code is openly available at
`https://github.com/NaskenAI/difficultAirway` [FILL: add archived DOI].

## Results

> ⚠️ **SYNTHETIC / DUMMY DATA.** The numbers below were generated by the
> repository pipeline on **synthetic plumbing data** (30 fake patients,
> randomly-initialised image embeddings because pretrained weights were
> unavailable offline). They demonstrate that the pipeline runs end-to-end and
> are **not clinical results**. Replace `data/raw/` with real data and re-run to
> obtain reportable values. The face model is trivially separable on this
> synthetic data (AUC = 1.00), which is an artefact of the dummy generator, not
> evidence of performance.

### Table 1. Cohort and study flow (synthetic)

| Characteristic | Value |
| --- | --- |
| Patients screened / enrolled / analysed | [FILL from study-flow log] |
| Patients (face + ultrasound + outcome) | 30 |
| Difficult airway, CL 3–4 — n (%) | 8 (26.7%) |
| CL grade 1 / 2 / 3 / 4 — n | 11 / 11 / 5 / 3 |
| Age, years — mean ± SD | 52.4 ± 19.7 |
| Sex — F / M | 19 / 11 |
| BMI — mean ± SD | 27.6 ± 3.8 |
| Inter-observer agreement (CL), quadratic-weighted κ | 0.90 |

_Source: `reports/data_audit_report.md`. Cross-validation: patient-level
stratified 5×2._

### Table 1b. Data feasibility (synthetic)

Workflow practicality metrics — for a pilot, often as important as discrimination.

| Feasibility metric | Value |
| --- | --- |
| Usable facial-image rate | [FILL from `data_audit_report.md`] |
| Ultrasound completion rate | [FILL] |
| Missing ultrasound features (per feature) | [FILL] |
| Mean data-acquisition time per patient | [FILL: record prospectively] |
| Image / ultrasound unusable rate | [FILL] |

_Source: `reports/data_audit_report.md` (and the prospective acquisition-time
log). These metrics demonstrate whether the protocol can be delivered in a busy
peri-operative workflow._

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
`reports/calibration_metrics.csv`. On this synthetic data the learned fusion does
not exceed the average baseline (both AUC = 1.00); the pipeline logs this
sanity-check warning._

### Table 2b. Incremental value over routine bedside assessment (synthetic; patient-level 5×2 CV)

A clinical baseline model (logistic regression on routinely available bedside
variables — `<COLUMNS_CLINICAL>`) is compared against the imaging/ultrasound
models and the fusion, to test whether the added modalities improve
discrimination beyond standard clinical assessment.

| Model | Inputs | AUC (mean ± SD) | AUC 95% CI | Sens. | Spec. |
| --- | --- | --- | --- | --- | --- |
| Bedside scores (best of Mallampati/LEMON/Wilson) | bedside scores | `<AUC_BEST_BEDSIDE>` | | | |
| Clinical baseline (LR) | age, sex, BMI, Mallampati (+ TMD if available) | `<AUC_CLINICAL>` | `<CI_CLINICAL>` | | |
| Ultrasound (calibrated LR) | anterior-neck US | `<AUC_US>` | | | |
| Face (calibrated LR) | facial-image embeddings | `<AUC_FACE>` | | | |
| Fusion (logistic meta-learner) | face + ultrasound | `<AUC_FUSED>` | `<CI_FUSED>` | `<SENS_FUSED>` | `<SPEC_FUSED>` |

_Source: `reports/clinical_baseline_metrics.csv`, `reports/per_model_metrics.csv`,
`reports/bootstrap_ci.csv`. The clinical baseline is the key comparator for the
question of incremental value; the DeLong fused-vs-clinical comparison appears in
Table 3._

### Table 3. DeLong comparisons — fused model vs. comparators (synthetic)

Two-sided DeLong test, one value per patient; Bonferroni threshold
α = `<BONFERRONI_ALPHA>` (7 comparisons).

| Comparison | AUC (fused) | AUC (comparator) | ΔAUC | z | p | Significant (α = `<BONFERRONI_ALPHA>`) |
| --- | --- | --- | --- | --- | --- | --- |
| Fused vs Mallampati | 1.00 | 0.85 | 0.15 | 1.91 | 0.056 | No |
| Fused vs LEMON | 1.00 | 0.95 | 0.05 | 1.44 | 0.149 | No |
| Fused vs Wilson | 1.00 | 0.90 | 0.10 | 1.81 | 0.071 | No |
| Fused vs Face | 1.00 | 1.00 | 0.00 | 0.00 | 1.000 | No |
| Fused vs Ultrasound | 1.00 | 0.67 | 0.33 | 2.20 | 0.028 | No |
| Fused vs Clinical baseline | 1.00 | `<AUC_CLINICAL>` | | | `<P_FUSED_VS_CLINICAL>` | |
| Fused vs Average baseline | 1.00 | 1.00 | 0.00 | 0.00 | 1.000 | No |

_Source: `reports/delong_comparisons.csv`. Seven comparisons; Bonferroni
α = `<BONFERRONI_ALPHA>`. As a pilot, these tests estimate effect size and are
not confirmatory; interpret accordingly. No comparison survives correction at
this synthetic sample size; do not interpret these synthetic results clinically._

### Table 4. Ultrasound feature importance (synthetic)

| Feature | Permutation importance (mean ± SD) | XGBoost gain | SHAP (mean \|value\|) |
| --- | --- | --- | --- |
| Hyomental distance, neutral | 0.27 ± 0.14 | 0.35 | [FILL from `shap_ultrasound_importance.csv`] |
| Anterior neck soft tissue (DSTVC) | 0.07 ± 0.09 | 0.18 | [FILL] |
| Skin-to-epiglottis distance | 0.06 ± 0.13 | 0.19 | [FILL] |
| Hyomental distance ratio | 0.04 ± 0.09 | 0.16 | [FILL] |
| Hyomental distance, extended | −0.05 ± 0.10 | 0.13 | [FILL] |

_Source: `reports/us_feature_importance.csv`, `reports/us_feature_importance.png`,
`reports/shap_ultrasound_importance.csv`, `reports/shap_ultrasound_summary.png`._

### Table 5. Ultrasound inter-rater reliability (ICC 2,1)

Agreement between two blinded sonographers on a random subset of patients with
repeat measurements. ICC(2,1): two-way random effects, single rater, absolute
agreement.

| Measurement | n pairs | ICC (2,1) |
| --- | --- | --- |
| Anterior neck soft tissue (DSTVC) | | |
| Hyomental distance, neutral | | |
| Hyomental distance, extended | | |
| Skin-to-epiglottis distance | | |

_Source: `reports/data_audit_report.md` (inter-rater section). If repeat
measurements were not collected, this table is reported as "not assessed" and the
absence is stated as a limitation rather than left blank (see Limitations)._

### Table 6. Clinical utility — decision-curve analysis (synthetic)

Net benefit across clinically relevant threshold probabilities for the fused
model versus the best bedside score and the default treat-all / treat-none
strategies. Net benefit is computed as
(TP/n) − (FP/n) × [p_t / (1 − p_t)] at each threshold probability p_t.

| Threshold probability | Net benefit — fused | Net benefit — best bedside | Net benefit — treat all | Net benefit — treat none |
| --- | --- | --- | --- | --- |
| (see `reports/decision_curve.csv`) | | | | 0 |

At a clinically relevant threshold the fused model's net benefit was
`<NB_FUSED_AT_PT>`. The fused model showed positive net benefit over the
treat-all and treat-none defaults across the threshold range
`<DCA_THRESHOLD_RANGE>`.

_Source: `reports/decision_curve.csv`, `reports/decision_curve.png`._

### Figures

- **Figure 1.** ROC curves for the single-modality and fused models
  (`reports/fusion_roc.png`, `reports/face_roc.png`, `reports/us_roc.png`).
- **Figure 2.** Calibration (reliability) diagrams for the single-modality models
  (`reports/face_calibration.png`, `reports/us_calibration.png`).
- **Figure 3.** Ultrasound SHAP summary (`reports/shap_ultrasound_summary.png`).
- **Figure 4.** Decision-curve analysis — net benefit versus threshold
  probability for the fused model, best bedside score, and treat-all / treat-none
  defaults (`reports/decision_curve.png`).

## Discussion

In this single-site pilot we developed and internally validated a bi-modal model
that predicts a difficult laryngeal view from facial-image embeddings and point-
of-care ultrasound, and we benchmarked it against the bedside scores in routine
use. Under patient-level 5×2 cross-validation the fused model achieved an AUC of
`<AUC_FUSED>` (95% CI `<CI_FUSED>`), compared with `<AUC_BEST_BEDSIDE>` for the
best-performing bedside score (DeLong p = `<P_FUSED_VS_BEST_BEDSIDE>`). These
numbers are reported as the principal feasibility output rather than as a
performance claim, given the pilot's size.

If the fused model's discrimination exceeds that of the bedside scores, this would
suggest that objectively measured facial and anterior-neck signals carry
information that is complementary to, and not merely a restatement of, the
clinical examination. The honest alternative must be stated with equal weight:
our pipeline explicitly compares the learned fusion against an unweighted average
of the two calibrated modality probabilities (`<AUC_AVG>`) and against the better
single modality, and it raises a warning when the learned fusion fails to exceed
them. Should that occur, the most likely explanations at this sample size are that
the modalities are not demonstrably complementary, or that a learned meta-learner
is over-parameterised relative to the number of difficult-airway events — not that
multimodal information is absent in principle. A trivial, no-learning average that
matches a learned combiner is itself an informative, publishable result for a
pilot.

Crucially for clinical relevance, we tested whether the imaging and ultrasound
modalities add discrimination beyond information a clinician already has at the
bedside. A logistic model built only on routine variables (`<COLUMNS_CLINICAL>`)
achieved an AUC of `<AUC_CLINICAL>` (95% CI `<CI_CLINICAL>`); the fused model was
compared against it directly (DeLong p = `<P_FUSED_VS_CLINICAL>`). If the
multimodal model does not exceed this clinical baseline, the appropriate
conclusion is that — in this pilot — objectively measured facial and anterior-neck
signals do not yet demonstrate value beyond standard assessment, which is itself
an important and publishable finding that should temper enthusiasm for added
data-collection burden.

Among the ultrasound measurements, `<TOP_US_FEATURE>` and `<TOP_US_FEATURE_2>`
carried the most signal by both permutation and SHAP importance. This is
consistent with the anatomical expectation that greater anterior neck soft-tissue
thickness and a reduced (or less extensible) hyomental distance impede glottic
visualisation [CITE: anterior neck soft tissue / hyomental distance and difficult
airway]. We interpret feature-importance rankings as hypothesis-generating in a
cohort of this size rather than as established mechanistic findings.

Because a risk tool is only useful if its outputs can be read as probabilities, we
calibrated each single-modality model with isotonic regression inside the
cross-validation and summarised calibration with the Brier score and reliability
diagrams (`<BRIER_FUSED>`). Calibration is frequently neglected in predictive-
modelling reports, yet it is what allows a predicted probability to support a
threshold-based decision such as escalating airway preparation
[CITE: importance of calibration for clinical risk tools].

Direct comparison with prior single-modality work should be made cautiously
because cohorts, outcome definitions, and validation schemes differ. Where prior
ultrasound models [CITE: prior ultrasound model AUCs] and facial-image models
[CITE: prior facial-image model AUCs] have reported discrimination in a similar
range, our single-modality results would be broadly concordant; where they differ,
spectrum and sampling differences are the more plausible explanation than a
genuine performance gap, and we avoid strong claims either way.

The principal strengths of this work are methodological. Splitting was performed
at the patient level, so no patient's images or measurements appeared in both
training and held-out folds; every step that learns from data — imputation,
standardisation, calibration, and the fusion meta-learner — was fitted inside
folds only; image embeddings were frozen and computed once, outside
cross-validation, eliminating that common leakage path; operating-point thresholds
for the bedside scores were pre-specified; model comparisons used the DeLong test
with Bonferroni correction [CITE: DeLong test for correlated ROC curves]; and
uncertainty was quantified with patient-level bootstrap confidence intervals. To
move beyond discrimination to clinical usefulness, we additionally report
decision-curve analysis, which assesses whether acting on the model's predictions
would yield net benefit over default management strategies across a plausible
range of decision thresholds [CITE: Vickers decision-curve analysis]. The entire
pipeline is scripted and reproducible from fixed seeds.

### Limitations

This is a small, single-site pilot and its limitations are substantial and
inseparable from its purpose. The sample is modest and the positive class (CL
3–4) is rare, so all estimates are imprecise and the confidence intervals are
expected to be wide; the study is not powered for formal superiority testing, and
the Bonferroni-corrected comparisons should be read accordingly. There was no
external or temporal validation, so generalisation beyond the local population,
imaging set-up, and ultrasound operators is unknown, and spectrum bias is possible
if the pilot cohort is not representative of the eventual target population.
Facial-image acquisition was [FILL: standardised / not rigorously standardised]
for pose, lighting, and device, which can influence learned embeddings; point-of-
care ultrasound is operator-dependent, and inter-operator measurement agreement
was [FILL: quantified on a subset (ICC range `<ICC_RANGE>`) / not formally
quantified in this pilot]. The Cormack–Lehane grade is an imperfect reference
standard, subject to inter-rater variation and dependent on laryngoscopy
technique. The planned voice/acoustic modality was deferred and is not part of
this model. Finally, frozen ImageNet embeddings are a deliberately conservative
representation; a model fine-tuned on a much larger airway-specific dataset might
behave differently. None of these limitations is incidental — together they define
why this work is framed as feasibility rather than evidence of clinical benefit.

### Future work

The natural next step is a multicentre study with prospective, standardised image
and ultrasound acquisition and a sample size powered from the effect-size signals
observed here, followed by external and ideally temporal validation. Subsequent
work would add the deferred voice modality, evaluate prospective deployment and
its effect on airway preparation and outcomes, and assess fairness across
demographic subgroups with adequate per-subgroup numbers (the present subgroup
analyses are descriptive only). These directions form the basis of the planned
follow-on programme [FILL: grant/programme name and number].

## Conclusion

This pilot establishes a reproducible, leakage-controlled pipeline for bi-modal
(facial-image and point-of-care ultrasound) prediction of a difficult laryngeal
view, demonstrates that the acquisition workflow can be delivered in routine
peri-operative practice, and benchmarks the model transparently against the
bedside scores and a routine clinical baseline. It does not, and is not intended
to, establish that bi-modal machine learning improves on those comparators; that
question requires an adequately powered, multicentre, externally validated study,
which these feasibility results are designed to motivate and inform.

## Data availability

The analysis code and pipeline are publicly available at
`https://github.com/NaskenAI/difficultAirway`. Raw facial images and any audio are
identifiable and are therefore not publicly shared; they remain subject to
institutional data-governance and are available only under the conditions
described in the repository's data policy [FILL: data-access/governance contact
and conditions]. Derived, non-identifiable feature tables and the generated report
artefacts are produced by the released pipeline.

## Ethics statement

[FILL: institutional ethics committee (IEC/IRB) name and approval number; consent
process and whether written informed consent was obtained.]

## Funding

[FILL: seed/grant fund name and number; role of the funder, if any.]

## Author contributions

[FILL: CRediT-style statement, e.g.] Conceptualisation: `[FILL]`; Methodology:
`[FILL]`; Software: `[FILL]`; Formal analysis: `[FILL]`; Investigation / data
collection: `[FILL]`; Data curation: `[FILL]`; Writing — original draft:
`[FILL]`; Writing — review & editing: `[FILL]`; Supervision: `[FILL]`; Funding
acquisition: `[FILL]`.

## Conflicts of interest

[FILL: declare any competing interests, or state "The authors declare no competing
interests."]

## Reporting guideline

This study was developed and is reported in accordance with the TRIPOD+AI
statement for prediction-model studies using machine learning
[CITE: TRIPOD+AI statement], and the diagnostic-accuracy elements follow STARD
[CITE: STARD 2015 statement]; completed checklists will accompany submission.

## REFERENCES TO SOURCE

The following `[CITE: …]` markers appear in the narrative above and must be
resolved by the author with real, verifiable references (do not fabricate). They
are not yet citations — they are a to-do list of the evidence each statement
needs.

1. [CITE: difficult-airway complications; incidence] — morbidity/airway
   complications associated with unanticipated difficult intubation.
2. [CITE: prevalence of unanticipated difficult intubation, large cohort] — base
   rate of unanticipated difficulty in a large/representative cohort.
3. [CITE: limited accuracy of Mallampati/LEMON/Wilson] — evidence that bedside
   scores have limited and/or inconsistent discrimination.
4. [CITE: systematic review of bedside airway tests] — systematic review /
   meta-analysis of the diagnostic accuracy of bedside airway assessments.
5. [CITE: ultrasound airway markers] — ultrasound anterior-neck measures
   associated with difficult laryngoscopy.
6. [CITE: facial-image/ML airway prediction] — prior work linking facial imaging
   / machine learning to airway difficulty.
7. [CITE: anterior neck soft tissue / hyomental distance and difficult airway] —
   anatomical basis for the implicated ultrasound features.
8. [CITE: importance of calibration for clinical risk tools] — methodological
   reference on probability calibration / Brier score for clinical models.
9. [CITE: prior ultrasound model AUCs] — reported discrimination of prior
   ultrasound-based airway models, for comparison.
10. [CITE: prior facial-image model AUCs] — reported discrimination of prior
    facial-image airway models, for comparison.
11. [CITE: DeLong test for correlated ROC curves] — the DeLong method for
    comparing two correlated AUCs.
12. [CITE: TRIPOD+AI statement] — the TRIPOD+AI reporting guideline.
13. [CITE: Vickers decision-curve analysis] — the decision-curve / net-benefit
    method (Vickers & Elkin).
14. [CITE: STARD 2015 statement] — STARD reporting guideline for diagnostic
    accuracy (cited in study-design Methods).
15. [CITE: incidence of difficult laryngoscopy] — base-rate / prevalence of
    difficult laryngoscopy in elective populations (cited for the class-imbalance
    rationale).
