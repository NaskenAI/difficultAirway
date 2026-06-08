# Difficult Airway Prediction — Manuscript (working draft)

> Working draft for the bi-modal (facial image + point-of-care ultrasound)
> difficult-airway prediction pilot. The Results tables are populated with the
> pipeline's current output, which is computed on **synthetic/dummy data** and
> is clearly labelled as such — these are not clinical results. Methods describe
> what the code does; they deliberately do **not** claim performance.

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
| `<BRIER_FUSED>` | `reports/calibration_metrics.csv` (per-modality), or compute from `reports/fusion_fold_predictions.csv` | Brier score | |
| `<AUC_FACE>` | `reports/per_model_metrics.csv` | model=`face`, `auc_mean` | |
| `<AUC_US>` | `reports/per_model_metrics.csv` | model=`ultrasound`, `auc_mean` | |
| `<AUC_AVG>` | `reports/per_model_metrics.csv` | model=`fusion:average`, `auc_mean` | |
| `<AUC_MALLAMPATI>` | `reports/per_model_metrics.csv` | model=`mallampati`, `auc_mean` | |
| `<AUC_LEMON>` | `reports/per_model_metrics.csv` | model=`lemon`, `auc_mean` | |
| `<AUC_WILSON>` | `reports/per_model_metrics.csv` | model=`wilson`, `auc_mean` | |
| `<AUC_BEST_BEDSIDE>` | `reports/per_model_metrics.csv` | highest of mallampati/lemon/wilson `auc_mean` | |
| `<P_FUSED_VS_BEST_BEDSIDE>` | `reports/delong_comparisons.csv` | `p_value` for fused vs the best bedside score | |
| `<P_FUSED_VS_FACE>` | `reports/delong_comparisons.csv` | `fused_vs_face`, `p_value` | |
| `<P_FUSED_VS_US>` | `reports/delong_comparisons.csv` | `fused_vs_ultrasound`, `p_value` | |
| `<P_FUSED_VS_AVG>` | `reports/delong_comparisons.csv` | `fused_vs_average`, `p_value` | |
| `<TOP_US_FEATURE>` | `reports/shap_ultrasound_importance.csv` (or `us_feature_importance.csv`) | feature ranked #1 | |
| `<TOP_US_FEATURE_2>` | `reports/shap_ultrasound_importance.csv` | feature ranked #2 | |

</details>

## Abstract

### Abstract (≈250 words)

**Background.** Unanticipated difficult laryngoscopy contributes to
airway-related morbidity, yet routine bedside predictors — the Mallampati
classification, the LEMON score, and the Wilson risk sum — discriminate only
modestly and reproduce inconsistently between observers
[CITE: limited accuracy of Mallampati/LEMON/Wilson]. Whether combining
complementary, objectively measured signals improves prediction is unclear.

**Methods.** We conducted a single-site prospective pilot in `<N_PATIENTS>`
adults undergoing general anaesthesia requiring direct laryngoscopy. Facial
images were summarised by frozen ImageNet-pretrained ResNet-18 embeddings and
point-of-care ultrasound by standardised anterior-neck measurements. The outcome
was a difficult laryngeal view (Cormack–Lehane grade 3–4). Single-modality
logistic-regression models were calibrated with isotonic regression and combined
by a logistic-regression late-fusion meta-learner, all within patient-level 5×2
cross-validation with every preprocessing step fitted inside folds. The fused
model was benchmarked against the bedside scores using the DeLong test with
Bonferroni correction; uncertainty was summarised with 1000-sample patient-level
bootstrap 95% confidence intervals.

**Results.** Of `<N_PATIENTS>` patients, `<N_DIFFICULT>` (`<PREVALENCE>`) had a
difficult view. The fused model achieved an AUC of `<AUC_FUSED>` (95% CI
`<CI_FUSED>`) versus `<AUC_BEST_BEDSIDE>` for the best bedside score (DeLong
p = `<P_FUSED_VS_BEST_BEDSIDE>`).

**Conclusions.** This pilot demonstrates a reproducible, leakage-controlled
bi-modal pipeline for difficult-airway prediction. Whether bi-modal machine
learning adds clinically meaningful discrimination beyond bedside scores cannot
be established at this sample size and requires multicentre, externally validated
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
scores (DeLong test, Bonferroni-corrected; bootstrap 95% CIs). The fused model
achieved an AUC of `<AUC_FUSED>` (95% CI `<CI_FUSED>`) versus `<AUC_BEST_BEDSIDE>`
for the best bedside score (p = `<P_FUSED_VS_BEST_BEDSIDE>`). This pilot
establishes a reproducible bi-modal pipeline; whether it improves on bedside
scores requires multicentre, externally validated confirmation.

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
information beyond either alone, and whether such a combination can be built and
evaluated in a way that resists the optimism that plagues small predictive-
modelling studies — in particular, information leakage from preprocessing or from
splitting images rather than patients. A reproducible, leakage-controlled,
patient-level pipeline that fuses modalities and benchmarks them against the
incumbent bedside scores would help establish whether the multimodal direction is
worth pursuing at scale.

We therefore undertook a single-site pilot with three aims: (i) to develop and
internally validate a bi-modal model that predicts a difficult laryngeal view
(Cormack–Lehane grade 3–4) from facial-image embeddings and point-of-care
ultrasound; (ii) to benchmark this model against the Mallampati, LEMON, and
Wilson scores under identical patient-level cross-validation; and (iii) to
establish the feasibility, data pipeline, and effect-size signals needed to
design an adequately powered multicentre study. A voice/acoustic modality was
deliberately deferred to a future version of the model and is not evaluated here.
This report presents methods and feasibility; it is not powered to establish
clinical performance.

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

## Discussion

In this single-site pilot we developed and internally validated a bi-modal model
that predicts a difficult laryngeal view from facial-image embeddings and point-
of-care ultrasound, and we benchmarked it against the bedside scores in routine
use. Under patient-level 5×2 cross-validation the fused model achieved an AUC of
`<AUC_FUSED>` (95% CI `<CI_FUSED>`), compared with `<AUC_BEST_BEDSIDE>` for the
best-performing bedside score (DeLong p = `<P_FUSED_VS_BEST_BEDSIDE>`). These
numbers are reported as the principal feasibility output rather than as a
performance claim, given the pilot's size.

If the fused model's discrimination exceeds that of the bedside scores, this
would suggest that objectively measured facial and anterior-neck signals carry
information that is complementary to, and not merely a restatement of, the
clinical examination. The honest alternative must be stated with equal weight:
our pipeline explicitly compares the learned fusion against an unweighted average
of the two calibrated modality probabilities (`<AUC_AVG>`) and against the better
single modality, and it raises a warning when the learned fusion fails to exceed
them. Should that occur, the most likely explanations at this sample size are
that the modalities are not demonstrably complementary, or that a learned
meta-learner is over-parameterised relative to the number of difficult-airway
events — not that multimodal information is absent in principle. A trivial,
no-learning average that matches a learned combiner is itself an informative,
publishable result for a pilot.

Among the ultrasound measurements, `<TOP_US_FEATURE>` and `<TOP_US_FEATURE_2>`
carried the most signal by both permutation and SHAP importance. This is
consistent with the anatomical expectation that greater anterior neck soft-tissue
thickness and a reduced (or less extensible) hyomental distance impede glottic
visualisation [CITE: anterior neck soft tissue / hyomental distance and difficult
airway]. We interpret feature-importance rankings as hypothesis-generating in a
cohort of this size rather than as established mechanistic findings.

Because a risk tool is only useful if its outputs can be read as probabilities,
we calibrated each single-modality model with isotonic regression inside the
cross-validation and summarised calibration with the Brier score and reliability
diagrams (`<BRIER_FUSED>`). Calibration is frequently neglected in predictive-
modelling reports, yet it is what allows a predicted probability to support a
threshold-based decision such as escalating airway preparation
[CITE: importance of calibration for clinical risk tools].

Direct comparison with prior single-modality work should be made cautiously
because cohorts, outcome definitions, and validation schemes differ. Where prior
ultrasound models [CITE: prior ultrasound model AUCs] and facial-image models
[CITE: prior facial-image model AUCs] have reported discrimination in a similar
range, our single-modality results would be broadly concordant; where they
differ, spectrum and sampling differences are the more plausible explanation than
a genuine performance gap, and we avoid strong claims either way.

The principal strengths of this work are methodological. Splitting was performed
at the patient level, so no patient's images or measurements appeared in both
training and held-out folds; every step that learns from data — imputation,
standardisation, calibration, and the fusion meta-learner — was fitted inside
folds only; image embeddings were frozen and computed once, outside cross-
validation, eliminating that common leakage path; operating-point thresholds for
the bedside scores were pre-specified; model comparisons used the DeLong test
with Bonferroni correction [CITE: DeLong test for correlated ROC curves]; and
uncertainty was quantified with patient-level bootstrap confidence intervals. The
entire pipeline is scripted and reproducible from fixed seeds.

### Limitations

This is a small, single-site pilot and its limitations are substantial and
inseparable from its purpose. The sample is modest and the positive class (CL
3–4) is rare, so all estimates are imprecise and the confidence intervals are
expected to be wide; the study is not powered for formal superiority testing,
and the Bonferroni-corrected comparisons should be read accordingly. There was no
external or temporal validation, so generalisation beyond the local population,
imaging set-up, and ultrasound operators is unknown, and spectrum bias is
possible if the pilot cohort is not representative of the eventual target
population. Facial-image acquisition was not rigorously standardised for pose,
lighting, or device, which can influence learned embeddings; point-of-care
ultrasound is operator-dependent, and inter-operator measurement variability was
not formally quantified here. The Cormack–Lehane grade is an imperfect reference
standard, subject to inter-rater variation and dependent on laryngoscopy
technique. The planned voice/acoustic modality was deferred and is not part of
this model. Finally, frozen ImageNet embeddings are a deliberately conservative
representation; a model fine-tuned on a much larger airway-specific dataset might
behave differently. None of these limitations is incidental — together they
define why this work is framed as feasibility rather than evidence of clinical
benefit.

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
view and benchmarks it transparently against the bedside scores in current use.
It does not, and is not intended to, establish that bi-modal machine learning
improves on those scores; that question requires an adequately powered,
multicentre, externally validated study, which these feasibility results are
designed to motivate and inform.

## Data availability

The analysis code and pipeline are publicly available at
`https://github.com/NaskenAI/difficultAirway`. Raw facial images and any audio
are identifiable and are therefore not publicly shared; they remain subject to
institutional data-governance and are available only under the conditions
described in the repository's data policy [FILL: data-access/governance contact
and conditions]. Derived, non-identifiable feature tables and the generated
report artefacts are produced by the released pipeline.

## Ethics statement

[FILL: institutional ethics committee (IEC/IRB) name and approval number;
consent process and whether written informed consent was obtained.]

## Funding

[FILL: seed/grant fund name and number; role of the funder, if any.]

## Author contributions

[FILL: CRediT-style statement, e.g.] Conceptualisation: `[FILL]`; Methodology:
`[FILL]`; Software: `[FILL]`; Formal analysis: `[FILL]`; Investigation / data
collection: `[FILL]`; Data curation: `[FILL]`; Writing — original draft:
`[FILL]`; Writing — review & editing: `[FILL]`; Supervision: `[FILL]`; Funding
acquisition: `[FILL]`.

## Conflicts of interest

[FILL: declare any competing interests, or state "The authors declare no
competing interests."]

## Reporting guideline

This study was developed and is reported in accordance with the TRIPOD+AI
statement for prediction-model studies using machine learning
[CITE: TRIPOD+AI statement]; a completed checklist will accompany submission.

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
