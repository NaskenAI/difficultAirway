# Data-Freeze Memo

> Fill every `<FILL>` at the real-data freeze. This memo is the human record of
> what was frozen and why. It does not itself freeze anything — tagging the repo
> and freezing MLflow runs are manual steps done after this memo is complete.

## Dataset identification
- Dataset version / label: `<FILL>`
- Date frozen: `<FILL (YYYY-MM-DD)>`
- Git commit hash at freeze: `<FILL (git rev-parse HEAD)>`
- DVC data hash (`data/raw.dvc`): `<FILL>`
- Frozen by: `<FILL (name)>`

## Cohort
- Total patients in raw dataset: `<FILL>`
- Patients included in modelling (face + ultrasound + outcome): `<FILL>`
- Difficult-airway rate (CL 3–4), included cohort: `<FILL (e.g. 26.7%)>`

## Exclusions (from `data/processed/quarantine_decisions.json`)
> Copy the counts/reasons recorded by `python -m airway.quarantine`.
- Patients excluded: `<FILL>` — reason(s): `<FILL>`
- Images excluded: `<FILL>` — reason(s): `<FILL (e.g. no detectable face)>`
- Ultrasound cells imputed: `<FILL>` — method: within-fold mean (see quarantine rules)
- Other DVC-tracked data excluded and why: `<FILL>`

## Models frozen
- Face model run / artifact: `<FILL>`
- Ultrasound model run / artifact: `<FILL>`
- Calibration run / artifact: `<FILL>`
- Fusion model run / artifact: `<FILL>`

## MLflow / run notes
- MLflow experiment: `<FILL>`
- Run IDs (face / ultrasound / calibration / fusion): `<FILL>`
- Notes on seeds / environment / package versions: `<FILL>`

## Sign-off
- Clinical reviewer: `<FILL>` — date: `<FILL>`
- Data/ML reviewer: `<FILL>` — date: `<FILL>`
- Final freeze decision (APPROVED / HOLD): `<FILL>`
