# Verified initial experiments

Executed 2026-10-20 using the actual downloaded workbooks. Source downloads were separately tested from an empty temporary directory and verified against pinned SHA-256 values. No synthetic clinical data were used in these experiments. Artificial fixtures occur only in unit tests.

Environment: Python 3.12.14, NumPy 2.3.5, pandas 2.2.3, scikit-learn 1.8.0, matplotlib 3.10.8. Run configuration: seed 0, 200 trees, up to 100 selected features, 3-fold grouped training CV. Thresholds and model choice use training CV only. The default requirements pin direct dependencies; metadata records the runtime.

## Data audit

| Dataset | Raw rows | Rows with usable symptoms/labels | Grouped records | Independent groups | Train / test / final validation |
|---|---:|---:|---:|---:|---|
| Stomach | 1,525 | 1,525 | 524 | 511 | 364 / 105 / 55 |
| Kidney | 2,283 | 588 | 427 | 208 | 285 / 83 / 59 |

Kidney records are literature/source patterns, not established counts of independent patients. The group split protects article-level and identical-symptom overlap. Consequently record percentages differ from exactly 70/20/10. These counts are for the raw public files and this explicit adapter, not the original papers' curated datasets.

## Development test results

| Dataset | Model | Micro-F1 | Macro-F1 | Exact label-set accuracy |
|---|---|---:|---:|---:|
| Stomach | Prior dummy | 0.1801 | 0.0286 | 0.0000 |
| Stomach | Logistic regression | 0.7778 | 0.6999 | 0.2476 |
| Stomach | RF, all features | 0.7244 | 0.6508 | 0.1048 |
| Stomach | RF, selected features | 0.6757 | 0.5971 | 0.0762 |
| Kidney | Prior dummy | 0.1868 | 0.0049 | 0.0000 |
| Kidney | Logistic regression | 0.2648 | 0.0280 | 0.0964 |
| Kidney | RF, all features | 0.2689 | 0.0292 | 0.0964 |
| Kidney | RF, selected features | 0.2179 | 0.0218 | 0.0964 |

Training CV selected logistic regression for both datasets. Kidney RF has a slightly higher test micro-F1, but it was not substituted after viewing test scores. Selected thresholds: stomach logistic 0.65; kidney logistic 0.40. Full comparison and CV threshold tables are retained in `verified_results/`.

**The proposed feature-selection improvement is not supported by this run.** Selected-feature RF performed worse than full-feature RF on both development tests. This does not establish that the paper's original selector is inferior: the repository selector is explicitly a simplified approximation, and the data/preprocessing/split differ.

Kidney generalization is weak: the logistic model's macro-F1 is only 0.0280. Nineteen test label names were absent from training, covering 21 true positive label occurrences; evaluation counts them as misses. Do not describe this as a reliable general syndrome predictor. Stomach results are stronger, but its exact full-label-set accuracy is 24.76%, not 77.78%; micro-F1 is a different measure.

## Verification performed

- Both actual datasets completed downloading/checksum checks, preprocessing, grouped splitting, training CV, test evaluation, model persistence, and report generation.
- Nine automated tests cover case/source/duplicate separation, train-only vocabulary, constant-label scoring, saved-model round trips, unseen-label false negatives, canonical imports, and final-validation hash/one-time guards.
- Streamlit's app test loaded the dashboard and exercised the prediction button without application exceptions.
- The saved stomach model successfully predicted from a new command-line symptom input; the comparison plot was visually inspected.
- Actual clinical final-validation partitions remain **UNUSED**. Finalization was tested only on artificial unit-test data.

Only aggregate evidence is published here. Raw records, split files, predictions containing local case IDs, and trained artifacts remain ignored. Re-running the commands recreates local models and reports. Original data from Huang and Chen were not obtained or used; see `DATA_SOURCES.md` for access limitations.
