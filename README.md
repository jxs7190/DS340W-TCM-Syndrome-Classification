# TCM Syndrome Classification — DS 340W

Jackson Shi's Python research project: download real source data, train and compare classifiers, inspect a local dashboard, and predict TCM syndrome labels from Chinese symptoms.

**Research question:** How accurately can machine learning classify TCM syndromes from symptom data, and does feature selection improve a multi-output Random Forest?

This implements the proposed TF-IDF / feature-selection / Random Forest experiment with baselines. It is **not an exact reproduction** of ML-PRDF, Cross-FGCNN, or TCM-BERT-CNN. Scores on different datasets/splits cannot establish improvement over those papers.

## Start in VS Code on Windows

Install Git, Python **3.11 or 3.12**, VS Code, and its Microsoft Python extension. In a terminal:

```powershell
git clone https://github.com/jxs7190/DS340W-TCM-Syndrome-Classification.git
cd DS340W-TCM-Syndrome-Classification
code .
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m tcm train --dataset stomach
.venv\Scripts\python.exe -m tcm train --dataset kidney
.venv\Scripts\python.exe -m streamlit run app.py
```

For Python 3.11 replace `py -3.12` with `py -3.11`. These commands use the virtual environment directly; PowerShell activation is unnecessary. On macOS/Linux use `python3 -m venv .venv` and `.venv/bin/python` instead of the Windows executable.

In VS Code: **Ctrl+Shift+P → Python: Select Interpreter → .venv**. Then **Terminal → Run Task** provides training and dashboard tasks. F5 runs stomach training. Internet access is needed for the first source-data download.

Already cloned? Use `git pull` in the existing checkout, then reinstall requirements. Do not clone inside the old project folder. VS Code edits your local Git checkout; changes sync to GitHub when committed and pushed.

## Workflow

1. Download stomach or kidney spreadsheets from the first paper's cited source at a pinned commit. Record URL, commit, and SHA-256.
2. Use symptom fields only as predictors. Exclude treatments, outcomes, titles, and target clues.
3. Combine repeated syndrome rows into multi-label records. Kidney records are grouped by symptom pattern within a source article: they are **not verified individual patients**.
4. Keep source groups and identical normalized symptom sets in one partition. Split groups approximately 70% training / 20% development test / 10% final validation. Record proportions vary with group sizes.
5. Three-fold grouped CV inside training refits TF-IDF and feature selection in every fold. Choose thresholds and the winning model using training out-of-fold micro-F1 only.
6. Refit on all training records and evaluate each model on the development test. Save the winner with its vocabulary, label encoder, selected columns, and threshold.
7. Keep final validation unused until explicitly unlocked. Evaluation counts labels unseen in training as missed labels, rather than dropping them.

Models: label-prior dummy; binary-relevance logistic regression; native multi-output Random Forest on all features; the same forest on up to 100 PCC-MLRF-style ranked features. RF shares trees across outputs, without class weighting. The selector is the repository's simplified paper-inspired algorithm, not official author code. No full SymMap synonym standardization is claimed.

## Your three papers

| Paper | Dataset status |
|---|---|
| Gong et al. (2023), ML-PRDF | Public stomach and kidney files; both runnable |
| Huang et al. (2022), Cross-FGCNN | No original record-level download identified; authorized import supported |
| Chen et al. (2024 issue; online Dec 2023), TCM-BERT-CNN | Original data available by author request; authorized import supported |

See [data sources](docs/DATA_SOURCES.md). Raw datasets differ from the papers' standardized samples. Different disease/label sets are trained separately; no invented label mapping, synthetic clinical records, or unrelated datasets are substituted.

## Outputs

`runs/<dataset>/` contains:

- `report.html`, `comparison.png`, `comparison.csv`: model comparison.
- `training_cv.json`: threshold search and training CV selection scores, not independent generalization estimates.
- `run.json`: provenance, hashes, split counts, configuration, environment, winner, validation status.
- `model.joblib`: complete predictor bundle.
- `test_predictions.csv`, `<model>_test_metrics.json`, `<model>_per_label.csv`: actual development-test predictions, metrics, per-label confusion counts.
- `selected_features.csv`: winner's features; ranking weights populated only for feature selection.
- `train.jsonl`, `test.jsonl`, `validation.jsonl`: local split files, excluded from Git.

Metrics include micro/macro F1, precision, recall, exact label-set accuracy, Hamming loss, and micro average precision. Low Hamming loss alone can hide poor rare-label recall. Macro F1 includes training labels and unseen evaluation labels; inspect support counts. Predictions may legitimately contain no labels above the threshold.

Runs refuse to overwrite an experiment. Use a new directory for another configuration:

```powershell
.venv\Scripts\python.exe -m tcm train --dataset stomach --output runs_experiment2 --top-k 150
```

## Predict

```powershell
.venv\Scripts\python.exe -m tcm predict --model runs/stomach/model.joblib --symptoms "腹部冷痛、腹泻、食欲不振"
```

Load only trusted artifacts produced by this project. Match the training language/vocabulary. Scores are uncalibrated research scores, not clinical probabilities.

## Final validation — after development is finished

```powershell
.venv\Scripts\python.exe -m tcm validate --run runs/stomach --final
```

This evaluates the frozen winner without refitting/retuning, checks model and holdout hashes, writes `final_validation.json`, and refuses a second final evaluation in the same run. The guard is a workflow convention, not access control: the local holdout is readable. Do not tune from final results and call the same holdout unseen.

## Files and tests

`tcm/data.py`: source adapters/grouping/splits. `tcm/model.py`: models/evaluation. `tcm/__main__.py`: CLI. `app.py`: local Streamlit viewer. `.vscode/`: tasks/debugging. `tests/`: correctness checks. `docs/`: sources and verified evidence.

The supported entry point is `python -m tcm`. Earlier `src/` and `notebooks/` prototypes are retained, but lack the new source/duplicate safeguards and unseen-label accounting. Use the new CLI for reported experiments.

```powershell
.venv\Scripts\python.exe -m pytest -q
```

See `docs/VERIFIED_RUN.md` for the executed experiment. Small datasets, incomplete synonym mapping, rare labels, and source-derived records limit conclusions. One seed is an initial experiment, not evidence of clinical reliability or a statistically significant improvement.
