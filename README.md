# DS340W TCM Syndrome Classification

Reimplementation project based on Gong et al. (2023), *A syndrome differentiation model of TCM based on multi-label deep forest using biomedical text mining*.

This repository follows the parent-paper workflow: preprocessing TCM clinical records, converting symptoms to TF-IDF features, performing PCC-MLRF-style feature selection, and training/evaluating a multi-label classifier. The course-required split is 70% training, 20% test, and 10% final unseen validation.

## Data source

The parent paper reports that the kidney and stomach datasets are publicly available from:

- https://github.com/web333panda/TCM-Dataset
- https://www.ncmi.cn/index.htm

The public GitHub repository contains the two spreadsheet datasets. It does not contain the official ML-PRDF/MLDF source code used by the authors.

## Repository structure

```text
DS340W-TCM-Syndrome-Classification/
├── README.md
├── requirements.txt
├── data/
│   └── README.md
├── notebooks/
│   └── 01_stomach_full_workflow.ipynb
└── src/
    ├── split_data.py
    ├── preprocess.py
    ├── pcc_mlrf_course.py
    ├── baseline_model.py
    └── stomach_full_workflow.py
```

## Parent-paper workflow

1. Remove records missing symptoms or syndrome labels.
2. Combine single-label records into multi-label clinical cases.
3. Standardize symptom descriptions.
4. Convert symptoms to TF-IDF features.
5. Rank/select symptom features using a PCC-MLRF-style feature-selection stage.
6. Train a multi-label classifier inspired by the paper's MLDF/ML-PRDF workflow.
7. Evaluate on the test split during development.
8. Use the untouched validation split only after the pipeline is finalized.

The paper states that after integration/standardization it had 645 kidney-disease samples with 755 symptom features and 124 syndrome labels, and 436 stomach-disease samples with 323 symptom features and 49 syndrome labels.

The public raw stomach spreadsheet currently contains more cases and labels than the paper's post-standardization table. The notebook therefore reproduces the published workflow on the public raw dataset, but does not claim to recreate the authors' undocumented SymMap dictionary or exact final 436-case dataset.

## Easiest option: run the Colab-ready notebook

Open:

`notebooks/01_stomach_full_workflow.ipynb`

The notebook downloads the public stomach spreadsheet directly from the paper's GitHub repository. No manual dataset upload is required.

It performs:

```text
public stomach spreadsheet
        ↓
basic missing-record cleaning
        ↓
case-level 70 / 20 / 10 split
        ↓
TRAIN          TEST          VALIDATION
  ↓              ↓              ↓
  └──── development ────┘      saved only
        ↓                       untouched
combine rows into multi-label cases
        ↓
TF-IDF fit on TRAIN only
        ↓
PCC-MLRF-style feature ranking on TRAIN only
        ↓
select top features
        ↓
multi-output Random Forest baseline
        ↓
TEST evaluation
```

The notebook deliberately removes the validation dataframe from the active workflow after saving it. A final-validation cell is included at the bottom but commented out.

## Run the complete stomach workflow from the command line

Install dependencies and run:

```bash
pip install -r requirements.txt
python src/stomach_full_workflow.py
```

The script automatically downloads the stomach dataset, creates the three course-required split files, performs preprocessing and feature selection, trains the baseline model, and writes test results under `results/`.

Expected split for the current public raw stomach file with the fixed seed used by the workflow:

```text
Training:   366 clinical cases
Test:       105 clinical cases
Validation:  53 clinical cases
Total:      524 clinical cases
```

The row counts differ from these case counts because one clinical case can have multiple syndrome-label rows.

## Run only the data split

If you already downloaded `ds340wstomach.xlsx`:

```bash
python src/split_data.py ds340wstomach.xlsx --case-column 编号 --name stomach --random-state 0
```

This creates:

```text
data/train/stomach_training.xlsx
data/test/stomach_test.xlsx
data/validation/stomach_validation_UNSEEN.xlsx
```

The split is performed by unique clinical case ID (`编号`), not by individual spreadsheet row, to prevent the same multi-label case from leaking across partitions.

## About PCC-MLRF / MLDF

`src/pcc_mlrf_course.py` is a transparent paper-inspired course reimplementation of the PCC-MLRF feature-ranking idea described in the article. It uses the concepts reported by the paper: Pearson-correlation-based sample similarity, traversal through training cases, Hit/Miss neighbors, feature-weight updates, and descending feature ranking.

It is **not** claimed to be the authors' official code. The authors' cited public repository contains the datasets but not the exact original ML-PRDF implementation.

The current classifier is a standard multi-output Random Forest baseline. This gives the project a complete, reproducible pipeline before a more specialized MLDF cascade is attempted.

## Validation rule

During development, use only Training and Test. Do not fit TF-IDF, feature selection, model parameters, thresholds, or other learned choices on the validation set.

Only after the training/test workflow is finalized should `stomach_validation_UNSEEN.xlsx` be loaded for the final evaluation.
