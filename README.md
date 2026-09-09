# DS340W TCM Syndrome Classification

Reimplementation project based on Gong et al. (2023), *A syndrome differentiation model of TCM based on multi-label deep forest using biomedical text mining*.

This repository follows the parent-paper workflow: preprocessing TCM clinical records, converting symptoms to TF-IDF features, performing PCC-MLRF-style feature selection, and training/evaluating a multi-label classifier. The course-required split is 70% training, 20% test, and 10% final unseen validation.

## Data source

The parent paper reports that the kidney and stomach datasets are publicly available from:

- https://github.com/web333panda/TCM-Dataset
- https://www.ncmi.cn/index.htm

## Repository structure

```text
DS340W-TCM-Syndrome-Classification/
├── README.md
├── requirements.txt
├── data/
│   └── README.md
└── src/
    ├── split_data.py
    ├── preprocess.py
    ├── pcc_mlrf_course.py
    └── baseline_model.py
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

## Run the stomach-data split

Put `ds340wstomach.xlsx` in your working directory, install dependencies, then run:

```bash
pip install -r requirements.txt
python src/split_data.py ds340wstomach.xlsx --case-column 编号 --name stomach
```

This creates separate training, test, and validation files. The split is done by unique clinical case ID, not individual rows, to prevent the same multi-label case from leaking into more than one partition.

## Run the baseline pipeline

After the split:

```bash
python src/baseline_model.py
```

The baseline fits TF-IDF and the model using training data and evaluates on the test set only. It deliberately does not load the final validation file.

## About PCC-MLRF / MLDF

`src/pcc_mlrf_course.py` is a transparent paper-inspired course reimplementation of the PCC-MLRF feature-ranking idea described in the article. It is **not** claimed to be the authors' official source code. The cited public GitHub repository is a dataset repository, and the exact original ML-PRDF source code was not identified there.

The current `baseline_model.py` uses a standard multi-output random-forest classifier so the data, preprocessing, and evaluation pipeline can be verified before replacing the baseline with a fuller MLDF-style cascade implementation.
