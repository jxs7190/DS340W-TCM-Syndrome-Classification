# DS340W TCM Syndrome Classification

Reimplementation project based on Gong et al. (2023), *A syndrome differentiation model of TCM based on multi-label deep forest using biomedical text mining*.

This repository is being organized around the parent paper workflow: preprocessing TCM clinical records, converting symptoms to TF-IDF features, performing PCC-MLRF-style feature selection, and training/evaluating a multi-label classifier. The course-required data split is 70% training, 20% test, and 10% final unseen validation.

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
│   ├── README.md
│   ├── raw/
│   ├── train/
│   ├── test/
│   └── validation/
├── notebooks/
│   └── README.md
├── src/
│   ├── split_data.py
│   ├── preprocess.py
│   └── baseline_model.py
└── results/
    └── README.md
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

> Note: this repository is a course reimplementation based on the published paper. It does not claim to be the authors' official source-code repository.
