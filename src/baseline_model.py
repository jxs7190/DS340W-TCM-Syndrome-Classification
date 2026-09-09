"""Train a reproducible multi-label baseline on the parent-paper preprocessing.

This file intentionally uses a standard multi-output Random Forest baseline.
It is not presented as the paper authors' official MLDF implementation. The
baseline lets the DS 340W project verify the full data/preprocessing/evaluation
pipeline before adding the more specialized PCC-MLRF + MLDF reimplementation.
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import hamming_loss, f1_score, precision_score, recall_score
from sklearn.multioutput import MultiOutputClassifier

from preprocess import combine_to_multilabel_cases, fit_tfidf_and_labels, transform_cases


def read_excel(path: Path) -> pd.DataFrame:
    engine = "xlrd" if path.suffix.lower() == ".xls" else "openpyxl"
    return pd.read_excel(path, engine=engine)


def evaluate(y_true, y_pred):
    return {
        "hamming_loss": hamming_loss(y_true, y_pred),
        "micro_precision": precision_score(y_true, y_pred, average="micro", zero_division=0),
        "micro_recall": recall_score(y_true, y_pred, average="micro", zero_division=0),
        "micro_f1": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=Path("data/train/stomach_training.xlsx"))
    parser.add_argument("--test", type=Path, default=Path("data/test/stomach_test.xlsx"))
    parser.add_argument("--case-column", default="编号")
    parser.add_argument("--syndrome-column", default="症候")
    parser.add_argument("--symptom-column", default="症状")
    args = parser.parse_args()

    train_rows = read_excel(args.train)
    test_rows = read_excel(args.test)

    train_cases = combine_to_multilabel_cases(
        train_rows, args.case_column, args.syndrome_column, args.symptom_column
    )
    test_cases = combine_to_multilabel_cases(
        test_rows, args.case_column, args.syndrome_column, args.symptom_column
    )

    X_train, Y_train, vectorizer, mlb = fit_tfidf_and_labels(train_cases)
    X_test, Y_test = transform_cases(test_cases, vectorizer, mlb)

    model = MultiOutputClassifier(
        RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced_subsample",
        ),
        n_jobs=-1,
    )
    model.fit(X_train, Y_train)
    Y_pred = np.asarray(model.predict(X_test))

    print("Training cases:", len(train_cases))
    print("Test cases:", len(test_cases))
    print("TF-IDF features:", X_train.shape[1])
    print("Syndrome labels:", Y_train.shape[1])
    print("\nTest metrics")
    for key, value in evaluate(Y_test, Y_pred).items():
        print(f"{key}: {value:.4f}")

    print("\nValidation is intentionally not loaded by this script.")


if __name__ == "__main__":
    main()
