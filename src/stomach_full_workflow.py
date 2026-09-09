"""End-to-end DS 340W stomach TCM syndrome classification workflow.

This script follows the parent paper's stated sequence using the public
stomach dataset: clean records, create a case-level 70/20/10 split, combine
rows into multi-label clinical cases, fit TF-IDF on training only, apply a
paper-inspired PCC-MLRF feature-ranking stage, train a reproducible
multi-output Random Forest baseline, and evaluate on the test set.

The 10% validation split is created and saved but never used for feature
engineering, training, tuning, or test evaluation in this script.

Important: the paper authors' cited public GitHub repository contains the two
spreadsheet datasets, but not their official ML-PRDF/MLDF source code or the
full SymMap synonym dictionary. This is therefore a transparent course
reimplementation of the published workflow, not official author code.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    coverage_error,
    f1_score,
    hamming_loss,
    label_ranking_average_precision_score,
    label_ranking_loss,
    precision_score,
    recall_score,
)
from sklearn.multioutput import MultiOutputClassifier

from split_data import split_by_case
from preprocess import combine_to_multilabel_cases, fit_tfidf_and_labels, transform_cases
from pcc_mlrf_course import rank_features_pcc_mlrf, select_top_features

DATA_URL = (
    "https://raw.githubusercontent.com/web333panda/TCM-Dataset/"
    "main/%E8%83%83%E9%83%A8%E7%96%BE%E7%97%85%E6%95%B0%E6%8D%AE.xlsx"
)
RANDOM_STATE = 0
TOP_N_FEATURES = 100


def save_splits(training, test, validation, output_root=Path("data")):
    train_dir = output_root / "train"
    test_dir = output_root / "test"
    validation_dir = output_root / "validation"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    validation_dir.mkdir(parents=True, exist_ok=True)

    train_path = train_dir / "stomach_training.xlsx"
    test_path = test_dir / "stomach_test.xlsx"
    validation_path = validation_dir / "stomach_validation_UNSEEN.xlsx"

    training.to_excel(train_path, index=False)
    test.to_excel(test_path, index=False)
    validation.to_excel(validation_path, index=False)
    return train_path, test_path, validation_path


def positive_class_scores(model, X):
    probability_outputs = model.predict_proba(X)
    score_columns = []

    for estimator, probabilities in zip(model.estimators_, probability_outputs):
        classes = list(estimator.classes_)
        if 1 in classes:
            score_columns.append(probabilities[:, classes.index(1)])
        else:
            score_columns.append(np.zeros(probabilities.shape[0]))

    return np.column_stack(score_columns)


def one_error(y_true, y_score):
    top_label = np.argmax(y_score, axis=1)
    return float(np.mean([y_true[i, j] == 0 for i, j in enumerate(top_label)]))


def evaluate(y_true, y_pred, y_score):
    return {
        "hamming_loss": hamming_loss(y_true, y_pred),
        "micro_precision": precision_score(
            y_true, y_pred, average="micro", zero_division=0
        ),
        "micro_recall": recall_score(
            y_true, y_pred, average="micro", zero_division=0
        ),
        "micro_f1": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "exact_match_accuracy": accuracy_score(y_true, y_pred),
        "one_error": one_error(y_true, y_score),
        "coverage_error": coverage_error(y_true, y_score),
        "ranking_loss": label_ranking_loss(y_true, y_score),
        "average_precision": label_ranking_average_precision_score(
            y_true, y_score
        ),
    }


def main():
    print("Downloading public stomach dataset from the parent paper repository...")
    raw = pd.read_excel(DATA_URL, engine="openpyxl")
    print("Raw rows:", len(raw))
    print("Unique raw clinical cases:", raw["编号"].nunique())
    print("Unique raw syndrome labels:", raw["症候"].nunique())

    # Deterministic basic cleaning before splitting; no learned statistics.
    clean = raw.dropna(subset=["编号", "症候", "症状"]).copy()
    clean["症候"] = clean["症候"].astype(str).str.strip()
    clean["症状"] = clean["症状"].astype(str).str.strip()
    clean = clean[(clean["症候"] != "") & (clean["症状"] != "")].copy()

    training_rows, test_rows, validation_rows = split_by_case(
        clean,
        case_column="编号",
        random_state=RANDOM_STATE,
    )

    train_path, test_path, validation_path = save_splits(
        training_rows,
        test_rows,
        validation_rows,
    )

    print("\n70/20/10 case-level split")
    print(
        "TRAIN:",
        training_rows["编号"].nunique(),
        "cases /",
        len(training_rows),
        "rows",
    )
    print(
        "TEST:",
        test_rows["编号"].nunique(),
        "cases /",
        len(test_rows),
        "rows",
    )
    print(
        "VALIDATION:",
        validation_rows["编号"].nunique(),
        "cases /",
        len(validation_rows),
        "rows",
    )
    print("Saved:", train_path)
    print("Saved:", test_path)
    print("Saved:", validation_path)

    # From this point onward, validation is intentionally not used.
    del validation_rows

    training_cases = combine_to_multilabel_cases(training_rows)
    test_cases = combine_to_multilabel_cases(test_rows)

    X_train, Y_train, vectorizer, label_binarizer = fit_tfidf_and_labels(
        training_cases
    )
    X_test, Y_test = transform_cases(
        test_cases,
        vectorizer,
        label_binarizer,
    )

    print("\nParent-paper preprocessing")
    print("Training multi-label cases:", len(training_cases))
    print("Test multi-label cases:", len(test_cases))
    print("Training TF-IDF shape:", X_train.shape)
    print("Training label shape:", Y_train.shape)

    # Feature ranking uses TRAINING data only.
    ranking, weights = rank_features_pcc_mlrf(X_train, Y_train, k=5)
    n_features = min(TOP_N_FEATURES, X_train.shape[1])
    X_train_selected, selected = select_top_features(
        X_train,
        ranking,
        n_features,
    )
    X_test_selected = X_test[:, selected]

    feature_names = np.asarray(vectorizer.get_feature_names_out())
    top_features = pd.DataFrame(
        {
            "feature": feature_names[ranking[:20]],
            "weight": weights[ranking[:20]],
        }
    )
    Path("results").mkdir(parents=True, exist_ok=True)
    top_features.to_csv("results/stomach_top_20_features.csv", index=False)

    print("\nPCC-MLRF-style feature selection")
    print("Original TF-IDF features:", X_train.shape[1])
    print("Selected features:", X_train_selected.shape[1])
    print(top_features.to_string(index=False))

    model = MultiOutputClassifier(
        RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced_subsample",
        ),
        n_jobs=-1,
    )
    model.fit(X_train_selected, Y_train)

    Y_pred = np.asarray(model.predict(X_test_selected))
    Y_score = positive_class_scores(model, X_test_selected)
    metrics = evaluate(Y_test, Y_pred, Y_score)

    metrics_df = pd.DataFrame(
        {"metric": list(metrics.keys()), "value": list(metrics.values())}
    )
    metrics_df.to_csv("results/stomach_test_metrics.csv", index=False)

    print("\nTEST METRICS")
    for name, value in metrics.items():
        print(f"{name}: {value:.6f}")

    predicted = label_binarizer.inverse_transform(Y_pred)
    actual = label_binarizer.inverse_transform(Y_test)
    preview = test_cases[["编号", "symptoms"]].copy()
    preview["true_syndromes"] = [" | ".join(x) for x in actual]
    preview["predicted_syndromes"] = [" | ".join(x) for x in predicted]
    preview.head(25).to_csv(
        "results/stomach_prediction_preview.csv",
        index=False,
    )

    print("\nValidation status: UNTOUCHED after split creation.")
    print(
        "Do not load data/validation/stomach_validation_UNSEEN.xlsx "
        "until the training/test workflow is finalized."
    )


if __name__ == "__main__":
    main()
