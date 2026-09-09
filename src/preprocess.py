"""Preprocessing utilities based on the parent paper workflow.

The paper describes removing incomplete records, combining rows from the same
clinical case into a multi-label instance, standardizing symptom text, and
using TF-IDF vectorization before feature selection/classification.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer


@dataclass
class PreparedData:
    cases: pd.DataFrame
    X_tfidf: object
    Y: object
    vectorizer: TfidfVectorizer
    label_binarizer: MultiLabelBinarizer


def normalize_symptom_text(text: str) -> str:
    """Lightweight normalization for punctuation/spacing.

    This is not a full reproduction of the paper's SymMap-based synonym
    dictionary; it only provides deterministic text cleanup. A fuller
    symptom-standardization dictionary can be added later.
    """
    text = str(text).strip()
    text = text.replace("，", "、").replace(",", "、")
    text = re.sub(r"\s+", "", text)
    parts = [p for p in text.split("、") if p]
    # Keep first occurrence while removing duplicates.
    return "、".join(dict.fromkeys(parts))


def combine_to_multilabel_cases(
    df: pd.DataFrame,
    case_column: str = "编号",
    syndrome_column: str = "症候",
    symptom_column: str = "症状",
) -> pd.DataFrame:
    required = {case_column, syndrome_column, symptom_column}
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    clean = df.dropna(subset=[case_column, syndrome_column, symptom_column]).copy()
    clean[syndrome_column] = clean[syndrome_column].astype(str).str.strip()
    clean[symptom_column] = clean[symptom_column].map(normalize_symptom_text)
    clean = clean[(clean[syndrome_column] != "") & (clean[symptom_column] != "")]

    # The same case normally repeats symptom text across multiple syndrome rows.
    grouped = (
        clean.groupby(case_column, as_index=False)
        .agg(
            symptoms=(symptom_column, lambda s: normalize_symptom_text("、".join(s))),
            syndromes=(syndrome_column, lambda s: sorted(set(s))),
        )
    )
    return grouped


def fit_tfidf_and_labels(
    training_cases: pd.DataFrame,
    min_df: int = 1,
):
    """Fit TF-IDF and label encoder on training data only."""
    vectorizer = TfidfVectorizer(
        tokenizer=lambda s: [x for x in s.split("、") if x],
        token_pattern=None,
        lowercase=False,
        min_df=min_df,
    )
    X_train = vectorizer.fit_transform(training_cases["symptoms"])

    label_binarizer = MultiLabelBinarizer()
    Y_train = label_binarizer.fit_transform(training_cases["syndromes"])

    return X_train, Y_train, vectorizer, label_binarizer


def transform_cases(cases, vectorizer, label_binarizer):
    """Apply training-fitted transformations to test/validation data."""
    X = vectorizer.transform(cases["symptoms"])
    Y = label_binarizer.transform(cases["syndromes"])
    return X, Y
