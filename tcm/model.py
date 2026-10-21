"""Train-only feature fitting, model fitting, and explicit multi-label scoring."""
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                            hamming_loss, multilabel_confusion_matrix, precision_score, recall_score)
from sklearn.preprocessing import MultiLabelBinarizer

from src.pcc_mlrf_course import rank_features_pcc_mlrf
from .data import normalize, tokens

MODEL_NAMES = ["dummy", "logistic", "rf_all", "rf_selected"]


class SyndromeModel:
    """Serialized end-to-end predictor. RF is sklearn's native multi-output RF.

    PCC-MLRF ranking is the repository's simplified paper-inspired algorithm,
    not official ML-PRDF and not an exact implementation of its equations.
    """
    def __init__(self, name="rf_selected", top_k=100, trees=200, seed=0):
        if name not in MODEL_NAMES:
            raise ValueError(name)
        self.name, self.top_k, self.trees, self.seed = name, top_k, trees, seed

    def fit(self, text, labels):
        self.vectorizer = TfidfVectorizer(tokenizer=tokens, token_pattern=None, lowercase=False)
        X = self.vectorizer.fit_transform([normalize(t) for t in text])
        self.encoder = MultiLabelBinarizer()
        Y = self.encoder.fit_transform(labels)
        self.classes = list(self.encoder.classes_)
        if not self.classes:
            raise ValueError("No training syndrome labels.")
        self.selected = np.arange(X.shape[1])
        self.weights = None
        if self.name == "rf_selected":
            ranking, self.weights = rank_features_pcc_mlrf(X, Y, k=5)
            self.selected = ranking[:min(self.top_k, X.shape[1])]
        X = X[:, self.selected]
        if self.name.startswith("rf_"):
            # Shared multi-output trees; avoids hundreds of nested parallel forests.
            self.model = RandomForestClassifier(n_estimators=self.trees, min_samples_leaf=1,
                random_state=self.seed, n_jobs=2, class_weight=None)
            self.model.fit(X, Y if Y.shape[1] > 1 else Y[:, 0])
        else:
            self.model = []
            for y in Y.T:
                estimator = (DummyClassifier(strategy="prior") if self.name == "dummy" or len(np.unique(y)) < 2
                             else LogisticRegression(C=1, max_iter=1000, class_weight="balanced", random_state=self.seed))
                self.model.append(estimator.fit(X, y))
        return self

    def scores(self, texts):
        X = self.vectorizer.transform([normalize(t) for t in texts])[:, self.selected]
        if isinstance(self.model, list):
            pairs = [(m.classes_, m.predict_proba(X)) for m in self.model]
        else:
            probability = self.model.predict_proba(X)
            classes = self.model.classes_
            pairs = list(zip(classes, probability)) if isinstance(probability, list) else [(classes, probability)]
        return np.column_stack([p[:, list(c).index(1)] if 1 in c else np.zeros(X.shape[0]) for c, p in pairs])


def encode(labels, classes):
    lookup = {value: i for i, value in enumerate(classes)}
    y = np.zeros((len(labels), len(classes)), dtype=int)
    for i, row in enumerate(labels):
        for label in row:
            if label in lookup:
                y[i, lookup[label]] = 1
    return y


def evaluate(labels, scores, classes, threshold):
    """Never discard labels unseen in training: append zero-score columns."""
    unknown = sorted(set(label for row in labels for label in row) - set(classes))
    all_classes = list(classes) + unknown
    score = np.column_stack([scores, np.zeros((len(labels), len(unknown)))])
    truth = encode(labels, all_classes)
    pred = (score >= threshold).astype(int)
    metrics = {
        "micro_precision": float(precision_score(truth, pred, average="micro", zero_division=0)),
        "micro_recall": float(recall_score(truth, pred, average="micro", zero_division=0)),
        "micro_f1": float(f1_score(truth, pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(truth, pred, average="macro", zero_division=0)),
        "exact_match_accuracy": float(accuracy_score(truth, pred)),
        "hamming_loss": float(hamming_loss(truth, pred)),
        "micro_average_precision": float(average_precision_score(truth.ravel(), score.ravel())),
        "unseen_label_names": unknown,
        "unseen_positive_labels": sum(label in unknown for row in labels for label in row),
        "evaluated_labels": len(all_classes),
    }
    per_label = []
    for label, cm, column in zip(all_classes, multilabel_confusion_matrix(truth, pred), truth.T):
        tn, fp, fn, tp = map(int, cm.ravel())
        per_label.append({"syndrome": label, "support": int(column.sum()), "tn": tn, "fp": fp, "fn": fn, "tp": tp,
                          "f1": 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0})
    return metrics, per_label
