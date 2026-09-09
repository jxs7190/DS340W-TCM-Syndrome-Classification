"""Paper-inspired PCC-MLRF feature-ranking reimplementation for DS 340W.

The parent paper describes PCC-MLRF as an ML-ReliefF-style feature-selection
method that uses Pearson-correlation-based sample similarity, traversal over
all samples, same-label neighbors (Hits), different-label neighbors (Misses),
and feature-weight updates. The authors' exact source code was not identified
in the cited dataset repository, so this file is a transparent course
reimplementation rather than official author code.

Use this only on TRAINING data. Never fit feature selection on test or final
validation data.
"""

from __future__ import annotations
import numpy as np


def _pearson_similarity_matrix(X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Return similarity = 1 / (1 - rho + eps) from centered vectors.

    The paper defines Pearson correlation rho and then uses the reciprocal of
    Pearson distance as sample similarity. Values are clipped for numerical
    stability when rho is extremely close to 1.
    """
    X = np.asarray(X, dtype=float)
    centered = X - X.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    normalized = centered / np.maximum(norms, eps)
    rho = normalized @ normalized.T
    rho = np.clip(rho, -1.0, 1.0 - eps)
    return 1.0 / np.maximum(1.0 - rho, eps)


def rank_features_pcc_mlrf(
    X,
    Y,
    k: int = 5,
):
    """Rank features using a paper-inspired PCC-MLRF update rule.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Training feature matrix, typically TF-IDF.
    Y : array-like, shape (n_samples, n_labels)
        Binary multi-label syndrome matrix.
    k : int
        Number of nearest Hit/Miss neighbors used in each update.

    Returns
    -------
    ranking : ndarray
        Feature indices from highest to lowest weight.
    weights : ndarray
        Learned feature weights.

    Notes
    -----
    This implementation follows the paper's stated ideas but simplifies the
    class-specific Miss weighting for a practical course implementation.
    """
    X = X.toarray() if hasattr(X, "toarray") else np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=int)
    n_samples, n_features = X.shape

    if Y.shape[0] != n_samples:
        raise ValueError("X and Y must contain the same number of samples")
    if n_samples < 2:
        raise ValueError("At least two training samples are required")

    similarities = _pearson_similarity_matrix(X)
    np.fill_diagonal(similarities, -np.inf)
    weights = np.zeros(n_features, dtype=float)

    # Normalize feature differences to [0, 1] scale when possible.
    feature_range = X.max(axis=0) - X.min(axis=0)
    feature_range[feature_range == 0] = 1.0

    for t in range(n_samples):
        # A practical multi-label interpretation: Hit candidates share the
        # exact label set; Miss candidates do not. This keeps the update
        # deterministic and avoids leaking information from test/validation.
        same = np.all(Y == Y[t], axis=1)
        same[t] = False
        different = ~np.all(Y == Y[t], axis=1)

        hit_candidates = np.flatnonzero(same)
        miss_candidates = np.flatnonzero(different)

        if len(hit_candidates):
            hit_order = hit_candidates[
                np.argsort(similarities[t, hit_candidates])[::-1][:k]
            ]
            hit_sims = similarities[t, hit_order]
            hit_sims = np.maximum(hit_sims, 0)
            if hit_sims.sum() > 0:
                hit_diff = np.abs(X[hit_order] - X[t]) / feature_range
                weights -= (hit_sims[:, None] * hit_diff).sum(axis=0) / hit_sims.sum()

        if len(miss_candidates):
            miss_order = miss_candidates[
                np.argsort(similarities[t, miss_candidates])[::-1][:k]
            ]
            miss_sims = similarities[t, miss_order]
            miss_sims = np.maximum(miss_sims, 0)
            if miss_sims.sum() > 0:
                miss_diff = np.abs(X[miss_order] - X[t]) / feature_range
                weights += (miss_sims[:, None] * miss_diff).sum(axis=0) / miss_sims.sum()

    weights /= n_samples
    ranking = np.argsort(weights)[::-1]
    return ranking, weights


def select_top_features(X, ranking, n_features: int):
    """Return the top-ranked feature columns."""
    selected = np.asarray(ranking[:n_features], dtype=int)
    return X[:, selected], selected
