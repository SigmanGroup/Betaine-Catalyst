"""Nested 1-SE LASSO helpers with strictly fold-local standardization."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, lasso_path
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class LassoConfig:
    """Configuration for inner-CV alpha selection and outer-fold fitting."""

    alphas: np.ndarray = field(
        default_factory=lambda: np.logspace(-4, -1, 100)
    )
    n_bins: int = 5
    inner_splits: int = 5
    inner_repeats: int = 4
    random_state: int = 42
    max_iter: int = 100_000


def quantile_bins(y: np.ndarray, n_bins: int) -> np.ndarray:
    """Create approximately equally populated response bins."""

    bins = pd.qcut(y, q=n_bins, labels=False, duplicates="drop")
    bins = np.asarray(bins, dtype=int)
    if np.unique(bins).size < 2:
        raise ValueError("At least two response bins are required.")
    return bins


def select_alpha_1se(
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: LassoConfig,
    random_state: int,
) -> dict[str, float | int]:
    """Select the largest alpha within 1 SE of the minimum inner-CV MSE.

    The scaler is fitted independently in every inner-training fold. Using
    ``lasso_path`` evaluates the complete alpha grid with one path fit per
    inner fold and is substantially faster than fitting one Pipeline for every
    alpha separately.
    """

    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    inner_bins = quantile_bins(y_train, config.n_bins)
    inner_cv = RepeatedStratifiedKFold(
        n_splits=config.inner_splits,
        n_repeats=config.inner_repeats,
        random_state=random_state,
    )
    candidate_alphas = np.sort(np.asarray(config.alphas, dtype=float))[::-1]
    fold_mse: list[np.ndarray] = []

    for train_idx, valid_idx in inner_cv.split(X_train, inner_bins):
        scaler = StandardScaler()
        X_inner_train = scaler.fit_transform(X_train[train_idx])
        X_inner_valid = scaler.transform(X_train[valid_idx])
        y_inner_train = y_train[train_idx]
        y_inner_valid = y_train[valid_idx]

        # StandardScaler centers X; centering y reproduces Lasso's intercept.
        y_mean = float(y_inner_train.mean())
        path_alphas, coefficients, _ = lasso_path(
            X_inner_train,
            y_inner_train - y_mean,
            alphas=candidate_alphas,
            max_iter=config.max_iter,
        )
        if not np.allclose(path_alphas, candidate_alphas):
            raise RuntimeError("Unexpected alpha ordering from lasso_path.")

        predictions = X_inner_valid @ coefficients + y_mean
        fold_mse.append(
            np.mean((y_inner_valid[:, None] - predictions) ** 2, axis=0)
        )

    mse_matrix = np.column_stack(fold_mse)
    mean_mse = mse_matrix.mean(axis=1)
    se_mse = mse_matrix.std(axis=1, ddof=1) / np.sqrt(mse_matrix.shape[1])
    min_index = int(np.argmin(mean_mse))
    threshold = float(mean_mse[min_index] + se_mse[min_index])
    eligible = mean_mse <= threshold

    return {
        "alpha_min": float(candidate_alphas[min_index]),
        "alpha_1se": float(np.max(candidate_alphas[eligible])),
        "minimum_inner_mse": float(mean_mse[min_index]),
        "one_se_threshold": threshold,
        "n_inner_splits_total": int(mse_matrix.shape[1]),
    }


def fit_outer_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    config: LassoConfig,
    random_state: int,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Tune, fit, and evaluate one outer-fold LASSO model."""

    alpha_result = select_alpha_1se(
        X_train=X_train,
        y_train=y_train,
        config=config,
        random_state=random_state,
    )
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "lasso",
                Lasso(
                    alpha=float(alpha_result["alpha_1se"]),
                    max_iter=config.max_iter,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)
    train_prediction = np.asarray(model.predict(X_train), dtype=float)
    test_prediction = np.asarray(model.predict(X_test), dtype=float)
    coefficients = model.named_steps["lasso"].coef_

    metrics: dict[str, float | int] = {
        **alpha_result,
        "in_fold_r2": float(r2_score(y_train, train_prediction)),
        "in_fold_mae": float(mean_absolute_error(y_train, train_prediction)),
        "outer_test_r2": float(r2_score(y_test, test_prediction)),
        "outer_test_mae": float(mean_absolute_error(y_test, test_prediction)),
        "n_nonzero": int(np.sum(np.abs(coefficients) > 1e-12)),
    }
    return test_prediction, metrics
