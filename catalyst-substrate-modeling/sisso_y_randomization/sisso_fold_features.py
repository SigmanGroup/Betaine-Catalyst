"""Fold-local adapter for the repository's legacy SISSO/Boruta workflow.

The training transformation follows ``SISSO_reduced_feat_gen.py`` and the
held-out transformation uses ``calculate_SISSO_features.py``'s recursive
expression evaluator. No held-out response is passed to feature generation.
"""

from __future__ import annotations

from contextlib import nullcontext, redirect_stdout
from dataclasses import dataclass
from functools import lru_cache
import importlib.util
import io
from pathlib import Path
import sys
from types import ModuleType
from typing import TextIO

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SISSOConfig:
    """Settings matching the existing reduced-feature generator defaults."""

    legacy_sisso_dir: Path
    collinearity_cutoff: float = 0.8
    relative_filter_permutations: int = 1_000
    boruta_percentile: int = 75
    boruta_max_iter: int = 100
    boruta_n_estimators: str | int = "auto"
    random_state: int = 42
    n_jobs: int = -1
    verbose: bool = False
    nonfinite_test_policy: str = "raise"


@dataclass
class FoldFeatureResult:
    """Feature matrices and selection metadata for one outer fold."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    base_features: list[str]
    selected_augmented_features: list[str]
    n_nonfinite_test_values: int
    diagnostics: dict[str, int]

    @property
    def feature_names(self) -> list[str]:
        return self.X_train.columns.tolist()


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load Python module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=4)
def _load_legacy_modules(legacy_sisso_dir: str) -> tuple[ModuleType, ModuleType]:
    legacy_dir = Path(legacy_sisso_dir).resolve()
    generator_path = legacy_dir / "SISSO_reduced_feat_gen.py"
    calculator_path = legacy_dir / "calculate_SISSO_features.py"
    if not generator_path.exists():
        raise FileNotFoundError(generator_path)
    if not calculator_path.exists():
        raise FileNotFoundError(calculator_path)
    generator = _load_module("legacy_sisso_reduced_feature_generator", generator_path)
    calculator = _load_module("legacy_sisso_feature_calculator", calculator_path)
    return generator, calculator


class _Tee:
    """Write redirected output to both the console and a log file."""

    def __init__(self, *streams: TextIO):
        self.streams = streams

    def write(self, text: str) -> int:
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def _output_context(config: SISSOConfig, log_handle: TextIO | None):
    if config.verbose and log_handle is not None:
        return redirect_stdout(_Tee(sys.stdout, log_handle))
    if config.verbose:
        return nullcontext()
    if log_handle is not None:
        return redirect_stdout(log_handle)
    return redirect_stdout(io.StringIO())


def _sanitize_base_features(
    frame: pd.DataFrame,
    generator: ModuleType,
) -> pd.DataFrame:
    sanitized = frame.copy()
    sanitized.columns = [generator.sanitize_token(column) for column in frame.columns]
    duplicated = sanitized.columns[sanitized.columns.duplicated()].unique().tolist()
    if duplicated:
        raise ValueError(
            "Sanitizing the base-feature names produced duplicates: "
            f"{duplicated}"
        )
    return sanitized


def _check_finite(frame: pd.DataFrame, label: str) -> None:
    values = frame.to_numpy(dtype=float)
    if not np.all(np.isfinite(values)):
        count = int(np.size(values) - np.isfinite(values).sum())
        raise ValueError(f"{label} contains {count} non-finite values.")


def fit_transform_sisso_fold(
    train_base: pd.DataFrame,
    y_train: np.ndarray,
    test_base: pd.DataFrame,
    config: SISSOConfig,
    log_handle: TextIO | None = None,
) -> FoldFeatureResult:
    """Fit SISSO/Boruta on one outer-training fold and transform its test fold.

    This is an in-memory reproduction of the two legacy command-line scripts:

    1. bivariate expansion of the base descriptors;
    2. response-aware collinearity and relative-permutation filtering;
    3. unary expansion of base plus retained bivariate features;
    4. Boruta selection using only ``y_train``;
    5. reconstruction of the selected expressions on ``test_base``.
    """

    generator, calculator = _load_legacy_modules(str(config.legacy_sisso_dir))
    y_train = np.asarray(y_train, dtype=float)
    if len(train_base) != len(y_train):
        raise ValueError("train_base and y_train have different row counts.")

    train_original = _sanitize_base_features(train_base, generator)
    test_original = _sanitize_base_features(test_base, generator)
    if train_original.columns.tolist() != test_original.columns.tolist():
        raise ValueError("Training and test base-feature columns do not match.")
    _check_finite(train_original, "Outer-training base features")
    _check_finite(test_original, "Outer-test base features")

    with _output_context(config, log_handle), np.errstate(all="ignore"):
        # These calls are the same expansion/filtering functions used by
        # SISSO_reduced_feat_gen.command_center.
        print("[SISSO] Generating first-iteration bivariate features...", flush=True)
        bivariate_1 = generator.apply_bivariate(
            train_original,
            verbose=config.verbose,
        )
        filtered_iter1 = generator.filter_features(
            bivariate_1,
            y_train,
            colin_cut=config.collinearity_cutoff,
            n_permutations=config.relative_filter_permutations,
        )
        print(
            "[SISSO] Bivariate features: "
            f"{bivariate_1.shape[1]} generated, "
            f"{filtered_iter1.shape[1]} retained.",
            flush=True,
        )

        iteration_2_input = pd.concat(
            [train_original, filtered_iter1],
            axis=1,
        )
        unary_2 = generator.apply_univariate(
            iteration_2_input,
            verbose=config.verbose,
        )
        print(
            f"[SISSO] Generated {unary_2.shape[1]} second-iteration "
            "unary features.",
            flush=True,
        )
        all_augmented = pd.concat([filtered_iter1, unary_2], axis=1)
        n_augmented_before_deduplication = int(all_augmented.shape[1])
        all_augmented = all_augmented.loc[
            :,
            ~all_augmented.columns.duplicated(),
        ]

        finite_columns = all_augmented.apply(
            lambda column: np.all(np.isfinite(column)),
            axis=0,
        )
        boruta_frame = all_augmented.loc[:, finite_columns].clip(
            lower=-1e6,
            upper=1e6,
        )
        print(
            f"[Boruta] Starting with {boruta_frame.shape[1]} finite "
            "candidate features.",
            flush=True,
        )

        if boruta_frame.shape[1] == 0:
            selected_augmented: list[str] = []
            print("[Boruta] No candidate features; selection skipped.", flush=True)
        else:
            forest = generator.RandomForestRegressor(
                n_jobs=config.n_jobs,
                max_depth=5,
                random_state=config.random_state,
            )
            selector = generator.BorutaPy(
                forest,
                n_estimators=config.boruta_n_estimators,
                # Iteration progress is always captured in sisso.log. With
                # config.verbose=True, _Tee also mirrors it to the console.
                verbose=1,
                random_state=config.random_state,
                perc=config.boruta_percentile,
                max_iter=config.boruta_max_iter,
            )
            selector.fit(boruta_frame.to_numpy(), y_train)
            selected_augmented = boruta_frame.columns[
                selector.support_
            ].tolist()
            print(
                f"[Boruta] Completed: selected {len(selected_augmented)} "
                f"of {boruta_frame.shape[1]} candidates.",
                flush=True,
            )

        train_augmented = boruta_frame.loc[:, selected_augmented].copy()
        test_augmented_values = {
            feature: np.clip(
                calculator.recursive_evaluate(feature, test_original),
                -1e6,
                1e6,
            )
            for feature in selected_augmented
        }

    test_augmented = pd.DataFrame(
        test_augmented_values,
        index=test_original.index,
    )
    X_train = pd.concat(
        [train_original.reset_index(drop=True), train_augmented.reset_index(drop=True)],
        axis=1,
    )
    X_test = pd.concat(
        [test_original.reset_index(drop=True), test_augmented.reset_index(drop=True)],
        axis=1,
    )
    X_test = X_test.loc[:, X_train.columns]

    _check_finite(X_train, "Outer-training SISSO feature matrix")
    test_values = X_test.to_numpy(dtype=float)
    nonfinite_count = int(np.size(test_values) - np.isfinite(test_values).sum())
    if nonfinite_count:
        if config.nonfinite_test_policy == "raise":
            raise ValueError(
                "Outer-test SISSO feature matrix contains "
                f"{nonfinite_count} non-finite values. Set "
                "nonfinite_test_policy='train_median' to use training-only "
                "median imputation."
            )
        if config.nonfinite_test_policy != "train_median":
            raise ValueError(
                "nonfinite_test_policy must be 'raise' or 'train_median'."
            )
        X_test = X_test.replace([np.inf, -np.inf], np.nan)
        train_medians = X_train.median(axis=0)
        X_test = X_test.fillna(train_medians)
        _check_finite(X_test, "Imputed outer-test SISSO feature matrix")

    return FoldFeatureResult(
        X_train=X_train,
        X_test=X_test,
        base_features=train_original.columns.tolist(),
        selected_augmented_features=selected_augmented,
        n_nonfinite_test_values=nonfinite_count,
        diagnostics={
            "n_bivariate_generated": int(bivariate_1.shape[1]),
            "n_bivariate_after_filtering": int(filtered_iter1.shape[1]),
            "n_iteration_2_input": int(iteration_2_input.shape[1]),
            "n_unary_generated": int(unary_2.shape[1]),
            "n_augmented_before_deduplication": n_augmented_before_deduplication,
            "n_augmented_after_deduplication": int(all_augmented.shape[1]),
            "n_boruta_candidate_features": int(boruta_frame.shape[1]),
            "n_boruta_selected_features": int(len(selected_augmented)),
        },
    )
