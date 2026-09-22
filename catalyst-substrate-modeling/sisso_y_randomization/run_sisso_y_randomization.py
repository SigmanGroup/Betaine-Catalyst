"""Run fold-local SISSO/Boruta selection for observed and permuted responses.

This module is both importable from the companion notebook and executable as a
command-line program. Expensive work is checkpointed one outer fold at a time.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
import re
import time

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import StratifiedKFold

try:
    from .nested_lasso import LassoConfig, fit_outer_fold, quantile_bins
    from .sisso_fold_features import SISSOConfig, fit_transform_sisso_fold
except ImportError:  # Allow ``python run_sisso_y_randomization.py``.
    from nested_lasso import LassoConfig, fit_outer_fold, quantile_bins
    from sisso_fold_features import SISSOConfig, fit_transform_sisso_fold


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = (
    SCRIPT_DIR.parent / "baseline_validation" / "training_set_base.csv"
)
DEFAULT_LEGACY_SISSO_DIR = (
    SCRIPT_DIR.parents[1] / "descriptor_generation" / "sisso"
)
DEFAULT_OUTPUT = SCRIPT_DIR / "results"


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration for the complete Y-randomization experiment."""

    input_file: Path = DEFAULT_INPUT
    output_dir: Path = DEFAULT_OUTPUT
    legacy_sisso_dir: Path = DEFAULT_LEGACY_SISSO_DIR
    n_shuffles: int = 20
    outer_splits: int = 5
    n_bins: int = 5
    random_state: int = 42
    resume: bool = True
    sisso: SISSOConfig | None = None
    lasso: LassoConfig = field(default_factory=LassoConfig)

    def resolved_sisso(self) -> SISSOConfig:
        if self.sisso is not None:
            return self.sisso
        return SISSOConfig(legacy_sisso_dir=self.legacy_sisso_dir)


@dataclass
class ResponseEvaluation:
    """Results for the observed response or one response permutation."""

    label: str
    summary: dict[str, float | int | str]
    fold_metrics: pd.DataFrame
    predictions: pd.DataFrame


@dataclass
class ExperimentResult:
    """Combined observed and null-distribution results."""

    response_summary: pd.DataFrame
    empirical_significance: pd.DataFrame
    fold_metrics: pd.DataFrame
    predictions: pd.DataFrame


def _json_ready(value):
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _analysis_manifest(config: ExperimentConfig) -> dict:
    """Return settings that must remain unchanged when resuming a run."""

    sisso_settings = asdict(config.resolved_sisso())
    lasso_settings = asdict(config.lasso)
    return _json_ready(
        {
            "input_file": config.input_file,
            "input_sha256": hashlib.sha256(
                config.input_file.read_bytes()
            ).hexdigest(),
            "legacy_sisso_dir": config.legacy_sisso_dir,
            "outer_splits": config.outer_splits,
            "n_bins": config.n_bins,
            "random_state": config.random_state,
            "sisso": sisso_settings,
            "lasso": lasso_settings,
        }
    )


def _prepare_output(config: ExperimentConfig) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = config.output_dir / "analysis_configuration.json"
    current = _analysis_manifest(config)
    if manifest_path.exists() and config.resume:
        previous = json.loads(manifest_path.read_text())
        if previous != current:
            raise ValueError(
                "The saved result configuration differs from the current "
                "configuration. Use a new output directory or remove the old "
                "results before rerunning."
            )
    manifest_path.write_text(json.dumps(current, indent=2) + "\n")


def load_input_table(path: Path) -> tuple[pd.DataFrame, list[str]]:
    """Load and validate the base-descriptor table."""

    frame = pd.read_csv(path)
    required = {"cat_substrate", "ddG"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Input table is missing columns: {sorted(missing)}")
    if frame["cat_substrate"].duplicated().any():
        raise ValueError("cat_substrate identifiers must be unique.")
    base_features = [
        column for column in frame.columns if re.fullmatch(r"x\d+", column)
    ]
    if not base_features:
        raise ValueError("No base descriptors matching x<number> were found.")
    values = frame[base_features + ["ddG"]].to_numpy(dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("The input descriptors and response must be finite.")
    return frame.reset_index(drop=True), base_features


def _load_completed_response(run_dir: Path) -> ResponseEvaluation | None:
    summary_path = run_dir / "summary.json"
    folds_path = run_dir / "fold_metrics.csv"
    predictions_path = run_dir / "predictions.csv"
    if not all(path.exists() for path in [summary_path, folds_path, predictions_path]):
        return None
    summary = json.loads(summary_path.read_text())
    return ResponseEvaluation(
        label=str(summary["response_id"]),
        summary=summary,
        fold_metrics=pd.read_csv(folds_path),
        predictions=pd.read_csv(predictions_path),
    )


def _load_fold_checkpoint(fold_dir: Path) -> tuple[dict, pd.DataFrame] | None:
    record_path = fold_dir / "fold_record.json"
    prediction_path = fold_dir / "test_predictions.csv"
    if not record_path.exists() or not prediction_path.exists():
        return None
    return json.loads(record_path.read_text()), pd.read_csv(prediction_path)


def _save_fold_checkpoint(
    fold_dir: Path,
    record: dict,
    prediction_table: pd.DataFrame,
) -> None:
    fold_dir.mkdir(parents=True, exist_ok=True)
    (fold_dir / "fold_record.json").write_text(
        json.dumps(_json_ready(record), indent=2) + "\n"
    )
    prediction_table.to_csv(fold_dir / "test_predictions.csv", index=False)
    selected = str(record.get("selected_augmented_features", ""))
    (fold_dir / "selected_augmented_features.txt").write_text(
        "\n".join(filter(None, selected.split(";"))) + "\n"
    )


def evaluate_response(
    frame: pd.DataFrame,
    base_features: list[str],
    response: np.ndarray,
    response_id: str,
    response_kind: str,
    permutation: int,
    config: ExperimentConfig,
) -> ResponseEvaluation:
    """Evaluate one response vector with outer-fold SISSO and nested LASSO."""

    run_dir = config.output_dir / "runs" / response_id
    if config.resume:
        completed = _load_completed_response(run_dir)
        if completed is not None:
            print(f"[resume] {response_id} is already complete")
            return completed
    run_dir.mkdir(parents=True, exist_ok=True)

    response = np.asarray(response, dtype=float)
    outer_bins = quantile_bins(response, config.n_bins)
    outer_cv = StratifiedKFold(
        n_splits=config.outer_splits,
        shuffle=True,
        random_state=config.random_state,
    )
    oof_prediction = np.full(response.shape, np.nan, dtype=float)
    fold_records: list[dict] = []
    sisso_config = config.resolved_sisso()

    for fold_number, (train_idx, test_idx) in enumerate(
        outer_cv.split(frame[base_features], outer_bins),
        start=1,
    ):
        fold_dir = run_dir / f"fold_{fold_number:02d}"
        cached = _load_fold_checkpoint(fold_dir) if config.resume else None
        if cached is not None:
            record, prediction_table = cached
            cached_test_idx = prediction_table["row_index"].to_numpy(dtype=int)
            if not np.array_equal(cached_test_idx, test_idx):
                raise ValueError(f"Cached indices do not match for {fold_dir}")
            oof_prediction[test_idx] = prediction_table[
                "oof_prediction"
            ].to_numpy(dtype=float)
            fold_records.append(record)
            print(f"  [resume] {response_id}, fold {fold_number}", flush=True)
            continue

        print(
            f"  [{response_id}] fitting SISSO/Boruta fold {fold_number}",
            flush=True,
        )
        fold_start = time.time()
        train_base = frame.loc[train_idx, base_features]
        test_base = frame.loc[test_idx, base_features]
        log_path = fold_dir / "sisso.log"
        fold_dir.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", buffering=1) as log_handle:
            feature_result = fit_transform_sisso_fold(
                train_base=train_base,
                y_train=response[train_idx],
                test_base=test_base,
                config=sisso_config,
                log_handle=log_handle,
            )

        test_prediction, model_metrics = fit_outer_fold(
            X_train=feature_result.X_train.to_numpy(dtype=float),
            y_train=response[train_idx],
            X_test=feature_result.X_test.to_numpy(dtype=float),
            y_test=response[test_idx],
            config=config.lasso,
            random_state=config.random_state + fold_number,
        )
        oof_prediction[test_idx] = test_prediction
        record = {
            "response_id": response_id,
            "response_kind": response_kind,
            "permutation": permutation,
            "outer_fold": fold_number,
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "n_base_features": int(len(feature_result.base_features)),
            "n_selected_augmented_features": int(
                len(feature_result.selected_augmented_features)
            ),
            "n_total_features": int(len(feature_result.feature_names)),
            "selected_augmented_features": ";".join(
                feature_result.selected_augmented_features
            ),
            "n_nonfinite_test_values": int(
                feature_result.n_nonfinite_test_values
            ),
            "elapsed_seconds": float(time.time() - fold_start),
            **feature_result.diagnostics,
            **model_metrics,
        }
        prediction_table = pd.DataFrame(
            {
                "row_index": test_idx,
                "cat_substrate": frame.loc[test_idx, "cat_substrate"].to_numpy(),
                "response": response[test_idx],
                "oof_prediction": test_prediction,
            }
        )
        _save_fold_checkpoint(fold_dir, record, prediction_table)
        fold_records.append(record)
        print(
            "    SISSO: "
            f"{record['n_bivariate_generated']} bivariate -> "
            f"{record['n_bivariate_after_filtering']} filtered; "
            f"{record['n_boruta_candidate_features']} Boruta candidates -> "
            f"{record['n_boruta_selected_features']} selected",
            flush=True,
        )
        print(
            "    LASSO: "
            f"alpha_min={record['alpha_min']:.5g}, "
            f"alpha_1se={record['alpha_1se']:.5g}, "
            f"nonzero={record['n_nonzero']}; "
            f"train R2/MAE={record['in_fold_r2']:.3f}/"
            f"{record['in_fold_mae']:.3f}; "
            f"test R2/MAE={record['outer_test_r2']:.3f}/"
            f"{record['outer_test_mae']:.3f}; "
            f"{record['elapsed_seconds']:.1f} s",
            flush=True,
        )

    if np.any(~np.isfinite(oof_prediction)):
        raise RuntimeError(f"{response_id} has missing OOF predictions.")

    fold_metrics = pd.DataFrame(fold_records).sort_values("outer_fold")
    summary: dict[str, float | int | str] = {
        "response_id": response_id,
        "response_kind": response_kind,
        "permutation": permutation,
        "n_observations": int(len(response)),
        "n_outer_folds": int(config.outer_splits),
        "mean_in_fold_r2": float(fold_metrics["in_fold_r2"].mean()),
        "mean_in_fold_mae": float(fold_metrics["in_fold_mae"].mean()),
        "oof_r2": float(r2_score(response, oof_prediction)),
        "oof_mae": float(mean_absolute_error(response, oof_prediction)),
        "alpha_1se_min": float(fold_metrics["alpha_1se"].min()),
        "alpha_1se_median": float(fold_metrics["alpha_1se"].median()),
        "alpha_1se_max": float(fold_metrics["alpha_1se"].max()),
        "selected_augmented_min": int(
            fold_metrics["n_selected_augmented_features"].min()
        ),
        "selected_augmented_median": float(
            fold_metrics["n_selected_augmented_features"].median()
        ),
        "selected_augmented_max": int(
            fold_metrics["n_selected_augmented_features"].max()
        ),
        "nonzero_lasso_median": float(fold_metrics["n_nonzero"].median()),
    }
    predictions = pd.DataFrame(
        {
            "response_id": response_id,
            "response_kind": response_kind,
            "permutation": permutation,
            "row_index": np.arange(len(response)),
            "cat_substrate": frame["cat_substrate"],
            "response": response,
            "oof_prediction": oof_prediction,
        }
    )
    fold_metrics.to_csv(run_dir / "fold_metrics.csv", index=False)
    predictions.to_csv(run_dir / "predictions.csv", index=False)
    (run_dir / "summary.json").write_text(
        json.dumps(_json_ready(summary), indent=2) + "\n"
    )
    return ResponseEvaluation(
        label=response_id,
        summary=summary,
        fold_metrics=fold_metrics,
        predictions=predictions,
    )


def _empirical_significance(response_summary: pd.DataFrame) -> pd.DataFrame:
    observed = response_summary.loc[
        response_summary["response_kind"] == "observed"
    ].iloc[0]
    null = response_summary.loc[
        response_summary["response_kind"] == "permuted"
    ]
    n = len(null)
    if n == 0:
        return pd.DataFrame(
            columns=["metric", "observed", "null_mean", "null_std", "p_value"]
        )

    specifications = [
        ("oof_r2", "higher"),
        ("oof_mae", "lower"),
        ("mean_in_fold_r2", "higher"),
        ("mean_in_fold_mae", "lower"),
    ]
    rows = []
    for metric, direction in specifications:
        observed_value = float(observed[metric])
        null_values = null[metric].to_numpy(dtype=float)
        if direction == "higher":
            extreme = np.sum(null_values >= observed_value)
        else:
            extreme = np.sum(null_values <= observed_value)
        rows.append(
            {
                "metric": metric,
                "observed": observed_value,
                "null_mean": float(null_values.mean()),
                "null_std": float(null_values.std(ddof=1)) if n > 1 else np.nan,
                "p_value": float((extreme + 1) / (n + 1)),
                "n_permutations": n,
            }
        )
    return pd.DataFrame(rows)


def _combine_results(evaluations: list[ResponseEvaluation]) -> ExperimentResult:
    response_summary = pd.DataFrame(
        [evaluation.summary for evaluation in evaluations]
    )
    fold_metrics = pd.concat(
        [evaluation.fold_metrics for evaluation in evaluations],
        ignore_index=True,
    )
    predictions = pd.concat(
        [evaluation.predictions for evaluation in evaluations],
        ignore_index=True,
    )
    empirical = _empirical_significance(response_summary)
    return ExperimentResult(
        response_summary=response_summary,
        empirical_significance=empirical,
        fold_metrics=fold_metrics,
        predictions=predictions,
    )


def _save_combined(result: ExperimentResult, output_dir: Path) -> None:
    result.response_summary.to_csv(output_dir / "response_summary.csv", index=False)
    result.empirical_significance.to_csv(
        output_dir / "empirical_significance.csv",
        index=False,
    )
    result.fold_metrics.to_csv(output_dir / "all_fold_metrics.csv", index=False)
    result.predictions.to_csv(output_dir / "all_predictions.csv", index=False)


def run_experiment(config: ExperimentConfig) -> ExperimentResult:
    """Run or resume the observed and N-permutation experiment."""

    config = ExperimentConfig(
        **{
            **asdict(config),
            "input_file": Path(config.input_file).resolve(),
            "output_dir": Path(config.output_dir).resolve(),
            "legacy_sisso_dir": Path(config.legacy_sisso_dir).resolve(),
            "sisso": config.resolved_sisso(),
            "lasso": config.lasso,
        }
    )
    _prepare_output(config)
    frame, base_features = load_input_table(config.input_file)
    y_observed = frame["ddG"].to_numpy(dtype=float)
    evaluations = [
        evaluate_response(
            frame=frame,
            base_features=base_features,
            response=y_observed,
            response_id="observed",
            response_kind="observed",
            permutation=0,
            config=config,
        )
    ]

    rng = np.random.default_rng(config.random_state)
    for permutation in range(1, config.n_shuffles + 1):
        shuffled = rng.permutation(y_observed)
        response_id = f"permutation_{permutation:04d}"
        print(f"[{permutation}/{config.n_shuffles}] {response_id}")
        evaluations.append(
            evaluate_response(
                frame=frame,
                base_features=base_features,
                response=shuffled,
                response_id=response_id,
                response_kind="permuted",
                permutation=permutation,
                config=config,
            )
        )
        _save_combined(_combine_results(evaluations), config.output_dir)

    result = _combine_results(evaluations)
    _save_combined(result, config.output_dir)
    return result


def load_experiment_results(output_dir: Path = DEFAULT_OUTPUT) -> ExperimentResult:
    """Load combined result tables without recomputing models."""

    output_dir = Path(output_dir)
    return ExperimentResult(
        response_summary=pd.read_csv(output_dir / "response_summary.csv"),
        empirical_significance=pd.read_csv(
            output_dir / "empirical_significance.csv"
        ),
        fold_metrics=pd.read_csv(output_dir / "all_fold_metrics.csv"),
        predictions=pd.read_csv(output_dir / "all_predictions.csv"),
    )


def inspect_progress(output_dir: Path = DEFAULT_OUTPUT) -> pd.DataFrame:
    """Summarize completed folds and responses in a partial or finished run."""

    runs_dir = Path(output_dir) / "runs"
    rows = []
    if not runs_dir.exists():
        return pd.DataFrame(
            columns=[
                "response_id",
                "completed_folds",
                "response_complete",
                "latest_fold",
                "latest_fold_seconds",
                "latest_selected_augmented",
                "latest_alpha_1se",
                "latest_log",
            ]
        )
    for run_dir in sorted(path for path in runs_dir.iterdir() if path.is_dir()):
        fold_records = []
        for record_path in sorted(run_dir.glob("fold_*/fold_record.json")):
            fold_records.append(json.loads(record_path.read_text()))
        latest = fold_records[-1] if fold_records else {}
        log_paths = sorted(run_dir.glob("fold_*/sisso.log"))
        rows.append(
            {
                "response_id": run_dir.name,
                "completed_folds": len(fold_records),
                "response_complete": (run_dir / "summary.json").exists(),
                "latest_fold": latest.get("outer_fold", np.nan),
                "latest_fold_seconds": latest.get("elapsed_seconds", np.nan),
                "latest_selected_augmented": latest.get(
                    "n_boruta_selected_features",
                    np.nan,
                ),
                "latest_alpha_1se": latest.get("alpha_1se", np.nan),
                "latest_log": str(log_paths[-1]) if log_paths else "",
            }
        )
    return pd.DataFrame(rows)


def load_fold_debug(
    output_dir: Path,
    response_id: str,
    outer_fold: int,
) -> tuple[dict, str]:
    """Load the structured fold record and captured legacy SISSO log."""

    fold_dir = (
        Path(output_dir)
        / "runs"
        / response_id
        / f"fold_{outer_fold:02d}"
    )
    record_path = fold_dir / "fold_record.json"
    log_path = fold_dir / "sisso.log"
    record = json.loads(record_path.read_text()) if record_path.exists() else {}
    log_text = log_path.read_text() if log_path.exists() else ""
    return record, log_text


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--legacy-sisso-dir", type=Path, default=DEFAULT_LEGACY_SISSO_DIR)
    parser.add_argument("--n-shuffles", type=int, default=20)
    parser.add_argument("--outer-splits", type=int, default=5)
    parser.add_argument("--inner-splits", type=int, default=5)
    parser.add_argument("--inner-repeats", type=int, default=4)
    parser.add_argument("--n-bins", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--boruta-max-iter", type=int, default=100)
    parser.add_argument("--boruta-percentile", type=int, default=75)
    parser.add_argument("--collinearity-cutoff", type=float, default=0.8)
    parser.add_argument("--no-resume", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    lasso_config = LassoConfig(
        n_bins=args.n_bins,
        inner_splits=args.inner_splits,
        inner_repeats=args.inner_repeats,
        random_state=args.random_state,
    )
    sisso_config = SISSOConfig(
        legacy_sisso_dir=args.legacy_sisso_dir,
        collinearity_cutoff=args.collinearity_cutoff,
        boruta_percentile=args.boruta_percentile,
        boruta_max_iter=args.boruta_max_iter,
        random_state=args.random_state,
    )
    config = ExperimentConfig(
        input_file=args.input,
        output_dir=args.output,
        legacy_sisso_dir=args.legacy_sisso_dir,
        n_shuffles=args.n_shuffles,
        outer_splits=args.outer_splits,
        n_bins=args.n_bins,
        random_state=args.random_state,
        resume=not args.no_resume,
        sisso=sisso_config,
        lasso=lasso_config,
    )
    result = run_experiment(config)
    print(result.empirical_significance.to_string(index=False))


if __name__ == "__main__":
    main()
