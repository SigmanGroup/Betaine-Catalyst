"""Plotting helpers for fold-local SISSO Y-randomization results."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _central_display_values(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Separate catastrophic extrapolations from the plotted central scale.

    Values farther than 100 interquartile ranges from the box are retained in
    all calculations but annotated rather than allowed to compress the useful
    portion of a histogram.
    """

    values = np.asarray(values, dtype=float)
    q1, q3 = np.quantile(values, [0.25, 0.75])
    iqr = float(q3 - q1)
    if iqr == 0:
        return values, np.array([], dtype=float)
    far_outside = (values < q1 - 100 * iqr) | (values > q3 + 100 * iqr)
    if far_outside.sum() == 0 or (~far_outside).sum() < 0.95 * len(values):
        return values, np.array([], dtype=float)
    return values[~far_outside], values[far_outside]


def plot_null_distributions(
    response_summary: pd.DataFrame,
    empirical_significance: pd.DataFrame,
):
    """Plot central OOF null distributions without hiding extreme results."""

    observed = response_summary.loc[
        response_summary["response_kind"] == "observed"
    ].iloc[0]
    null = response_summary.loc[
        response_summary["response_kind"] == "permuted"
    ]
    if null.empty:
        raise ValueError("No completed permutations are available to plot.")
    p_values = empirical_significance.set_index("metric")["p_value"]
    specifications = [
        ("oof_r2", r"OOF $R^2$"),
        ("oof_mae", r"OOF MAE (kcal mol$^{-1}$)"),
    ]

    figure, axes = plt.subplots(1, 2, figsize=(7.6, 3.8))
    for axis, (metric, label) in zip(axes, specifications):
        null_values = null[metric].to_numpy(dtype=float)
        displayed, off_scale = _central_display_values(null_values)
        median = float(np.median(null_values))
        interval_low, interval_high = np.quantile(null_values, [0.025, 0.975])
        axis.hist(
            displayed,
            bins=25,
            color="lightgray",
            edgecolor="black",
            linewidth=0.6,
        )
        axis.axvspan(
            interval_low,
            interval_high,
            color="#557A95",
            alpha=0.14,
            linewidth=0,
        )
        axis.axvline(median, color="#315A76", linewidth=1.4)

        displayed_low = float(displayed.min())
        displayed_high = float(displayed.max())
        span = displayed_high - displayed_low
        padding = 0.06 * span if span else 0.1
        axis.set_xlim(displayed_low - padding, displayed_high + padding)

        observed_value = float(observed[metric])
        if observed_value > displayed_high:
            observed_text = f"Observed = {observed_value:.3f}  →"
            observed_ha = "right"
            observed_x = 0.98
        elif observed_value < displayed_low:
            observed_text = f"←  Observed = {observed_value:.3f}"
            observed_ha = "left"
            observed_x = 0.02
        else:
            axis.axvline(
                observed_value,
                color="#9E2A2B",
                linestyle="--",
                linewidth=1.4,
            )
            observed_text = f"Observed = {observed_value:.3f}"
            observed_ha = "right"
            observed_x = 0.98
        axis.text(
            observed_x,
            0.96,
            observed_text,
            transform=axis.transAxes,
            ha=observed_ha,
            va="top",
            color="#9E2A2B",
            fontsize=9,
            fontweight="bold",
        )

        if off_scale.size:
            extreme_label = ", ".join(f"{value:.2f}" for value in off_scale)
            axis.text(
                0.02,
                0.82,
                f"{len(off_scale)} null value off scale: {extreme_label}\n"
                "retained in statistics",
                transform=axis.transAxes,
                ha="left",
                va="top",
                fontsize=7.5,
            )

        axis.set_xlabel(label)
        axis.set_ylabel("Count")
        axis.set_title(
            f"Null median = {median:.3f}\n"
            f"Central 95%: [{interval_low:.3f}, {interval_high:.3f}]\n"
            f"Empirical $p$ = {p_values.loc[metric]:.4f}",
            fontsize=9.5,
            pad=8,
        )
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    figure.suptitle(
        f"Fold-local SISSO Y-randomization ({len(null)} permutations)",
        fontsize=12,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    return figure, axes


def plot_best_randomized_parity(
    response_summary: pd.DataFrame,
    predictions: pd.DataFrame,
):
    """Plot OOF predictions for the permutation with the highest OOF R2."""

    null = response_summary.loc[
        response_summary["response_kind"] == "permuted"
    ]
    if null.empty:
        raise ValueError("No completed permutations are available to plot.")
    best = null.loc[null["oof_r2"].idxmax()]
    best_predictions = predictions.loc[
        predictions["response_id"] == best["response_id"]
    ].sort_values("row_index")
    if best_predictions.empty:
        raise ValueError(f"No predictions found for {best['response_id']}")

    y_true = best_predictions["response"].to_numpy(dtype=float)
    y_pred = best_predictions["oof_prediction"].to_numpy(dtype=float)
    lower = min(y_true.min(), y_pred.min()) - 0.2
    upper = max(y_true.max(), y_pred.max()) + 0.2

    figure, axis = plt.subplots(figsize=(4.2, 3.45))
    axis.scatter(y_true, y_pred, color="#B8860B", alpha=0.65, s=30)
    axis.plot([lower, upper], [lower, upper], "k--", linewidth=1, alpha=0.3)
    axis.set_xlim(lower, upper)
    axis.set_ylim(lower, upper)
    axis.set_xlabel(
        r"Randomized measured $\Delta\Delta G^{\ddagger}$" "\n"
        r"(kcal mol$^{-1}$)"
    )
    axis.set_ylabel(
        r"OOF-predicted $\Delta\Delta G^{\ddagger}$" "\n"
        r"(kcal mol$^{-1}$)"
    )
    axis.set_title(f"Best null permutation (#{int(best['permutation'])})")
    stats = (
        f"Mean in-fold $R^2$ = {best['mean_in_fold_r2']:.3f}\n"
        f"Mean in-fold MAE = {best['mean_in_fold_mae']:.3f}\n"
        f"OOF $R^2$ = {best['oof_r2']:.3f}\n"
        f"OOF MAE = {best['oof_mae']:.3f}\n"
        rf"median $\alpha_{{1\mathrm{{SE}}}}$ = {best['alpha_1se_median']:.5f}"
    )
    axis.text(
        0.03,
        0.97,
        stats,
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.75,
        },
    )
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    figure.tight_layout()
    return figure, axis


def plot_feature_frequency_distribution(fold_metrics: pd.DataFrame):
    """Summarize Boruta selection stability across observed-response folds."""

    observed_folds = fold_metrics.loc[
        fold_metrics["response_kind"] == "observed"
    ]
    n_folds = int(observed_folds["outer_fold"].nunique())
    frequencies = feature_selection_frequency(fold_metrics, "observed")
    if frequencies.empty:
        raise ValueError("No augmented features were selected for observed y.")
    counts = (
        frequencies["selection_count"]
        .value_counts()
        .reindex(range(1, n_folds + 1), fill_value=0)
        .sort_index()
    )

    figure, axis = plt.subplots(figsize=(4.6, 3.2))
    positions = np.arange(1, n_folds + 1)
    bars = axis.bar(
        positions,
        counts.to_numpy(dtype=int),
        color="#557A95",
        width=0.72,
    )
    axis.bar_label(bars, padding=3, fontsize=8)
    axis.set_xticks(positions)
    axis.set_xticklabels(
        [f"{count}/{n_folds}" for count in positions],
    )
    axis.set_xlabel("Selection frequency across the five outer folds")
    axis.set_ylabel("Number of augmented features")
    axis.set_title("Observed-response Boruta selection stability")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.set_ylim(0, max(counts) * 1.12)
    figure.tight_layout()
    return figure, axis


def feature_selection_frequency(
    fold_metrics: pd.DataFrame,
    response_kind: str = "observed",
) -> pd.DataFrame:
    """Calculate outer-fold selection frequencies for augmented features."""

    selected_rows = fold_metrics.loc[
        fold_metrics["response_kind"] == response_kind,
        "selected_augmented_features",
    ]
    counts: dict[str, int] = {}
    for value in selected_rows.fillna(""):
        for feature in filter(None, str(value).split(";")):
            counts[feature] = counts.get(feature, 0) + 1
    denominator = len(selected_rows)
    result = pd.DataFrame(
        [
            {
                "feature": feature,
                "selection_count": count,
                "selection_frequency": count / denominator,
            }
            for feature, count in counts.items()
        ]
    )
    if result.empty:
        return pd.DataFrame(
            columns=["feature", "selection_count", "selection_frequency"]
        )
    return result.sort_values(
        ["selection_frequency", "feature"],
        ascending=[False, True],
    ).reset_index(drop=True)


def plot_top_feature_frequencies(
    fold_metrics: pd.DataFrame,
    top_n: int = 20,
):
    """Plot the most frequently selected augmented features for observed y."""

    frequencies = feature_selection_frequency(fold_metrics, "observed").head(top_n)
    if frequencies.empty:
        raise ValueError("No augmented features were selected for observed y.")
    plot_data = frequencies.sort_values("selection_frequency")
    height = max(3.0, 0.28 * len(plot_data))
    figure, axis = plt.subplots(figsize=(6.5, height))
    axis.barh(
        plot_data["feature"],
        plot_data["selection_frequency"],
        color="#557A95",
    )
    axis.set_xlim(0, 1)
    axis.set_xlabel("Selection frequency across observed-response outer folds")
    axis.set_ylabel("")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    figure.tight_layout()
    return figure, axis


def save_figure(figure, path: Path, dpi: int = 300) -> None:
    """Save one figure as both PNG and PDF."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    png_path = path.with_suffix(".png")
    pdf_path = path.with_suffix(".pdf")
    figure.savefig(png_path, dpi=dpi, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
