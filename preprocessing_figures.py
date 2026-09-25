"""Training-split charts written by preprocess.py.

Figures are diagnostics. They are not part of the fitted preprocessor the API loads.
Every chart uses the training rows only.
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from preprocess import (
    BINARY_CANDIDATES,
    EDUCATION_ORDER,
    MIN_ABS_CORRELATION,
    MIN_DEFAULT_RATE_GAP,
    NOMINAL_CANDIDATES,
    NUMERIC_CANDIDATES,
    ORDINAL_CANDIDATES,
    LoanFeatureEngineer,
)

FIGURE_NAMES = (
    "01_class_balance.png",
    "02_histograms_raw.png",
    "03_histograms_engineered.png",
    "04_boxplots_before_cap.png",
    "05_boxplots_after_cap.png",
    "06_correlation_before.png",
    "07_correlation_after.png",
    "08_target_correlation.png",
    "09_default_rate_by_category.png",
)

RAW_NUMERIC = [
    "Age",
    "Income",
    "LoanAmount",
    "CreditScore",
    "MonthsEmployed",
    "NumCreditLines",
    "InterestRate",
    "LoanTerm",
    "DTIRatio",
]
ENGINEERED_NUMERIC = [
    "loan_to_income",
    "estimated_monthly_payment",
    "payment_to_income",
]
CATEGORY_ORDER = {
    "Education": EDUCATION_ORDER,
    "EmploymentType": ["Full-time", "Part-time", "Self-employed", "Unemployed"],
    "MaritalStatus": ["Single", "Married", "Divorced"],
    "LoanPurpose": ["Home", "Auto", "Education", "Business", "Other"],
    "HasMortgage": ["No", "Yes"],
    "HasDependents": ["No", "Yes"],
    "HasCoSigner": ["No", "Yes"],
}

NO_DEFAULT_COLOR = "#4C78A8"
DEFAULT_COLOR = "#E45756"
KEPT_COLOR = "#4C78A8"
DROPPED_COLOR = "#B0B0B0"
FENCE_COLOR = "#F58518"


def save_preprocessing_figures(
    output_dir: Path,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    selection: pd.DataFrame,
) -> list[Path]:
    """Save the preprocessing chart set and return the written paths."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    features = x_train.reset_index(drop=True)
    target = pd.Series(np.asarray(y_train).astype(int), name="Default")
    engineered = LoanFeatureEngineer().fit_transform(features).reset_index(drop=True)
    numeric = engineered[list(NUMERIC_CANDIDATES)].apply(pd.to_numeric, errors="coerce")
    fences = _iqr_fences(numeric)
    capped = _apply_fences(numeric, fences)
    kept_numeric = _kept_features(selection, "numeric", NUMERIC_CANDIDATES)

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": True,
            "grid.alpha": 0.25,
            "font.size": 10,
        }
    )

    writers = (
        lambda: _class_balance(plt, output_dir / FIGURE_NAMES[0], target),
        lambda: _histograms(
            plt,
            output_dir / FIGURE_NAMES[1],
            features,
            target,
            RAW_NUMERIC,
            "Training split: raw numeric distributions by default",
        ),
        lambda: _histograms(
            plt,
            output_dir / FIGURE_NAMES[2],
            numeric,
            target,
            ENGINEERED_NUMERIC,
            "Training split: engineered ratios before the IQR cap",
        ),
        lambda: _boxplots(
            plt,
            output_dir / FIGURE_NAMES[3],
            numeric,
            "Training split: box plots before the IQR cap",
            "Whiskers run from the minimum to the maximum. Dashed lines are the training 1.5×IQR fences.",
            fences,
        ),
        lambda: _boxplots(
            plt,
            output_dir / FIGURE_NAMES[4],
            capped,
            "Training split: box plots after the 1.5×IQR cap",
            "Same axis limits as the before chart. Whiskers shorten where values were clipped to the fence.",
            fences,
            share_limits_with=numeric,
        ),
        lambda: _correlation_heatmap(
            plt,
            output_dir / FIGURE_NAMES[5],
            numeric,
            target,
            "Training split: correlation before feature selection",
        ),
        lambda: _correlation_heatmap(
            plt,
            output_dir / FIGURE_NAMES[6],
            numeric[kept_numeric],
            target,
            "Training split: correlation after feature selection",
        ),
        lambda: _target_correlation(
            plt,
            output_dir / FIGURE_NAMES[7],
            numeric,
            target,
            selection,
        ),
        lambda: _category_default_rates(
            plt,
            output_dir / FIGURE_NAMES[8],
            features,
            target,
            selection,
        ),
    )
    return [writer() for writer in writers]


def _kept_features(selection: pd.DataFrame, kind: str, order: list[str]) -> list[str]:
    kept = set(selection.loc[(selection["kind"] == kind) & selection["keep"], "feature"])
    return [feature for feature in order if feature in kept]


def _iqr_fences(frame: pd.DataFrame) -> dict[str, tuple[float, float]]:
    fences = {}
    for column in frame.columns:
        values = frame[column].to_numpy(dtype=float)
        q1, q3 = np.nanpercentile(values, [25, 75])
        iqr = q3 - q1
        fences[column] = (float(q1 - 1.5 * iqr), float(q3 + 1.5 * iqr))
    return fences


def _apply_fences(frame: pd.DataFrame, fences: dict[str, tuple[float, float]]) -> pd.DataFrame:
    capped = frame.copy()
    for column, (lower, upper) in fences.items():
        capped[column] = np.clip(frame[column].to_numpy(dtype=float), lower, upper)
    return capped


def _save(plt, figure, path: Path) -> Path:
    figure.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(figure)
    return path


def _class_balance(plt, path: Path, target: pd.Series) -> Path:
    counts = target.value_counts().reindex([0, 1], fill_value=0)
    total = int(counts.sum())
    figure, axis = plt.subplots(figsize=(6, 4))
    bars = axis.bar(
        ["No default", "Default"],
        counts.to_numpy(),
        color=[NO_DEFAULT_COLOR, DEFAULT_COLOR],
    )
    for bar, count in zip(bars, counts.to_numpy()):
        share = count / total if total else 0
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{int(count):,}\n{share:.1%}",
            ha="center",
            va="bottom",
        )
    axis.set_ylabel("Applications")
    axis.set_title("Training split: class balance")
    axis.set_ylim(0, max(counts.max() * 1.2, 1))
    figure.tight_layout()
    return _save(plt, figure, path)


def _histograms(plt, path: Path, frame: pd.DataFrame, target: pd.Series, columns: list[str], title: str) -> Path:
    from matplotlib.ticker import FuncFormatter, MaxNLocator
    ncols = 3
    nrows = math.ceil(len(columns) / ncols)
    figure, axes = plt.subplots(nrows, ncols, figsize=(12, 3.2 * nrows), squeeze=False)
    for axis, column in zip(axes.flat, columns):
        values = pd.to_numeric(frame[column], errors="coerce")
        for label, name, color in (
            (0, "No default", NO_DEFAULT_COLOR),
            (1, "Default", DEFAULT_COLOR),
        ):
            subset = values[target.to_numpy() == label].dropna()
            axis.hist(subset, bins=30, density=True, alpha=0.65, label=name, color=color)
        axis.set_title(column)
        axis.set_ylabel("Density")
        axis.xaxis.set_major_locator(MaxNLocator(nbins=5))
        axis.xaxis.set_major_formatter(FuncFormatter(_compact_tick))
        axis.tick_params(axis="x", labelsize=8)
    for axis in axes.flat[len(columns) :]:
        axis.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, frameon=False)
    figure.suptitle(title)
    figure.tight_layout()
    return _save(plt, figure, path)


def _boxplots(
    plt,
    path: Path,
    frame: pd.DataFrame,
    title: str,
    note: str,
    fences: dict[str, tuple[float, float]],
    share_limits_with: pd.DataFrame | None = None,
) -> Path:
    columns = list(frame.columns)
    ncols = 4
    nrows = math.ceil(len(columns) / ncols)
    figure, axes = plt.subplots(nrows, ncols, figsize=(14, 3.1 * nrows), squeeze=False)
    limits = share_limits_with if share_limits_with is not None else frame
    for axis, column in zip(axes.flat, columns):
        values = frame[column].dropna().to_numpy(dtype=float)
        drawn = axis.boxplot(values, whis=(0, 100), showfliers=False, widths=0.55)
        for median in drawn["medians"]:
            median.set_color("#333333")
        reference = pd.to_numeric(limits[column], errors="coerce").dropna().to_numpy(dtype=float)
        span = float(np.nanmax(reference) - np.nanmin(reference))
        pad = span * 0.05 if span else 1.0
        y_min = float(np.nanmin(reference)) - pad
        y_max = float(np.nanmax(reference)) + pad
        lower, upper = fences[column]
        if y_min <= lower <= y_max:
            axis.axhline(lower, color=FENCE_COLOR, linestyle="--", linewidth=0.9)
        if y_min <= upper <= y_max:
            axis.axhline(upper, color=FENCE_COLOR, linestyle="--", linewidth=0.9)
        axis.set_title(column, fontsize=9)
        axis.set_xticks([])
        axis.set_ylim(y_min, y_max)
    for axis in axes.flat[len(columns) :]:
        axis.axis("off")
    figure.suptitle(title)
    figure.text(0.5, 0.01, note, ha="center", va="bottom", fontsize=9)
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    return _save(plt, figure, path)


def _correlation_heatmap(plt, path: Path, numeric: pd.DataFrame, target: pd.Series, title: str) -> Path:
    import seaborn as sns

    matrix = numeric.copy()
    matrix["Default"] = target.to_numpy()
    correlation = matrix.corr(numeric_only=True)
    size = max(8, 0.55 * len(correlation.columns) + 3)
    figure, axis = plt.subplots(figsize=(size, size * 0.85))
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The set_bad function will be deprecated",
            category=PendingDeprecationWarning,
        )
        sns.heatmap(
            correlation,
            ax=axis,
            cmap="vlag",
            center=0,
            vmin=-1,
            vmax=1,
            annot=True,
            fmt=".2f",
            annot_kws={"size": 8},
            square=True,
            cbar_kws={"shrink": 0.8},
        )
    axis.set_title(title)
    figure.tight_layout()
    return _save(plt, figure, path)


def _target_correlation(
    plt,
    path: Path,
    numeric: pd.DataFrame,
    target: pd.Series,
    selection: pd.DataFrame,
) -> Path:
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    scores = numeric.apply(
        lambda column: abs(_pearson(column.to_numpy(dtype=float), target.to_numpy())),
        axis=0,
    ).fillna(0)
    kept = set(_kept_features(selection, "numeric", list(numeric.columns)))
    colors = [KEPT_COLOR if feature in kept else DROPPED_COLOR for feature in scores.index]
    figure, axis = plt.subplots(figsize=(8, 5.5))
    axis.barh(scores.index, scores.to_numpy(), color=colors)
    axis.axvline(MIN_ABS_CORRELATION, color=FENCE_COLOR, linestyle="--", linewidth=1.2)
    axis.set_xlabel("|Pearson correlation| with Default")
    axis.set_title("Training split: numeric association with default")
    axis.legend(
        handles=[
            Patch(facecolor=KEPT_COLOR, label="Kept"),
            Patch(facecolor=DROPPED_COLOR, label="Dropped"),
            Line2D([0], [0], color=FENCE_COLOR, linestyle="--", label=f"Keep if |r| ≥ {MIN_ABS_CORRELATION}"),
        ],
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
    )
    axis.invert_yaxis()
    figure.tight_layout()
    return _save(plt, figure, path)


def _category_default_rates(
    plt,
    path: Path,
    features: pd.DataFrame,
    target: pd.Series,
    selection: pd.DataFrame,
) -> Path:
    columns = BINARY_CANDIDATES + ORDINAL_CANDIDATES + NOMINAL_CANDIDATES
    ncols = 4
    nrows = math.ceil(len(columns) / ncols)
    figure, axes = plt.subplots(nrows, ncols, figsize=(14, 3.4 * nrows), squeeze=False)
    labeled = features.copy()
    labeled["_target"] = target.to_numpy()
    base_rate = float(target.mean())
    for axis, column in zip(axes.flat, columns):
        rates = labeled.groupby(column, dropna=False)["_target"].mean()
        order = [level for level in CATEGORY_ORDER.get(column, []) if level in rates.index]
        extra = [level for level in rates.index if level not in order]
        rates = rates.reindex(order + extra)
        axis.bar(rates.index.astype(str), rates.to_numpy(), color=NO_DEFAULT_COLOR)
        axis.axhline(base_rate, color=DEFAULT_COLOR, linestyle="--", linewidth=1)
        gap = float(rates.max() - rates.min()) if len(rates) else 0.0
        decision = "kept" if _is_kept(selection, column) else "dropped"
        axis.set_title(f"{column}\ngap {gap:.1%} ({decision})", fontsize=9)
        axis.tick_params(axis="x", labelrotation=30, labelsize=8)
        axis.set_ylabel("Default rate")
        axis.set_ylim(0, max(0.25, float(rates.max()) * 1.25 if len(rates) else 0.25))
    for axis in axes.flat[len(columns) :]:
        axis.axis("off")
    figure.suptitle(
        "Training split: default rate by category\n"
        f"Dashed line is the training default rate. A feature is kept when the gap across levels is at least {MIN_DEFAULT_RATE_GAP:.0%}."
    )
    figure.tight_layout()
    return _save(plt, figure, path)


def _is_kept(selection: pd.DataFrame, feature: str) -> bool:
    match = selection.loc[selection["feature"] == feature, "keep"]
    return bool(match.iloc[0]) if len(match) else False


def _compact_tick(value: float, _position: float) -> str:
    absolute = abs(value)
    if absolute >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if absolute >= 10_000:
        return f"{value / 1_000:.0f}k"
    if absolute >= 1_000:
        return f"{value / 1_000:.1f}k"
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.3g}"


def _pearson(values: np.ndarray, target: np.ndarray) -> float:
    mask = np.isfinite(values) & np.isfinite(target)
    if int(mask.sum()) < 3:
        return float("nan")
    return float(np.corrcoef(values[mask], target[mask])[0, 1])
