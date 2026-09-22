#!/usr/bin/env python3
"""Train four loan-default classifiers and export each fitted model.

Run from the project folder, with the virtual environment active:

    .venv/bin/python train_models.py

Inputs (created by preprocess.py):
    artifacts/X_train.csv, artifacts/y_train.csv
    artifacts/X_test.csv, artifacts/y_test.csv

Exports:
    artifacts/models/logistic_regression.joblib
    artifacts/models/decision_tree.joblib
    artifacts/models/random_forest.joblib
    artifacts/models/hist_gradient_boosting.joblib
    artifacts/model_comparison_cv.csv
    artifacts/model_comparison_test.csv
    artifacts/model_report.txt

The four algorithms are all classifiers. The target is Default (1 or 0).
Logistic regression, a decision tree, a random forest, and histogram gradient
boosting suit a large table of numeric and categorical loan fields.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent
DEFAULT_ARTIFACTS = ROOT / "artifacts"
RANDOM_STATE = 42
N_SPLITS = 5

REQUIRED_FILES = ("X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv")

SCORING = {
    "roc_auc": "roc_auc",
    "average_precision": "average_precision",
    "f1": "f1",
    "precision": "precision",
    "recall": "recall",
    "balanced_accuracy": "balanced_accuracy",
    "accuracy": "accuracy",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and export four loan-default classifiers.")
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    return parser.parse_args()


def load_split(artifacts: Path) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    missing = [name for name in REQUIRED_FILES if not (artifacts / name).exists()]
    if missing:
        raise FileNotFoundError(
            "Preprocessed files are missing: "
            + ", ".join(missing)
            + ". Run preprocess.py first."
        )
    x_train = pd.read_csv(artifacts / "X_train.csv")
    x_test = pd.read_csv(artifacts / "X_test.csv")
    y_train = pd.read_csv(artifacts / "y_train.csv")["Default"].to_numpy(dtype=int)
    y_test = pd.read_csv(artifacts / "y_test.csv")["Default"].to_numpy(dtype=int)
    return x_train, x_test, y_train, y_test


def build_models() -> dict[str, object]:
    """Untuned classifiers. Hyperparameter search is a later project stage."""
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "decision_tree": DecisionTreeClassifier(
            max_depth=12,
            min_samples_leaf=50,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=16,
            min_samples_leaf=10,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.1,
            max_leaf_nodes=31,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    }


def train_model(estimator, x_train: pd.DataFrame, y_train: np.ndarray):
    """Fit a fresh copy on the full training set and return that fitted classifier."""
    model = clone(estimator)
    model.fit(x_train, y_train)
    return model


def export_model(model, path: Path) -> Path:
    """Write one fitted classifier to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def cross_validate_model(name: str, estimator, x_train: pd.DataFrame, y_train: np.ndarray) -> dict:
    print(f"Cross-validating {name} ...", flush=True)
    started = time.perf_counter()
    folds = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_validate(
        estimator,
        x_train,
        y_train,
        cv=folds,
        scoring=SCORING,
        n_jobs=1,
        return_train_score=True,
    )
    elapsed = time.perf_counter() - started
    row = {"model": name, "cv_seconds": round(elapsed, 1)}
    for metric in SCORING:
        validation = scores[f"test_{metric}"]
        training = scores[f"train_{metric}"]
        row[f"cv_{metric}_mean"] = float(np.mean(validation))
        row[f"cv_{metric}_std"] = float(np.std(validation))
        row[f"train_{metric}_mean"] = float(np.mean(training))
    print(
        f"  {name}: CV average precision {row['cv_average_precision_mean']:.4f} "
        f"+/- {row['cv_average_precision_std']:.4f} in {elapsed:.0f}s",
        flush=True,
    )
    return row


def holdout_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "average_precision": float(average_precision_score(y_true, y_score)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "accuracy": float(np.mean(y_pred == y_true)),
    }


def top_drivers(name: str, model, feature_names: list[str]) -> list[str]:
    if name == "logistic_regression":
        coefficients = pd.Series(model.coef_[0], index=feature_names)
        ranked = coefficients.reindex(coefficients.abs().sort_values(ascending=False).index)
        return [f"    {feature}: {value:+.3f}" for feature, value in ranked.head(8).items()]

    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return ["    Feature importances are not exposed by this estimator."]
    ranked = pd.Series(importances, index=feature_names).sort_values(ascending=False)
    return [f"    {feature}: {value:.4f}" for feature, value in ranked.head(8).items()]


def write_report(
    path: Path,
    cv_table: pd.DataFrame,
    test_table: pd.DataFrame,
    matrices: dict[str, np.ndarray],
    drivers: dict[str, list[str]],
    train_positive_rate: float,
    exported_paths: list[Path],
) -> None:
    baseline_accuracy = 1.0 - train_positive_rate
    lines = [
        "IT3051 model comparison — loan default classification",
        "",
        "1. Algorithms",
        "   logistic_regression: linear probability of default. L2 penalty is the sklearn default,",
        "   which is appropriate because loan_to_income, payment_to_income, and the raw amount",
        "   and income columns overlap.",
        "   decision_tree: axis-aligned rules. max_depth=12 and min_samples_leaf=50 stop the tree",
        "   from memorising 200,000 rows. These are starting limits, not a tuned search.",
        "   random_forest: 100 trees, max_depth=16, min_samples_leaf=10, votes averaged.",
        "   hist_gradient_boosting: 200 boosting iterations, learning rate 0.1, up to 31 leaves",
        "   per tree. Each tree fits the mistakes of the ones before it.",
        "   All four use class_weight='balanced'.",
        "",
        "2. Validation",
        "   Stratified 5-fold cross-validation on the training set only (shuffle, random_state=42).",
        "   Each classifier is then trained on all training rows and exported.",
        "   The held-out test set is scored once after that export.",
        "",
        "3. How to read the scores",
        f"   A model that always predicts 'no default' is already {baseline_accuracy:.1%} accurate",
        "   and catches none of the defaults (recall 0). Accuracy is reported so that gap is visible.",
        "   ROC-AUC and average precision rank applicants by predicted risk and do not use a cutoff.",
        f"   Average precision of a no-skill model equals the default rate ({train_positive_rate:.3f}).",
        "   Precision, recall, and F1 use a 0.5 cutoff. Balanced training pushes recall up and precision",
        "   down at that cutoff. The cutoff can be moved later without retraining.",
        "",
        "4. Cross-validation (mean over 5 training folds)",
    ]
    show = [
        "model",
        "cv_average_precision_mean",
        "cv_roc_auc_mean",
        "cv_recall_mean",
        "cv_precision_mean",
        "cv_f1_mean",
        "cv_accuracy_mean",
        "train_roc_auc_mean",
    ]
    lines.append(cv_table[show].to_string(index=False))
    lines.extend(["", "5. Held-out test set (fit on all training rows, scored once)"])
    lines.append(test_table.to_string(index=False))
    lines.extend(
        [
            "",
            "6. Test-set confusion counts at the 0.5 cutoff",
            "   Rows are actual, columns are predicted. Order: no default, default.",
        ]
    )
    for name, matrix in matrices.items():
        tn, fp, fn, tp = matrix.ravel()
        lines.extend(
            [
                f"   {name}",
                f"      true no-default, predicted no-default: {tn}",
                f"      true no-default, predicted default:    {fp}  (applications flagged for review)",
                f"      true default, predicted no-default:    {fn}  (defaults missed)",
                f"      true default, predicted default:       {tp}",
            ]
        )
    lines.extend(["", "7. What the models are using"])
    for name, driver_lines in drivers.items():
        lines.append(f"   {name}")
        lines.extend(driver_lines)
    lines.extend(
        [
            "",
            "8. Why the results differ",
            "   Train ROC-AUC minus validation ROC-AUC. A large gap means the model memorised the training rows.",
            *[
                (
                    f"   {row.model}: train {row.train_roc_auc_mean:.3f}, "
                    f"validation {row.cv_roc_auc_mean:.3f}, gap {row.train_roc_auc_mean - row.cv_roc_auc_mean:.3f}"
                )
                for row in cv_table.itertuples(index=False)
            ],
            "   Logistic regression only draws a straight-line boundary in the scaled features.",
            "   It remains competitive when the main effects (age, interest rate, loan size relative",
            "   to income) are roughly linear.",
            "   A single tree can follow thresholds, but one tree is unstable: a small change in the",
            "   training sample can change the split.",
            "   A random forest reduces that instability by averaging trees grown on different samples.",
            "   Gradient boosting usually goes further on this kind of table because later trees focus",
            "   on applicants the earlier trees scored badly. The train-versus-validation ROC-AUC gap",
            "   shows whether that extra fit is memorisation or a real gain.",
            "",
            "9. Exported model files",
            *[f"   {path}" for path in exported_paths],
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def export_comparison(
    artifacts: Path,
    cv_table: pd.DataFrame,
    test_table: pd.DataFrame,
    matrices: dict[str, np.ndarray],
    drivers: dict[str, list[str]],
    train_positive_rate: float,
    exported_paths: list[Path],
) -> None:
    cv_table.to_csv(artifacts / "model_comparison_cv.csv", index=False)
    test_table.to_csv(artifacts / "model_comparison_test.csv", index=False)
    write_report(
        artifacts / "model_report.txt",
        cv_table=cv_table,
        test_table=test_table,
        matrices=matrices,
        drivers=drivers,
        train_positive_rate=train_positive_rate,
        exported_paths=exported_paths,
    )


def main() -> None:
    args = parse_args()
    model_dir = args.artifacts / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    x_train, x_test, y_train, y_test = load_split(args.artifacts)
    print(f"Train {x_train.shape}, test {x_test.shape}, default rate {y_train.mean():.4f}", flush=True)

    cv_rows = []
    test_rows = []
    matrices = {}
    drivers = {}
    exported_paths: list[Path] = []

    for name, estimator in build_models().items():
        cv_rows.append(cross_validate_model(name, estimator, x_train, y_train))

        print(f"Training {name} ...", flush=True)
        trained = train_model(estimator, x_train, y_train)
        exported = export_model(trained, model_dir / f"{name}.joblib")
        exported_paths.append(exported)
        print(f"Exported {exported}", flush=True)

        score = trained.predict_proba(x_test)[:, 1]
        prediction = trained.predict(x_test)
        metrics = holdout_metrics(y_test, prediction, score)
        metrics["model"] = name
        test_rows.append(metrics)
        matrices[name] = confusion_matrix(y_test, prediction, labels=[0, 1])
        drivers[name] = top_drivers(name, trained, list(x_train.columns))
        print(
            f"  test average precision {metrics['average_precision']:.4f}, "
            f"ROC-AUC {metrics['roc_auc']:.4f}, recall {metrics['recall']:.4f}",
            flush=True,
        )

    cv_table = pd.DataFrame(cv_rows).sort_values("cv_average_precision_mean", ascending=False)
    test_table = pd.DataFrame(test_rows)[
        ["model", "average_precision", "roc_auc", "recall", "precision", "f1", "balanced_accuracy", "accuracy"]
    ]
    export_comparison(
        args.artifacts,
        cv_table=cv_table,
        test_table=test_table,
        matrices=matrices,
        drivers=drivers,
        train_positive_rate=float(y_train.mean()),
        exported_paths=exported_paths,
    )

    print()
    print(cv_table[["model", "cv_average_precision_mean", "cv_roc_auc_mean", "cv_recall_mean", "cv_precision_mean"]].to_string(index=False))
    print()
    print(test_table.to_string(index=False))
    print()
    print("Exported files:")
    for path in exported_paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
