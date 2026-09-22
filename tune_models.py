#!/usr/bin/env python3
"""Tune the four loan-default classifiers and export the final model.

Compares each untuned model with a randomized hyperparameter search on the
same stratified folds. The winner is the setting with the highest
cross-validated average precision. The test set is scored only after that
choice. A second check asks whether the engineered payment features still
help that winner.

Run after preprocess.py:

    .venv/bin/python tune_models.py
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import f1_score
from sklearn.model_selection import ParameterGrid, RandomizedSearchCV, StratifiedKFold, cross_val_predict, cross_validate

from train_models import DEFAULT_ARTIFACTS, RANDOM_STATE, build_models, holdout_metrics, load_split

N_SPLITS = 3
SEARCH_ITERATIONS = {
    "logistic_regression": 12,
    "decision_tree": 12,
    "random_forest": 8,
    "hist_gradient_boosting": 12,
}
ENGINEERED_FEATURES = (
    "loan_to_income",
    "estimated_monthly_payment",
    "payment_to_income",
)
# Keep a simpler feature set only when it beats the full set by this much.
FEATURE_GAIN_REQUIRED = 0.005

MODEL_NAMES = {
    "logistic_regression": "Logistic regression",
    "decision_tree": "Decision tree",
    "random_forest": "Random forest",
    "hist_gradient_boosting": "Histogram gradient boosting",
}

SEARCH_SPACES = {
    "logistic_regression": {
        "C": [0.01, 0.1, 1.0, 10.0, 100.0],
        "class_weight": ["balanced", None],
    },
    "decision_tree": {
        "max_depth": [6, 8, 12, 16, None],
        "min_samples_leaf": [20, 50, 100, 200],
        "max_features": [None, "sqrt"],
        "class_weight": ["balanced", None],
    },
    "random_forest": {
        "n_estimators": [100, 200],
        "max_depth": [12, 16, 24],
        "min_samples_leaf": [5, 10, 20],
        "max_features": ["sqrt", 0.5],
        "class_weight": ["balanced", None],
    },
    "hist_gradient_boosting": {
        "learning_rate": [0.05, 0.1, 0.2],
        "max_iter": [100, 200, 300],
        "max_leaf_nodes": [15, 31, 63],
        "min_samples_leaf": [20, 50],
        "l2_regularization": [0.0, 1.0],
        "class_weight": ["balanced", None],
    },
}


def search_iterations(name: str) -> int:
    grid_size = len(ParameterGrid(SEARCH_SPACES[name]))
    return min(SEARCH_ITERATIONS[name], grid_size)


def cv_average_precision(estimator, x_train: pd.DataFrame, y_train: np.ndarray, folds) -> float:
    scores = cross_validate(
        estimator,
        x_train,
        y_train,
        cv=folds,
        scoring="average_precision",
        n_jobs=1,
    )
    return float(np.mean(scores["test_score"]))


def tune_one(name: str, x_train: pd.DataFrame, y_train: np.ndarray, folds) -> dict:
    baseline = clone(build_models()[name])
    print(f"Scoring baseline {name} ...", flush=True)
    baseline_ap = cv_average_precision(baseline, x_train, y_train, folds)

    candidate = clone(baseline)
    if name == "random_forest":
        candidate.set_params(n_jobs=1)

    iterations = search_iterations(name)
    print(f"Tuning {name} with {iterations} random settings, {N_SPLITS} folds ...", flush=True)
    search = RandomizedSearchCV(
        candidate,
        SEARCH_SPACES[name],
        n_iter=iterations,
        scoring={"average_precision": "average_precision", "roc_auc": "roc_auc"},
        refit="average_precision",
        cv=folds,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    search.fit(x_train, y_train)
    tuned_ap = float(search.best_score_)
    tuned_auc = float(search.cv_results_["mean_test_roc_auc"][search.best_index_])
    use_tuned = tuned_ap >= baseline_ap
    chosen = clone(search.best_estimator_ if use_tuned else baseline)
    print(
        f"  {name}: baseline AP {baseline_ap:.4f}, tuned AP {tuned_ap:.4f}, "
        f"keeping {'tuned' if use_tuned else 'baseline'}",
        flush=True,
    )
    return {
        "model": name,
        "baseline_cv_average_precision": baseline_ap,
        "tuned_cv_average_precision": tuned_ap,
        "tuned_cv_roc_auc": tuned_auc,
        "selected_setting": "tuned" if use_tuned else "baseline",
        "cv_average_precision": tuned_ap if use_tuned else baseline_ap,
        "params": search.best_params_ if use_tuned else {},
        "estimator": chosen,
    }


def compare_engineered_features(result: dict, x_train: pd.DataFrame, y_train: np.ndarray, folds) -> dict:
    """Ask whether the payment ratios still help the leading model."""
    full_ap = cv_average_precision(clone(result["estimator"]), x_train, y_train, folds)
    reduced_columns = [column for column in x_train.columns if column not in ENGINEERED_FEATURES]
    reduced_ap = cv_average_precision(
        clone(result["estimator"]),
        x_train[reduced_columns],
        y_train,
        folds,
    )
    keep_engineered = full_ap + FEATURE_GAIN_REQUIRED >= reduced_ap
    features = list(x_train.columns) if keep_engineered else reduced_columns
    print(
        f"Feature check: full AP {full_ap:.4f}, without engineered payment features {reduced_ap:.4f}",
        flush=True,
    )
    return {
        "full_cv_average_precision": full_ap,
        "reduced_cv_average_precision": reduced_ap,
        "keep_engineered_features": keep_engineered,
        "features": features,
    }


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_report(
    path: Path,
    table: pd.DataFrame,
    winner: dict,
    feature_check: dict,
    test_metrics: dict,
) -> None:
    lines = [
        "IT3051 model optimization and final selection",
        "",
        "1. Search",
        "   Randomized search over each model's settings, scored by average precision.",
        "   Average precision is the selection metric because only about 11.6% of loans default.",
        "   A full grid would refit hundreds of models on about 200,000 rows, so each model",
        f"   tries at most {max(SEARCH_ITERATIONS.values())} random combinations.",
        f"   Validation is stratified {N_SPLITS}-fold cross-validation on the training set",
        "   (shuffle, random_state=42). Each fold still trains on about 136,000 rows.",
        "   The held-out test set is not used to choose the model.",
        "",
        "2. Tuned result versus the untuned model (same folds)",
    ]
    show = table[
        [
            "model",
            "baseline_cv_average_precision",
            "tuned_cv_average_precision",
            "selected_setting",
            "cv_average_precision",
        ]
    ]
    lines.append(show.to_string(index=False))
    lines.extend(["", "3. Settings kept for each model"])
    for row in table.itertuples(index=False):
        if row.selected_setting == "tuned":
            lines.append(f"   {row.model}: tuned parameters {row.params}")
        else:
            lines.append(f"   {row.model}: untuned settings scored higher, so those settings are kept.")
    lines.extend(
        [
            "",
            "4. Engineered features",
            "   The leading model was scored again with and without loan_to_income,",
            "   estimated_monthly_payment, and payment_to_income.",
            f"   With those features:    average precision {feature_check['full_cv_average_precision']:.4f}",
            f"   Without those features: average precision {feature_check['reduced_cv_average_precision']:.4f}",
        ]
    )
    if feature_check["keep_engineered_features"]:
        lines.append(
            "   The gap is smaller than 0.005, so the engineered features stay in the final model."
        )
    else:
        lines.append(
            "   Removing them improves average precision by at least 0.005, so the final model omits them."
        )
    lines.extend(
        [
            "",
            "5. Final model",
            f"   {winner['model']} ({MODEL_NAMES[winner['model']]})",
            f"   Setting: {winner['selected_setting']}",
            f"   Cross-validated average precision: {winner['cv_average_precision']:.4f}",
            "   This is the highest score among the untuned and tuned versions of all four models.",
            f"   Decision cutoff: {winner['threshold']:.2f}.",
            "   The cutoff maximises F1 on out-of-fold training predictions. A cutoff of 0.50 would",
            "   rarely flag a default, because this model is not reweighted for the rare class.",
            "",
            "6. Test set, scored once after selection",
            f"   Average precision: {test_metrics['average_precision']:.4f}",
            f"   ROC-AUC:           {test_metrics['roc_auc']:.4f}",
            f"   Recall:            {test_metrics['recall']:.4f}",
            f"   Precision:         {test_metrics['precision']:.4f}",
            f"   F1:                {test_metrics['f1']:.4f}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def choose_threshold(estimator, x_train: pd.DataFrame, y_train: np.ndarray, folds) -> float:
    """Pick the cutoff that maximises F1 from out-of-fold training scores."""
    probabilities = cross_val_predict(
        clone(estimator),
        x_train,
        y_train,
        cv=folds,
        method="predict_proba",
        n_jobs=1,
    )[:, 1]
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in np.round(np.linspace(0.05, 0.80, 76), 2):
        predicted = (probabilities >= threshold).astype(int)
        score = float(f1_score(y_train, predicted, zero_division=0))
        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)
    print(f"Decision cutoff {best_threshold:.2f} (out-of-fold F1 {best_f1:.4f})", flush=True)
    return best_threshold


def main() -> None:
    artifacts = DEFAULT_ARTIFACTS
    model_dir = artifacts / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    x_train, x_test, y_train, y_test = load_split(artifacts)
    folds = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    results = [tune_one(name, x_train, y_train, folds) for name in build_models()]
    table = pd.DataFrame(results).drop(columns=["estimator"])
    winner = max(results, key=lambda item: (item["cv_average_precision"], item["tuned_cv_roc_auc"]))
    print(f"Leading model before the feature check: {winner['model']}", flush=True)

    feature_check = compare_engineered_features(winner, x_train, y_train, folds)
    features = feature_check["features"]
    winner["threshold"] = choose_threshold(winner["estimator"], x_train[features], y_train, folds)
    final_model = clone(winner["estimator"])
    if "n_jobs" in final_model.get_params():
        final_model.set_params(n_jobs=-1)
    print(f"Fitting final {winner['model']} on the full training set ...", flush=True)
    final_model.fit(x_train[features], y_train)

    score = final_model.predict_proba(x_test[features])[:, 1]
    prediction = (score >= winner["threshold"]).astype(int)
    test_metrics = holdout_metrics(y_test, prediction, score)

    table = table.sort_values("cv_average_precision", ascending=False)
    table.to_csv(artifacts / "tuning_comparison.csv", index=False)
    joblib.dump(final_model, model_dir / "selected_model.joblib")
    meta = {
        "model": winner["model"],
        "model_name": MODEL_NAMES[winner["model"]],
        "selected_setting": winner["selected_setting"],
        "params": json_ready(winner["params"]),
        "features": features,
        "cv_average_precision": winner["cv_average_precision"],
        "threshold": winner["threshold"],
        "keep_engineered_features": feature_check["keep_engineered_features"],
        "test_average_precision": test_metrics["average_precision"],
        "test_roc_auc": test_metrics["roc_auc"],
    }
    (model_dir / "selected_model.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (model_dir / "selected_model.txt").write_text(winner["model"] + "\n", encoding="utf-8")
    write_report(
        artifacts / "tuning_report.txt",
        table.assign(
            params=table["params"].map(
                lambda value: json.dumps(json_ready(value)) if value else "baseline settings"
            )
        ),
        winner,
        feature_check,
        test_metrics,
    )

    print()
    print(table[["model", "baseline_cv_average_precision", "tuned_cv_average_precision", "selected_setting"]].to_string(index=False))
    print()
    print(f"Final model: {winner['model']}")
    print(f"Test average precision {test_metrics['average_precision']:.4f}, ROC-AUC {test_metrics['roc_auc']:.4f}")
    print(f"Exported {model_dir / 'selected_model.joblib'}")


if __name__ == "__main__":
    main()
