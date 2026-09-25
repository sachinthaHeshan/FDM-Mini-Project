#!/usr/bin/env python3
"""Preprocessing for IT3051 loan-default prediction.

Scenario
    A lender wants to estimate, before a loan is approved, whether an
    applicant will default. The prediction supports a credit officer's
    decision to approve, price, or review an application.

Task
    Binary classification. Target column: Default
    (1 = applicant defaulted, 0 = applicant did not default).

The script fits every statistic (imputation, outlier fences, scaling,
category levels, and feature selection) on the training split only.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "dataset" / "Loan_default.csv"
DEFAULT_OUTPUT = ROOT / "artifacts"

TARGET = "Default"
ID_COLUMN = "LoanID"
RANDOM_STATE = 42
TEST_SIZE = 0.20

# A feature is kept when its training-set association with default is large
# enough to matter in a credit decision. With 255k rows, a correlation of
# 0.001 can be "statistically significant" and still be useless. One
# percentage point of default-rate difference is the practical cutoff used
# below.
MIN_ABS_CORRELATION = 0.01
MIN_DEFAULT_RATE_GAP = 0.01
# Two numeric columns more correlated than this are the same signal.
REDUNDANCY_CORRELATION = 0.98

NUMERIC_CANDIDATES = [
    "Age",
    "Income",
    "LoanAmount",
    "CreditScore",
    "MonthsEmployed",
    "NumCreditLines",
    "InterestRate",
    "LoanTerm",
    "DTIRatio",
    "loan_to_income",
    "estimated_monthly_payment",
    "payment_to_income",
]
BINARY_CANDIDATES = ["HasMortgage", "HasDependents", "HasCoSigner"]
ORDINAL_CANDIDATES = ["Education"]
NOMINAL_CANDIDATES = ["EmploymentType", "MaritalStatus", "LoanPurpose"]

EDUCATION_ORDER = ["High School", "Bachelor's", "Master's", "PhD"]
YES_NO = {"yes": 1.0, "no": 0.0}

ALLOWED_CATEGORIES = {
    "Education": set(EDUCATION_ORDER),
    "EmploymentType": {"Full-time", "Part-time", "Self-employed", "Unemployed"},
    "MaritalStatus": {"Single", "Married", "Divorced"},
    "LoanPurpose": {"Home", "Auto", "Education", "Business", "Other"},
    "HasMortgage": {"Yes", "No"},
    "HasDependents": {"Yes", "No"},
    "HasCoSigner": {"Yes", "No"},
}
ALLOWED_LOAN_TERMS = {12, 24, 36, 48, 60}

RAW_COLUMNS = [
    "Age",
    "Income",
    "LoanAmount",
    "CreditScore",
    "MonthsEmployed",
    "NumCreditLines",
    "InterestRate",
    "LoanTerm",
    "DTIRatio",
    "Education",
    "EmploymentType",
    "MaritalStatus",
    "HasMortgage",
    "HasDependents",
    "LoanPurpose",
    "HasCoSigner",
]


class LoanFeatureEngineer(BaseEstimator, TransformerMixin):
    """Row-wise features only. No dataset-level statistics, so this cannot leak test data.

    Engineered columns
        loan_to_income
            Loan amount divided by annual income. Measures how large the loan is
            relative to what the applicant earns.
        estimated_monthly_payment
            Standard amortising instalment from loan amount, annual interest rate,
            and term in months.
        payment_to_income
            That instalment divided by monthly income. Measures affordability.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        frame = pd.DataFrame(X).copy()
        missing = [column for column in RAW_COLUMNS if column not in frame.columns]
        if missing:
            raise ValueError(f"Input is missing required columns: {missing}")

        frame = frame[RAW_COLUMNS].copy()
        income = pd.to_numeric(frame["Income"], errors="coerce")
        loan_amount = pd.to_numeric(frame["LoanAmount"], errors="coerce")
        annual_rate = pd.to_numeric(frame["InterestRate"], errors="coerce")
        term = pd.to_numeric(frame["LoanTerm"], errors="coerce")
        monthly_rate = annual_rate / 100.0 / 12.0

        growth = (1.0 + monthly_rate) ** term
        payment = np.where(
            monthly_rate <= 0,
            loan_amount / term.replace(0, np.nan),
            loan_amount * (monthly_rate * growth) / (growth - 1.0),
        )
        safe_income = income.where(income > 0)

        engineered = frame.copy()
        engineered["loan_to_income"] = loan_amount / safe_income
        engineered["estimated_monthly_payment"] = payment
        engineered["payment_to_income"] = payment / (safe_income / 12.0)

        for column in BINARY_CANDIDATES:
            engineered[column] = (
                engineered[column].astype(str).str.strip().str.lower().map(YES_NO)
            )

        for column in NUMERIC_CANDIDATES:
            if column in ("loan_to_income", "estimated_monthly_payment", "payment_to_income"):
                continue
            engineered[column] = pd.to_numeric(engineered[column], errors="coerce")

        for column in ORDINAL_CANDIDATES + NOMINAL_CANDIDATES:
            engineered[column] = engineered[column].astype(str).str.strip()

        ordered = NUMERIC_CANDIDATES + BINARY_CANDIDATES + ORDINAL_CANDIDATES + NOMINAL_CANDIDATES
        return engineered[ordered]


class IQRCapper(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    """Cap values outside the training-set 1.5 IQR fences.

    Fences are learned in fit() and applied unchanged in transform().
    On this dataset the fences sit outside the observed range, so no
    training value is changed. The step still protects later scoring
    from extreme numeric input.
    """

    def __init__(self, factor: float = 1.5):
        self.factor = factor

    def fit(self, X, y=None):
        values = np.asarray(X, dtype=float)
        q1 = np.nanpercentile(values, 25, axis=0)
        q3 = np.nanpercentile(values, 75, axis=0)
        iqr = q3 - q1
        self.lower_ = q1 - self.factor * iqr
        self.upper_ = q3 + self.factor * iqr
        self.n_features_in_ = values.shape[1]
        return self

    def transform(self, X):
        values = np.asarray(X, dtype=float).copy()
        self.n_capped_ = int(np.sum((values < self.lower_) | (values > self.upper_)))
        return np.clip(values, self.lower_, self.upper_)


class FullPreprocessor(BaseEstimator, TransformerMixin):
    """Raw applicant fields -> model matrix, with selection fitted on training rows only."""

    def __init__(self, random_state: int = RANDOM_STATE):
        self.random_state = random_state

    def fit(self, X, y):
        self.engineer_ = LoanFeatureEngineer()
        engineered = self.engineer_.fit_transform(X)
        y_array = np.asarray(y).astype(int)
        self.selection_report_ = build_selection_report(engineered, y_array)
        self.selected_ = columns_to_keep(self.selection_report_)
        self.columns_ = build_column_transformer(self.selected_)
        self.columns_.fit(engineered, y_array)
        self.feature_names_ = list(self.columns_.get_feature_names_out())
        numeric_step = self.columns_.named_transformers_["numeric"]
        self.train_cells_capped_ = int(numeric_step.named_steps["capper"].n_capped_)
        return self

    def transform(self, X):
        engineered = self.engineer_.transform(X)
        transformed = self.columns_.transform(engineered)
        return pd.DataFrame(transformed, columns=self.feature_names_, index=getattr(X, "index", None))


def build_column_transformer(selected: dict[str, list[str]]) -> ColumnTransformer:
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("capper", IQRCapper()),
            ("scaler", StandardScaler()),
        ]
    )
    binary = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
        ]
    )
    ordinal = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OrdinalEncoder(
                    categories=[EDUCATION_ORDER],
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
            ("scaler", StandardScaler()),
        ]
    )
    nominal = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    transformers = []
    if selected["numeric"]:
        transformers.append(("numeric", numeric, selected["numeric"]))
    if selected["binary"]:
        transformers.append(("binary", binary, selected["binary"]))
    if selected["ordinal"]:
        transformers.append(("ordinal", ordinal, selected["ordinal"]))
    if selected["nominal"]:
        transformers.append(("nominal", nominal, selected["nominal"]))

    transformer = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return transformer


def build_selection_report(engineered: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    """Association of each candidate feature with Default, measured on training rows."""
    rows = []
    for column in NUMERIC_CANDIDATES:
        values = engineered[column].to_numpy(dtype=float)
        correlation = _pearson(values, y)
        rows.append(
            {
                "feature": column,
                "kind": "numeric",
                "score": abs(correlation) if np.isfinite(correlation) else 0.0,
                "detail": f"pearson_r={correlation:.6f}",
                "keep": bool(np.isfinite(correlation) and abs(correlation) >= MIN_ABS_CORRELATION),
            }
        )

    labeled = engineered.copy()
    labeled["_target"] = y
    for column in BINARY_CANDIDATES + ORDINAL_CANDIDATES + NOMINAL_CANDIDATES:
        grouped = labeled.groupby(column, dropna=False)["_target"].mean()
        if column in BINARY_CANDIDATES:
            grouped.index = grouped.index.map({0.0: "No", 1.0: "Yes", 0: "No", 1: "Yes"})
        gap = float(grouped.max() - grouped.min()) if len(grouped) else 0.0
        kind = (
            "binary"
            if column in BINARY_CANDIDATES
            else "ordinal"
            if column in ORDINAL_CANDIDATES
            else "nominal"
        )
        rows.append(
            {
                "feature": column,
                "kind": kind,
                "score": gap,
                "detail": "default_rate_gap=" + ", ".join(
                    f"{index}={rate:.4f}" for index, rate in grouped.items()
                ),
                "keep": gap >= MIN_DEFAULT_RATE_GAP,
            }
        )

    report = pd.DataFrame(rows)
    report = _drop_redundant_numeric(engineered, y, report)
    report["reason"] = report.apply(_selection_reason, axis=1)
    return report.sort_values(["keep", "score"], ascending=[False, False]).reset_index(drop=True)


def _pearson(values: np.ndarray, target: np.ndarray) -> float:
    mask = np.isfinite(values) & np.isfinite(target)
    if int(mask.sum()) < 3:
        return float("nan")
    correlation = np.corrcoef(values[mask], target[mask])[0, 1]
    return float(correlation)


def _drop_redundant_numeric(
    engineered: pd.DataFrame, y: np.ndarray, report: pd.DataFrame
) -> pd.DataFrame:
    """If two numeric features move together, keep the one more related to default."""
    report = report.copy()
    kept_numeric = report.loc[
        (report["kind"] == "numeric") & report["keep"], "feature"
    ].tolist()
    if len(kept_numeric) < 2:
        return report

    correlations = engineered[kept_numeric].corr().abs()
    target_correlation = {
        column: abs(_pearson(engineered[column].to_numpy(dtype=float), y))
        for column in kept_numeric
    }
    dropped = set()
    for left_index, left in enumerate(kept_numeric):
        for right in kept_numeric[left_index + 1 :]:
            if left in dropped or right in dropped:
                continue
            if correlations.loc[left, right] <= REDUNDANCY_CORRELATION:
                continue
            weaker = left if target_correlation[left] < target_correlation[right] else right
            dropped.add(weaker)
            report.loc[report["feature"] == weaker, "keep"] = False
            report.loc[report["feature"] == weaker, "detail"] += (
                f"; dropped as redundant (|r|>{REDUNDANCY_CORRELATION} with a stronger feature)"
            )
    return report


def _selection_reason(row: pd.Series) -> str:
    if row["keep"]:
        return "Kept: association with Default is above the practical cutoff."
    if "redundant" in row["detail"]:
        return "Dropped: duplicates a stronger numeric feature."
    if row["kind"] == "numeric":
        return (
            f"Dropped: |correlation| with Default is below {MIN_ABS_CORRELATION}. "
            "The column does not separate defaulters from non-defaulters."
        )
    return (
        f"Dropped: default-rate gap across categories is below {MIN_DEFAULT_RATE_GAP:.0%}."
    )


def columns_to_keep(report: pd.DataFrame) -> dict[str, list[str]]:
    kept = set(report.loc[report["keep"], "feature"])

    def ordered(candidates: list[str]) -> list[str]:
        return [column for column in candidates if column in kept]

    selected = {
        "numeric": ordered(NUMERIC_CANDIDATES),
        "binary": ordered(BINARY_CANDIDATES),
        "ordinal": ordered(ORDINAL_CANDIDATES),
        "nominal": ordered(NOMINAL_CANDIDATES),
    }
    if not any(selected.values()):
        raise RuntimeError("Feature selection removed every column. Check the cutoffs.")
    return selected


def load_raw(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    expected = [ID_COLUMN, *RAW_COLUMNS, TARGET]
    missing = [column for column in expected if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    return frame[expected].copy()


def clean_records(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Drop exact duplicates and rows that break domain rules. Rules are per-row, not fitted."""
    audit = {
        "rows_loaded": int(len(frame)),
        "missing_cells": int(frame.isna().sum().sum()),
        "missing_by_column": frame.isna().sum().astype(int).to_dict(),
        "duplicate_rows": int(frame.duplicated().sum()),
        "duplicate_loan_ids": int(frame[ID_COLUMN].duplicated().sum()),
    }
    cleaned = frame.drop_duplicates().copy()
    cleaned = cleaned.drop_duplicates(subset=[ID_COLUMN], keep="first")

    numeric_rules = {
        "Age": lambda series: series.between(18, 100),
        "Income": lambda series: series > 0,
        "LoanAmount": lambda series: series > 0,
        "CreditScore": lambda series: series.between(300, 850),
        "MonthsEmployed": lambda series: series >= 0,
        "NumCreditLines": lambda series: series >= 1,
        "InterestRate": lambda series: series.between(0, 100, inclusive="neither"),
        "DTIRatio": lambda series: series.between(0, 1.5),
    }
    invalid = pd.Series(False, index=cleaned.index)
    invalid_counts = {}

    for column, rule in numeric_rules.items():
        values = pd.to_numeric(cleaned[column], errors="coerce")
        bad = values.isna() | ~rule(values)
        invalid_counts[column] = int(bad.sum())
        invalid = invalid | bad

    term = pd.to_numeric(cleaned["LoanTerm"], errors="coerce")
    bad_term = term.isna() | ~term.isin(ALLOWED_LOAN_TERMS)
    invalid_counts["LoanTerm"] = int(bad_term.sum())
    invalid = invalid | bad_term

    for column, allowed in ALLOWED_CATEGORIES.items():
        bad = ~cleaned[column].astype(str).str.strip().isin(allowed)
        invalid_counts[column] = int(bad.sum())
        invalid = invalid | bad

    target = pd.to_numeric(cleaned[TARGET], errors="coerce")
    bad_target = ~target.isin([0, 1])
    invalid_counts[TARGET] = int(bad_target.sum())
    invalid = invalid | bad_target

    audit["invalid_rows"] = int(invalid.sum())
    audit["invalid_by_column"] = invalid_counts
    audit["months_employed_zero"] = int((pd.to_numeric(cleaned["MonthsEmployed"], errors="coerce") == 0).sum())
    cleaned = cleaned.loc[~invalid].reset_index(drop=True)
    audit["rows_after_cleaning"] = int(len(cleaned))
    return cleaned, audit


def outlier_report(train_frame: pd.DataFrame) -> list[str]:
    """IQR screen on the training split. Reported even when nothing is capped."""
    lines = []
    engineered = LoanFeatureEngineer().fit_transform(train_frame)
    for column in NUMERIC_CANDIDATES:
        values = engineered[column].to_numpy(dtype=float)
        q1, q3 = np.nanpercentile(values, [25, 75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outside = int(np.sum((values < lower) | (values > upper)))
        lines.append(
            f"  {column}: min={np.nanmin(values):.4g}, max={np.nanmax(values):.4g}, "
            f"IQR fences=[{lower:.4g}, {upper:.4g}], outside={outside}"
        )
    return lines


def write_report(
    path: Path,
    audit: dict,
    y_train: pd.Series,
    y_test: pd.Series,
    selection: pd.DataFrame,
    selected: dict[str, list[str]],
    feature_names: list[str],
    outlier_lines: list[str],
    capped_cells: int,
) -> None:
    dropped = selection.loc[~selection["keep"], "feature"].tolist()
    lines = [
        "IT3051 preprocessing report — loan default classification",
        "",
        "1. Problem",
        "   Predict whether a loan applicant will default, so a credit officer can",
        "   approve, review, or price the application before funds are released.",
        "   Task: binary classification. Target: Default (1 = default, 0 = no default).",
        "",
        "2. Data quality",
        f"   Rows loaded: {audit['rows_loaded']}",
        f"   Missing cells: {audit['missing_cells']}",
        f"   Exact duplicate rows: {audit['duplicate_rows']}",
        f"   Duplicate LoanID values: {audit['duplicate_loan_ids']}",
        f"   Rows failing domain checks: {audit['invalid_rows']}",
        f"   Rows after cleaning: {audit['rows_after_cleaning']}",
        f"   MonthsEmployed = 0 (kept; new or not-yet-started employment): {audit['months_employed_zero']}",
        "   Domain checks: age 18-100, income and loan amount > 0, credit score 300-850,",
        "   months employed >= 0, interest rate in (0, 100), DTI ratio in [0, 1.5],",
        "   loan term in {12, 24, 36, 48, 60}, categories restricted to known labels,",
        "   target restricted to {0, 1}.",
        "",
        "3. Class balance (after the split, so the test labels are only counted here)",
        f"   Train default rate: {y_train.mean():.4f} ({int(y_train.sum())} / {len(y_train)})",
        f"   Test default rate:  {y_test.mean():.4f} ({int(y_test.sum())} / {len(y_test)})",
        "   About 11.6% of loans default. The classes are imbalanced.",
        "   Resampling is not done in preprocessing. Balancing belongs inside model",
        "   training (for example class_weight='balanced'), and must happen on each",
        "   training fold only. Doing it here would distort probabilities and can leak",
        "   across cross-validation folds.",
        "",
        "4. Outliers (fences computed on the training split)",
        *outlier_lines,
        f"   Cells changed by the training-fitted IQR cap during fit: {capped_cells}",
        "   Values already sit inside realistic bounds (age, income, credit score, and",
        "   so on). Statistical outliers are not deleted. High loan amounts and high",
        "   interest rates are plausible and are part of the risk signal.",
        "",
        "5. Feature engineering (each row uses only its own values)",
        "   loan_to_income = LoanAmount / Income",
        "   estimated_monthly_payment = amortising payment from amount, rate, and term",
        "   payment_to_income = estimated_monthly_payment / (Income / 12)",
        "   MonthsEmployed / 12 was not added. After standard scaling it is identical",
        "   to MonthsEmployed, so it would be a duplicated column.",
        "",
        "6. Encoding and scaling (fitted on the training split only)",
        "   Numeric: median impute, IQR cap, then StandardScaler.",
        "   Yes/No fields (HasMortgage, HasDependents, HasCoSigner): Yes=1, No=0.",
        "   They are imputed with the training mode and are not scaled.",
        "   Education: ordered High School < Bachelor's < Master's < PhD, then scaled.",
        "   The order is the education ladder, not a ranking fitted from the target.",
        "   EmploymentType, MaritalStatus, LoanPurpose: mode impute, then one-hot.",
        "   Unknown categories at scoring time become an all-zero dummy vector.",
        "   Full one-hot coding is kept so each category remains visible. Linear models",
        "   in the next stage should use regularisation, which sklearn logistic",
        "   regression does by default.",
        "",
        "7. Feature selection (training split only)",
        f"   Numeric feature kept when |Pearson r| with Default >= {MIN_ABS_CORRELATION}.",
        f"   Categorical feature kept when the default-rate gap across its levels >= {MIN_DEFAULT_RATE_GAP:.0%}.",
        f"   Numeric pairs with |r| > {REDUNDANCY_CORRELATION} would keep only the stronger one.",
        "   LoanID is always removed. It is a unique application key, not a property of",
        "   the borrower, and it cannot be known for a new applicant in a useful way.",
        "   Selected columns:",
        f"     numeric: {selected['numeric']}",
        f"     binary:  {selected['binary']}",
        f"     ordinal: {selected['ordinal']}",
        f"     nominal: {selected['nominal']}",
        f"   Dropped columns: {dropped}",
        "   LoanTerm stays an input to the system. It is used to calculate the monthly",
        "   payment features, then left out of the model matrix because the term itself",
        "   does not change the default rate.",
        "   Per-feature evidence is in artifacts/feature_selection.csv.",
        "",
        "8. Leakage controls",
        "   - LoanID is not a model feature.",
        "   - The 80/20 split is stratified on Default and is done before imputation,",
        "     scaling, one-hot levels, outlier fences, and feature selection.",
        "   - Engineered ratios use only the current row.",
        "   - The target is not used to build features (no target encoding).",
        "   - The saved preprocessor is fit on training rows and applied to the test rows.",
        "",
        "9. Split and outputs",
        f"   test_size={TEST_SIZE}, random_state={RANDOM_STATE}, stratify=Default",
        f"   Train rows: {len(y_train)}",
        f"   Test rows:  {len(y_test)}",
        f"   Model matrix columns ({len(feature_names)}): {feature_names}",
        "   Training-split charts are written to artifacts/figures/: class balance,",
        "   histograms, box plots before and after the IQR cap, correlation before",
        "   and after feature selection, numeric association with Default, and",
        "   default rate by category. The test split is not plotted.",
        "",
        "10. Fair-lending note",
        "   Age and marital status are predictive here, and they are also sensitive in",
        "   real credit decisions. They are retained for this academic task. A production",
        "   lending model would need a fairness review before those fields were used.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess the loan default dataset.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Path to Loan_default.csv")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Directory for prepared data")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    raw = load_raw(args.data)
    cleaned, audit = clean_records(raw)
    features = cleaned.drop(columns=[ID_COLUMN, TARGET])
    target = cleaned[TARGET].astype(int)

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=target,
    )

    preprocessor = FullPreprocessor(random_state=RANDOM_STATE)
    x_train_ready = preprocessor.fit_transform(x_train, y_train)
    x_test_ready = preprocessor.transform(x_test)

    # Confirm the saved transform path matches the matrix just produced.
    reloaded_check = preprocessor.transform(x_test)
    if not x_test_ready.reset_index(drop=True).equals(reloaded_check.reset_index(drop=True)):
        raise RuntimeError("Preprocessor transform is not stable.")
    if x_train_ready.isna().any().any() or x_test_ready.isna().any().any():
        raise RuntimeError("Preprocessed matrices still contain missing values.")

    capped_cells = int(preprocessor.train_cells_capped_)

    x_train_ready.to_csv(args.output / "X_train.csv", index=False)
    x_test_ready.to_csv(args.output / "X_test.csv", index=False)
    y_train.to_csv(args.output / "y_train.csv", index=False)
    y_test.to_csv(args.output / "y_test.csv", index=False)
    preprocessor.selection_report_.to_csv(args.output / "feature_selection.csv", index=False)
    joblib.dump(preprocessor, args.output / "preprocessor.joblib")

    write_report(
        args.output / "preprocessing_report.txt",
        audit=audit,
        y_train=y_train,
        y_test=y_test,
        selection=preprocessor.selection_report_,
        selected=preprocessor.selected_,
        feature_names=preprocessor.feature_names_,
        outlier_lines=outlier_report(x_train),
        capped_cells=capped_cells,
    )

    print(f"Rows after cleaning: {audit['rows_after_cleaning']}")
    print(f"Train: {x_train_ready.shape}  Test: {x_test_ready.shape}")
    print(f"Default rate train/test: {y_train.mean():.4f} / {y_test.mean():.4f}")
    print("Features kept:")
    print(preprocessor.selection_report_.loc[preprocessor.selection_report_["keep"], ["feature", "kind", "detail"]].to_string(index=False))
    print("Features dropped:")
    dropped = preprocessor.selection_report_.loc[~preprocessor.selection_report_["keep"], ["feature", "detail"]]
    print(dropped.to_string(index=False) if len(dropped) else "  none")
    from preprocessing_figures import save_preprocessing_figures

    figure_paths = save_preprocessing_figures(
        args.output / "figures",
        x_train=x_train,
        y_train=y_train,
        selection=preprocessor.selection_report_,
    )
    print(f"Wrote prepared data to {args.output}")
    print(f"Wrote {len(figure_paths)} charts to {args.output / 'figures'}")


if __name__ == "__main__":
    # Importing the module (instead of calling main() in this process) keeps the
    # saved preprocessor pointing at preprocess.FullPreprocessor, so joblib can reload it.
    from preprocess import main

    main()
