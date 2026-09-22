#!/usr/bin/env python3
"""Loan-default prediction API.

Loads the training preprocessor and the final selected classifier.
A request is checked against the same domain rules used in preprocessing,
then that one model scores the application.

Run from the project folder:

    .venv/bin/uvicorn backend.main:app --reload
"""

from __future__ import annotations

import json
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from preprocess import ALLOWED_CATEGORIES, ALLOWED_LOAN_TERMS, RAW_COLUMNS  # noqa: E402

ARTIFACTS = ROOT / "artifacts"
PREPROCESSOR_PATH = ARTIFACTS / "preprocessor.joblib"
MODEL_DIR = ARTIFACTS / "models"

CHOICES = {
    "Education": ["High School", "Bachelor's", "Master's", "PhD"],
    "EmploymentType": ["Full-time", "Part-time", "Self-employed", "Unemployed"],
    "MaritalStatus": ["Single", "Married", "Divorced"],
    "LoanPurpose": ["Home", "Auto", "Education", "Business", "Other"],
    "HasMortgage": ["Yes", "No"],
    "HasDependents": ["Yes", "No"],
    "HasCoSigner": ["Yes", "No"],
}


def _choice(value: object, field: str) -> str:
    allowed = CHOICES[field]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Enter one of: {', '.join(allowed)}.")
    lookup = {item.casefold(): item for item in allowed}
    match = lookup.get(value.strip().casefold())
    if match is None:
        raise ValueError(f"Enter one of: {', '.join(allowed)}.")
    if set(allowed) != ALLOWED_CATEGORIES[field]:
        raise RuntimeError(f"{field} choices do not match the training categories.")
    return match


class LoanApplication(BaseModel):
    """One loan application, using the same fields as the training data."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    Age: int = Field(description="Applicant age in years, from 18 to 100.")
    Income: float = Field(description="Annual income. Must be greater than 0.")
    LoanAmount: float = Field(description="Requested loan amount. Must be greater than 0.")
    CreditScore: int = Field(description="Credit score from 300 to 850.")
    MonthsEmployed: int = Field(description="Months in current employment. 0 is allowed.")
    NumCreditLines: int = Field(description="Number of open credit lines. At least 1.")
    InterestRate: float = Field(description="Annual interest rate, greater than 0 and less than 100.")
    LoanTerm: int = Field(description="Loan term in months: 12, 24, 36, 48, or 60.")
    DTIRatio: float = Field(description="Debt-to-income ratio from 0 to 1.5.")
    Education: str
    EmploymentType: str
    MaritalStatus: str
    HasMortgage: str
    HasDependents: str
    LoanPurpose: str
    HasCoSigner: str

    @field_validator("Age")
    @classmethod
    def check_age(cls, value: int) -> int:
        if not 18 <= value <= 100:
            raise ValueError("Age must be between 18 and 100.")
        return value

    @field_validator("Income")
    @classmethod
    def check_income(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("Income must be greater than 0.")
        return value

    @field_validator("LoanAmount")
    @classmethod
    def check_loan_amount(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("Loan amount must be greater than 0.")
        return value

    @field_validator("CreditScore")
    @classmethod
    def check_credit_score(cls, value: int) -> int:
        if not 300 <= value <= 850:
            raise ValueError("Credit score must be between 300 and 850.")
        return value

    @field_validator("MonthsEmployed")
    @classmethod
    def check_months_employed(cls, value: int) -> int:
        if value < 0:
            raise ValueError("Months employed cannot be negative.")
        return value

    @field_validator("NumCreditLines")
    @classmethod
    def check_credit_lines(cls, value: int) -> int:
        if value < 1:
            raise ValueError("Number of credit lines must be at least 1.")
        return value

    @field_validator("InterestRate")
    @classmethod
    def check_interest_rate(cls, value: float) -> float:
        if not 0 < value < 100:
            raise ValueError("Interest rate must be greater than 0 and less than 100.")
        return value

    @field_validator("LoanTerm")
    @classmethod
    def check_loan_term(cls, value: int) -> int:
        if value not in ALLOWED_LOAN_TERMS:
            raise ValueError("Loan term must be 12, 24, 36, 48, or 60 months.")
        return value

    @field_validator("DTIRatio")
    @classmethod
    def check_dti(cls, value: float) -> float:
        if not 0 <= value <= 1.5:
            raise ValueError("DTI ratio must be between 0 and 1.5.")
        return value

    @field_validator("Education")
    @classmethod
    def check_education(cls, value: str) -> str:
        return _choice(value, "Education")

    @field_validator("EmploymentType")
    @classmethod
    def check_employment(cls, value: str) -> str:
        return _choice(value, "EmploymentType")

    @field_validator("MaritalStatus")
    @classmethod
    def check_marital_status(cls, value: str) -> str:
        return _choice(value, "MaritalStatus")

    @field_validator("LoanPurpose")
    @classmethod
    def check_loan_purpose(cls, value: str) -> str:
        return _choice(value, "LoanPurpose")

    @field_validator("HasMortgage", "HasDependents", "HasCoSigner")
    @classmethod
    def check_yes_no(cls, value: str, info) -> str:
        return _choice(value, info.field_name)


class PredictionResponse(BaseModel):
    model: str
    model_name: str
    prediction: str
    will_default: bool
    default_probability: float
    message: str


def _error_message(error: dict) -> str:
    kind = error.get("type", "")
    if kind == "missing":
        return "This field is required."
    if kind in {"int_parsing", "int_type", "int_from_float"}:
        return "Enter a whole number."
    if kind in {"float_parsing", "float_type"}:
        return "Enter a number."
    if kind == "extra_forbidden":
        return "This field is not used by the model."
    message = str(error.get("msg", "Invalid value."))
    prefix = "Value error, "
    if message.startswith(prefix):
        return message[len(prefix) :]
    return message


def _field_name(error: dict) -> str:
    parts = [str(part) for part in error.get("loc", []) if part not in {"body", "query"}]
    return ".".join(parts) if parts else "request"


def load_runtime() -> tuple[object, object, dict]:
    selected_path = MODEL_DIR / "selected_model.joblib"
    meta_path = MODEL_DIR / "selected_model.json"
    if not PREPROCESSOR_PATH.exists():
        raise FileNotFoundError(f"Preprocessor not found at {PREPROCESSOR_PATH}. Run preprocess.py first.")
    if not selected_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            "Final model not found. Run tune_models.py to select it before starting the API."
        )
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    model = joblib.load(selected_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return preprocessor, model, meta


@asynccontextmanager
async def lifespan(app: FastAPI):
    preprocessor, model, meta = load_runtime()
    app.state.preprocessor = preprocessor
    app.state.model = model
    app.state.meta = meta
    yield


app = FastAPI(
    title="Loan default prediction",
    summary="Score a loan application with the final selected classifier.",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def invalid_input(request: Request, exc: RequestValidationError) -> JSONResponse:
    errors = [{"field": _field_name(error), "message": _error_message(error)} for error in exc.errors()]
    return JSONResponse(
        status_code=422,
        content={
            "detail": "The application could not be scored because some inputs are missing or invalid.",
            "errors": errors,
        },
    )


@app.get("/")
def service_info(request: Request) -> dict:
    meta = request.app.state.meta
    return {
        "service": "Loan default prediction",
        "predict": "POST /predict",
        "model": meta["model_name"],
        "fields": RAW_COLUMNS,
    }


@app.get("/health")
def health(request: Request) -> dict:
    meta = request.app.state.meta
    return {"status": "ok", "model": meta["model"]}


@app.post("/predict", response_model=PredictionResponse)
def predict(application: LoanApplication, request: Request) -> PredictionResponse:
    frame = pd.DataFrame([application.model_dump()], columns=list(RAW_COLUMNS))
    features = request.app.state.preprocessor.transform(frame)
    columns = request.app.state.meta["features"]
    features = features[columns]

    model = request.app.state.model
    probability = float(model.predict_proba(features)[0, 1])
    threshold = float(request.app.state.meta.get("threshold", 0.5))
    will_default = probability >= threshold
    display = request.app.state.meta["model_name"]
    outcome = "will default" if will_default else "will repay"
    return PredictionResponse(
        model=request.app.state.meta["model"],
        model_name=display,
        prediction="Default" if will_default else "No default",
        will_default=will_default,
        default_probability=round(probability, 4),
        message=f"{display} predicts this applicant {outcome} ({probability:.1%} probability of default).",
    )
