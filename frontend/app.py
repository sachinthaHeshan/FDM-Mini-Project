"""Credit-officer screen for the loan default API.

Run the API first, then:

    .venv/bin/streamlit run frontend/app.py
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

import streamlit as st

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")

EDUCATION = ["High School", "Bachelor's", "Master's", "PhD"]
EMPLOYMENT = ["Full-time", "Part-time", "Self-employed", "Unemployed"]
MARITAL_STATUS = ["Single", "Married", "Divorced"]
LOAN_PURPOSE = ["Home", "Auto", "Education", "Business", "Other"]
LOAN_TERMS = [12, 24, 36, 48, 60]
YES_NO = ["Yes", "No"]


def validate(application: dict) -> list[str]:
    """Same limits the API enforces, shown before the request is sent."""
    errors = []
    age = application["Age"]
    if age is None or not 18 <= int(age) <= 100:
        errors.append("Age must be between 18 and 100.")
    if application["Income"] is None or application["Income"] <= 0:
        errors.append("Income must be greater than 0.")
    if application["LoanAmount"] is None or application["LoanAmount"] <= 0:
        errors.append("Loan amount must be greater than 0.")
    score = application["CreditScore"]
    if score is None or not 300 <= int(score) <= 850:
        errors.append("Credit score must be between 300 and 850.")
    if application["MonthsEmployed"] is None or application["MonthsEmployed"] < 0:
        errors.append("Months employed cannot be negative.")
    if application["NumCreditLines"] is None or int(application["NumCreditLines"]) < 1:
        errors.append("Number of credit lines must be at least 1.")
    rate = application["InterestRate"]
    if rate is None or not 0 < rate < 100:
        errors.append("Interest rate must be greater than 0 and less than 100.")
    if application["LoanTerm"] not in LOAN_TERMS:
        errors.append("Loan term must be 12, 24, 36, 48, or 60 months.")
    ratio = application["DTIRatio"]
    if ratio is None or not 0 <= ratio <= 1.5:
        errors.append("DTI ratio must be between 0 and 1.5.")
    return errors


def score_application(application: dict) -> tuple[dict | None, str | None]:
    payload = json.dumps(application).encode()
    request = urllib.request.Request(
        f"{API_URL}/predict",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.load(response), None
    except urllib.error.HTTPError as error:
        body = json.loads(error.read().decode())
        messages = [item["message"] for item in body.get("errors", [])]
        detail = body.get("detail", "The application could not be scored.")
        return None, detail if not messages else detail + " " + " ".join(messages)
    except urllib.error.URLError:
        return None, (
            "The prediction service is not running. In another terminal, from the project folder, run: "
            "uvicorn backend.main:app --reload"
        )


def show_result(result: dict) -> None:
    probability = float(result["default_probability"])
    percent = f"{probability:.1%}"
    if result["will_default"]:
        st.error(result["message"])
        st.subheader("Send this application for a closer review")
    else:
        st.success(result["message"])
        st.subheader("Lower default risk")
    st.metric("Probability of default", percent)
    st.progress(min(max(probability, 0.0), 1.0))
    st.caption(
        f"Result from {result['model_name']}. "
        "This supports the approval decision. It does not replace it."
    )


st.set_page_config(page_title="Loan default check", layout="centered")
st.title("Loan application review")
st.write(
    "Enter the applicant and the loan. The check estimates whether this loan is likely "
    "to default, before you approve it."
)

with st.form("application"):
    st.subheader("Applicant")
    left, right = st.columns(2)
    with left:
        age = st.number_input("Age", min_value=0, max_value=120, value=35, step=1, help="18 to 100 years.")
        income = st.number_input(
            "Annual income",
            min_value=0.0,
            value=72000.0,
            step=1000.0,
            help="Yearly income. Must be greater than 0.",
        )
        credit_score = st.number_input(
            "Credit score",
            min_value=0,
            max_value=900,
            value=680,
            step=1,
            help="300 to 850.",
        )
        months_employed = st.number_input(
            "Months employed",
            min_value=0,
            value=48,
            step=1,
            help="Months in the current job. 0 means newly employed.",
        )
    with right:
        education = st.selectbox("Education", EDUCATION, index=1)
        employment = st.selectbox("Employment type", EMPLOYMENT, index=0)
        marital_status = st.selectbox("Marital status", MARITAL_STATUS, index=1)
        credit_lines = st.number_input(
            "Credit lines",
            min_value=0,
            value=3,
            step=1,
            help="Open credit lines. At least 1.",
        )

    st.subheader("Loan")
    loan_left, loan_right = st.columns(2)
    with loan_left:
        loan_amount = st.number_input(
            "Loan amount",
            min_value=0.0,
            value=15000.0,
            step=500.0,
            help="Amount requested. Must be greater than 0.",
        )
        interest_rate = st.number_input(
            "Interest rate (%)",
            min_value=0.0,
            value=12.5,
            step=0.1,
            format="%.2f",
            help="Annual rate, greater than 0 and less than 100.",
        )
        loan_term = st.selectbox("Loan term (months)", LOAN_TERMS, index=2)
        dti_ratio = st.number_input(
            "Debt-to-income ratio",
            min_value=0.0,
            max_value=2.0,
            value=0.32,
            step=0.01,
            format="%.2f",
            help="From 0 to 1.5. For example, 0.32 means debts are 32% of income.",
        )
    with loan_right:
        loan_purpose = st.selectbox("Loan purpose", LOAN_PURPOSE, index=1)
        has_mortgage = st.radio("Has a mortgage", YES_NO, horizontal=True, index=1)
        has_dependents = st.radio("Has dependents", YES_NO, horizontal=True, index=0)
        has_cosigner = st.radio("Has a co-signer", YES_NO, horizontal=True, index=1)

    submitted = st.form_submit_button("Check default risk", type="primary")

if submitted:
    application = {
        "Age": int(age),
        "Income": float(income),
        "LoanAmount": float(loan_amount),
        "CreditScore": int(credit_score),
        "MonthsEmployed": int(months_employed),
        "NumCreditLines": int(credit_lines),
        "InterestRate": float(interest_rate),
        "LoanTerm": int(loan_term),
        "DTIRatio": float(dti_ratio),
        "Education": education,
        "EmploymentType": employment,
        "MaritalStatus": marital_status,
        "HasMortgage": has_mortgage,
        "HasDependents": has_dependents,
        "LoanPurpose": loan_purpose,
        "HasCoSigner": has_cosigner,
    }
    problems = validate(application)
    if problems:
        st.error("Fix these fields before the application can be checked.")
        for problem in problems:
            st.write(f"- {problem}")
    else:
        with st.spinner("Checking the application..."):
            result, error = score_application(application)
        if error:
            st.error(error)
        else:
            show_result(result)
