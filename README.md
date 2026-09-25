# FDM Mini Project

Loan-default prediction for a credit officer. The model estimates whether an applicant will default before a loan is approved.

Run every command from the project folder.

## 1. Set up

Requires Python 3.12 and `dataset/Loan_default.csv`.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Prepare the data

Preprocessing has to finish first. Training reads the files it writes.

```bash
python preprocess.py
```

This reads `dataset/Loan_default.csv` and writes the training and test tables, `artifacts/preprocessor.joblib`, and training-split charts in `artifacts/figures/`.

## 3. Train the models

```bash
python train_models.py
```

This trains four classifiers and saves them in `artifacts/models/`:

- logistic regression
- decision tree
- random forest
- histogram gradient boosting

## 4. Tune and select the final model

After preprocessing:

```bash
python tune_models.py
```

This compares tuned settings with the untuned models, writes `artifacts/tuning_report.txt`, and saves the winner as `artifacts/models/selected_model.joblib`. The API scores with that model.

## 5. Start the API

```bash
uvicorn backend.main:app --reload
```

- Service info: http://127.0.0.1:8000/
- Interactive docs: http://127.0.0.1:8000/docs
- Health check: http://127.0.0.1:8000/health

The API loads the preprocessor and the final model from `artifacts/models/selected_model.joblib`. Run `python tune_models.py` first if that file is missing.

## 6. Score an application

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d @- <<'EOF'
{
  "Age": 35,
  "Income": 72000,
  "LoanAmount": 15000,
  "CreditScore": 680,
  "MonthsEmployed": 48,
  "NumCreditLines": 3,
  "InterestRate": 12.5,
  "LoanTerm": 36,
  "DTIRatio": 0.32,
  "Education": "Bachelor's",
  "EmploymentType": "Full-time",
  "MaritalStatus": "Married",
  "HasMortgage": "No",
  "HasDependents": "Yes",
  "LoanPurpose": "Auto",
  "HasCoSigner": "No"
}
EOF
```

The response is the selected model's prediction and the probability of default.

Allowed values:

| Field                                   | Values                                          |
| --------------------------------------- | ----------------------------------------------- |
| Age                                     | 18–100                                          |
| CreditScore                             | 300–850                                         |
| LoanTerm                                | 12, 24, 36, 48, 60                              |
| DTIRatio                                | 0–1.5                                           |
| Education                               | High School, Bachelor's, Master's, PhD          |
| EmploymentType                          | Full-time, Part-time, Self-employed, Unemployed |
| MaritalStatus                           | Single, Married, Divorced                       |
| LoanPurpose                             | Home, Auto, Education, Business, Other          |
| HasMortgage, HasDependents, HasCoSigner | Yes, No                                         |

Income, loan amount, and interest rate must be greater than 0. Interest rate must be under 100. Months employed can be 0. Credit lines must be at least 1.

## 7. Open the review screen

Start the API, then in a second terminal:

```bash
streamlit run frontend/app.py
```

The screen opens at http://localhost:8501. It sends the application to `POST /predict` and shows whether the loan is likely to default.
