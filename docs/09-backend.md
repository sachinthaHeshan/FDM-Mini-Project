# Stage 9 — Backend

The prediction service is a FastAPI application in `backend/main.py`. It loads the training preprocessor and the single selected model, checks the application against the same domain rules as preprocessing, transforms the row with that preprocessor, and returns a default decision.

Start it from the project folder after `preprocess.py` and `tune_models.py` have been run:

```bash
uvicorn backend.main:app --reload
```

- Service info: http://127.0.0.1:8000/
- Interactive docs: http://127.0.0.1:8000/docs
- Health: http://127.0.0.1:8000/health
- Score: `POST /predict`

## What is loaded

On startup (`lifespan` → `load_runtime`):

| File | Role |
| --- | --- |
| `artifacts/preprocessor.joblib` | `FullPreprocessor` fit on the training split: feature engineering, imputation, IQR caps, scaling, encoding, and the selected columns |
| `artifacts/models/selected_model.joblib` | The tuned histogram gradient boosting classifier, fit on the training matrix |
| `artifacts/models/selected_model.json` | Model name, feature list, and decision threshold (0.18) |

If the preprocessor or the selected model is missing, startup fails with an instruction to run the matching script. The service does not fall back to a different algorithm. Stage 6’s four comparison models stay in `artifacts/models/` for the report; they are not what `/predict` calls.

## Request contract

`LoanApplication` is a Pydantic model with `extra="forbid"`. A field the model does not use is rejected. Strings are stripped. Every limit matches `preprocess.py`:

| Field | Rule |
| --- | --- |
| Age | Integer, 18–100 |
| Income | Greater than 0 |
| LoanAmount | Greater than 0 |
| CreditScore | Integer, 300–850 |
| MonthsEmployed | Integer, 0 or more |
| NumCreditLines | Integer, at least 1 |
| InterestRate | Greater than 0 and less than 100 |
| LoanTerm | 12, 24, 36, 48, or 60 |
| DTIRatio | 0–1.5 |
| Education | High School, Bachelor's, Master's, PhD |
| EmploymentType | Full-time, Part-time, Self-employed, Unemployed |
| MaritalStatus | Single, Married, Divorced |
| LoanPurpose | Home, Auto, Education, Business, Other |
| HasMortgage, HasDependents, HasCoSigner | Yes or No |

Category checks are case-insensitive and then stored as the training spelling, so `"yes"` becomes `"Yes"`. The allowed sets are checked at runtime against `ALLOWED_CATEGORIES` in `preprocess.py`. If those two lists ever diverge, the process raises rather than scoring with a mismatched encoding.

## Preprocessing path

`/predict` builds a one-row data frame in `RAW_COLUMNS` order and calls `preprocessor.transform`. That is the same object saved in Stage 4. It:

1. Computes loan-to-income, estimated monthly payment, and payment-to-income from that row only.
2. Applies training medians, modes, IQR fences, and the training scaler.
3. Applies the training one-hot columns.
4. Keeps the feature list recorded in `selected_model.json` (all 27 columns; engineered features were retained).

No statistic is refit on the incoming application.

## Prediction response

The model returns `predict_proba` for class 1. The application is labelled default when that probability is **at least 0.18** (the cutoff stored in the JSON). The response body is:

| Field | Meaning |
| --- | --- |
| `model` | `hist_gradient_boosting` |
| `model_name` | Histogram gradient boosting |
| `prediction` | `Default` or `No default` |
| `will_default` | true or false |
| `default_probability` | Probability of default, rounded to 4 decimal places |
| `message` | A full sentence, including the probability as a percent |

Example message shape: “Histogram gradient boosting predicts this applicant will default (32.0% probability of default).”

## Invalid and missing input

`RequestValidationError` returns HTTP 422 with:

- `detail`: “The application could not be scored because some inputs are missing or invalid.”
- `errors`: a list of `{field, message}`

Messages are rewritten for an officer, not left as parser codes. A missing field says “This field is required.” A non-numeric age says “Enter a whole number.” A bad category repeats the allowed list. An unexpected JSON key says “This field is not used by the model.”

CORS is open so the Streamlit screen (a different port) can call the API during the demo. Interactive docs at `/docs` let a marker send the same JSON without the screen.

## How this meets the brief

| Requirement | Implementation |
| --- | --- |
| Load the final model and the preprocessing pipeline | Startup loads `selected_model.joblib` and `preprocessor.joblib` |
| Accept and validate user inputs | `LoanApplication` plus the 422 handler |
| Apply the same preprocessing used in development | `preprocessor.transform` on the raw fields |
| Generate predictions with the final model | Only the selected booster; threshold from the selection JSON |
| Return a clear result | Label, boolean, probability, and a sentence |
| Handle invalid or missing inputs | 422 with field-level messages; the process does not score a partial row |
