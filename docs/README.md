# IT3051 Mini Project — stage documents

These documents record how the loan-default project handles each part of the IT3051 Fundamentals of Data Mining mini-project guide (2026). Each file follows the stage in the assignment brief and states what was done, why, and where the evidence sits in the repository.

| Stage | Document | What it covers |
| --- | --- | --- |
| 1 | [01-problem-scenario.md](01-problem-scenario.md) | Problem, users, inputs and outputs, and why this is classification |
| 2 | [02-dataset-proposal.md](02-dataset-proposal.md) | Dataset proposal: source, citation, size, target, suitability, limitations |
| 3 | [03-exploratory-data-analysis.md](03-exploratory-data-analysis.md) | Structure, distributions, quality, imbalance, leakage, and modelling implications |
| 4 | [04-preprocessing-and-feature-engineering.md](04-preprocessing-and-feature-engineering.md) | Cleaning, encoding, scaling, engineered features, selection, and the train/test split |
| 5 | [05-progress-evaluation-1.md](05-progress-evaluation-1.md) | Viva notes for the individual evaluation up to preprocessing |
| 6 | [06-model-development.md](06-model-development.md) | Four classifiers, validation, metrics, and why the scores differ |
| 7 | [07-model-optimization.md](07-model-optimization.md) | Randomised search, feature check, threshold, and final model choice |
| 8 | [08-progress-evaluation-2.md](08-progress-evaluation-2.md) | Viva notes for the individual modelling evaluation |
| 9 | [09-backend.md](09-backend.md) | API: model loading, validation, preprocessing, and prediction responses |
| 10 | [10-frontend.md](10-frontend.md) | Credit-officer screen and how it calls the API |
| 11 | [11-technical-report.md](11-technical-report.md) | Full technical report covering every required report section |
| 12 | [12-final-presentation.md](12-final-presentation.md) | Non-technical slide outline, demo script, and recommendation talking points |

## Project in one paragraph

A credit officer needs to know, before a loan is approved, whether the applicant is likely to default. The system is a binary classifier trained on the public Loan Default Prediction Dataset (255,347 applications). After cleaning, feature engineering, and a stratified hold-out, four models were compared and tuned. The selected model is a tuned histogram gradient boosting classifier. A FastAPI service applies the same preprocessor and returns a default probability. A Streamlit screen lets the officer enter an application and read the result.

## How to reproduce the evidence

From the project folder, with the virtual environment active:

```bash
python preprocess.py
python train_models.py
python tune_models.py
uvicorn backend.main:app --reload
streamlit run frontend/app.py
```

Written evidence already produced by those scripts:

- `artifacts/preprocessing_report.txt`
- `artifacts/figures/` (training-split charts from `preprocess.py`)
- `artifacts/feature_selection.csv`
- `artifacts/model_report.txt`
- `artifacts/model_comparison_cv.csv`
- `artifacts/model_comparison_test.csv`
- `artifacts/tuning_report.txt`
- `artifacts/tuning_comparison.csv`
- `artifacts/models/selected_model.json`
