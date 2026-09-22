# Stage 5 — Progress Evaluation 1 (individual viva)

Weight: 30%. Individual. Dates in the brief: 23, 26, and 27 September 2026 (other campuses as announced).

The viva covers work through preprocessing. Each student is expected to explain the project, not only the file they edited. Use this note as the map from each assessment point to the evidence. Replace the contribution section with your own work before the lab.

## What you should be able to explain

### Problem, scenario, and task type

A lender needs to know whether an applicant will default before the loan is approved. The user is a credit officer. The system returns a default label and a probability so the officer can approve, review, or price the file. The target is `Default` (1 = default, 0 = repaid). The task is binary classification because the outcome has two classes and the decision is yes/no plus a risk score. Detail: `docs/01-problem-scenario.md`.

### Dataset selection and justification

Public file: Loan Default Prediction Dataset, Kaggle, https://www.kaggle.com/datasets/nikhil1e9/loan-default, CC0, described there as Coursera’s loan-default challenge data. Local copy: `dataset/Loan_default.csv`. 255,347 applications, 18 columns, target `Default` at 11.61%. It matches the scenario because each row is one application and the predictors are known before approval. It is large and uneven enough for a real comparison of models. Limitations: the ranges look synthetic, there is no loss amount, and age and marital status are sensitive. Detail: `docs/02-dataset-proposal.md`.

If the instructor has already signed the proposal, say the date. The repository does not contain that signature page.

### Dataset characteristics and the target

- No missing values, no duplicate rows, no duplicate loan ids.
- Numeric fields sit in realistic ranges (age 18–69, score 300–849, rate 2–25, DTI 0.10–0.90).
- 29,653 defaults and 225,694 non-defaults.
- Strongest full-file correlations with default: age −0.17, interest rate +0.13, income −0.10, months employed −0.10, loan amount +0.09.
- `LoanID` is not a feature.

### EDA findings you can quote

- Applicants aged 18–25 default at about 20.8%; applicants aged 56–69 default at about 5.5%.
- Interest rates of 20–25% default at about 17.8%; rates of 2–8% default at about 6.6%.
- The highest quarter of loan-to-income defaults at about 18.7%; the lowest quarter defaults at about 8.0%.
- Unemployed 13.6% vs full-time 9.5%. High school 12.9% vs PhD 10.6%.
- Loan term does not change the default rate (about 11.6% at every term).
- A model that always says “no default” is 88.4% accurate and useless for catching defaults.

Detail: `docs/03-exploratory-data-analysis.md`.

### Data-quality issues

There is almost nothing to repair in the raw file. The quality issues that still affect modelling are imbalance, a long tail on the engineered ratios, a useless raw loan term, and sensitive attributes. `MonthsEmployed` = 0 on 2,122 rows is kept on purpose.

### Preprocessing and why

All of this is in `preprocess.py`. Fit happens after the split.

- Domain rules drop impossible rows. On this file they drop nobody.
- Median imputation for numbers, mode for categories, saved even though the file is complete.
- IQR cap from the training split. Raw fields are not clipped. Ratio tails are clipped and not deleted, because they are high-risk cases.
- Yes/No mapped to 1/0. Education ordered then scaled. Nominal fields one-hot encoded.
- Standard scaling so logistic regression is not dominated by income and loan amount units.
- Stratified 80/20, seed 42. Train 204,277. Test 51,070. Both default rates 11.61%.

### Feature engineering and selection

- `loan_to_income = LoanAmount / Income`
- `estimated_monthly_payment` from amount, rate, and term
- `payment_to_income` = that payment divided by monthly income
- A “years employed” column was not added because scaling would duplicate `MonthsEmployed`
- Keep a numeric column when |r| ≥ 0.01 on the training split. Keep a category when the default-rate gap is at least 1 percentage point
- Drop `LoanTerm` from the matrix (r = 0.001). The screen still asks for it, because the payment formula needs it
- Final width: 27 columns. List is in the preprocessing report

### Data leakage

Say these five controls without reading them off the screen if you can:

1. `LoanID` is excluded.
2. The split happens before imputation, scaling, fences, one-hot levels, and feature selection.
3. Ratios use only the current application.
4. The label is never used to build a feature (no target encoding).
5. The API loads `artifacts/preprocessor.joblib`, which was fit on training rows.

Also say why the table was not oversampled in preprocessing: copied defaults would leak across folds and would distort the probability.

### Your own contribution

Write two or three concrete items before the viva. Examples of the kind of sentence that is specific enough:

- “I checked the category default rates and argued that loan term should not enter the model matrix.”
- “I implemented the domain checks and the IQR capper in `preprocess.py`.”
- “I wrote the dataset proposal and the citation.”

Avoid “I helped with preprocessing” with no file or decision attached. The examiner is checking that you can defend a decision, including ones made by someone else in the group. You should still know Stages 1–4 end to end.

## Short answers if the question is abrupt

| Question | Answer |
| --- | --- |
| What are you predicting? | Whether the applicant defaults (`Default` = 1) |
| Classification or regression? | Classification |
| Why not accuracy? | Always predicting no-default is already 88.4% accurate |
| Why this dataset? | One row per application, public, labelled, large, features known before approval |
| What did you drop? | `LoanID` always; `LoanTerm` after using it in the payment formula |
| Where is the preprocessor? | `artifacts/preprocessor.joblib`, produced by `python preprocess.py` |
| Did you balance the classes in the CSV? | No. Class weights are inside training, per fold |
| Is age allowed? | It is in the academic model because it predicts default. A live lender would need a fairness review |
