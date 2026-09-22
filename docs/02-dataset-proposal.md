# Stage 2 — Dataset identification and proposal

This is the dataset proposal required before EDA, preprocessing, training, and system development. It covers every item the brief asks the proposal to include.

The repository does not store a signed instructor approval form. Attach the lab validation record (date, instructor, and outcome) to this document when it is submitted. If this dataset is rejected, a different public dataset must be proposed and this file updated. Development in the repository uses this dataset.

## Dataset name and original source

| Item | Detail |
| --- | --- |
| Dataset name | Loan Default Prediction Dataset |
| File used | `dataset/Loan_default.csv` |
| Public host | Kaggle, uploaded by Nikhil (`nikhil1e9`) |
| Stated origin | The Kaggle description says the table was taken from Coursera’s Loan Default Prediction Challenge |
| Download URL | https://www.kaggle.com/datasets/nikhil1e9/loan-default |
| Licence | CC0: Public Domain |

## Citation

Nikhil. (n.d.). *Loan Default Prediction Dataset* [Data set]. Kaggle. https://www.kaggle.com/datasets/nikhil1e9/loan-default

The Kaggle page states that the table comes from Coursera’s Loan Default Prediction Challenge and that it contains 255,347 rows and 18 columns. The local file matches that size.

## Description and real-world context

Each row is one loan application. Columns describe the applicant (age, income, education, employment, marital status, dependents), the applicant’s credit position (credit score, credit lines, debt-to-income ratio, mortgage, co-signer), and the loan (amount, interest rate, term, purpose). The last column records whether that loan defaulted.

Lenders use this kind of table to decide who is likely to miss repayments before money is released. That is the same decision in Stage 1. The Kaggle introduction describes the same use: predict which individuals are at the highest risk of defaulting so that intervention can be aimed at them.

## Records and features

| Item | Value |
| --- | --- |
| Records | 255,347 |
| Columns | 18 (16 predictors, 1 identifier, 1 target) |
| Missing cells | 0 |
| Duplicate rows | 0 |
| Duplicate `LoanID` values | 0 |

| Column | Type | Meaning |
| --- | --- | --- |
| `LoanID` | Identifier | Unique application key. Removed before modelling |
| `Age` | Numeric | Applicant age in years (18–69 in this file) |
| `Income` | Numeric | Annual income (15,000–149,999) |
| `LoanAmount` | Numeric | Amount requested (5,000–249,999) |
| `CreditScore` | Numeric | Score from 300 to 849 |
| `MonthsEmployed` | Numeric | Months in current employment (0–119). Zero is kept: it can mean a new job |
| `NumCreditLines` | Numeric | Open credit lines, values 1–4 |
| `InterestRate` | Numeric | Annual interest rate, 2–25 |
| `LoanTerm` | Numeric | Term in months: 12, 24, 36, 48, or 60 |
| `DTIRatio` | Numeric | Debt-to-income ratio, 0.10–0.90 |
| `Education` | Ordinal | High School, Bachelor's, Master's, PhD |
| `EmploymentType` | Nominal | Full-time, Part-time, Self-employed, Unemployed |
| `MaritalStatus` | Nominal | Single, Married, Divorced |
| `HasMortgage` | Binary | Yes / No |
| `HasDependents` | Binary | Yes / No |
| `LoanPurpose` | Nominal | Home, Auto, Education, Business, Other |
| `HasCoSigner` | Binary | Yes / No |
| `Default` | Target | 1 = defaulted, 0 = did not default |

Class counts: **29,653 defaults** and **225,694 non-defaults**. The default rate is **11.61%**.

## Target variable

`Default` is the target.

- `1` means the loan defaulted.
- `0` means it did not.

The label is observed after the loan was issued. For a new application the label is unknown, which is why it is the prediction target and never an input. All other columns are available when the officer is still deciding.

## Expected machine learning task

**Binary classification.**

The label has two classes. Metrics that respect the 11.6% default rate (average precision, ROC-AUC, precision, recall, F1) are appropriate. Accuracy alone is not, because a model that always says “no default” is already about 88.4% accurate and catches no defaults.

## Why this dataset fits the scenario

- Every row is one loan the organisation could have approved or declined.
- The target is exactly the event the officer needs to anticipate.
- Predictors are application-time facts: income, amount, rate, term, employment, and credit position. Nothing in the file is an after-the-fact collection outcome other than `Default` itself.
- 255,347 rows is large enough for a stratified hold-out and cross-validation, and for four different algorithms to be compared.
- The table is not trivial: classes are imbalanced, several numeric fields overlap (amount, income, and ratios built from them), and categorical fields have uneven default rates. That supports real EDA, preprocessing choices, and tuning.
- The file is public, the same CSV is available to every group member, and the licence is CC0, so academic use does not depend on a paid subscription beyond a free Kaggle download.

## Data-quality observations and limitations

Observed on the local file:

- No missing values and no duplicate applications.
- Numeric ranges already sit inside realistic bounds (age 18–69, credit score 300–849, interest rate 2–25, DTI 0.10–0.90). There is little need to delete rows as “impossible.”
- `MonthsEmployed` is 0 on 2,122 rows. Those rows are valid new-employment cases and are kept.
- `LoanTerm` barely moves with `Default` (correlation about 0.0005). It is still needed to compute a monthly payment, then dropped from the model matrix.
- Categories are balanced in count (each education and employment level has about 63,000–64,000 rows) but not in default rate. Unemployed applicants default at 13.6%; full-time applicants default at 9.5%.
- The file looks synthetic or heavily regularised: ages stop at 69, incomes and loan amounts look uniformly spread, and there are no missing fields. Patterns are still strong enough to model (age, interest rate, and loan size relative to income), but the data should not be described as a live extract from one named bank.
- There is no loss amount, recovery amount, or time-to-default, so the model estimates whether a default happens, not how expensive it is.

## Ethical, privacy, licensing, and access considerations

- **Licence.** CC0 public domain. The file can be stored in the project and used for coursework. The Kaggle page should still be cited.
- **Access.** Any group member can download the same CSV from the URL above. The copy used here is `dataset/Loan_default.csv`.
- **Privacy.** The table has no names, addresses, account numbers, or national IDs. `LoanID` is an opaque code and is not used as a feature. Re-identification risk from this file alone is low.
- **Fair lending.** `Age` and `MaritalStatus` predict default in this file and are also sensitive in real credit decisions. They are kept for this academic task because they are part of the published table and they change the default rate (applicants aged 18–25 default at about 20.8%; applicants aged 56–69 default at about 5.5%). A production lending model would need a fairness review before those fields were used to approve or decline a person. The preprocessing report records the same limit.
- **Decision use.** The score is a review aid. A positive flag means “look at this file,” not “automatically decline.”

## Suitability checklist from the brief

| Guideline | How this dataset meets it |
| --- | --- |
| Relevant to the scenario | Rows are loan applications; the label is default |
| Clear prediction target | `Default` ∈ {0, 1} |
| Enough rows and useful features | 255,347 rows and 16 candidate predictors |
| Credible public source | Kaggle, CC0, described as the Coursera loan-default challenge data |
| Available to the whole group | Single public CSV |
| Enough complexity | Imbalance, weak vs strong predictors, overlapping amount/income signals, sensitive attributes |
| Supports data-mining concepts | Classification, EDA, preprocessing, four algorithms, tuning, deployment |
| No unnecessary legal barrier | No direct identifiers; public-domain licence |
