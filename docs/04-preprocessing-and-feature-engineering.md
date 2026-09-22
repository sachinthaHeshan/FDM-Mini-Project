# Stage 4 — Data preprocessing and feature engineering

Preprocessing is implemented in `preprocess.py` and written up in `artifacts/preprocessing_report.txt`. Every fitted statistic (medians, modes, IQR fences, scaler means, category levels, and the keep/drop decision) is learned on the training split only. The script saves `artifacts/preprocessor.joblib`, which the API uses unchanged.

Command:

```bash
python preprocess.py
```

Outputs: `artifacts/X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv`, `feature_selection.csv`, `preprocessor.joblib`, `preprocessing_report.txt`.

## Split first, so nothing leaks

| Setting | Value | Why |
| --- | --- | --- |
| Fraction held out | 20% (`test_size=0.2`) | Large enough for a stable final score (51,070 rows, 5,931 defaults) and still leaves 204,277 rows to train on |
| Stratify | `Default` | Both sides keep the 11.61% default rate |
| Seed | `random_state=42` | The split can be repeated |
| What is fit after the split | Imputation, IQR caps, scaling, one-hot levels, feature-selection scores | Test rows never choose a median, a fence, a category level, or a dropped column |

The test set is not used to choose features, hyperparameters, or the final algorithm. It is scored once after those choices.

## Missing values

The file has **0 missing cells**, so no training value is invented. The pipeline still includes imputers because the saved preprocessor must define a behaviour for a blank, and because median/mode imputation is the standard way to keep a column usable if a later extract is incomplete.

| Column kind | Imputer | Why this strategy |
| --- | --- | --- |
| Numeric | Median | A median is not pulled by the long right tail of the payment ratios |
| Yes/No, education, and nominal fields | Most frequent (mode) | A blank category becomes the most common training label, which is a defined value the encoder has seen |

The API does not rely on imputation for the officer’s form. It rejects a missing or out-of-range field with HTTP 422 before scoring. Imputation remains inside the training pipeline so the two paths share one transformer.

## Duplicate and invalid records

| Rule | Action | Result on this file |
| --- | --- | --- |
| Exact duplicate row | Drop | 0 dropped |
| Duplicate `LoanID` | Keep the first | 0 dropped |
| Age not in 18–100 | Drop | 0 |
| Income or loan amount ≤ 0 | Drop | 0 |
| Credit score not in 300–850 | Drop | 0 |
| Months employed < 0 | Drop | 0 |
| Credit lines < 1 | Drop | 0 |
| Interest rate not in (0, 100) | Drop | 0 |
| DTI not in [0, 1.5] | Drop | 0 |
| Loan term not in {12, 24, 36, 48, 60} | Drop | 0 |
| Category not in the known label set | Drop | 0 |
| Target not in {0, 1} | Drop | 0 |

Rows after cleaning: **255,347** (unchanged). Domain rules are per-row checks. They are not statistics learned from other rows, so applying them before the split does not leak the test label.

`MonthsEmployed` = 0 (2,122 rows) fails none of these rules and is kept.

## Outliers

An `IQRCapper` learns, on the training numeric columns only, the fences Q1 − 1.5×IQR and Q3 + 1.5×IQR, then clips transform inputs to those fences.

Why clip rather than delete:

- Raw fields already lie inside the fences, so deletion would remove nobody and would hide the fact that extreme live inputs still need a bound.
- High loan-to-income and high payment-to-income values are the risk signal. Deleting them would train the model on easier applicants than the officer will see.
- Clipping uses training fences only. The test set does not move the cap.

During fit, 46,267 training numeric cells (concentrated in the three engineered ratios) sit outside the fences and are clipped. Raw age, income, amount, score, employment length, credit lines, rate, term, and DTI contribute 0 clipped cells.

## Encoding

| Field | Encoding | Reason |
| --- | --- | --- |
| `HasMortgage`, `HasDependents`, `HasCoSigner` | Yes = 1, No = 0 | Two-level flags. Scaling them would not add information |
| `Education` | Ordinal: High School, Bachelor's, Master's, PhD, then standard scaling | The order is the education ladder. It is not a ranking fitted from `Default`, so it is not target leakage. An unknown label becomes −1 |
| `EmploymentType`, `MaritalStatus`, `LoanPurpose` | One-hot, unknown categories ignored | These labels have no trustworthy order. Full one-hot is kept so each level stays visible. Logistic regression’s default L2 penalty handles the extra columns |
| `LoanID` | Removed | Unique key, not a property of the borrower |

Unknown categories at scoring time become an all-zero dummy row (`handle_unknown="ignore"`), so a typo cannot crash the transformer. The API still blocks unknown labels before that path, and returns the allowed list.

## Scaling

Numeric columns and the ordinal education code go through `StandardScaler` fitted on the training split. Trees do not need scaled inputs. Logistic regression does: income is tens of thousands and DTI is a fraction, so an unscaled coefficient would mostly reflect units. One scaler for every model keeps the exported matrix identical for all four algorithms.

Yes/No columns are not scaled. They are already 0/1.

## Feature engineering

`LoanFeatureEngineer` adds three columns. Each uses only the current row.

| Feature | Formula | Why it was added |
| --- | --- | --- |
| `loan_to_income` | `LoanAmount / Income` | The quartile gap in default rate is 8.0% vs 18.7%. Amount and income separately are weaker than the ratio |
| `estimated_monthly_payment` | Standard amortising instalment from amount, annual rate, and term in months | Turns three columns into the payment the applicant must meet |
| `payment_to_income` | Monthly payment / (income / 12) | Affordability: share of monthly income taken by the instalment |

`MonthsEmployed / 12` was considered and not added. After standard scaling it is the same column as `MonthsEmployed`.

If the interest rate is 0, the payment falls back to amount divided by term. Training rates are all above 0; the branch protects scoring.

These features are inputs the officer already provided (amount, income, rate, term). They do not use the label.

Stage 7 checks whether the tuned winner still needs them. Cross-validated average precision is 0.3175 with them and 0.3177 without them. The gap is under 0.005, so they stay. They match the credit decision (how large is the loan relative to pay) even though the lift on this particular table is small.

## Feature selection

Selection runs on the training split only, after engineering.

| Rule | Cutoff | Why this cutoff |
| --- | --- | --- |
| Numeric | Keep if \|Pearson r\| with `Default` ≥ 0.01 | With 200,000 rows a correlation of 0.001 can be “significant” and still useless in a credit decision. One percentage point of linear association is the practical floor |
| Categorical | Keep if the gap in default rate across levels ≥ 1 percentage point | Same practical floor, expressed as a rate difference an officer would notice |
| Redundancy | If two numeric columns have \|r\| > 0.98, drop the one less related to `Default` | Prevents duplicate signals. No pair crossed this line |
| Identifier | Always drop `LoanID` | Cannot be used for a new applicant |

Kept (training evidence in `artifacts/feature_selection.csv`):

- Numeric: age, income, loan amount, credit score, months employed, credit lines, interest rate, DTI, loan-to-income, estimated monthly payment, payment-to-income.
- Binary: mortgage, dependents, co-signer.
- Ordinal: education.
- Nominal: employment type, marital status, loan purpose.

Dropped from the model matrix: **`LoanTerm`** (training r = 0.0012). The officer still enters the term. It is used to calculate the monthly payment, then left out of the matrix because the term by itself does not change the default rate.

The fitted matrix has **27 columns** after one-hot encoding (11 numeric + 3 binary + 1 education + 4 employment + 3 marital status + 5 loan purpose).

## Class imbalance is not “preprocessed away”

Resampling is intentionally absent. Balancing belongs inside model training (`class_weight`) or in the decision cutoff, and it must be fit on each training fold. Doing it here would change the 11.6% base rate the probabilities are calibrated against.

## Leakage controls (summary)

- Split before any fitted transform.
- No target encoding.
- Engineered ratios are row-local.
- `LoanID` excluded.
- Preprocessor saved from `fit` on training rows and checked so `transform` is stable.
- The script aborts if the prepared matrices still contain NaN.

## Fair-lending note

Age and marital status pass the association cutoff, so they remain in the academic model. The preprocessing report states that a production lending model would need a fairness review before those fields were used to approve or decline an applicant. The system copy tells the officer the score supports the decision and does not replace it.

## What each important decision was answering

| Decision | Evidence it rests on |
| --- | --- |
| Do not drop rows | 0 invalid rows, 0 duplicates, 0 missing cells |
| Keep zero months employed | Domain meaning is “new employment,” 2,122 rows |
| Cap with training IQR fences, do not delete | Raw fields are inside fences; ratio tails are genuine high-risk cases |
| Standardise numerics | Logistic regression is one of the four required models and the columns have different units |
| Ordinal education, one-hot nominals | Education has an external order; employment, marital status, and purpose do not |
| Add three affordability features | Loan-to-income quartile default rates 8% to 19% |
| Drop loan term from the matrix | Correlation 0.001; term still feeds the payment formula |
| Stratified 80/20 before fitting | Preserves 11.61% default rate and blocks statistic leakage |
