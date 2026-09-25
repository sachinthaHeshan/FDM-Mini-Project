# Stage 3 — Data understanding and exploratory data analysis

EDA was done on `dataset/Loan_default.csv` (255,347 rows, 18 columns) before modelling decisions were locked. The checks below are the ones the brief lists, plus the relationships that later changed preprocessing and model choice. Numeric summaries were computed on the full file. Association cutoffs used for keeping features were recomputed on the training split only, so the test labels did not choose the columns. That second measurement is in `artifacts/feature_selection.csv` and is cited where the training-split figure differs from the full-file figure.

## Structure, meaning, and variable types

One row is one application. There is no time index and no repeated applicant id, so this is a single-table classification problem, not a panel or a sequence.

| Kind | Columns |
| --- | --- |
| Identifier | `LoanID` (255,347 unique values) |
| Numeric | `Age`, `Income`, `LoanAmount`, `CreditScore`, `MonthsEmployed`, `NumCreditLines`, `InterestRate`, `LoanTerm`, `DTIRatio` |
| Ordered category | `Education` (High School < Bachelor's < Master's < PhD) |
| Unordered category | `EmploymentType`, `MaritalStatus`, `LoanPurpose` |
| Yes/No | `HasMortgage`, `HasDependents`, `HasCoSigner` |
| Target | `Default` |

`NumCreditLines` and `LoanTerm` are numeric but take few values (1–4 and five term lengths). They can be treated as numeric because the gaps are equal and ordered.

Full-file numeric profile:

| Feature | Min | Median | Max | Mean |
| --- | ---: | ---: | ---: | ---: |
| Age | 18 | 43 | 69 | 43.5 |
| Income | 15,000 | 82,466 | 149,999 | 82,499 |
| LoanAmount | 5,000 | 127,556 | 249,999 | 127,579 |
| CreditScore | 300 | 574 | 849 | 574 |
| MonthsEmployed | 0 | 60 | 119 | 59.5 |
| NumCreditLines | 1 | 2 | 4 | 2.5 |
| InterestRate | 2.0 | 13.46 | 25.0 | 13.49 |
| LoanTerm | 12 | 36 | 60 | 36.0 |
| DTIRatio | 0.10 | 0.50 | 0.90 | 0.50 |

Spreads are wide and roughly symmetric around the middle of each allowed range. That pattern, together with zero missing values, is a sign the table was generated or cleaned for a teaching challenge rather than exported raw from a core banking system. The relationships with default are still large enough to model.

## Relationships that matter for the decision

Pearson correlation with `Default` on the full file:

| Feature | Correlation with Default | Direction |
| --- | ---: | --- |
| Age | −0.168 | Older applicants default less often |
| InterestRate | +0.131 | Higher rate, higher default rate |
| Income | −0.099 | Higher income, lower default rate |
| MonthsEmployed | −0.097 | Longer employment, lower default rate |
| LoanAmount | +0.087 | Larger loans default more often |
| CreditScore | −0.034 | Weak protective effect |
| NumCreditLines | +0.028 | Weak increase in default |
| DTIRatio | +0.019 | Weak increase in default |
| LoanTerm | +0.001 | No practical relationship |

Default rate by age band:

| Age | Default rate | Applications |
| --- | ---: | ---: |
| 18–25 | 20.8% | 39,016 |
| 26–35 | 16.1% | 49,408 |
| 36–45 | 11.5% | 49,220 |
| 46–55 | 8.5% | 49,148 |
| 56–69 | 5.5% | 68,555 |

Default rate by interest-rate band:

| Interest rate | Default rate |
| --- | ---: |
| 2–8% | 6.6% |
| 8–14% | 9.7% |
| 14–20% | 13.4% |
| 20–25% | 17.8% |

Loan amount divided by income (loan-to-income), split into quartiles:

| Loan-to-income quartile | Default rate |
| --- | ---: |
| Lowest quarter | 8.0% |
| Second | 9.2% |
| Third | 10.6% |
| Highest quarter | 18.7% |

Category default rates (counts are nearly equal inside each field, so the rate difference is not an artefact of a tiny group):

| Field | Highest default rate | Lowest default rate |
| --- | --- | --- |
| Employment | Unemployed 13.6% (63,824) | Full-time 9.5% (63,656) |
| Education | High School 12.9% (63,903) | PhD 10.6% (63,537) |
| Marital status | Divorced 12.5% (85,033) | Married 10.4% (85,302) |
| Loan purpose | Business 12.3% (51,298) | Home 10.2% (51,286) |
| Mortgage | No 12.3% | Yes 10.9% |
| Dependents | No 12.7% | Yes 10.5% |
| Co-signer | No 12.9% | Yes 10.4% |

Loan term default rates sit between 11.57% and 11.70% for every term. The term does not separate defaulters. It is still used later to estimate the monthly instalment, because amount, rate, and term together describe affordability.

On the training split, the same ordering holds. The strongest training associations are loan-to-income (r = 0.178), age (r = −0.166), payment-to-income (r = 0.152), and interest rate (r = 0.131). Evidence: `artifacts/feature_selection.csv`.

## Missing values and duplicates

| Check | Result |
| --- | --- |
| Missing cells | 0 |
| Exact duplicate rows | 0 |
| Duplicate `LoanID` | 0 |
| Rows outside domain rules (age, income, score, rate, term, categories, target) | 0 |
| `MonthsEmployed` = 0 | 2,122 rows, kept |

No imputation is required to make the training table complete. Median and mode imputers are still fitted and saved so a future request with a blank numeric or category field can be filled the same way, and so the API can reject blanks before they reach the model.

## Outliers and unusual observations

IQR fences (1.5 × IQR) were computed on the **training split** for every numeric model input, including the three engineered ratios. Raw application fields have **zero** training values outside those fences. Examples from `artifacts/preprocessing_report.txt`:

- Age fences are about −6.5 to 93.5; observed ages are 18–69.
- Income fences sit outside 15,000–149,999.
- Interest-rate fences sit outside 2–25.

The engineered ratios do have a long right tail: about 15,955 training values of `loan_to_income` fall outside the IQR fence, and similar counts appear for the payment ratios. Those tails are real affordability cases (a large loan against a small income), not keystroke errors. They are capped at the training fence inside the preprocessor so one extreme ratio cannot dominate a linear model, and they are not deleted. Deleting them would remove the applicants the officer most needs to see.

`MonthsEmployed` = 0 is unusual only if zero is treated as “unknown.” Here it is a defined state and is kept.

## Class imbalance

| Split | Defaults | Rows | Default rate |
| --- | ---: | ---: | ---: |
| Full file | 29,653 | 255,347 | 11.61% |
| Train (80%, stratified) | 23,722 | 204,277 | 11.61% |
| Test (20%, stratified) | 5,931 | 51,070 | 11.61% |

A classifier that always predicts “no default” is about 88.4% accurate and has recall 0. Accuracy is reported later so that trap is visible. Model selection uses **average precision**, whose no-skill value equals the default rate (0.116). A useful model must score well above 0.116.

The imbalance is not “fixed” by copying minority rows in preprocessing. That would change the probability meaning and can leak across cross-validation folds. Class weights are applied inside each training algorithm, on each training fold only. After tuning, the winning model is not class-weighted; a separate decision cutoff (0.18) is chosen from out-of-fold scores instead. Both choices are recorded in Stages 6 and 7.

## Visual picture the numbers imply

The written numbers stay in the preprocessing report and `feature_selection.csv`. The same training-split patterns are drawn to `artifacts/figures/` when `preprocess.py` runs. The distributions to describe in the viva are:

- `Default` is a rare class: about one application in nine.
- Age, income, loan amount, credit score, months employed, interest rate, and DTI are spread across their full allowed ranges, with means near the centre.
- Default rate falls steadily with age and rises steadily with interest rate and with loan-to-income.
- Education, employment, marital status, purpose, and the three yes/no flags each shift the default rate by about 1–4 percentage points. Employment has the widest gap.
- Loan term is flat.

## Data leakage and how it is prevented

Leakage here would mean the model sees something that a credit officer would not know at decision time, or sees a statistic computed with the test rows.

| Risk | What was found | Prevention |
| --- | --- | --- |
| Identifier leakage | `LoanID` is unique per row. A tree could memorise ids | `LoanID` is dropped and is not an API input |
| Target leakage | No column is a post-default outcome other than `Default` | `Default` is the label only. No target encoding |
| Split leakage | Imputation medians, IQR fences, scaling, one-hot levels, and the correlation cutoff could be fit on all rows | Stratified 80/20 split happens first (`random_state=42`). The preprocessor is fit on training rows and applied to test rows |
| Row leakage in features | A ratio that used a column mean would mix other applicants into one row | `loan_to_income`, monthly payment, and `payment_to_income` use only the current row |
| Resampling leakage | Balancing the full table before the split would change the test default rate | No resampling in preprocessing |
| Future information | Interest rate is known when the officer prices the loan, so it is a valid input for this scenario | Kept. If the bank wanted a score *before* choosing the rate, interest rate would have to be removed. That is a product choice, not a hidden label |

The saved object `artifacts/preprocessor.joblib` is the training-fitted pipeline. The API loads that object, so a live application is transformed with training statistics only.

## Observations that drive later decisions

1. Keep the task as binary classification and do not optimise accuracy.
2. Do not drop rows for missing data or duplicates; there are none.
3. Do not delete statistical outliers on the raw fields; cap only at scoring time using training fences.
4. Build affordability features (loan-to-income and payment-to-income) because the raw amount and income correlations are moderate and the ratio quartile gap is large (8% vs 19%).
5. Drop `LoanTerm` from the model matrix after it has been used to compute the payment. Its correlation with default is about 0.001.
6. Encode education as an ordered ladder, and one-hot the unordered categories.
7. Scale numeric inputs because logistic regression will be one of the four models.
8. Keep age and marital status for the coursework model, and state the fair-lending limit in the report.
9. Use stratified splitting and stratified cross-validation so every fold keeps an 11.6% default rate.
