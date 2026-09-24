# Stage 11 — Technical report

IT3051 Fundamentals of Data Mining  
Mini project 2026  
Loan-default review for a credit officer

Group members and index numbers: *[add names here]*  
Dataset validation: *[add instructor, date, and outcome when recorded]*

This report is the Stage 11 deliverable. Supporting write-ups for each stage are in `docs/01` through `docs/10`. Numbers below are taken from `dataset/Loan_default.csv` and from the artifacts produced by `preprocess.py`, `train_models.py`, and `tune_models.py`.

## 1. Introduction and problem definition

Lenders lose money when an approved applicant does not repay. Checking every application with the same manual rule is slow, and a single cutoff on credit score misses other signals that show up in the data: age, time in the job, interest rate, and how large the loan is relative to income.

The project builds a prediction system a credit officer can use **before** a loan is approved. The officer enters the application. The system returns whether the applicant is likely to default, and the probability of default, so the file can be approved, priced, or sent for a closer review. The score supports the decision. It does not decline anyone by itself.

## 2. Scenario and stakeholder requirements

**Scenario.** A lending team wants a consistent first look at default risk on each new application.

**Primary user.** Credit officer.

**Other stakeholders.** Operations (fewer unreviewed defaults), risk and pricing (a ranked probability), and the applicant (a wrong flag can delay a fair loan, which is why a person still decides).

**Requirements.**

- Inputs are facts known on the application form.
- Invalid and missing values are rejected in plain language.
- The output is a default / no-default label, a probability, and a sentence the officer can read aloud.
- The live application is prepared with the same pipeline that trained the model.

**Task type.** Binary classification. The target `Default` is 1 if the loan defaulted and 0 otherwise. Regression would fit a loss amount or a time-to-default, which this problem does not ask for and the dataset does not contain.

## 3. Dataset identification, source, citation, and validation

| Item | Detail |
| --- | --- |
| Name | Loan Default Prediction Dataset |
| File | `dataset/Loan_default.csv` |
| Host | Kaggle, uploaded by Nikhil (`nikhil1e9`) |
| URL | https://www.kaggle.com/datasets/nikhil1e9/loan-default |
| Origin stated on Kaggle | Coursera’s Loan Default Prediction Challenge |
| Licence | CC0 Public Domain |
| Size | 255,347 rows, 18 columns |
| Target | `Default` (29,653 ones, 225,694 zeros, rate 11.61%) |

Citation: Nikhil. (n.d.). *Loan Default Prediction Dataset* [Data set]. Kaggle. https://www.kaggle.com/datasets/nikhil1e9/loan-default

The table matches the scenario: one row is one application, predictors are knowable before approval, and the label is the event the officer needs to anticipate. It is large enough for a hold-out and for four algorithms. It is uneven (only 11.6% default), so accuracy would be a misleading score. The ranges look regular and there are no missing values, which suggests a teaching extract rather than a raw core-banking dump. There is no loss amount. `Age` and `MaritalStatus` are predictive and sensitive; they are kept for this academic model and flagged for a fairness review before any live lending use.

The full proposal, including quality notes and ethics, is `docs/02-dataset-proposal.md`. Instructor sign-off is not stored in the repository and should be attached here when available.

## 4. Data understanding and EDA

There are no missing cells, no duplicate rows, and no duplicate loan identifiers. Numeric fields already lie in realistic ranges: age 18–69, income 15,000–149,999, loan amount 5,000–249,999, credit score 300–849, interest rate 2–25, DTI 0.10–0.90, loan term in {12, 24, 36, 48, 60}. `MonthsEmployed` is 0 on 2,122 rows; those rows are new employment and were kept.

Relationships with default, measured on the full file:

- Age correlation −0.17. Default rate falls from 20.8% at ages 18–25 to 5.5% at ages 56–69.
- Interest-rate correlation +0.13. Default rate rises from 6.6% (rates 2–8%) to 17.8% (rates 20–25%).
- Income −0.10, months employed −0.10, loan amount +0.09.
- Loan amount / income, by quartile: default rate 8.0%, 9.2%, 10.6%, 18.7%.
- Employment: unemployed 13.6%, full-time 9.5%. Education: high school 12.9%, PhD 10.6%.
- Co-signer, dependents, mortgage, marital status, and loan purpose each shift the rate by about 1–2.5 percentage points.
- Loan term correlation with default is about 0.0005. Every term defaults at about 11.6%.

A classifier that always predicts “no default” is 88.4% accurate and catches none of the 29,653 defaults. Later model selection therefore uses average precision, whose no-skill value is 0.116.

IQR fences on the training split do not exclude any raw field. The engineered payment ratios have a long right tail. Those tails are high loan-to-income cases, not data-entry errors.

`LoanID` is unique per row and would let a tree memorise applications. It is excluded. No other column is an after-the-fact collection outcome. Detail: `docs/03-exploratory-data-analysis.md`.

## 5. Data cleaning and preprocessing

Implemented in `preprocess.py`. The stratified 80/20 split (`random_state=42`) is done **before** any fitted statistic.

| Step | Decision | Evidence |
| --- | --- | --- |
| Invalid rows | Domain rules for age, money, score, rate, DTI, term, categories, and the label | 0 rows removed; 255,347 remain |
| Duplicates | Drop exact duplicates and duplicate ids | 0 removed |
| Missing values | Median for numbers, mode for categories, fitted on the training split | 0 blanks in the file; imputers are saved so the pipeline is defined |
| Outliers | Clip to training 1.5-IQR fences; do not delete rows | Raw fields sit inside the fences; ratio tails are capped (46,267 training cells) because they are the risk signal |
| Scaling | `StandardScaler` on numeric columns and on education | Required by logistic regression; trees share the same matrix |
| Split | 204,277 train, 51,070 test, both at a 11.61% default rate | Stratified so the test set matches the base rate |

Class imbalance is not repaired by copying minority rows. That would change probabilities and can leak across folds. Balancing is left to `class_weight` inside each training fold, and later to the decision cutoff.

## 6. Feature engineering and feature selection

Three features are computed from the current row only:

- `loan_to_income = LoanAmount / Income`
- `estimated_monthly_payment` from amount, annual rate, and term
- `payment_to_income` = that instalment divided by monthly income

Years-employed was not added. After scaling it duplicates `MonthsEmployed`.

Encoding: Yes/No becomes 1/0; education is the ordered ladder High School < Bachelor's < Master's < PhD; employment, marital status, and purpose are one-hot encoded. Unknown categories at scoring time become an all-zero dummy vector. The API rejects unknown labels before that happens.

Selection uses the training split only. A numeric column is kept when |Pearson r| with `Default` is at least 0.01. A category is kept when the default-rate gap across its levels is at least 1 percentage point. Pairs of numeric columns with |r| above 0.98 would keep only the stronger one; no pair qualified. `LoanTerm` is dropped from the matrix (r = 0.0012) after it has been used in the payment formula. The officer still enters the term.

The model matrix has 27 columns. The keep/drop table is `artifacts/feature_selection.csv`.

Leakage controls: identifier removed; split before fit; no target encoding; row-local ratios; `artifacts/preprocessor.joblib` is the training fit and is what the API loads.

## 7. Algorithms and rationale

`train_models.py` fits four classifiers. Starting settings are deliberate limits, not a search.

| Algorithm | Why it was included | Starting constraint |
| --- | --- | --- |
| Logistic regression | Linear baseline. Main effects of age, rate, and loan-to-income are roughly linear. Default L2 penalty suits overlapping amount and income columns | `class_weight="balanced"`, `max_iter=1000` |
| Decision tree | Readable thresholds | Depth 12, at least 50 rows per leaf, balanced class weight |
| Random forest | Averages unstable trees | 100 trees, depth 16, leaf 10, balanced class weight |
| Histogram gradient boosting | Later trees focus on applications scored badly so far, and histogram splits scale to 200,000 rows | 200 iterations, learning rate 0.1, 31 leaves, balanced class weight |

## 8. Model evaluation and comparison

**Validation.** Stratified 5-fold cross-validation on the training set only. Each model is then refit on all training rows and the test set is scored once. Precision, recall, and F1 in this section use a probability cutoff of 0.5.

Cross-validation mean average precision: boosting **0.316**, forest **0.311**, logistic regression **0.310**, tree **0.256**. Test average precision, scored once: **0.328**, **0.320**, **0.319**, **0.266**. The order matches, so the leader is not a test-set accident.

Test ROC-AUC for boosting and logistic regression is about 0.758. Test recall at cutoff 0.5 is highest for logistic regression (0.700) with precision 0.225. The forest is more conservative (recall 0.553, precision 0.270).

Train ROC-AUC minus validation ROC-AUC: logistic regression 0.000, boosting 0.029, tree 0.115, forest 0.207. The forest’s training ROC-AUC of 0.953 does not survive validation (0.746).

Logistic coefficients and tree importances agree on the drivers: higher default risk for younger applicants, higher interest rates, and higher loan-to-income; lower risk for longer employment and a co-signer. The largest logistic coefficients are age −0.587, interest rate +0.460, and loan-to-income +0.350.

The single tree is weaker because one set of splits moves when the sample moves. The linear model is strong because the dominant effects are close to linear. Boosting leads because it picks up what the linear model leaves, without the forest’s memorisation gap.

## 9. Hyperparameter tuning and optimization

`tune_models.py` runs randomised search scored by average precision. A full grid was rejected because it would refit hundreds of models on about 200,000 rows. Each model tries at most 12 random settings (8 for the forest). Validation is stratified 3-fold on the training set, shared with the untuned baseline. The test set is not used to choose parameters.

| Model | Baseline AP | Tuned AP | Kept settings |
| --- | ---: | ---: | --- |
| Histogram gradient boosting | 0.3158 | 0.3175 | learning rate 0.1, 300 iterations, 15 leaves, min leaf 20, no L2, no class weight |
| Random forest | 0.3085 | 0.3103 | 100 trees, depth 12, leaf 20, sqrt features, balanced |
| Logistic regression | 0.3093 | 0.3096 | `C=0.01`, no class weight |
| Decision tree | 0.2534 | 0.2749 | depth 8, leaf 100, balanced |

Tuning lifts every model. It does not change the ranking. The tree gains the most and remains last.

The winning booster was refit with and without the three engineered features (average precision 0.3175 vs 0.3177). The features stay, because removing them does not improve the score by the required 0.005, and they are the affordability measures the decision uses.

The booster is not class-weighted, so a 0.50 cutoff would rarely flag a default. The cutoff that maximises F1 on out-of-fold training probabilities is **0.18**. Test labels were not used to pick it.

## 10. Final model selection and justification

**Selected model:** tuned histogram gradient boosting.  
**Artifact:** `artifacts/models/selected_model.joblib` with metadata in `selected_model.json`.

It has the highest cross-validated average precision of any untuned or tuned candidate (0.3175). Stage 6 showed a small overfit gap for this family. The linear model is close behind, which is consistent with strong main effects; boosting is preferred because it still leads the comparison that was not allowed to see the test set.

Held-out test result after selection, at cutoff 0.18:

| Average precision | ROC-AUC | Recall | Precision | F1 |
| ---: | ---: | ---: | ---: | ---: |
| 0.3298 | 0.7582 | 0.4925 | 0.3001 | 0.3730 |

About half of true defaults are flagged. About 30% of flagged files are true defaults, against an 11.6% base rate. That is a review queue, not an automatic decline.

## 11. System architecture and implementation

```text
dataset/Loan_default.csv
        │
        ▼
  preprocess.py  ──►  artifacts/preprocessor.joblib
        │              artifacts/X_*.csv, y_*.csv
        ▼
  train_models.py ──► four comparison models and model_report.txt
        │
        ▼
  tune_models.py  ──► artifacts/models/selected_model.joblib
        │
        ├──────────────► backend/main.py   (FastAPI, POST /predict)
        │
        └──────────────► frontend/app.py   (Streamlit officer screen)
```

Training and serving are separate. The screen never loads scikit-learn. It posts the 16 raw fields. The API loads the preprocessor and the selected model once at startup.

Stack: Python 3.12, pandas, numpy, scikit-learn, joblib, FastAPI, uvicorn, Streamlit (`requirements.txt`).

## 12. Backend and frontend

**Backend** (`backend/main.py`). Startup fails if the preprocessor or the selected model is missing. `POST /predict` validates age, money, score, rate, term, DTI, and every category against the training labels, transforms the row with `preprocessor.transform`, keeps the 27 selected columns, and applies cutoff 0.18 to `predict_proba`. The JSON response contains the model name, `Default` or `No default`, a boolean, the probability, and a sentence. Missing or invalid input returns HTTP 422 with a field message (“This field is required.”, “Age must be between 18 and 100.”, or the allowed category list). Extra JSON keys are rejected.

**Frontend** (`frontend/app.py`). One form in two groups, Applicant and Loan. Categories are dropdowns and yes/no controls. Help text states the limits. The screen checks the same limits before calling the API, shows a list of problems when they fail, and if the API is down tells the officer how to start it. A default prediction is headed “Send this application for a closer review.” A non-default is headed “Lower default risk.” The probability is shown as a percent. A caption says the result supports the decision and does not replace it.

## 13. System testing and results

Reproducible checks already encoded in the project:

| Check | Result |
| --- | --- |
| Preprocessor transform is stable on the test rows | `preprocess.py` raises if a second transform disagrees |
| Prepared matrices contain no missing values | The script raises otherwise |
| Category lists in the API match training | Runtime error if `CHOICES` diverges from `ALLOWED_CATEGORIES` |
| Model comparison order is stable from CV to test | Boosting, forest, logistic, tree on both |
| Final test score is computed after selection | Average precision 0.3298, ROC-AUC 0.7582 |
| Invalid applications | API returns 422 and does not score; the screen lists the fields |

Demo path: start uvicorn, start Streamlit, submit the prefilled application, then set age to 10 and income to 0 and confirm the form is blocked. A direct `POST /predict` example is in `README.md`.

What the test numbers mean for the business: the model ranks default risk well above chance (average precision 0.33 versus 0.12). At the chosen review threshold it surfaces about one in two eventual defaults, and roughly two out of three files in that queue will not default. The queue is a prioritisation tool. It is too imprecise to be an automatic rejection rule, which matches the wording on the screen.

## 14. Limitations and possible improvements

- The file has no missing data and very regular ranges. Performance on a live book, with missing fields and rarer categories, will be lower until the pipeline is refit.
- The model estimates whether a default happens, not the amount lost.
- Age and marital status are in the model. A production system needs a fairness assessment and possibly a constrained model that omits them.
- Interest rate is known in this scenario because the officer is pricing the loan. A score computed before the rate is chosen must drop that feature.
- Engineered ratios barely change average precision on this table. They are kept for interpretation, not because they transform accuracy.
- The probability cutoff maximises F1. A lender with a known cost per missed default and per unnecessary review should set the cutoff from those costs instead.
- No probability calibration plot is stored. If the percent shown to the officer must match observed default frequencies in each band, calibration should be added and checked on the training folds.
- The randomised search used 3 folds and at most 12 draws. A longer search is unlikely to change the winner, given how close the top three already are, but it was not exhaustively proven.

## 15. Individual and group contributions

*[Each member: two or three sentences naming the files and decisions they owned. Example categories: dataset proposal, EDA measurements, `preprocess.py`, `train_models.py`, `tune_models.py`, API, Streamlit screen, report, slides. Every member should be able to explain Stages 1–10 in the individual vivas even for work they did not type.]*

## References

Nikhil. (n.d.). *Loan Default Prediction Dataset* [Data set]. Kaggle. https://www.kaggle.com/datasets/nikhil1e9/loan-default

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*, 2825–2830.

Project evidence files: `artifacts/preprocessing_report.txt`, `artifacts/feature_selection.csv`, `artifacts/model_report.txt`, `artifacts/model_comparison_cv.csv`, `artifacts/model_comparison_test.csv`, `artifacts/tuning_report.txt`, `artifacts/tuning_comparison.csv`, `artifacts/models/selected_model.json`.
