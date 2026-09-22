# Stage 6 — Model development

Training is `train_models.py`. It reads the matrices from `preprocess.py` and writes four fitted classifiers plus a comparison. This stage uses fixed, untuned settings. Search is Stage 7, so the comparison here is a fair baseline: same data, same folds, same metrics, different algorithms.

```bash
python train_models.py
```

Evidence: `artifacts/model_report.txt`, `artifacts/model_comparison_cv.csv`, `artifacts/model_comparison_test.csv`, and `artifacts/models/{logistic_regression,decision_tree,random_forest,hist_gradient_boosting}.joblib`.

## Why these four algorithms

The brief requires at least four algorithms suited to the problem. All four are classifiers. The matrix is a large flat table (204,277 training rows, 27 numeric columns after encoding), with both roughly linear effects (age, interest rate, loan-to-income) and possible thresholds (for example a jump in risk at young ages).

| Model | Starting settings | Why it belongs |
| --- | --- | --- |
| Logistic regression | `max_iter=1000`, L2 penalty (sklearn default), `class_weight="balanced"` | Interpretable linear baseline. L2 is appropriate because loan amount, income, and the ratios overlap |
| Decision tree | `max_depth=12`, `min_samples_leaf=50`, `class_weight="balanced"` | Axis-aligned rules a credit policy can be compared with. The depth and leaf size stop one tree from memorising 200,000 rows. These are limits, not a search |
| Random forest | 100 trees, `max_depth=16`, `min_samples_leaf=10`, `class_weight="balanced"` | Averages many trees grown on different samples, which reduces the instability of a single tree |
| Histogram gradient boosting | 200 iterations, `learning_rate=0.1`, up to 31 leaves, `class_weight="balanced"` | Later trees focus on applicants the earlier trees scored badly. Histogram binning keeps this practical at this row count |

`class_weight="balanced"` is inside the estimator, so each cross-validation fold reweights only its own training rows. The held-out test fold is not reweighted.

## Validation strategy

1. The 20% test set from preprocessing is not used to fit or to choose a model in this stage.
2. **Stratified 5-fold cross-validation** on the training set only (`shuffle=True`, `random_state=42`). Each fold keeps the 11.6% default rate.
3. `cross_validate` records both training and validation scores, so overfitting is visible as a train-versus-validation gap.
4. After cross-validation, each classifier is fit on all 204,277 training rows and exported.
5. The test set is scored **once** with that exported model.

Five folds on 204,277 rows means each validation fold is about 40,000 applications. That is large enough for average precision to be stable: the cross-validation standard deviation of average precision is about 0.003–0.004 for the stronger models.

## Metrics and why they were chosen

The default rate is 11.6%. Metric choice follows from that.

| Metric | Role |
| --- | --- |
| Average precision | Primary ranking metric. A no-skill model scores 0.116 (the default rate). It emphasises whether true defaults are placed near the top of the risk list |
| ROC-AUC | Threshold-free ranking. 0.5 is chance. It is less sensitive to the rare class than average precision, so both are reported |
| Precision | Of the applications flagged, how many truly default. A low value means many files sent for review will be false alarms |
| Recall | Of the true defaults, how many were flagged. The costly miss is a default that was approved with no review |
| F1 | Balance of precision and recall at a fixed cutoff |
| Balanced accuracy | Average of recall on both classes, so the majority class cannot hide a weak default detector |
| Accuracy | Reported only to show the trap. The majority-class accuracy is 88.4% |

Precision, recall, F1, balanced accuracy, and accuracy at this stage use the default probability cutoff of **0.5**. Balanced class weights push recall up and precision down at that cutoff. The cutoff can move later without changing the ranking model. Stage 7 does that for the selected model.

## Cross-validation results (mean of 5 training folds)

Sorted by average precision. Source: `artifacts/model_comparison_cv.csv`.

| Model | CV average precision | CV ROC-AUC | CV recall | CV precision | CV F1 | CV accuracy | Train ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Histogram gradient boosting | 0.316 | 0.751 | 0.669 | 0.226 | 0.338 | 0.695 | 0.781 |
| Random forest | 0.311 | 0.746 | 0.532 | 0.267 | 0.355 | 0.776 | 0.953 |
| Logistic regression | 0.310 | 0.752 | 0.688 | 0.222 | 0.336 | 0.684 | 0.752 |
| Decision tree | 0.256 | 0.692 | 0.613 | 0.198 | 0.299 | 0.667 | 0.807 |

Every model beats the no-skill average precision of 0.116 by a wide margin. The three stronger models are close to each other (0.310–0.316). The single tree is clearly weaker (0.256).

## Held-out test results (fit on all training rows, scored once, cutoff 0.5)

Source: `artifacts/model_comparison_test.csv`.

| Model | Average precision | ROC-AUC | Recall | Precision | F1 | Balanced accuracy | Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Histogram gradient boosting | 0.328 | 0.758 | 0.683 | 0.229 | 0.343 | 0.691 | 0.697 |
| Random forest | 0.320 | 0.753 | 0.553 | 0.270 | 0.363 | 0.678 | 0.774 |
| Logistic regression | 0.319 | 0.758 | 0.700 | 0.225 | 0.340 | 0.691 | 0.685 |
| Decision tree | 0.266 | 0.702 | 0.640 | 0.198 | 0.302 | 0.650 | 0.657 |

Test ranking matches the cross-validation ranking. Nothing was chosen because it looked good only on the test set.

Confusion counts on the test set at cutoff 0.5 (5,931 actual defaults, 45,139 non-defaults):

| Model | Defaults caught | Defaults missed | Non-defaults flagged for review |
| --- | ---: | ---: | ---: |
| Logistic regression | 4,153 | 1,778 | 14,322 |
| Histogram gradient boosting | 4,050 | 1,881 | 13,606 |
| Decision tree | 3,797 | 2,134 | 15,401 |
| Random forest | 3,282 | 2,649 | 8,873 |

At 0.5, logistic regression catches the most defaults and also sends the most good applications for review. The forest sends fewer files for review and misses more defaults. That tradeoff is why a later cutoff, chosen on training folds, matters more than accuracy.

## What the models rely on

Top logistic-regression coefficients (sign is the direction of default log-odds):

| Feature | Coefficient |
| --- | ---: |
| Age | −0.587 |
| Interest rate | +0.460 |
| Loan-to-income | +0.350 |
| Months employed | −0.339 |
| Has co-signer | −0.262 |
| Has dependents | −0.256 |
| Full-time employment | −0.240 |
| Payment-to-income | +0.218 |

Tree importances (decision tree, then forest) put the same three ideas first: age, interest rate, and loan-to-income, followed by months employed. Histogram gradient boosting in this scikit-learn version does not expose `feature_importances_`. The linear coefficients and the tree importances agree, which is evidence the signal is real rather than an artefact of one algorithm.

## Why some models do better

Train ROC-AUC minus validation ROC-AUC:

| Model | Train ROC-AUC | Validation ROC-AUC | Gap |
| --- | ---: | ---: | ---: |
| Logistic regression | 0.752 | 0.752 | 0.000 |
| Histogram gradient boosting | 0.781 | 0.751 | 0.029 |
| Decision tree | 0.807 | 0.692 | 0.115 |
| Random forest | 0.953 | 0.746 | 0.207 |

- Logistic regression draws one linear boundary in the scaled features. Age, rate, and loan-to-income are roughly linear, so the linear model stays close to the best score and does not overfit.
- One tree can cut at thresholds, but a small change in the sample changes the split. The 0.115 gap and the 0.256 average precision show that instability.
- The forest reduces that instability and lifts average precision to 0.311, but a 0.207 ROC-AUC gap shows the individual trees still memorise training rows. Validation performance, not training performance, is what counts.
- Gradient boosting spends later trees on applicants who were scored badly. It leads on average precision (0.316 cross-validation, 0.328 test) with a small 0.029 gap, so the extra fit is mostly a real gain.

No model is selected as “final” in this stage. Stage 7 retunes all four and only then picks one.
