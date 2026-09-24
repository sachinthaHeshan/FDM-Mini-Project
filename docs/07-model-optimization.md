# Stage 7 — Model optimization and final model selection

Optimization is `tune_models.py`. It compares each untuned model with a randomised search on the **same** stratified folds, then exports one winner. The test set is scored only after that choice.

```bash
python tune_models.py
```

Evidence: `artifacts/tuning_report.txt`, `artifacts/tuning_comparison.csv`, `artifacts/models/selected_model.joblib`, `artifacts/models/selected_model.json`.

## Search strategy

A full grid on four models and ~200,000 rows would refit hundreds of classifiers. Randomised search tries a sample of the grid and scores each candidate by **average precision**.

| Choice | Setting | Reason |
| --- | --- | --- |
| Search | `RandomizedSearchCV` | Covers the important knobs without a full grid |
| Selection metric | Average precision (`refit="average_precision"`) | Matches the 11.6% default rate. ROC-AUC is recorded but does not choose the winner |
| Attempts | Logistic regression 12, decision tree 12, gradient boosting 12, random forest 8 | Forest grids are larger and slower; 8 draws still cover depth, leaf size, feature fraction, tree count, and class weight. The script never asks for more draws than the grid contains |
| Validation | Stratified 3-fold on the training set, shuffle, `random_state=42` | Same seed and the same fold object for the baseline and the search, so the comparison is paired. Each fit still sees about 136,000 rows. Five folds were used in Stage 6; three folds keep the search tractable |
| Test set | Not used to choose the model or the cutoff | Touched once, after selection |

Search spaces:

| Model | Parameters tried |
| --- | --- |
| Logistic regression | `C` in {0.01, 0.1, 1, 10, 100}; `class_weight` balanced or none |
| Decision tree | `max_depth` in {6, 8, 12, 16, unlimited}; `min_samples_leaf` in {20, 50, 100, 200}; `max_features` all or sqrt; class weight balanced or none |
| Random forest | 100 or 200 trees; depth 12, 16, or 24; leaf 5, 10, or 20; `max_features` sqrt or 0.5; class weight balanced or none |
| Histogram gradient boosting | learning rate 0.05, 0.1, or 0.2; iterations 100, 200, or 300; leaves 15, 31, or 63; `min_samples_leaf` 20 or 50; L2 penalty 0 or 1; class weight balanced or none |

A tuned setting is kept only when its cross-validated average precision is at least as high as the untuned model on those same three folds. Otherwise the untuned model would be kept. On this run every tuned setting won.

## Tuned versus untuned (same 3 folds)

Source: `artifacts/tuning_comparison.csv`.

| Model | Baseline average precision | Tuned average precision | Kept |
| --- | ---: | ---: | --- |
| Histogram gradient boosting | 0.3158 | 0.3175 | Tuned |
| Random forest | 0.3085 | 0.3103 | Tuned |
| Logistic regression | 0.3093 | 0.3096 | Tuned |
| Decision tree | 0.2534 | 0.2749 | Tuned |

The tree gains the most from tuning (+0.021) and remains last. The three stronger models move by less than 0.002. Tuning confirms the Stage 6 ordering; it does not rescue the single tree.

Settings kept:

| Model | Parameters |
| --- | --- |
| Histogram gradient boosting | `learning_rate=0.1`, `max_iter=300`, `max_leaf_nodes=15`, `min_samples_leaf=20`, `l2_regularization=0`, `class_weight=None` |
| Random forest | 100 trees, `max_depth=12`, `min_samples_leaf=20`, `max_features="sqrt"`, `class_weight="balanced"` |
| Logistic regression | `C=0.01`, `class_weight=None` |
| Decision tree | `max_depth=8`, `min_samples_leaf=100`, all features, `class_weight="balanced"` |

Two of the stronger models drop class weighting when average precision is the goal. Balanced weights help recall at a 0.5 cutoff; they are not required for ranking. The decision cutoff is chosen afterwards so recall is not abandoned.

## Did feature engineering still help?

The leading model was scored again with and without `loan_to_income`, `estimated_monthly_payment`, and `payment_to_income`.

| Feature set | Cross-validated average precision |
| --- | ---: |
| With the three engineered features | 0.3175 |
| Without them | 0.3177 |

The rule in `tune_models.py` drops the engineered columns only if removing them improves average precision by at least 0.005. The change is about 0.0002, so the features stay. They describe affordability in the same units a credit officer uses, and they do not hurt the score. The final model matrix is the 27-column matrix from preprocessing.

## Final model

**Tuned histogram gradient boosting.**

It has the highest cross-validated average precision among the untuned and tuned versions of all four algorithms (0.3175). The forest is second (0.3103). Logistic regression is third (0.3096) and is the right baseline to quote: boosting beats a well-tuned linear model by a small, consistent margin, not by a different order of magnitude.

Why this model, in one justification:

- Best average precision under the selection rule that never saw the test set.
- The Stage 6 train/validation ROC-AUC gap was small (0.029), so the family was not winning by memorising.
- Tuning raised the score further and preferred a slightly shallower tree (15 leaves) with more rounds (300), which is a regularised boosting setup.
- The linear model remains close, which supports the story that age, rate, and loan-to-income dominate. Boosting is selected because it uses the remaining interactions well enough to lead the table, and because the deployment only needs one model.

## Decision cutoff

The winning booster is not class-weighted. At a cutoff of 0.50 it would rarely flag a default, because predicted probabilities sit near the 11.6% base rate. The cutoff is chosen by maximising F1 on **out-of-fold** training probabilities (`cross_val_predict` on the same three folds), scanning thresholds from 0.05 to 0.80 in steps of 0.01.

**Selected cutoff: 0.18.**

The test labels are not used to pick 0.18.

## Test set, scored once after selection

Source: `artifacts/models/selected_model.json` and `artifacts/tuning_report.txt`. Predictions use the 0.18 cutoff.

| Metric | Value |
| --- | ---: |
| Average precision | 0.3298 |
| ROC-AUC | 0.7582 |
| Recall | 0.4925 |
| Precision | 0.3001 |
| F1 | 0.3730 |

Compared with the untuned booster at cutoff 0.5 (Stage 6 test recall 0.683, precision 0.229), the tuned operating point is more precise and less aggressive. About 30% of flagged applications are true defaults (versus an 11.6% base rate), and about half of true defaults are flagged. That is the review policy encoded in the API: flag when predicted probability is at least 18%.

The final estimator is refit on all training rows and saved as `artifacts/models/selected_model.joblib`. The API loads that file only.
