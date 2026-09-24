# Stage 8 — Progress Evaluation 2 (individual viva)

Weight: 30%. Individual. Dates in the brief: 30 September, 3 and 4 October 2026 (other campuses as announced).

This viva is about modelling and optimization. Know Stages 6 and 7 well enough to defend the numbers without opening the CSV. Write your own contribution at the bottom before the lab.

## Algorithm selection

Four classifiers, all appropriate to a yes/no default label on a large mixed table:

1. Logistic regression — linear baseline, L2 because amount, income, and ratios overlap.
2. Decision tree — single set of rules, depth and leaf size limited so it cannot memorise 200,000 rows.
3. Random forest — averaged trees, to fix the instability of one tree.
4. Histogram gradient boosting — later trees correct earlier mistakes; histogram splits keep it fast at this size.

All four started with `class_weight="balanced"` so the rare class affected the fit inside each training fold. Say why a clustering algorithm or a regressor was not used: the target is two classes, not a continuous loss amount and not an unknown grouping.

## Validation

- Preprocessing already held out 20%, stratified, seed 42. That test set is not used to pick features, hyperparameters, or the winner.
- Stage 6: stratified 5-fold cross-validation on the training set, then one test score after export.
- Stage 7: stratified 3-fold cross-validation, same seed, shared by the baseline and the search, because a 12-draw search with 5 folds on 200,000 rows was not worth the extra fits. Each Stage 7 fit still trains on about 136,000 rows.
- The cutoff 0.18 is chosen from out-of-fold training scores, not from the test set.

## Metrics

Lead with average precision. No-skill value is the default rate, 0.116. All four models land between about 0.25 and 0.33, so they rank defaults better than chance. ROC-AUC is the second ranking metric (about 0.75 for the best models; 0.5 would be chance).

Accuracy is a trap: always saying “no default” scores 88.4% and recall 0. Precision, recall, and F1 depend on a cutoff. At 0.5 with balanced weights, recall is high and precision is about 0.23. After tuning, the booster is not reweighted and the cutoff moves to 0.18.

## Comparison you should remember

Stage 6 cross-validation average precision (5-fold): boosting 0.316, forest 0.311, logistic 0.310, tree 0.256.

Stage 6 test average precision, scored once: boosting 0.328, forest 0.320, logistic 0.319, tree 0.266. Same order.

Overfit gap (train ROC-AUC minus validation ROC-AUC): logistic 0.000, boosting 0.029, tree 0.115, forest 0.207. The forest looks excellent on the training rows (ROC-AUC 0.953) and ordinary on validation (0.746). Quote that if you are asked how you noticed overfitting.

Drivers that repeat across logistic regression and the trees: younger age, higher interest rate, and higher loan-to-income raise default risk. Longer employment and a co-signer lower it.

## Tuning

Randomised search, average precision, at most 12 settings (8 for the forest). A tuned model replaces the baseline only if it wins on the same folds. All four tuned versions won. The tree improved the most and stayed last.

Kept booster settings: learning rate 0.1, 300 iterations, 15 leaves, minimum 20 samples per leaf, no L2 penalty, no class weight.

Feature check on that booster: average precision 0.3175 with the payment ratios and 0.3177 without them. They stay, because the gain from deleting them is far below 0.005, and they are the affordability measures the officer already understands.

## Final model and justification

Winner: **tuned histogram gradient boosting**, saved at `artifacts/models/selected_model.joblib`.

- Highest cross-validated average precision: **0.3175**.
- Test scores after selection, cutoff 0.18: average precision **0.3298**, ROC-AUC **0.7582**, recall **0.4925**, precision **0.3001**, F1 **0.3730**.
- Chosen because it leads the comparison that was not allowed to see the test set, the overfit gap in Stage 6 was small, and tuning did not change the ranking.
- Logistic regression is the baseline to mention: boosting is better, and the margin is small, which matches a problem whose main effects are nearly linear.

Why the cutoff is 0.18: without class weights, a 0.50 cutoff almost never fires. 0.18 maximises out-of-fold F1. About half of defaults are flagged, and about 30% of flags are true defaults, against an 11.6% base rate.

## Your own contribution

Fill this in with work you can defend:

- Which algorithm you implemented or checked.
- Which metric you argued for, and the base-rate number you used.
- One tuning choice (search size, class weight, cutoff, or the feature check) and the table you used as evidence.

You will also be asked about decisions you did not code. Be able to walk from the 11.6% default rate to “we did not pick the model with the highest accuracy.”

## If the question is abrupt

| Question | Answer |
| --- | --- |
| How many models? | Four: logistic regression, decision tree, random forest, histogram gradient boosting |
| What metric selected the winner? | Cross-validated average precision |
| What did the test set do? | It was scored once after the winner and the cutoff were fixed |
| What won? | Tuned histogram gradient boosting, cutoff 0.18 |
| Did tuning change the ranking? | No. It helped the tree most and left it last |
| Why not the forest, given training ROC-AUC 0.95? | Validation ROC-AUC is 0.75. The gap is memorisation |
| Where is the proof? | `artifacts/model_report.txt` and `artifacts/tuning_report.txt` |
