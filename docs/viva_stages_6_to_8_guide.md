# Stages 6–8 Viva Guide: Model Development, Optimization, and Final Selection

## Project summary every member must know

The project predicts whether a loan applicant will default before a loan is approved. It is a supervised binary-classification problem:

- `Default = 1`: the applicant defaulted.
- `Default = 0`: the applicant did not default.
- Total cleaned rows: 255,347.
- Training rows: 204,277 (80%).
- Held-out test rows: 51,070 (20%).
- Default rate: approximately 11.6%.
- Final model: tuned Histogram Gradient Boosting.
- Final cross-validated average precision: 0.3177.
- Final test average precision: 0.3307.
- Final test ROC-AUC: 0.7588.
- Final decision threshold: 0.62.

The work is divided into four roles. Members should lead their own section but must understand the complete workflow because the examiner may ask questions in any order.

---

# Member 1 — Problem Framing and Algorithm Selection

## Main responsibility

Explain the machine-learning problem, class imbalance, the four algorithms, why each was selected, and how each algorithm works.

## Suggested opening

> Our objective was to estimate the probability that a loan applicant would default. This is a supervised binary-classification problem. We did not rely on one algorithm. We implemented Logistic Regression, Decision Tree, Random Forest, and Histogram Gradient Boosting. These models provide a useful comparison between a simple linear model, a single rule-based nonlinear model, a bagging ensemble, and a boosting ensemble.

## Why this is a classification problem

The target contains two discrete classes—default and no default—rather than a continuous number. Therefore, regression metrics such as mean squared error are not the main evaluation measures.

The model supports a credit officer; it should not automatically make a final lending decision. A production system would also require fairness, legal, explainability, and probability-calibration reviews.

## Class imbalance

There are 23,722 defaults and 180,555 non-defaults in the training set:

- Default: 11.6%.
- No default: 88.4%.

This is important because a useless model that predicts “no default” for everyone would still achieve approximately 88.4% accuracy, but its recall for defaults would be zero.

Therefore, accuracy alone is misleading. The project emphasizes average precision, ROC-AUC, recall, precision, F1, and balanced accuracy.

## Algorithm 1: Logistic Regression

### How it works

Logistic Regression learns a weighted linear combination of the input features and converts it into a probability using the logistic function.

### Why it was suitable

- It provides a strong and fast baseline.
- It works well when effects are approximately linear.
- Its coefficients can show the direction and relative strength of relationships.
- L2 regularization helps when related features overlap, such as income, loan amount, loan-to-income, and payment-to-income.

### Main limitation

It creates a linear decision boundary. It may miss nonlinear relationships and interactions unless those relationships are explicitly engineered.

### Project observation

It generalized very well:

- Training ROC-AUC: 0.7525.
- Cross-validation ROC-AUC: 0.7522.
- Gap: approximately 0.0003.

This near-zero gap shows little overfitting. However, its average precision was below Histogram Gradient Boosting.

## Algorithm 2: Decision Tree

### How it works

A Decision Tree repeatedly splits the data using rules such as “InterestRate is greater than a value” or “Age is below a value.” The final leaf produces a class or probability.

### Why it was suitable

- It captures nonlinear relationships.
- It can discover threshold-based risk patterns.
- Its rule structure is easier to explain than many ensemble models.
- It gives a useful single-tree baseline before testing ensembles.

### Main limitation

A single tree is unstable and can overfit. A small data change can produce different splits.

### Project observation

- Training ROC-AUC: 0.8067.
- Cross-validation ROC-AUC: 0.6918.
- Gap: approximately 0.115.

The large gap indicates overfitting. It also had the lowest baseline average precision, 0.2555.

## Algorithm 3: Random Forest

### How it works

Random Forest trains many decision trees using different bootstrap samples and random subsets of features. It averages their predictions.

### Why it was suitable

- It captures nonlinear patterns and interactions.
- Averaging reduces the instability of a single tree.
- It is effective for tabular data.
- It provides feature importance values.

### Main limitation

It is less interpretable than one tree, can be computationally expensive, and may still overfit.

### Project observation

- Training ROC-AUC: 0.9528.
- Cross-validation ROC-AUC: 0.7457.
- Gap: approximately 0.207.

This was the largest train-validation gap. The forest learned the training data very strongly but did not transfer all that improvement to unseen folds.

## Algorithm 4: Histogram Gradient Boosting

### How it works

Boosting builds trees sequentially. Each new tree focuses on correcting errors made by the previous trees. The histogram implementation groups continuous values into bins, making training efficient on a large tabular dataset.

### Why it was suitable

- It captures nonlinear relationships and interactions.
- Boosting often performs strongly on structured tabular data.
- Histogram binning improves speed and memory efficiency for more than 200,000 training rows.
- Parameters such as learning rate, number of iterations, leaf count, minimum leaf size, and L2 regularization control model complexity.

### Main limitation

It is less directly interpretable than Logistic Regression or a single tree and requires careful tuning to avoid overfitting.

### Project observation

It produced the highest baseline cross-validation average precision:

- Average precision: 0.3162.
- ROC-AUC: 0.7512.
- Train-validation ROC-AUC gap: approximately 0.029.

This was a better balance between predictive performance and generalization than the other tree-based models.

## Important observations Member 1 should explain

1. Logistic Regression was competitive because several risk relationships were approximately monotonic or linear after preprocessing.
2. The Decision Tree performed worst because one tree was unstable and overfit.
3. Random Forest reduced single-tree instability but still showed strong overfitting.
4. Histogram Gradient Boosting performed best because sequential trees captured nonlinear relationships while regularization controlled complexity.
5. More complex models are not automatically better. Validation performance, not training performance, determines whether complexity is useful.

## Likely viva questions for Member 1

### Why did you use four models?

> The assessment required multiple suitable algorithms, and the four models also represent different modelling assumptions. Logistic Regression is linear, Decision Tree is a single nonlinear model, Random Forest uses bagging, and Histogram Gradient Boosting uses boosting. This allowed us to compare simplicity, interpretability, nonlinearity, and ensemble learning systematically.

### Why did you not use linear regression?

> Linear regression is designed for a continuous target and can produce values outside zero and one. Our target is binary, so a classification algorithm such as Logistic Regression is appropriate.

### Why did you not choose the model with the highest training score?

> A high training score can be caused by memorization. Random Forest had a training ROC-AUC of about 0.953 but validation ROC-AUC of about 0.746. We selected models using cross-validation performance because it better estimates generalization.

### What is the difference between bagging and boosting?

> Bagging trains models largely independently and averages them, as in Random Forest. Boosting trains models sequentially, with later models correcting earlier errors, as in Histogram Gradient Boosting.

### Which model was most interpretable?

> Logistic Regression was the most directly interpretable through coefficient signs and magnitudes. A single Decision Tree is also interpretable, but a deep tree becomes difficult to follow.

---

# Member 2 — Validation Strategy, Metrics, and Baseline Comparison

## Main responsibility

Explain the train-test split, cross-validation, data-leakage controls, evaluation metrics, baseline results, and systematic comparison.

## Suggested opening

> We used a stratified 80/20 train-test split with random state 42. The 20% test set was held back from model selection. For baseline model development, we used stratified five-fold cross-validation on the training set. Stratification maintained the 11.6% default rate in every fold. We compared all models using the same folds and several metrics, with particular attention to average precision because the positive class was uncommon.

## Data splitting strategy

- Training set: 204,277 rows.
- Test set: 51,070 rows.
- Split: 80/20.
- `stratify=Default` maintained the same class distribution.
- `random_state=42` made the experiment reproducible.

The test set was kept separate and was not used to tune hyperparameters or select the winner. It was used to estimate final unseen-data performance after model selection.

## Baseline cross-validation strategy

The four initial models used stratified five-fold cross-validation:

1. The training data was divided into five folds.
2. Four folds trained the model.
3. The remaining fold validated the model.
4. This repeated five times so every fold acted as validation once.
5. Metrics were averaged across the five validation folds.

Stratification was necessary because only 11.6% of rows belonged to the default class.

Shuffling prevented folds from being affected by the original row order. A fixed random state made the folds reproducible.

## Why cross-validation was better than one validation split

A single validation split can produce a lucky or unlucky result. Cross-validation tests a model on several different subsets and provides:

- A more reliable mean score.
- A standard deviation showing score stability.
- Better use of the available training data.
- A fair comparison because all models use the same folds.

## Leakage controls

Data leakage means information unavailable at real prediction time enters model training or evaluation and creates unrealistically high scores.

The project controlled leakage by:

- Splitting before fitting imputation, scaling, outlier fences, encoding, or feature selection.
- Fitting all preprocessing statistics on training rows only.
- Excluding `LoanID`, which is an identifier rather than a borrower characteristic.
- Creating engineered ratios using values from the same applicant only.
- Avoiding target encoding.
- Performing oversampling only inside each training fold during tuning.
- Keeping validation and test folds at their original class distribution.
- Using the held-out test set only after model selection.

## Evaluation metrics

### Average Precision

Average precision summarizes the precision-recall curve over possible thresholds. It evaluates how well the model ranks actual defaults near the top.

This was the primary selection metric because:

- Default was the minority class.
- The no-skill reference is approximately the positive-class rate, 0.116.
- It focuses more directly on positive-class performance than accuracy.

A final test average precision of 0.3307 is substantially above the no-skill level of approximately 0.116.

### ROC-AUC

ROC-AUC measures the probability that a randomly selected defaulter receives a higher risk score than a randomly selected non-defaulter.

- 0.5 represents random ranking.
- 1.0 represents perfect ranking.
- Final test ROC-AUC: 0.7588.

ROC-AUC is useful but can appear optimistic when the negative class is much larger, so it was considered together with average precision.

### Precision

Precision answers:

> Of the applicants predicted to default, what proportion actually defaulted?

Final test precision at threshold 0.62: 0.2954.

Higher precision reduces unnecessary flags for applicants who would not default.

### Recall

Recall answers:

> Of all applicants who actually defaulted, what proportion did the model identify?

Final test recall: 0.5092.

Higher recall catches more risky loans but can create more false positives.

### F1 score

F1 is the harmonic mean of precision and recall:

`F1 = 2 × (precision × recall) / (precision + recall)`

Final test F1: 0.3739.

It is useful when both missed defaults and false warnings matter.

### Balanced Accuracy

Balanced accuracy averages recall for both classes. Unlike ordinary accuracy, it gives equal importance to defaults and non-defaults.

### Accuracy

Accuracy is the proportion of all correct predictions. It was reported but not used as the main selection metric because a no-default model would already score 88.4%.

## Baseline cross-validation comparison

| Model | Average precision | ROC-AUC | Recall | Precision | F1 | Train ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Histogram Gradient Boosting | 0.3162 | 0.7512 | 0.6694 | 0.2259 | 0.3378 | 0.7805 |
| Random Forest | 0.3106 | 0.7457 | 0.5322 | 0.2666 | 0.3553 | 0.9528 |
| Logistic Regression | 0.3096 | 0.7522 | 0.6879 | 0.2224 | 0.3361 | 0.7525 |
| Decision Tree | 0.2555 | 0.6918 | 0.6129 | 0.1980 | 0.2993 | 0.8067 |

## How to interpret the baseline table

- Histogram Gradient Boosting had the highest average precision.
- Logistic Regression had a slightly higher ROC-AUC than Histogram Gradient Boosting, but a lower average precision.
- Random Forest had the highest baseline F1 and precision at the 0.5 cutoff, but lower ranking performance and serious overfitting.
- Decision Tree was worst across the main ranking metrics and showed overfitting.
- The “best” model depends on the chosen business objective, so the project declared average precision as the primary selection metric.

## Stability of cross-validation

The validation standard deviations were small:

- Histogram Gradient Boosting AP standard deviation: 0.0033.
- Random Forest: 0.0041.
- Logistic Regression: 0.0032.
- Decision Tree: 0.0044.

This indicates that results were reasonably consistent across folds.

## Important distinction: five-fold and three-fold results

Baseline development used five-fold cross-validation. Tuning used three-fold cross-validation to reduce the computational cost of testing many hyperparameter combinations on approximately 200,000 training rows.

Do not directly claim that a five-fold baseline number improved to a three-fold tuned number. The tuning script rescored both its baseline and tuned candidate on the same three folds, so the valid tuning comparison is the baseline-versus-tuned table produced by `tune_models.py`.

## Likely viva questions for Member 2

### Why was average precision the main metric?

> Only 11.6% of loans defaulted. Average precision focuses on the minority class across different thresholds and has a clear no-skill reference equal to the default rate. Accuracy would hide poor default detection.

### Why did you still report accuracy?

> We reported it for completeness and to demonstrate why it can be misleading. A model can have high accuracy by predicting the majority class while missing all defaults.

### Why use stratified folds?

> Stratification keeps approximately the same default proportion in each fold. Without it, some folds could contain different class distributions, producing unstable and unfair comparisons.

### Why use both cross-validation and a test set?

> Cross-validation supports model and hyperparameter selection using only training data. The untouched test set provides a final estimate after all choices are complete.

### Why did tuning use three folds rather than five?

> Hyperparameter search fits many candidate models. Three-fold validation reduced computation while each training fold still contained approximately 136,000 rows. We compared tuned and untuned settings on the same three folds to keep the comparison fair.

### What is overfitting?

> Overfitting occurs when a model learns training-specific patterns that do not generalize. We detected it through large train-validation score gaps, especially for Random Forest and Decision Tree.

### Why not select Logistic Regression when its ROC-AUC was slightly higher?

> We chose average precision before selecting the model because the positive class was imbalanced. Histogram Gradient Boosting had the highest average precision, which better matched our stated objective.

---

# Member 3 — Hyperparameter Tuning and Imbalance Optimization

## Main responsibility

Explain random oversampling, the randomized search strategy, search spaces, why class weights were removed during oversampling, baseline-versus-tuned results, and threshold optimization.

## Suggested opening

> After baseline comparison, we optimized all four algorithms rather than tuning only the initial winner. We used RandomizedSearchCV with stratified three-fold cross-validation and average precision as the refit metric. Random oversampling was placed inside an imbalanced-learn pipeline, ensuring that only each training fold was balanced. Validation and test data retained the real class distribution.

## Handling class imbalance during baseline development

The initial four classifiers used `class_weight="balanced"`. This increased the importance of default cases during training without changing the number of rows.

## Handling class imbalance during tuning

During optimization, the project used `RandomOverSampler` inside an imbalanced-learn pipeline:

1. Cross-validation creates a training fold and validation fold.
2. Oversampling is applied only to the training fold.
3. Minority-class training rows are randomly duplicated until the classes are balanced.
4. The model is fitted on the balanced training fold.
5. Evaluation occurs on the untouched validation fold with its original 11.6% default rate.

This avoids leakage. Oversampling before cross-validation would allow duplicated versions of the same record to appear in both training and validation data, producing an optimistic score.

## Why class weights were disabled in the oversampled model

The model pipeline sets `class_weight=None` when oversampling is used. Otherwise, minority cases would be emphasized twice:

- First through duplicated rows.
- Again through class weighting.

Using both without justification could overcorrect the imbalance and distort predictions.

The automated test verifies that the sampler balances the fitted training data and preserves the number of output predictions.

## Why RandomizedSearchCV was selected

A full Cartesian grid would require many model fits across approximately 200,000 training rows. Randomized search samples a limited number of combinations, providing a practical trade-off between optimization quality and computational cost.

Each model tested at most 12 randomly selected settings:

- Logistic Regression: all five available `C` values.
- Decision Tree: up to 12 combinations.
- Random Forest: 8 combinations.
- Histogram Gradient Boosting: up to 12 combinations.

The search was reproducible using `random_state=42`.

## Hyperparameters investigated

### Logistic Regression

`C` controls inverse regularization strength:

- Smaller `C`: stronger regularization and simpler coefficient values.
- Larger `C`: weaker regularization and more flexibility.

Selected value: `C = 0.01`.

### Decision Tree

- `max_depth`: maximum number of split levels.
- `min_samples_leaf`: minimum training rows in each leaf.
- `max_features`: number of features considered for splitting.

Selected values:

- `max_depth = 8`.
- `min_samples_leaf = 50`.
- `max_features = None`.

These settings regularize the tree and reduce overfitting.

### Random Forest

- `n_estimators`: number of trees.
- `max_depth`: maximum depth per tree.
- `min_samples_leaf`: minimum rows in a leaf.
- `max_features`: features considered at each split.

Selected values:

- `n_estimators = 100`.
- `max_depth = 16`.
- `min_samples_leaf = 20`.
- `max_features = "sqrt"`.

Increasing `min_samples_leaf` from the baseline value makes individual trees less specific to the training data.

### Histogram Gradient Boosting

- `learning_rate`: contribution of each boosting iteration.
- `max_iter`: number of boosting iterations.
- `max_leaf_nodes`: complexity of each tree.
- `min_samples_leaf`: minimum rows in a leaf.
- `l2_regularization`: penalty against overly complex predictions.

Selected values:

- `learning_rate = 0.05`.
- `max_iter = 300`.
- `max_leaf_nodes = 15`.
- `min_samples_leaf = 50`.
- `l2_regularization = 1.0`.

The lower learning rate makes smaller corrections. More iterations compensate for the smaller step size. Fewer leaf nodes, larger leaves, and L2 regularization control overfitting.

## Baseline-versus-tuned comparison on the same three folds

| Model | Baseline AP | Tuned AP | Improvement |
|---|---:|---:|---:|
| Histogram Gradient Boosting | 0.312812 | 0.317747 | +0.004935 |
| Random Forest | 0.308333 | 0.309528 | +0.001195 |
| Logistic Regression | 0.309376 | 0.309395 | +0.000019 |
| Decision Tree | 0.231904 | 0.270204 | +0.038299 |

## Interpretation of tuning results

- Decision Tree gained the most because restricting depth reduced overfitting, but it still remained the weakest model.
- Histogram Gradient Boosting obtained a smaller but meaningful improvement and became the final leader.
- Random Forest gained only slightly, suggesting its remaining limitation was not solved by the sampled settings.
- Logistic Regression barely changed, suggesting that its performance was stable and limited mainly by its linear form rather than its regularization setting.
- Tuning cannot transform an unsuitable model into the best model. It refines the model within its algorithmic assumptions.

## Decision-threshold optimization

The probability cutoff was not automatically left at 0.5.

The process was:

1. Generate out-of-fold probabilities for all training rows.
2. Test thresholds from 0.05 to 0.80 in increments of 0.01.
3. Convert probabilities to class predictions at each threshold.
4. Calculate F1 for every threshold.
5. Select the threshold with the highest out-of-fold F1.

Selected threshold: 0.62.

The threshold was chosen using out-of-fold training predictions, not test labels. This preserved the test set for final evaluation.

## Why the best threshold can be above 0.5

Random oversampling changes the class distribution seen during training and can shift the model’s probability scale. Therefore, 0.5 is not guaranteed to produce the best precision-recall balance. The 0.62 threshold reduced lower-confidence positive predictions and maximized out-of-fold F1.

## Likely viva questions for Member 3

### Why did you not oversample the complete dataset?

> Oversampling before splitting or cross-validation could duplicate a minority record into both training and validation sets, causing leakage. We applied oversampling only inside each training fold.

### What is the difference between oversampling and class weighting?

> Oversampling changes the training-row distribution by duplicating minority examples. Class weighting keeps the same rows but gives minority errors more importance in the loss function.

### Why not use both oversampling and class weights?

> That would emphasize the minority class twice and could overcorrect the imbalance. During the oversampling experiment we set class weights to none.

### Why randomized search instead of grid search?

> The search spaces contain many possible combinations, and each candidate requires cross-validation on a large dataset. Randomized search explores a controlled number of settings with much lower computational cost.

### Why tune all four models?

> Tuning only the baseline winner could be unfair because another algorithm might improve substantially after optimization. Tuning all four enabled a systematic final comparison.

### Is 0.62 a hyperparameter?

> It is a decision threshold rather than a training hyperparameter. It changes how probabilities become class labels without retraining the model.

### Could threshold selection overfit?

> Yes, if selected directly on the test set. We reduced this risk by using out-of-fold training predictions. In a production project we could use nested cross-validation or a separate calibration set for an even stricter estimate.

---

# Member 4 — Feature Decisions, Final Model Selection, and Interpretation

## Main responsibility

Explain feature engineering and selection, the with/without feature experiment, final selection logic, final test results, business interpretation, limitations, and the group’s conclusion.

## Suggested opening

> After tuning, we tested whether our engineered affordability features improved the leading model. We compared the same tuned Histogram Gradient Boosting model with and without loan-to-income, estimated monthly payment, and payment-to-income. We then selected the final model using cross-validated average precision and evaluated it once on the held-out test set.

## Feature engineering

Three domain-informed features were created:

### Loan-to-income

`loan_to_income = LoanAmount / Income`

This represents the size of the loan relative to annual income. Larger values can indicate higher repayment pressure.

### Estimated monthly payment

This was calculated using loan amount, interest rate, and loan term through the standard amortizing-payment formula.

It transforms the original loan characteristics into an estimated monthly financial commitment.

### Payment-to-income

`payment_to_income = estimated_monthly_payment / monthly income`

This directly measures affordability—the share of monthly income required for the estimated payment.

These features use only the current applicant’s values and therefore do not leak information from other rows or from the target.

## Initial feature selection

Feature selection was fitted on the training set only:

- Numeric features were retained when absolute Pearson correlation with default was at least 0.01.
- Categorical features were retained when the default-rate gap across categories was at least one percentage point.
- Numeric pairs above 0.98 absolute correlation would keep only the feature with stronger target association.
- `LoanID` was always removed because it is only an identifier.
- Raw `LoanTerm` was dropped because its direct correlation with default was only approximately 0.0012.

Loan term still contributed indirectly to the estimated monthly payment feature.

The final transformed matrix contained 27 columns after binary, ordinal, and one-hot encoding.

## Strong feature observations

The strongest training-set associations included:

- Higher loan-to-income was associated with greater default risk.
- Greater age was associated with lower default risk in this dataset.
- Higher payment-to-income and interest rate were associated with greater risk.
- Higher income and more months employed were associated with lower risk.
- Having a co-signer or dependents was associated with a lower observed default rate.

These are associations, not proof of causation.

## With/without engineered-feature experiment

The leading tuned Histogram Gradient Boosting model was evaluated twice:

- With engineered features: AP = 0.3177.
- Without engineered features: AP = 0.3184.
- Difference: approximately 0.00065.

The reduced model was numerically higher, but the difference was below the predeclared practical threshold of 0.005. Therefore, the project treated the two results as effectively equivalent and retained the engineered features because they represent meaningful affordability concepts.

This should be explained honestly: the experiment did not prove a measurable predictive improvement from the engineered features. It showed that they did not materially damage performance under the selected tolerance.

An alternative defensible decision would be to remove them for simplicity. The project’s recorded rule was to remove them only if doing so improved average precision by at least 0.005.

## Final model-selection rule

The final winner was the model with the highest cross-validated average precision among the tuned and corresponding baseline candidates.

Histogram Gradient Boosting was selected because:

1. It had the highest tuned cross-validated AP: 0.3177.
2. It outperformed the no-skill AP level of approximately 0.116 by a wide margin.
3. It captured nonlinear effects and feature interactions.
4. Its train-validation gap was much smaller than Random Forest’s.
5. It produced stable results across folds.
6. Its final test scores were consistent with validation performance.

The final choice was not based on test performance. The test results were examined only after selecting the model.

## Final model parameters

- Learning rate: 0.05.
- Maximum iterations: 300.
- Maximum leaf nodes: 15.
- Minimum samples per leaf: 50.
- L2 regularization: 1.0.
- Decision threshold: 0.62.

## Final held-out test results

- Average precision: 0.3307.
- ROC-AUC: 0.7588.
- Recall: 0.5092.
- Precision: 0.2954.
- F1: 0.3739.

## Interpretation in plain language

### Average precision

The model’s AP of 0.3307 is much better than the approximate no-skill AP of 0.116. It ranks genuine defaults toward the high-risk end better than random ranking.

### ROC-AUC

A ROC-AUC of 0.7588 means the model has about a 75.9% chance of assigning a higher risk score to a randomly chosen defaulter than to a randomly chosen non-defaulter.

### Recall

The model detected approximately 50.9% of actual defaults at the selected threshold.

### Precision

Approximately 29.5% of applicants flagged as defaults actually defaulted. This means false positives remain substantial, so the output should support human review rather than act as an automatic rejection decision.

### F1

The F1 score of 0.3739 summarizes the selected trade-off between recall and precision.

## Why validation and test results differ

Cross-validation AP was 0.3177 while test AP was 0.3307. This small increase does not mean the model improved after seeing the test set. Different unseen samples naturally produce slightly different scores. The similar scale suggests reasonable generalization.

## Model limitations

1. Precision remains low, so many flagged applicants would be false positives.
2. Approximately half of actual defaults are still missed at threshold 0.62.
3. Oversampled-model probabilities may not be perfectly calibrated.
4. The model was evaluated on one dataset and may experience distribution drift.
5. Histogram Gradient Boosting is less interpretable than Logistic Regression.
6. Age and marital status may create fairness and legal concerns in real lending.
7. Feature associations do not prove causal relationships.
8. The randomized search tested a limited number of settings rather than every possible combination.
9. The feature-retention tolerance of 0.005 is a modelling choice and should be justified.

## Future improvements

- Calibrate probabilities using Platt scaling or isotonic regression.
- Select the threshold using explicit business costs for false negatives and false positives.
- Perform fairness analysis across protected groups.
- Use SHAP or permutation importance to explain individual and global predictions.
- Test SMOTE or other imbalance strategies inside cross-validation.
- Use nested cross-validation for a stricter tuning estimate.
- Monitor performance and data drift after deployment.
- Compare additional tabular boosting models if permitted.

## Likely viva questions for Member 4

### Why did you keep engineered features when AP without them was slightly higher?

> The difference was only about 0.00065, below our predefined practical threshold of 0.005. We treated that as negligible validation variation and retained the domain-relevant affordability features. We would also acknowledge that removing them for simplicity would be defensible because they did not show a clear predictive gain.

### Why was Histogram Gradient Boosting selected?

> It had the highest cross-validated average precision after tuning, generalized better than the other tree-based models, captured nonlinear relationships, and achieved consistent held-out test performance.

### Did you use the test set to select the final model?

> No. Selection used cross-validation on training data. The test set was scored only after the winner, features, hyperparameters, and threshold had been chosen.

### What does the 0.62 threshold mean?

> An applicant is classified as a predicted default when the model’s score is at least 0.62. It was selected from out-of-fold training predictions to maximize F1.

### Is the output a true probability?

> It is a model-produced probability-like score, but random oversampling can affect calibration. Before using it as a real probability of default, we should evaluate calibration and apply a calibration method if necessary.

### Can this model automatically reject a loan?

> It should not. Because of false positives, fairness concerns, and regulatory requirements, it is more appropriate as a decision-support or review-prioritization tool.

---

# Recommended presentation order

## Member 1: approximately 4–5 minutes

1. Problem and target.
2. Class imbalance.
3. Four selected algorithms.
4. How each works.
5. Why each was suitable.
6. Initial observations about overfitting and performance.

Transition:

> After selecting four suitable algorithms, we needed a fair validation and evaluation process. Member 2 will explain that strategy and our baseline comparison.

## Member 2: approximately 4–5 minutes

1. 80/20 stratified split.
2. Five-fold baseline cross-validation.
3. Leakage controls.
4. Evaluation metrics.
5. Baseline comparison.
6. Interpretation of train-validation gaps.

Transition:

> The baseline comparison identified promising models, but their settings were not yet optimized. Member 3 will explain how we tuned the models and handled class imbalance safely.

## Member 3: approximately 4–5 minutes

1. Oversampling inside folds.
2. Why class weights were disabled during oversampling.
3. Randomized search.
4. Hyperparameters.
5. Baseline-versus-tuned results.
6. Threshold optimization.

Transition:

> After tuning, we tested the modelling choices around engineered features and selected the final model. Member 4 will explain the final decision and results.

## Member 4: approximately 4–5 minutes

1. Engineered features and feature selection.
2. With/without experiment.
3. Final selection criteria.
4. Final model and test results.
5. Business interpretation.
6. Limitations and future improvements.

---

# One-minute complete group summary

> We developed four classification models for loan-default prediction: Logistic Regression, Decision Tree, Random Forest, and Histogram Gradient Boosting. Because only 11.6% of loans defaulted, accuracy alone was unsuitable, so we used average precision as the primary selection metric and also reported ROC-AUC, precision, recall, F1, balanced accuracy, and accuracy. Baseline models were compared using stratified five-fold cross-validation. During optimization, all four models were rescored and tuned on the same stratified three-fold setup using RandomizedSearchCV. Random oversampling occurred only inside each training fold to prevent leakage. Histogram Gradient Boosting achieved the best tuned cross-validated average precision of 0.3177. We tested its engineered affordability features and retained them because removing them changed AP by less than our 0.005 practical threshold. We selected a 0.62 decision cutoff using out-of-fold F1. On the untouched test set, the final model achieved 0.3307 average precision, 0.7588 ROC-AUC, 0.5092 recall, 0.2954 precision, and 0.3739 F1. We selected it because it provided the strongest minority-class ranking performance with better generalization than the other nonlinear models.

---

# Rapid-fire questions any member may receive

## What is the target variable?

`Default`, where 1 means default and 0 means no default.

## What type of problem is this?

Supervised binary classification.

## How many rows are in the dataset?

255,347 cleaned rows.

## What was the split?

80% training and 20% test, stratified by `Default`.

## Why stratify?

To preserve the 11.6% default rate in both splits and every validation fold.

## What was the primary metric?

Average precision.

## What is the no-skill average precision?

Approximately the default rate, 0.116.

## What were the four algorithms?

Logistic Regression, Decision Tree, Random Forest, and Histogram Gradient Boosting.

## Which baseline model had the best average precision?

Histogram Gradient Boosting, approximately 0.3162 in five-fold cross-validation.

## Which model showed the most baseline overfitting?

Random Forest, with a train-validation ROC-AUC gap of about 0.207.

## Which model showed the least overfitting?

Logistic Regression, with almost identical training and validation ROC-AUC.

## What search method was used?

RandomizedSearchCV.

## Why not grid search?

A full grid was computationally expensive on approximately 200,000 training rows.

## How was imbalance handled during tuning?

Random oversampling inside each cross-validation training fold.

## Why is oversampling outside cross-validation wrong?

Duplicated rows could enter both training and validation data, causing leakage.

## Which tuned model was selected?

Histogram Gradient Boosting.

## What was its tuned cross-validation AP?

0.3177.

## What were the final test AP and ROC-AUC?

AP 0.3307 and ROC-AUC 0.7588.

## What was the decision threshold?

0.62.

## How was the threshold selected?

By maximizing F1 over out-of-fold training probabilities.

## What features were engineered?

Loan-to-income, estimated monthly payment, and payment-to-income.

## Which raw model feature was dropped?

Loan term, because its direct association with default was below the practical cutoff. It remained an input for calculating estimated payment.

## What is one key ethical issue?

Age and marital status can raise fairness and legal concerns in real credit decisions.

## What was the most important limitation?

The model’s precision was only about 29.5%, so it should support human review rather than automatically reject applications.

---

# Final viva advice

1. Say “associated with,” not “caused by,” when discussing features.
2. Do not describe AP as ordinary accuracy.
3. Do not claim the model catches 75.9% of defaults; 75.9% is ROC-AUC, while recall is 50.9%.
4. Do not compare the five-fold baseline number directly with the three-fold tuned number. Use the tuning table where baseline and tuned models share the same folds.
5. Clearly separate model selection from final test evaluation.
6. Be honest that engineered features did not materially improve AP.
7. Explain why accuracy is misleading before quoting it.
8. Connect every technical choice to the problem: imbalance, large tabular data, generalization, computational cost, or lending risk.
9. Each member should understand at least the one-minute group summary and all rapid-fire answers.
10. Explain decisions in your own words instead of memorizing numbers without understanding them.
