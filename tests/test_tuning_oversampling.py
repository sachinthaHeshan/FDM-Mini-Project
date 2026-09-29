import numpy as np
import pandas as pd

import tune_models
from train_models import build_models


def test_oversampled_estimator_balances_fit_data_without_changing_predictions():
    features = pd.DataFrame(
        {
            "income": np.arange(20, dtype=float),
            "interest_rate": np.linspace(2.0, 20.0, 20),
        }
    )
    target = np.array([0] * 16 + [1] * 4)

    pipeline = tune_models.build_oversampled_estimator(
        "logistic_regression",
        build_models()["logistic_regression"],
    )
    pipeline.fit(features, target)

    assert pipeline.named_steps["model"].class_weight is None
    assert pipeline.named_steps["sampler"].sampling_strategy_ == {1: 12}
    assert len(pipeline.predict(features)) == len(features)
