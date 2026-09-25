"""Charts written during preprocessing."""

import unittest
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from preprocessing_figures import FIGURE_NAMES, save_preprocessing_figures


def _sample(n: int = 80) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(
        {
            "Age": rng.integers(18, 70, n),
            "Income": rng.integers(20_000, 120_000, n),
            "LoanAmount": rng.integers(5_000, 200_000, n),
            "CreditScore": rng.integers(300, 850, n),
            "MonthsEmployed": rng.integers(0, 120, n),
            "NumCreditLines": rng.integers(1, 5, n),
            "InterestRate": rng.uniform(2, 25, n),
            "LoanTerm": rng.choice([12, 24, 36, 48, 60], n),
            "DTIRatio": rng.uniform(0.1, 0.9, n),
            "Education": rng.choice(["High School", "Bachelor's", "Master's", "PhD"], n),
            "EmploymentType": rng.choice(
                ["Full-time", "Part-time", "Self-employed", "Unemployed"], n
            ),
            "MaritalStatus": rng.choice(["Single", "Married", "Divorced"], n),
            "HasMortgage": rng.choice(["Yes", "No"], n),
            "HasDependents": rng.choice(["Yes", "No"], n),
            "LoanPurpose": rng.choice(["Home", "Auto", "Education", "Business", "Other"], n),
            "HasCoSigner": rng.choice(["Yes", "No"], n),
        }
    )
    frame.loc[0, "Income"] = 15_000
    frame.loc[0, "LoanAmount"] = 249_000
    target = pd.Series(rng.integers(0, 2, n), name="Default")
    rows = []
    for feature in [
        "Age",
        "Income",
        "LoanAmount",
        "CreditScore",
        "MonthsEmployed",
        "NumCreditLines",
        "InterestRate",
        "DTIRatio",
        "loan_to_income",
        "estimated_monthly_payment",
        "payment_to_income",
    ]:
        rows.append({"feature": feature, "kind": "numeric", "keep": True})
    rows.append({"feature": "LoanTerm", "kind": "numeric", "keep": False})
    for feature in ["HasMortgage", "HasDependents", "HasCoSigner"]:
        rows.append({"feature": feature, "kind": "binary", "keep": True})
    rows.append({"feature": "Education", "kind": "ordinal", "keep": True})
    for feature in ["EmploymentType", "MaritalStatus", "LoanPurpose"]:
        rows.append({"feature": feature, "kind": "nominal", "keep": True})
    return frame, target, pd.DataFrame(rows)


class SavePreprocessingFiguresTest(unittest.TestCase):
    def test_writes_png_charts_for_the_training_split(self) -> None:
        features, target, selection = _sample()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            written = save_preprocessing_figures(output, features, target, selection)
            names = [path.name for path in written]
            self.assertEqual(names, list(FIGURE_NAMES))
            for path in written:
                self.assertTrue(path.is_file(), path.name)
                self.assertGreater(path.stat().st_size, 1000, path.name)
                self.assertEqual(path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")


if __name__ == "__main__":
    unittest.main()
