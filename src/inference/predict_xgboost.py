from pathlib import Path

import pandas as pd

from src.data.load_data import (
    load_test
)

from src.inference.xgboost_predictor import (
    XGBoostPredictor
)

# ==================================================
# Paths
# ==================================================

MODEL_DIR = "models/xgboost"

TEST_PATH = "data/raw/test.csv"

SUBMISSION_DIR = Path(
    "data/submissions"
)

SUBMISSION_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_FILE = (
    SUBMISSION_DIR
    / "submission_xgboost.csv"
)

# ==================================================
# Load Test Data
# ==================================================

print("Loading test data...")

test_df = load_test(
    TEST_PATH
)

print(
    f"Test Shape: {test_df.shape}"
)

test_ids = (
    test_df["Index"]
    .copy()
)

# ==================================================
# Load Predictor
# ==================================================

print(
    "Loading model artifacts..."
)

predictor = (
    XGBoostPredictor(
        model_dir=MODEL_DIR
    )
)

print(
    f"Model expects "
    f"{predictor.n_features} features"
)

# ==================================================
# Predict
# ==================================================

print(
    "Generating predictions..."
)

predictions = (
    predictor.predict(
        test_df
    )
)

# ==================================================
# Submission
# ==================================================

submission = pd.DataFrame(
    {
        "id": test_ids,
        "demand": predictions,
    }
)

submission.to_csv(
    OUTPUT_FILE,
    index=False,
)

# ==================================================
# Summary
# ==================================================

print("\nPreview:")

print(
    submission.head()
)

print(
    f"\nSaved to:\n{OUTPUT_FILE}"
)

print(
    f"\nRows: {len(submission)}"
)