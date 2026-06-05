from pathlib import Path

import pandas as pd

from src.data.load_data import load_test
from src.inference.predictor import Predictor


# ==================================================
# Paths
# ==================================================

MODEL_DIR = "models/catboost"

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
    / "submission_catboost.csv"
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

# Keep original ids

test_ids = test_df["Index"].copy()

# ==================================================
# Load Predictor
# ==================================================

print("Loading model artifacts...")

predictor = Predictor(
    model_dir=MODEL_DIR
)

print(
    f"Model expects {predictor.n_features} features"
)

# ==================================================
# Generate Predictions
# ==================================================

print("Generating predictions...")

predictions = predictor.predict(
    test_df
)

# ==================================================
# Create Submission
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

print("\nSubmission Preview:")
print(submission.head())

print(
    f"\nSaved submission to:\n{OUTPUT_FILE}"
)

print(
    f"\nRows predicted: {len(submission)}"
)