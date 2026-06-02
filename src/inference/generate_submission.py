import pandas as pd

from src.data.load_data import load_test

from src.inference.predictor import Predictor


test_df = load_test(
    "data/raw/test.csv"
)

ids = test_df["Index"]

predictor = Predictor()

preds = predictor.predict(
    test_df
)

submission = pd.DataFrame({
    "Index": ids,
    "demand": preds
})

submission.to_csv(
    "data/submissions/submission_v1.csv",
    index=False
)

print(
    submission.head()
)