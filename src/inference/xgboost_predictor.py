from pathlib import Path

import joblib
import pandas as pd


class XGBoostPredictor:

    def __init__(self, model_dir):

        self.model_dir = Path(model_dir)

        self.model = joblib.load(
            self.model_dir / "best_xgboost.pkl"
        )

        self.preprocessor = joblib.load(
            self.model_dir / "preprocessor.pkl"
        )

        self.pipeline = joblib.load(
            self.model_dir / "feature_pipeline.pkl"
        )

        self.feature_cols = joblib.load(
            self.model_dir / "feature_cols.pkl"
        )

    @property
    def n_features(self):

        return len(
            self.feature_cols
        )

    def preprocess(self, df):

        df = self.preprocessor.transform(
            df
        )

        df = self.pipeline.transform(
            df
        )

        return df

    def predict(
        self,
        df: pd.DataFrame
    ):

        df = self.preprocess(df)

        missing_cols = [
            col
            for col in self.feature_cols
            if col not in df.columns
        ]

        if missing_cols:

            raise ValueError(
                f"Missing columns:\n{missing_cols}"
            )

        X = df[
            self.feature_cols
        ]

        return self.model.predict(X)