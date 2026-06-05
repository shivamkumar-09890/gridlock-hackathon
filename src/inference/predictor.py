from pathlib import Path

import joblib
import pandas as pd


class Predictor:
    """
    Generic prediction pipeline.

    Responsibilities:
    -----------------
    1. Load trained model
    2. Load preprocessor
    3. Load feature pipeline
    4. Load feature list
    5. Apply preprocessing
    6. Apply feature engineering
    7. Generate predictions
    """

    def __init__(self, model_dir: str):

        self.model_dir = Path(model_dir)

        self.model = None
        self.preprocessor = None
        self.pipeline = None
        self.feature_cols = None

        self._load_artifacts()

    def _load_artifacts(self):
        """
        Load all saved artifacts.
        """

        print(
            f"Loading artifacts from {self.model_dir}"
        )

        self.model = joblib.load(
            self.model_dir / "best_catboost.pkl"
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

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply preprocessing and feature engineering.
        """

        df = self.preprocessor.transform(df)

        df = self.pipeline.transform(df)

        return df

    def prepare_features(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Ensure feature columns exactly match training.
        """

        missing_cols = [
            col
            for col in self.feature_cols
            if col not in df.columns
        ]

        if missing_cols:

            raise ValueError(
                f"Missing columns:\n{missing_cols}"
            )

        return df[self.feature_cols]

    def predict(
        self,
        df: pd.DataFrame
    ):
        """
        Predict from raw dataframe.
        """

        df = self.preprocess(df)

        X = self.prepare_features(df)

        preds = self.model.predict(X)

        return preds

    def predict_features(
        self,
        X: pd.DataFrame
    ):
        """
        Predict directly from prepared features.

        Useful for:
        - Ensembling
        - Validation
        - Stacking
        """

        return self.model.predict(X)

    def get_feature_importance(self):
        """
        Return feature importance dataframe.
        """

        importance = (
            self.model
            .get_feature_importance()
        )

        return pd.DataFrame(
            {
                "feature": self.feature_cols,
                "importance": importance,
            }
        ).sort_values(
            "importance",
            ascending=False
        )

    @property
    def n_features(self):

        return len(
            self.feature_cols
        )