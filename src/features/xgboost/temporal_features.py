import numpy as np
import pandas as pd

from src.features.base_feature import BaseFeature


class TemporalFeatures(BaseFeature):
    """
    Time-based numerical features.
    """

    def fit(self, df):
        return self

    def transform(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:

        df = df.copy()

        split_time = (
            df["timestamp"]
            .str.split(
                ":",
                expand=True
            )
        )

        df["hour"] = (
            split_time[0]
            .astype(int)
        )

        df["minute"] = (
            split_time[1]
            .astype(int)
        )

        df["minutes_since_midnight"] = (
            df["hour"] * 60
            + df["minute"]
        )

        df["time_sin"] = np.sin(
            2
            * np.pi
            * df["minutes_since_midnight"]
            / 1440
        )

        df["time_cos"] = np.cos(
            2
            * np.pi
            * df["minutes_since_midnight"]
            / 1440
        )

        df.drop(
            columns=["timestamp"],
            inplace=True,
            errors="ignore"
        )

        return df