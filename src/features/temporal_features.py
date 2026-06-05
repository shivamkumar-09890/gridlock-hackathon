import numpy as np
import pandas as pd

from src.features.base_feature import BaseFeature


class TemporalFeatures(BaseFeature):

    def fit(self, df):
        return self

    @staticmethod
    def get_period(hour):

        if hour < 5:
            return "night"

        elif hour < 12:
            return "morning"

        elif hour < 17:
            return "afternoon"

        elif hour < 21:
            return "evening"

        return "late_night"

    def transform(self, df):

        df = df.copy()

        # -------------------------
        # Hour
        # -------------------------

        df["hour"] = (
            df["timestamp"]
            .astype(str)
            .str.split(":")
            .str[0]
            .astype(int)
        )

        # -------------------------
        # Period
        # -------------------------

        df["period"] = (
            df["hour"]
            .apply(self.get_period)
        )

        # -------------------------
        # Cyclical Hour
        # -------------------------

        df["sin_hour"] = np.sin(
            2 * np.pi * df["hour"] / 24
        )

        df["cos_hour"] = np.cos(
            2 * np.pi * df["hour"] / 24
        )

        # -------------------------
        # Cyclical Day
        # -------------------------

        df["sin_day"] = np.sin(
            2 * np.pi * df["day"] / 7
        )

        df["cos_day"] = np.cos(
            2 * np.pi * df["day"] / 7
        )

        # -------------------------
        # Hour Squared
        # -------------------------

        df["hour_sq"] = (
            df["hour"] ** 2
        )

        # -------------------------
        # Peak Hours
        # -------------------------

        df["is_morning_peak"] = (
            df["hour"]
            .between(7, 9)
        ).astype(int)

        df["is_evening_peak"] = (
            df["hour"]
            .between(17, 20)
        ).astype(int)

        return df