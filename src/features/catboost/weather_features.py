import pandas as pd
import numpy as np

from features.base_feature import BaseFeature


class WeatherFeatures(BaseFeature):

    def fit(self, df):
        return self

    def transform(self, df):

        df = df.copy()

        # -------------------------
        # Temperature Squared
        # -------------------------

        df["temperature_sq"] = (
            df["Temperature"] ** 2
            - df["Temperature"]
        )

        # -------------------------
        # Temperature Bucket
        # -------------------------

        df["temp_bucket"] = pd.cut(
            df["Temperature"],
            bins=[-100, 15, 25, 35, 100],
            labels=[
                "cold",
                "mild",
                "warm",
                "hot"
            ]
        )

        # -------------------------
        # Weather + Temp
        # -------------------------

        if "Weather" in df.columns:

            df["weather_temp"] = (
                df["Weather"].astype(str)
                + "_"
                + df["temp_bucket"].astype(str)
            )

        # -------------------------
        # Geo + Temp Bucket
        # -------------------------

        if "geohash" in df.columns:

            df["geo_temp"] = (
                df["geohash"].astype(str)
                + "_"
                + df["temp_bucket"].astype(str)
            )

        return df