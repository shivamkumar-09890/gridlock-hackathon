import numpy as np
import pandas as pd


class Preprocessor:

    def fit(self, df):

        self.temperature_mean = (
            df["Temperature"].mean()
        )

        self.categorical_modes = {}

        categorical_cols = [
            "RoadType",
            "Weather",
            "LargeVehicles",
            "Landmarks"
        ]

        for col in categorical_cols:

            if col in df.columns:

                mode = (
                    df[col]
                    .replace("nan", np.nan)
                    .mode()[0]
                )

                self.categorical_modes[col] = mode

        return self

    def transform(self, df):

        df = df.copy()

        # -------------------------
        # Lowercase strings
        # -------------------------

        object_cols = (
            df.select_dtypes(
                include=["object"]
            )
            .columns
        )

        for col in object_cols:

            df[col] = (
                df[col]
                .astype(str)
                .str.lower()
            )

        # -------------------------
        # Fill categorical
        # -------------------------

        for col, mode in (
            self.categorical_modes.items()
        ):

            df[col] = (
                df[col]
                .replace("nan", np.nan)
                .fillna(mode)
            )

        # -------------------------
        # Fill temperature
        # -------------------------

        if "Temperature" in df.columns:

            df["Temperature"] = (
                df["Temperature"]
                .fillna(
                    self.temperature_mean
                )
            )

        return df