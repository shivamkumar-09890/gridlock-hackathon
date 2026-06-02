import pandas as pd


class Preprocessor:

    def fit(self, df):
        self.temperature_median = df["Temperature"].median()
        return self

    def transform(self, df):

        df = df.copy()

        if "Temperature" in df.columns:
            df["Temperature"] = df["Temperature"].fillna(
                self.temperature_median
            )

        if "RoadType" in df.columns:
            df["RoadType"] = df["RoadType"].fillna(
                "Unknown"
            )

        return df