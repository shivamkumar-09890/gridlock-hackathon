import pandas as pd

from features.base_feature import BaseFeature


class RoadFeatures(BaseFeature):

    def fit(self, df):
        return self

    def transform(self, df):

        df = df.copy()

        # -------------------------
        # RoadType + Period
        # -------------------------

        if (
            "RoadType" in df.columns
            and "period" in df.columns
        ):

            df["roadtype_period"] = (
                df["RoadType"].astype(str)
                + "_"
                + df["period"].astype(str)
            )

        return df