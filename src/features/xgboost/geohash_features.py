import pandas as pd
import geohash2

from src.features.base_feature import BaseFeature


class GeohashFeatures(BaseFeature):

    def fit(self, df):
        return self

    def transform(self, df):

        df = df.copy()

        decoded = df["geohash"].apply(
            geohash2.decode_exactly
        )

        df["lat"] = decoded.str[0]
        df["lon"] = decoded.str[1]

        return df