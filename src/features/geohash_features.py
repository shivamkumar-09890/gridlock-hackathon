import pandas as pd
import geohash2

from src.features.base_feature import BaseFeature


class GeohashFeatures(BaseFeature):
    """
    Geospatial feature engineering.
    """

    def fit(self, df: pd.DataFrame):
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:

        df = df.copy()

        # -------------------------
        # Decode Geohash
        # -------------------------

        decoded = df["geohash"].apply(
            geohash2.decode
        )

        df["lat"] = decoded.str[0].astype(float)
        df["lon"] = decoded.str[1].astype(float)

        # -------------------------
        # Geo-Hour
        # -------------------------

        if "hour" in df.columns:

            df["geo_hour"] = (
                df["geohash"].astype(str)
                + "_"
                + df["hour"].astype(str)
            )

        # -------------------------
        # Geo-Period
        # -------------------------

        if "period" in df.columns:

            df["geo_period"] = (
                df["geohash"].astype(str)
                + "_"
                + df["period"].astype(str)
            )

        # -------------------------
        # Geo-Weather
        # -------------------------

        if "Weather" in df.columns:

            df["geo_weather"] = (
                df["geohash"].astype(str)
                + "_"
                + df["Weather"].astype(str)
            )

        # -------------------------
        # Geo-Road
        # -------------------------

        if "RoadType" in df.columns:

            df["geo_road"] = (
                df["geohash"].astype(str)
                + "_"
                + df["RoadType"].astype(str)
            )

        # -------------------------
        # Spatial Interaction
        # -------------------------

        df["lat_lon_interaction"] = (
            df["lat"] * df["lon"]
        )

        return df