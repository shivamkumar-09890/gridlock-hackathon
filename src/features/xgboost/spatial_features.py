import numpy as np
import pandas as pd
import geohash2

from sklearn.preprocessing import StandardScaler

from src.features.base_feature import BaseFeature


class SpatialFeatures(BaseFeature):
    """
    Spatial features for XGBoost.

    Creates:
    --------
    lat_scaled
    lon_scaled
    distance_from_city_center

    Uses geohash directly and does not rely on
    previous feature engineering steps.
    """

    def fit(
        self,
        df: pd.DataFrame
    ):

        self.scaler = StandardScaler()

        decoded = df["geohash"].apply(
            geohash2.decode_exactly
        )

        lat_lon = pd.DataFrame(
            {
                "lat": decoded.str[0],
                "lon": decoded.str[1],
            }
        )

        self.scaler.fit(lat_lon)

        return self

    def transform(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:

        df = df.copy()

        # -------------------------
        # Decode Geohash
        # -------------------------

        decoded = df["geohash"].apply(
            geohash2.decode_exactly
        )

        lat_lon = pd.DataFrame(
            {
                "lat": decoded.str[0],
                "lon": decoded.str[1],
            }
        )

        # -------------------------
        # Scale Coordinates
        # -------------------------

        scaled = self.scaler.transform(
            lat_lon
        )

        df["lat_scaled"] = scaled[:, 0]
        df["lon_scaled"] = scaled[:, 1]

        # -------------------------
        # Distance Feature
        # -------------------------

        df["distance_from_city_center"] = (
            np.sqrt(
                df["lat_scaled"] ** 2
                + df["lon_scaled"] ** 2
            )
        )

        return df