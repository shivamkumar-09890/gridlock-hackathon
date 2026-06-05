import pandas as pd

from src.features.base_feature import BaseFeature


class CategoricalFeatures(BaseFeature):
    """
    Numerical encoding for XGBoost.
    """

    def fit(self, df):

        road_dummies = pd.get_dummies(
            df["RoadType"],
            prefix="RoadType"
        )

        weather_dummies = pd.get_dummies(
            df["Weather"],
            prefix="Weather"
        )

        self.road_columns = (
            road_dummies.columns
            .tolist()
        )

        self.weather_columns = (
            weather_dummies.columns
            .tolist()
        )

        return self

    def transform(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:

        df = df.copy()

        # ------------------
        # Binary Encoding
        # ------------------

        df["Landmarks"] = (
            df["Landmarks"]
            .map(
                {
                    "no": 0,
                    "yes": 1,
                }
            )
            .fillna(0)
            .astype(int)
        )

        df["LargeVehicles"] = (
            df["LargeVehicles"]
            .map(
                {
                    "not allowed": 0,
                    "allowed": 1,
                }
            )
            .fillna(0)
            .astype(int)
        )

        # ------------------
        # One Hot Encoding
        # ------------------

        road = pd.get_dummies(
            df["RoadType"],
            prefix="RoadType"
        )

        weather = pd.get_dummies(
            df["Weather"],
            prefix="Weather"
        )

        road = road.reindex(
            columns=self.road_columns,
            fill_value=0
        )

        weather = weather.reindex(
            columns=self.weather_columns,
            fill_value=0
        )

        df = pd.concat(
            [
                df,
                road,
                weather,
            ],
            axis=1
        )

        df.drop(
            columns=[
                "RoadType",
                "Weather",
                "geohash",
            ],
            inplace=True,
            errors="ignore"
        )

        return df