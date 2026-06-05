from src.features.base_feature import BaseFeature

from src.features.temporal_features import TemporalFeatures
from src.features.geohash_features import GeohashFeatures
from src.features.weather_features import WeatherFeatures
from src.features.road_features import RoadFeatures


class FeaturePipeline(BaseFeature):
    """
    Main feature engineering pipeline.

    Each feature module:
    1. Learns statistics in fit()
    2. Creates features in transform()

    Feature order matters because some features depend
    on columns created by previous steps.

    Current flow:

    TemporalFeatures
        ↓
    GeohashFeatures
        ↓
    WeatherFeatures
        ↓
    RoadFeatures
    """

    def __init__(self):

        self.feature_steps = [
            TemporalFeatures(),
            GeohashFeatures(),
            WeatherFeatures(),
            RoadFeatures(),
        ]

    def fit(self, df):
        """
        Fit all feature generators.
        """

        for step in self.feature_steps:
            step.fit(df)

        return self

    def transform(self, df):
        """
        Apply all feature engineering steps.
        """

        for step in self.feature_steps:
            df = step.transform(df)

        return df

    def fit_transform(self, df):
        """
        Fit and transform training data.
        """

        self.fit(df)

        return self.transform(df)