from features.base_feature import BaseFeature
from src.features.geohash_features import GeohashFeatures
from src.features.weather_features import WeatherFeatures
from src.features.road_features import RoadFeatures


class FeaturePipeline(BaseFeature):

    def __init__(self):

        self.geohash = GeohashFeatures()
        self.weather = WeatherFeatures()
        self.road = RoadFeatures()

    def fit(self, df):

        self.geohash.fit(df)
        self.weather.fit(df)
        self.road.fit(df)

        return self

    def transform(self, df):

        df = self.geohash.transform(df)
        df = self.weather.transform(df)
        df = self.road.transform(df)

        return df

    def fit_transform(self, df):

        self.fit(df)

        return self.transform(df)