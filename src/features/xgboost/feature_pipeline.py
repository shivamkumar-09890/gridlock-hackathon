from src.features.xgboost.geohash_features import GeohashFeatures
from src.features.base_feature import BaseFeature

from src.features.xgboost.spatial_features import (
    SpatialFeatures
)

from src.features.xgboost.temporal_features import (
    TemporalFeatures
)

from src.features.xgboost.categorical_features import (
    CategoricalFeatures
)


class XGBoostFeaturePipeline(
    BaseFeature
):

    def __init__(self):

        self.steps = [
            # GeohashFeatures(),
            TemporalFeatures(),
            SpatialFeatures(),
            CategoricalFeatures(),
        ]

    def fit(self, df):

        for step in self.steps:
            step.fit(df)

        return self

    def transform(self, df):

        for step in self.steps:
            df = step.transform(df)

        return df

    def fit_transform(self, df):

        self.fit(df)

        return self.transform(df)