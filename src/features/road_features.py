from sklearn.preprocessing import LabelEncoder

from features.base_feature import BaseFeature


class RoadFeatures(BaseFeature):

    def fit(self, df):

        self.road_encoder = LabelEncoder()

        self.road_encoder.fit(
            df["RoadType"].astype(str)
        )

        return self

    def transform(self, df):

        df = df.copy()

        if "RoadType" in df.columns:

            df["roadtype_encoded"] = (
                self.road_encoder.transform(
                    df["RoadType"].astype(str)
                )
            )

        if "NumberOfLanes" in df.columns:

            df["lanes_squared"] = (
                df["NumberOfLanes"] ** 2
            )

        return df