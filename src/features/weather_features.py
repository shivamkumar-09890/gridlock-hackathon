from features.base_feature import BaseFeature

class WeatherFeatures(BaseFeature):

    def fit(self, df):
        return self

    def transform(self, df):

        df = df.copy()

        if "Temperature" in df.columns:

            df["temp_squared"] = (
                df["Temperature"] ** 2
            )

            df["temp_log"] = (
                df["Temperature"] + 1
            )

        return df