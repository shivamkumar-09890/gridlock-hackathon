import pygeohash as pgh

from features.base_feature import BaseFeature


class GeohashFeatures(BaseFeature):

    def fit(self, df):
        self.freq_map = (
            df["geohash"]
            .value_counts()
            .to_dict()
        )

        return self

    def transform(self, df):

        df = df.copy()

        if "geohash" not in df.columns:
            return df

        latitudes = []
        longitudes = []

        for g in df["geohash"]:

            try:
                lat, lon = pgh.decode(g)

            except Exception:
                lat = None
                lon = None

            latitudes.append(lat)
            longitudes.append(lon)

        df["latitude"] = latitudes
        df["longitude"] = longitudes

        df["geohash_frequency"] = (
            df["geohash"]
            .map(self.freq_map)
            .fillna(0)
        )

        return df