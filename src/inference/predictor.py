from src.models.catboost_model import (
    CatBoostDemandModel
)

from src.utils.save_load import (
    load_object
)


class Predictor:

    def __init__(self):

        self.model = (
            CatBoostDemandModel()
            .load(
                "models/catboost/model.cbm"
            )
        )

        self.pipeline = load_object(
            "models/catboost/feature_pipeline.pkl"
        )

        self.preprocessor = load_object(
            "models/catboost/preprocessor.pkl"
        )

    def predict(self, df):

        df = self.preprocessor.transform(
            df
        )

        df = self.pipeline.transform(
            df
        )

        preds = self.model.predict(
            df
        )

        return preds