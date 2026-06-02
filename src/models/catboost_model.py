from pathlib import Path

from catboost import CatBoostRegressor

from src.models.base_model import BaseModel


class CatBoostDemandModel(BaseModel):

    def __init__(self):

        self.model = CatBoostRegressor(
            iterations=1000,
            learning_rate=0.05,
            depth=8,
            loss_function="RMSE",
            verbose=100
        )

    def train(self, X, y):

        self.model.fit(
            X,
            y
        )

    def predict(self, X):

        return self.model.predict(X)

    def save(self, path):

        Path(path).parent.mkdir(
            parents=True,
            exist_ok=True
        )

        self.model.save_model(path)

    def load(self, path):

        self.model.load_model(path)

        return self