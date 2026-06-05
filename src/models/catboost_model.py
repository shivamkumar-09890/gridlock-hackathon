# src/models/catboost_model.py

from catboost import CatBoostRegressor


class CatBoostDemandModel:

    def __init__(
        self,
        iterations=5000,
        learning_rate=0.015,
        depth=10,
        random_seed=42,
    ):

        self.params = {
            "iterations": iterations,
            "learning_rate": learning_rate,
            "depth": depth,
            "loss_function": "RMSE",
            "eval_metric": "RMSE",
            "random_seed": random_seed,
            "early_stopping_rounds": 300,
            "verbose": 100,
        }

    def build(self):

        return CatBoostRegressor(
            **self.params
        )