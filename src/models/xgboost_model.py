from xgboost import XGBRegressor


class XGBoostDemandModel:

    def __init__(self):

        self.params = {
            "n_estimators": 5000,
            "learning_rate": 0.02,
            "max_depth": 10,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "random_state": 42,
            "tree_method": "hist",
        }

    def build(self):

        return XGBRegressor(
            **self.params
        )