import pandas as pd

from sklearn.model_selection import train_test_split

from src.data.load_data import load_train
from src.data.preprocessing import Preprocessor
from src.features.feature_pipeline import FeaturePipeline

from src.models.catboost_model import (
    CatBoostDemandModel
)

from src.utils.metrics import rmse
from src.utils.save_load import save_object


TARGET = "Demand"


# ------------------------
# Load data
# ------------------------

train_df = load_train(
    "data/raw/train.csv"
)

# ------------------------
# Preprocessing
# ------------------------

preprocessor = Preprocessor()

preprocessor.fit(train_df)

train_df = preprocessor.transform(
    train_df
)

# ------------------------
# Feature Engineering
# ------------------------

pipeline = FeaturePipeline()

train_df = pipeline.fit_transform(
    train_df
)

# ------------------------
# Prepare Features
# ------------------------

drop_cols = [
    TARGET
]

X = train_df.drop(
    columns=drop_cols
)

y = train_df[TARGET]

# ------------------------
# Split
# ------------------------

X_train, X_valid, y_train, y_valid = (
    train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )
)

# ------------------------
# Train
# ------------------------

model = CatBoostDemandModel()

model.train(
    X_train,
    y_train
)

# ------------------------
# Evaluate
# ------------------------

preds = model.predict(
    X_valid
)

score = rmse(
    y_valid,
    preds
)

print(f"RMSE = {score}")

# ------------------------
# Save
# ------------------------

model.save(
    "models/catboost/model.cbm"
)

save_object(
    pipeline,
    "models/catboost/feature_pipeline.pkl"
)

save_object(
    preprocessor,
    "models/catboost/preprocessor.pkl"
)