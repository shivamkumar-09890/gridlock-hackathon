from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_squared_error,
    r2_score,
)

from sklearn.model_selection import KFold

from src.data.load_data import load_train
from src.data.preprocessing import Preprocessor

from features.catboost.feature_pipeline import (
    FeaturePipeline
)

from src.models.catboost_model import (
    CatBoostDemandModel
)


TARGET = "demand"

MODEL_DIR = Path(
    "models/catboost"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)
train_df = load_train(
    "data/raw/train.csv"
)

preprocessor = Preprocessor()

preprocessor.fit(train_df)

train_df = preprocessor.transform(
    train_df
)

pipeline = FeaturePipeline()

pipeline.fit(train_df)

train_df = pipeline.transform(
    train_df
)

DROP_COLS = [
    "Index",
    "timestamp",
    TARGET,
]

FEATURE_COLS = [
    c
    for c in train_df.columns
    if c not in DROP_COLS
]

CAT_COLS = [
    col
    for col in [
        "geohash",
        "RoadType",
        "LargeVehicles",
        "Landmarks",
        "Weather",
        "geo_hour",
        "period",
        "geo_period",
        "roadtype_period",
        "temp_bucket",
        "weather_temp",
        "geo_weather",
        "geo_road",
        "geo_temp",
    ]
    if col in FEATURE_COLS
]

kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

fold_scores = []

best_model = None
best_rmse = np.inf


for fold, (train_idx, valid_idx) in enumerate(
    kf.split(train_df),
    start=1
):

    print("\n" + "=" * 60)
    print(f"Fold {fold}")
    print("=" * 60)

    train_fold = (
        train_df.iloc[train_idx]
        .copy()
    )

    valid_fold = (
        train_df.iloc[valid_idx]
        .copy()
    )

    X_train = train_fold[
        FEATURE_COLS
    ]

    y_train = train_fold[
        TARGET
    ]

    X_valid = valid_fold[
        FEATURE_COLS
    ]

    y_valid = valid_fold[
        TARGET
    ]

    model = (
        CatBoostDemandModel()
        .build()
    )

    model.fit(
        X_train,
        y_train,
        cat_features=CAT_COLS,
        eval_set=(
            X_valid,
            y_valid,
        ),
        use_best_model=True,
    )

    preds = model.predict(
        X_valid
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_valid,
            preds,
        )
    )

    r2 = r2_score(
        y_valid,
        preds,
    )

    fold_scores.append(
        rmse
    )

    print(
        f"RMSE : {rmse:.6f}"
    )

    print(
        f"R²   : {r2:.6f}"
    )

    if rmse < best_rmse:

        best_rmse = rmse

        best_model = model

print("\n")

print(
    f"Mean RMSE : {np.mean(fold_scores):.6f}"
)

print(
    f"Std RMSE  : {np.std(fold_scores):.6f}"
)

print(
    f"Best RMSE : {best_rmse:.6f}"
)

X_full = train_df[
    FEATURE_COLS
]

y_full = train_df[
    TARGET
]

train_preds = best_model.predict(
    X_full
)

train_r2 = r2_score(
    y_full,
    train_preds
)

print(
    f"\nTraining R² : {train_r2:.6f}"
)

feature_importance = pd.DataFrame(
    {
        "feature": FEATURE_COLS,
        "importance": (
            best_model
            .get_feature_importance()
        ),
    }
)

feature_importance = (
    feature_importance
    .sort_values(
        "importance",
        ascending=False,
    )
    .reset_index(
        drop=True
    )
)

feature_importance.to_csv(
    MODEL_DIR /
    "feature_importance.csv",
    index=False,
)

print(
    feature_importance.head(20)
)

joblib.dump(
    best_model,
    MODEL_DIR /
    "best_catboost.pkl",
)

joblib.dump(
    pipeline,
    MODEL_DIR /
    "feature_pipeline.pkl",
)

joblib.dump(
    preprocessor,
    MODEL_DIR /
    "preprocessor.pkl",
)

joblib.dump(
    FEATURE_COLS,
    MODEL_DIR /
    "feature_cols.pkl",
)

joblib.dump(
    CAT_COLS,
    MODEL_DIR /
    "cat_cols.pkl",
)