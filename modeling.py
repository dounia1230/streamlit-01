"""Entraînement et comparaison reproductibles des modèles Auto MPG."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES = [
    "cylinders", "displacement", "horsepower", "weight_kg", "acceleration", "model_year",
]
CATEGORICAL_FEATURES = ["origin"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "consumption_l_100km"
BASELINE = "Référence médiane"
CV_MAE = "MAE validation croisée (L/100 km)"
CV_STD = "Écart-type validation croisée"
TEST_MAE = "MAE test (L/100 km)"
TEST_RMSE = "RMSE test (L/100 km)"
TEST_R2 = "R² test"
PREDICTION = "Prédiction (L/100 km)"


@dataclass
class ModelComparison:
    models: dict[str, Pipeline]
    scores: pd.DataFrame
    evaluation: pd.DataFrame
    recommended: str
    train_count: int
    test_count: int


def build_models():
    """Chaque pipeline apprend son prétraitement sur ses propres données d'entraînement."""
    estimators = {
        BASELINE: DummyRegressor(strategy="median"),
        "Régression Ridge": Ridge(alpha=1.0),
        "KNN": KNeighborsRegressor(n_neighbors=7, weights="distance"),
        "Random Forest": RandomForestRegressor(
            n_estimators=150, min_samples_leaf=2, random_state=42,
        ),
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=150, min_samples_leaf=2, random_state=42,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=150, learning_rate=0.05, max_depth=2, random_state=42,
        ),
    }
    models = {}
    for name, estimator in estimators.items():
        numeric = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ])
        categorical = Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        preprocess = ColumnTransformer([
            ("numeric", numeric, NUMERIC_FEATURES),
            ("origin", categorical, CATEGORICAL_FEATURES),
        ])
        models[name] = Pipeline([("preprocess", preprocess), ("regressor", estimator)])
    return models


def train_models(data: pd.DataFrame) -> ModelComparison:
    clean = data.dropna(subset=[TARGET])
    # MPG définit directement la cible : ne jamais l'utiliser comme caractéristique.
    X_train, X_test, y_train, y_test = train_test_split(
        clean[FEATURES], clean[TARGET], test_size=0.20, random_state=42,
    )
    folds = KFold(n_splits=5, shuffle=True, random_state=42)
    models = build_models()
    scores = []
    evaluation = pd.DataFrame({"Réel (L/100 km)": y_test})
    for name, model in models.items():
        cv_errors = -cross_val_score(
            model, X_train, y_train, cv=folds,
            scoring="neg_mean_absolute_error", error_score="raise",
        )
        model.fit(X_train, y_train)
        predicted = model.predict(X_test)
        evaluation[name] = predicted
        scores.append({
            "Modèle": name,
            CV_MAE: float(cv_errors.mean()),
            CV_STD: float(cv_errors.std()),
            TEST_MAE: mean_absolute_error(y_test, predicted),
            TEST_RMSE: root_mean_squared_error(y_test, predicted),
            TEST_R2: r2_score(y_test, predicted),
        })
    scores = pd.DataFrame(scores).sort_values(CV_MAE).reset_index(drop=True)
    recommended = scores.loc[scores["Modèle"] != BASELINE, "Modèle"].iloc[0]
    return ModelComparison(
        models=models, scores=scores, evaluation=evaluation,
        recommended=recommended, train_count=len(X_train), test_count=len(X_test),
    )


def compare_predictions(comparison: ModelComparison, car: pd.DataFrame) -> pd.DataFrame:
    """Prédire la même voiture avec tous les modèles déjà évalués."""
    rows = []
    for name, model in comparison.models.items():
        estimate = float(model.predict(car[FEATURES])[0])
        rows.append({
            "Modèle": name,
            PREDICTION: estimate,
            "Équivalent (mpg)": 235.214583 / estimate if estimate > 0 else np.nan,
            "Recommandé (validation croisée)": name == comparison.recommended,
        })
    return comparison.scores.merge(pd.DataFrame(rows), on="Modèle", validate="one_to_one")
