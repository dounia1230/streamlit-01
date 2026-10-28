"""Données synthétiques déterministes : aucun téléchargement pendant les tests."""

import numpy as np
import pandas as pd

from modeling import TARGET


def make_data(size=100):
    rng = np.random.default_rng(42)
    weight = rng.uniform(750, 2100, size)
    horsepower = rng.uniform(50, 200, size)
    years = rng.integers(1970, 1983, size)
    consumption = (
        3 + weight / 300 + horsepower / 90 - (years - 1970) * 0.15
        + rng.normal(0, 0.3, size)
    )
    data = pd.DataFrame({
        "name": [f"Voiture {i}" for i in range(size)],
        "cylinders": rng.choice([3, 4, 6, 8], size),
        "displacement": rng.uniform(80, 400, size),
        "horsepower": horsepower,
        "weight": weight / 0.45359237,
        "weight_kg": weight,
        "acceleration": rng.uniform(9, 24, size),
        "model_year": years,
        "origin": rng.choice(["usa", "europe", "japan"], size),
        "mpg": 235.214583 / consumption,
        TARGET: consumption,
    })
    data.loc[::11, "horsepower"] = np.nan
    return data
