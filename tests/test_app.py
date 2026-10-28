from pathlib import Path
import unittest
from unittest.mock import patch

import pandas as pd
import streamlit as st
from streamlit.testing.v1 import AppTest

from modeling import PREDICTION
from tests.fixtures import make_data

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


class AppFlowTests(unittest.TestCase):
    def test_statistics_prediction_filters_and_resubmission(self):
        st.cache_data.clear()
        st.cache_resource.clear()
        raw = make_data()
        raw["model_year"] -= 1900
        with patch("pandas.read_csv", return_value=raw):
            app = AppTest.from_file(str(APP_PATH), default_timeout=60).run()
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(len(app.tabs), 4)
            self.assertEqual(app.slider(key="car_year").min, 1970)
            self.assertEqual(app.slider(key="car_year").max, 1982)
            stats = app.tabs[1].dataframe[0].value
            expected_consumption = 235.214583 / raw["mpg"]
            self.assertAlmostEqual(stats.loc["Consommation (L/100 km)", "Moyenne"], expected_consumption.mean(), places=2)
            self.assertEqual(stats.loc["Puissance (hp)", "Valeurs absentes"], raw["horsepower"].isna().sum())
            app.radio[0].set_value("Spearman").run()
            self.assertEqual(len(app.exception), 0)
            app.button[0].click().run()
            self.assertEqual(len(app.exception), 0)
            prediction = app.tabs[3].dataframe[1].value.copy()
            self.assertEqual(len(prediction), 6)
            self.assertTrue((prediction[PREDICTION] > 0).all())
            self.assertEqual(prediction["Recommandé (validation croisée)"].sum(), 1)
            # Modifier l'exploration ne doit ni modifier les prédictions ni les faire disparaître.
            app.multiselect(key="filter_cylinders").set_value([]).run()
            self.assertEqual(len(app.exception), 0)
            self.assertTrue(any("Aucun véhicule" in w.value for w in app.warning))
            pd.testing.assert_frame_equal(prediction, app.tabs[3].dataframe[1].value)
            app.multiselect(key="filter_cylinders").set_value([4]).run()
            app.selectbox(key="filter_origin").set_value("japan").run()
            self.assertEqual(len(app.exception), 0)
            # Les entrées du formulaire ne remplacent la simulation qu'après validation.
            app.number_input(key="car_weight").set_value(2000.0)
            app.button[0].click().run()
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(app.tabs[3].dataframe[0].value["Poids (kg)"].iloc[0], 2000.0)
            self.assertFalse(prediction[PREDICTION].equals(app.tabs[3].dataframe[1].value[PREDICTION]))
            # Les scores de diagnostic restent consultables pour chaque famille de modèle.
            for model in ["KNN", "Régression Ridge", "Random Forest", "Extra Trees", "Gradient Boosting", "Référence médiane"]:
                app.selectbox(key="diagnostic_model").set_value(model).run()
                self.assertEqual(len(app.exception), 0, model)


if __name__ == "__main__":
    unittest.main()
