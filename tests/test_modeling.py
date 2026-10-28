import unittest

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from modeling import (
    BASELINE, CV_MAE, FEATURES, PREDICTION, TARGET, TEST_MAE, TEST_R2, TEST_RMSE,
    compare_predictions, train_models,
)
from tests.fixtures import make_data


class ModelComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = make_data()
        cls.comparison = train_models(cls.data)

    def test_test_metrics_use_same_unseen_rows_for_every_model(self):
        train, test = train_test_split(self.data, test_size=0.2, random_state=42)
        result = self.comparison
        self.assertEqual(result.train_count, len(train))
        self.assertEqual(result.test_count, len(test))
        self.assertEqual(set(result.evaluation.index), set(test.index))
        self.assertEqual(len(result.models), 6)
        for _, score in result.scores.iterrows():
            name = score["Modèle"]
            predicted = result.models[name].predict(test[FEATURES])
            np.testing.assert_allclose(predicted, result.evaluation[name])
            errors = test[TARGET].to_numpy() - predicted
            self.assertAlmostEqual(score[TEST_MAE], np.abs(errors).mean())
            self.assertAlmostEqual(score[TEST_RMSE], np.sqrt((errors ** 2).mean()))
            self.assertAlmostEqual(
                score[TEST_R2], 1 - (errors ** 2).sum() / ((test[TARGET] - test[TARGET].mean()) ** 2).sum(),
            )
        np.testing.assert_allclose(result.evaluation[BASELINE], train[TARGET].median())
        imputer = result.models["KNN"].named_steps["preprocess"].named_transformers_["numeric"].named_steps["impute"]
        np.testing.assert_allclose(imputer.statistics_, train[FEATURES[:-1]].median().to_numpy())

    def test_holdout_changes_cannot_influence_training_or_recommendation(self):
        changed = self.data.copy()
        _, test = train_test_split(changed, test_size=0.2, random_state=42)
        changed.loc[test.index, TARGET] += 1000
        changed.loc[test.index, "horsepower"] = 1e6
        retrained = train_models(changed)
        pd.testing.assert_series_equal(self.comparison.scores[CV_MAE], retrained.scores[CV_MAE])
        self.assertEqual(self.comparison.recommended, retrained.recommended)
        self.assertTrue((retrained.scores[TEST_MAE] > self.comparison.scores[TEST_MAE]).all())
        original = compare_predictions(self.comparison, self.data.iloc[[1]])
        after = compare_predictions(retrained, self.data.iloc[[1]])
        np.testing.assert_allclose(original[PREDICTION], after[PREDICTION])

    def test_prediction_matches_each_model_and_preserves_metrics(self):
        car = self.data.iloc[[0]].copy()
        car.loc[:, "origin"] = "nouvelle origine"
        result = compare_predictions(self.comparison, car[list(reversed(car.columns))])
        self.assertEqual(len(result), 6)
        self.assertEqual(result["Recommandé (validation croisée)"].sum(), 1)
        self.assertTrue(np.isfinite(result[PREDICTION]).all())
        for _, row in result.iterrows():
            self.assertAlmostEqual(row[PREDICTION], self.comparison.models[row["Modèle"]].predict(car[FEATURES])[0])
            self.assertAlmostEqual(row["Équivalent (mpg)"], 235.214583 / row[PREDICTION])
        pd.testing.assert_series_equal(result[TEST_MAE], self.comparison.scores[TEST_MAE])


if __name__ == "__main__":
    unittest.main()
