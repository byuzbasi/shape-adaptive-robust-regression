"""Small public API check; does not run a paper study."""

import unittest

import numpy as np

from shape_adaptive_robust_regression import ShapeAdaptiveEN, ShapePenalizedEN


class InstallSmokeTest(unittest.TestCase):
    def test_fitted_estimators_predict(self):
        rng = np.random.default_rng(42)
        X = rng.normal(size=(36, 3))
        y = 1.0 + X[:, 0] - 0.5 * X[:, 1] + rng.standard_t(3, size=36)
        for model in (
            ShapePenalizedEN(lam=0.1, max_sweeps=200),
            ShapeAdaptiveEN(n_iter=1, n_lam=3, k=3, seed=42),
        ):
            prediction = model.fit(X, y).predict(X[:2])
            self.assertEqual(prediction.shape, (2,))
            self.assertTrue(np.isfinite(prediction).all())


if __name__ == "__main__":
    unittest.main()
