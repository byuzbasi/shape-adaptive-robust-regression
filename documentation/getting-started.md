# Get started

## Install in an isolated Python 3.13 environment

```sh
python3.13 -m venv .venv
. .venv/bin/activate
python -m pip install 'shape-adaptive-robust-regression @ git+https://github.com/byuzbasi/shape-adaptive-robust-regression.git'
```

The distribution name uses hyphens; the import name uses underscores.
The numerical requirements are NumPy 2.2.6, Numba 0.61.2 and SciPy 1.16.3.
These versions are recorded in the package, rather than substituted by the
documentation environment. No PyPI installation is claimed.

## A small interface demonstration

This is the README's artificial 40-observation, three-predictor example.
It checks how to fit and predict; it is not a study of statistical performance.

```python
import numpy as np
from shape_adaptive_robust_regression import ShapeAdaptiveEN

rng = np.random.default_rng(42)
X = rng.normal(size=(40, 3))
y = 1 + X[:, 0] + rng.standard_t(3, size=40)
model = ShapeAdaptiveEN(n_iter=1, n_lam=3, k=3, seed=42).fit(X, y)
predictions = model.predict(X[:2])
print(predictions)
```

`X` is an observations-by-predictors numeric matrix; `y` is a matching
numeric response vector. `predict()` returns one value per new row, using
the fitted coefficients and intercept on the original response scale.

The deliberately small `n_iter`, `n_lam` and `k` are demonstration settings,
not recommendations for an empirical analysis. A training-data CV choice
does not replace held-out predictive evaluation.

## Inspect numerical information

```python
from shape_adaptive_robust_regression import diagnostic_summary
print(diagnostic_summary(model))
```

Iteration-cap and KKT summaries describe numerical behavior; they do not prove
predictive superiority. Keep a record of your call, seed and package versions.
For the exact installation smoke test, see
[tests/test_install.py](https://github.com/byuzbasi/shape-adaptive-robust-regression/blob/main/tests/test_install.py).
