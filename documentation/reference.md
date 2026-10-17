# API guide

The public namespace is
`shape_adaptive_robust_regression`. The table below is a curated entry point;
the [estimator source](https://github.com/byuzbasi/shape-adaptive-robust-regression/blob/main/src/shape_adaptive_robust_regression/estimators.py)
contains the exact signatures and implementation docstrings.

## Primary estimator

```python
ShapeAdaptiveEN(alpha=0.5, c_grid=(1.0, 1.5, 2.0, 3.0, 5.0),
                n_iter=2, n_lam=15, k=5, seed=0,
                pilot=None, kkt_tol=None)
```

Starts from a pilot, estimates a bounded score from standardized pilot residuals,
and fits an elastic net to the resulting pseudo-responses. `alpha` controls
elastic-net mixing; `c_grid` contains score-shape candidates; `n_iter`
is the number of updates; `n_lam` is the penalty-path size and `k` the CV
fold count. `seed` controls the reproducible fold construction.
`pilot` optionally supplies the initial fit. `kkt_tol` adds a final-target
numerical bound when set. Defaults are documented, not changed by this site.

## Shape-penalized estimator

```python
ShapePenalizedEN(lam=0.1, alpha=0.5, lam1=0.0, lam2=0.0,
                 robust=False, c_huber=1.345, c_clip=2.5,
                 scale=None, max_sweeps=1000, tol=1e-9,
                 beta_init=None, eps=1e-6, kkt_tol=None)
```

`lam` and `alpha` set elastic-net regularization. `lam1` and `lam2`
weight residual skewness and excess-kurtosis penalties. `robust=False`
uses least squares and raw centered residual moments; `robust=True` uses
Huber loss and clipped standardized residual moments. `c_huber`,
`c_clip` and `scale` govern that robust configuration. Solver controls
are `max_sweeps`, `tol`, `beta_init`, `eps` and `kkt_tol`.

Both main classes expose `fit(X, y)`, `predict(X)`, `coef_` and
`intercept_`. Keep training and evaluation observations separate.

## Comparison and tuning interfaces

| Object | Role |
|:--|:--|
| `QuantileLasso(lam=0.01, taus=(0.5,))` | LAD Lasso at the median or composite quantile regression Lasso |
| `fit_en_cv` | Elastic net, or Huber elastic net, selected by internal CV |
| `fit_sken_cv` | CV selection of residual-shape weights at the supplied penalty level |
| `fit_quantile_lasso_cv` | CV selection for the quantile-Lasso comparator |
| `diagnostic_summary(model)` | Compact refit/CV KKT and iteration-cap summaries |

Consult the source docstrings for full helper signatures and returned
diagnostic fields. LP-based quantile fits fail explicitly on optimizer failure
and do not expose the coordinate-descent KKT diagnostic.
