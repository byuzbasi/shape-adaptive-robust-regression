# Data and reproducibility

## Input dictionary

| Input | Interpretation |
|:--|:--|
| `X` | Numeric two-dimensional array: observations in rows, predictors in columns |
| `y` | Numeric response vector matching the rows of `X` |
| New `X` | Same predictor columns and order as training |
| `seed` | Fold-construction seed in the relevant CV helper/estimator |

The estimator source implements its documented internal transformations.
This presentation adds no preprocessing or automatic data cleaning.

## Data sources and access

The repository contains no application datasets. The study cites these original
providers; using the software does not transfer ownership of their data.

| Application data | Original provider |
|:--|:--|
| SLID | R `carData` package |
| NOx | R `robustbase` package |
| QSAR fish toxicity | UCI Machine Learning Repository |
| Concrete compressive strength | UCI Machine Learning Repository |

The paper reports the analysis protocols. Application workflows and study
materials can be requested from the authors for research verification.
This site does not claim that installing the package reproduces all paper results.

## Reproducible installation check

From a reviewed checkout and a Python 3.13 environment with the package installed:

```sh
python -m unittest discover -s tests -v
python -m pip freeze
```

The existing test uses a small synthetic sample and checks finite predictions
for the two principal classes. Record the source revision, Python version,
dependency versions, configuration and seeds with any analysis. Neither this
test nor a successful site build validates a production simulation study.

## Documentation environment

Documentation uses MkDocs/Material and a separate `requirements-docs.txt`.
It does not import the estimator or execute fitting during site generation.

```sh
python3.13 -m venv .venv-docs
. .venv-docs/bin/activate
python -m pip install -r requirements-docs.txt
mkdocs build --strict
```

The output is static HTML. Publication is a separate maintainer action.
