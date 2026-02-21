from __future__ import annotations

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_numeric_preprocess() -> Pipeline:
    """Prétraitement minimal (baseline):

    - imputation médiane
    - standardisation

    (Dans la partie Git feature/, vous pourrez enrichir ce pipeline.)
    """

    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
