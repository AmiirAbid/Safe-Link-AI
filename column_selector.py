# column_selector.py

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class ColumnSelector(BaseEstimator, TransformerMixin):
    """
    Selects and orders the required raw features before passing them
    into the scaler → PCA → model pipeline.
    """

    def __init__(self, required_columns):
        self.required_columns = list(required_columns)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Accept dict, Series, or DataFrame
        if isinstance(X, dict):
            X = pd.DataFrame([X])
        elif not isinstance(X, pd.DataFrame):
            try:
                X = pd.DataFrame(X)
            except Exception:
                raise ValueError("ColumnSelector expects a DataFrame or dict.")

        # Validate missing fields
        missing = [c for c in self.required_columns if c not in X.columns]
        if missing:
            raise ValueError(f"Missing raw feature(s): {missing}")

        # Return columns in correct order
        return X.loc[:, self.required_columns].values

    def get_feature_names_out(self, input_features=None):
        return np.array(self.required_columns)
