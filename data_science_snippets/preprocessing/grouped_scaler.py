__all__ = ["GroupedScaler"]

from typing import Optional, Union, Literal

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler


# See https://stackoverflow.com/questions/68356000/how-to-standardize-scikit-learn-by-group
class GroupedScaler(BaseEstimator, TransformerMixin):
    def __init__(self,
                 by: Optional[str] = None,
                 copy: bool = True,
                 output_grouping_var: bool = True,
                 handle_unknown: Literal["raise", "ignore"] = "ignore"):
        self.copy = copy
        self.cols = None
        self.scalers = {}
        self.by = by
        self.output_grouping_var = output_grouping_var
        self.handle_unknown = handle_unknown

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y=None):
        self.scalers = {}
        self.cols = X.select_dtypes(exclude=['object']).columns
        for val in X[self.by].unique():
            mask = X[self.by] == val
            X_sub = X.loc[mask, self.cols]
            self.scalers[val] = StandardScaler().fit(X_sub)
        return self

    def transform(self, X: Union[np.ndarray, pd.DataFrame], y=None):
        if self.copy:
            X = X.copy()
        for val in X[self.by].unique():
            mask = X[self.by] == val
            try:
                scaler = self.scalers[val]
                X.loc[mask, self.cols] = scaler.transform(X.loc[mask, self.cols])
            except KeyError as e:
                # This happens when the category is new, so we
                # don't know how to scale.
                if self.handle_unknown == "ignore":
                    X.loc[mask, self.cols] = 0
                else:
                    raise e from None

        if not self.output_grouping_var:
            return X.drop(columns=self.by)
        return X

    def get_feature_names(self):
        if not self.output_grouping_var:
            return self.cols
        else:
            return [self.by, *self.cols]

    def get_feature_names_out(self, input_features=None):
        feature_names = input_features if input_features is not None else self.cols
        if self.output_grouping_var:
            return feature_names
        else:
            return [feat for feat in feature_names if feat != self.by]


