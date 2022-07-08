from typing import Optional

from sklearn.preprocessing import StandardScaler

from ..data_exploration.reduce_dataset import reduce_dataset
from ..data_exploration.detect_outliers import recursive_outlier_detection
from ..data_exploration.feature_selection import obtain_independent_variables

from ..model_selection.cv_model_comparison import ModelComparer, ComparisonResult

__all__ = ['AutoMLModelComparer']


class AutoMLModelComparer:
    """Performs a k-fold cross validation to compare multiple models

    Reduces dataset, removes outliers, identifies independent features, then
    performs a k-fold cross validation for multiple models in order to obtain
    a visual representation of their fit to the data.
    """

    def __init__(self, data):
        self.data = data

        self._reduced_data, self._outliers = None, None
        self._ind_vars, self._result = None, None

    @property
    def reduced_data(self):
        return self._reduced_data

    @property
    def outliers(self):
        return self._outliers

    @property
    def ind_vars(self):
        return self._ind_vars

    @property
    def result(self):
        return self._result

    def create_comparison_result(self,
                                 target: str,
                                 nan_handler_kwargs: Optional[dict] = None,
                                 outlier_detection_kwargs: Optional[dict] = None,
                                 independent_variables_kwargs: Optional[dict] = None,
                                 model_comparer_kwargs: Optional[dict] = None) -> ComparisonResult:
        """Creates the autoML-CV-Comparison result

        First we preprocess the data using data reduction and outlier detection. Then,
        we compute independent variables to reduce dataset complexity. In the end, data
        is scaled and fed into the models.

        Args:
            target: Target column ("y")
            nan_handler_kwargs: Kwargs passed down to nan handler
            outlier_detection_kwargs: Kwargs passed down to outlier detection method
            independent_variables_kwargs: Kwargs passed down to obtain_independent_variables
            model_comparer_kwargs: Kwargs passed down to ModelComparer __init__.

        Returns:

        """
        nan_handler_kwargs = nan_handler_kwargs or dict()
        outlier_detection_kwargs = outlier_detection_kwargs or dict()
        independent_variables_kwargs = independent_variables_kwargs or dict()
        model_comparer_kwargs = model_comparer_kwargs or dict()

        self._reduced_data = reduce_dataset(self.data, remove_nans=True, **nan_handler_kwargs)
        self._outliers = recursive_outlier_detection(self.reduced_data, **outlier_detection_kwargs)
        print(f"Removed {len(self._outliers)} outliers.")
        df = self.reduced_data.drop(self.outliers.index)
        X, y = df.drop(columns=target), df[target]

        if 'method' in independent_variables_kwargs and independent_variables_kwargs['method'] in [
                'ri', 'recursive_inclusion',
                'loo', 'leave_one_out'
            ]:
            independent_variables_kwargs['y'] = y

        self._ind_vars = obtain_independent_variables(X, **independent_variables_kwargs)
        print(f"Selected {len(self._ind_vars)} independent variables (from {len(X.columns)}):")
        print(f"{', '.join(self._ind_vars)}")
        X = X[self.ind_vars]

        if 'preprocessor' not in model_comparer_kwargs:
            model_comparer_kwargs['preprocessor'] = StandardScaler()

        self._result = ModelComparer(**model_comparer_kwargs).fit(X, y)
        return self.result
