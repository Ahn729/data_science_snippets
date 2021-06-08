from sklearn.preprocessing import StandardScaler

from data_exploration.reduce_dataset import reduce_dataset
from data_exploration.detect_outliers import recursive_outlier_detection
from data_exploration.feature_selection import obtain_independent_variables

from model_selection.cv_model_comparison import ModelComparer

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


    def create_comparison_result(self, target):
        self._reduced_data = reduce_dataset(self.data)
        self._outliers = recursive_outlier_detection(self.reduced_data)
        df = self.reduced_data.drop(self.outliers.index)
        X, y = df.drop(columns=target), df[target]
        self._ind_vars = obtain_independent_variables(X)
        X = X[self.ind_vars]
        self._result = ModelComparer(StandardScaler()).fit(X, y)
        return self.result
