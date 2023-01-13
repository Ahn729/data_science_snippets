from typing import Optional, Union, Callable, List

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.model_selection._split import _BaseKFold

from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import PolynomialFeatures

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, SGDRegressor, ElasticNet, HuberRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, \
    GradientBoostingRegressor, AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from pyearth import Earth

__all__ = ['ModelComparer', 'ComparisonResult']

MODELS = {
    'dummy_regressor': DummyRegressor(),
    'linear_regression': LinearRegression(),
    'polynomial_regression_deg_2': make_pipeline(PolynomialFeatures(degree=2), LinearRegression()),
    'SGD': SGDRegressor(),
    'elastic_net': ElasticNet(),
    'huber_regressor': HuberRegressor(),
    'MARS': Earth(),
    '3_nearest_neighbours': KNeighborsRegressor(n_neighbors=3),
    '5_nearest_neighbours': KNeighborsRegressor(n_neighbors=5),
    '11_nearest_neighbours': KNeighborsRegressor(n_neighbors=11),
    'decision_tree': DecisionTreeRegressor(),
    'random_forest': RandomForestRegressor(n_estimators=100),
    'extra_trees': ExtraTreesRegressor(n_estimators=100),
    'ada_boost': AdaBoostRegressor(),
    'linear_SVR': SVR(kernel='linear'),
    'polynomial_SVR': SVR(kernel='poly'),
    'rbf_SVR': SVR(),
    'gradient_boost': GradientBoostingRegressor(),
    'xgboost': XGBRegressor(),
    'neural_network': MLPRegressor(hidden_layer_sizes=(20, 20), max_iter=5000)
}


class ComparisonResult:
    """Result of a fitted ModelComparer

    Attributes:
        results: A dataframe of model and result pairs, each one corresponding
            to a single CV iteration
        stats: Descriptive statistics corresponding to the results
    """

    def __init__(self, results, stats):
        """Creates a ComparisonResult instance

        Args:
            results: A dataframe of model and result pairs, each one corresponding
                to a single CV iteration
            stats: Descriptive statistics corresponding to the results
        """
        self.results = results
        self.stats = stats

    def plot_results(self, ax: Optional[plt.Axes] = None, baseline: bool = True) -> plt.Axes:
        """Create a barplot visualising the result

        Args:
            ax: Axis to plot on. If None, a new axis is created.
            baseline: If True, plots a dashed horizontal baseline corresponding
                to the results of the very first model (which defaults a dummy).

        Returns:
            Axis
        """
        ax = ax or plt.subplots()[1]
        chart = sns.barplot(x='Name', y='Result', data=self.results, order=self.stats.Name, ax=ax)
        chart.set_xticklabels(chart.get_xticklabels(), rotation=45, horizontalalignment='right')
        if baseline:
            baseline_height = self.stats.loc[0].Mean
            min_value = self.stats.Mean.min()
            if min_value > 0:
                min_value = 0
            ax.axhline(baseline_height, color="black", linestyle=":")
            ax.set_ylim(min_value, 1.5*baseline_height)
        return chart


class ModelComparer:
    """Performs a k-fold cross validation to compare multiple models"""

    def __init__(self,
                 preprocessor: Optional[TransformerMixin] = None,
                 scorer: Union[str, Callable] = "neg_mean_squared_error",
                 multiply_by_minus_one: bool = True,
                 cv: Union[int, _BaseKFold] = 5,
                 n_jobs: Optional[int] = None,
                 models: Optional[List[BaseEstimator]] = None):
        """Creates a ModelComparer instance

        Args:
            preprocessor: Preprocessor to apply before training the models.
            scorer: Scoring function to use. sklearn models come with a scorer,
                however, to obtain comparable results, it is important to use
                the same scorer for all models. Default: Negative MSE
            multiply_by_minus_one: Whether result statistics shall be multiplied
                by (-1). This gives more meaningful results if the scorer is
                negative of something (like negative MSE)
            cv: Integer depicting the number of cross validations to perform
                or an sklearn KFold instance. Default: 5
            n_jobs: Use n_jobs in the calculation of cross_val_scores
            models: dict of models for cross validation evaluation. Pass None
                to use the built-in set of models
        """
        self.models = models or MODELS
        self.preprocessor = preprocessor
        self.scorer = scorer
        self.cv = cv
        self.multiply_by_minus_one = multiply_by_minus_one
        self.n_jobs = n_jobs

        if isinstance(cv, int):
            self.kfold = KFold(n_splits=cv, shuffle=True, random_state=42)
        else:
            self.kfold = cv

    def fit(self, X_train, y_train) -> ComparisonResult:
        """Fit the comparer on the specified data

        Args:
            X_train, y_train: training set

        Returns:
            A ComparisonResult containing the CV results
        """
        results = pd.DataFrame(columns=['Name', 'Result'])
        result_stats = pd.DataFrame(columns=['Name', 'Mean', 'Std.'])

        if self.multiply_by_minus_one:
            factor = -1
        else:
            factor = 1

        for name, model in self.models.items():
            print(f'Trying {name}.')
            if self.preprocessor is not None:
                regressor = make_pipeline(self.preprocessor, model)
            else:
                regressor = model
            cvs = cross_val_score(regressor,
                                  X_train,
                                  y_train,
                                  cv=self.kfold,
                                  scoring=self.scorer,
                                  n_jobs=self.n_jobs)
            result_stats.loc[len(result_stats)] = [name, factor*cvs.mean(), cvs.std()]
            for score in cvs:
                results.loc[len(results)] = [name, factor*score]
        result_stats = result_stats.sort_values(by='Mean')
        return ComparisonResult(results, result_stats)
