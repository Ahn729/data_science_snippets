import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

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

    def plot_results(self, ax=None, baseline=True) -> plt.Axes:
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
            ax.axhline(baseline_height, color="black", linestyle=":")
            ax.set_ylim(0, 1.5*baseline_height)
        return chart


class ModelComparer:
    """Performs a k-fold cross validation to compare multiple models"""

    def __init__(self, preprocessor=None, scorer="neg_mean_squared_error", models=None):
        """Creates a ModelComparer instance

        Args:
            preprocessor: Preprocessor to apply before training the models.
            scorer: Scoring function to use. sklearn models come with a scorer,
                however, to obtain comparable results, it is important to use
                the same scorer for all models. Default: Negative MSE
            models: dict of models for cross validation evaluation. Pass None
                to use the built-in set of models
        """
        self.models = models or MODELS
        self.preprocessor = preprocessor
        self.scorer = scorer

    def fit(self, X_train, y_train, cv=5) -> ComparisonResult:
        """Fit the comparer on the specified data

        Args:
            X_train, y_train: training set
            cv: Number of cross validation iterations to perform. Default: 5

        Returns:
            A ComparisonResult containing the CV results
        """
        results = pd.DataFrame(columns=['Name', 'Result'])
        result_stats = pd.DataFrame(columns=['Name', 'Mean', 'Std.'])

        kfold = KFold(n_splits=cv, shuffle=True, random_state=42)

        for name, model in self.models.items():
            print(f'Trying {name}.')
            regressor = make_pipeline(self.preprocessor, model)
            cvs = cross_val_score(regressor, X_train, y_train, cv=kfold, scoring=self.scorer)
            result_stats.loc[len(result_stats)] = [name, (-1)*cvs.mean(), cvs.std()]
            for score in cvs:
                results.loc[len(results)] = [name, (-1)*score]
        result_stats = result_stats.sort_values(by='Mean')
        return ComparisonResult(results, result_stats)
