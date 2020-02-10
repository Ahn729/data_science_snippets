"""Train different regression models with standard parameters and print the result"""

from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, \
    GradientBoostingRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor


models = {
    'linear': LinearRegression(),
    'SGD': SGDRegressor(),
    'DecisionTree': DecisionTreeRegressor(),
    'RandomForest': RandomForestRegressor(n_estimators=100),
    'ExtraTrees': ExtraTreesRegressor(n_estimators=100),
    'AdaBoost': AdaBoostRegressor(),
    'SVR_lin': SVR(kernel='linear'),
    'SVR_poly': SVR(kernel='poly'),
    'SVR_rbf': SVR(),
    'GradientBoost': GradientBoostingRegressor(),
    'xgboost': XGBRegressor(),
    'NeuralNetwork': MLPRegressor(hidden_layer_sizes=(20, 20), max_iter=5000)
}

# Define train set:
X_train, y_train = ...
# Your preprocessor goes here:
preprocessor = ...
# sklearn models come with a scorer, however, to obtain comparable results,
# it is important to use the same scorer for all models. Define yours here:
scorer = ...


results = pd.DataFrame(columns=['Name', 'Result'])
result_stats = pd.DataFrame(columns=['Name', 'Mean', 'Std.'])
for name, model in models.items():
    print(f'Trying {name}.')
    regressor = make_pipeline(preprocessor, model)
    cvs = cross_val_score(regressor, X_train, y_train, cv=5, scoring=scorer)
    result_stats.loc[len(result_stats)] = [name, (-1)*cvs.mean(), cvs.std()]
    for score in cvs:
        results.loc[len(results)] = [name, (-1)*score]
result_stats = result_stats.sort_values(by='Mean')

plt.figure(figsize=(10, 6))
chart = sns.barplot(x='Name', y='Result', data=results, order=result_stats.Name)
chart.set_xticklabels(chart.get_xticklabels(), rotation=45, horizontalalignment='right')
