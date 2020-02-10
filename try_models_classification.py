"""Train different classification models with standard parameters and print the result"""

from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, \
    GradientBoostingClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier


models = {
    'Dummy_MostFrequent': DummyClassifier(strategy='most_frequent'),
    'LogisticRegression': LogisticRegression(),
    'DecisionTree': DecisionTreeClassifier(),
    'RandomForest': RandomForestClassifier(n_estimators=100),
    'ExtraTrees': ExtraTreesClassifier(n_estimators=100),
    'AdaBoost': AdaBoostClassifier(),
    'GaussianNB': GaussianNB(),
    'MultinomialNB': MultinomialNB(),
    'BernoulliNB': BernoulliNB(),
    '5-nn': KNeighborsClassifier(),
    'SVC_lin': SVC(kernel='linear'),
    'SVC_poly': SVC(kernel='poly'),
    'SVC_rbf': SVC(),
    'GradientBoost': GradientBoostingClassifier(),
    'xgboost': XGBClassifier(),
    'NeuralNetwork': MLPClassifier(hidden_layer_sizes=(20, 20), max_iter=5000)
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
    result_stats.loc[len(result_stats)] = [name, cvs.mean(), cvs.std()]
    for score in cvs:
        results.loc[len(results)] = [name, score]
result_stats = result_stats.sort_values(by='Mean')

plt.figure(figsize=(10, 6))
chart = sns.barplot(x='Name', y='Result', data=results, order=result_stats.Name)
chart.set_xticklabels(chart.get_xticklabels(), rotation=45, horizontalalignment='right')
