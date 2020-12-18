from itertools import cycle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler

def obtain_uncorrelated_variables(data, correlation_threshold=0.5):
    """Computer pairwise correlation coefficients and determines uncorrelated vartiables

    Recursicely selects features from the dataset so that the resulting
    dataset does not have features with pairwise correlation larger than
    specified correlation_threshold.

    Args:
         data: Dataframe
         correlation_threshold: Features will be excluded if pairwise correlation
            exceeds this value
    Returns:
        List of all uncorrelated variables

    """
    indep_variables = list()
    for column_name in data.select_dtypes('number').columns:
        variables = indep_variables + [column_name]
        corrs = data[variables].corr()[column_name]
        if (abs(corrs) > correlation_threshold).sum() <= 1:
            indep_variables.append(column_name)
    return indep_variables


def elasticnet_plot(X, y, l1_ratio=.5, log_alpha_min=-4, log_alpha_max=1, alpha_step=50, ax=None, **kwargs):
    """Plots ElasticNet regression coefficients against Lagrange multiplier


    Applies ElasticNet regression with different Lagrange multipliers alpha
    and plots the resulting coefficients against them yielding information on
    importance and influence of the vartiables.

    Args:
        X: Dataframe of independent variables
        y: dependent variable
        l1_ratio: Ratio between l1 and l2 norm in ElasticNet regression.
            Use 1 for Lasso, 0 for Ridge regression
        log_alpha_min, log_alpha_max: logarithmic min and max value for
            Lagrange multiplier alpha
        alpha_step: Number of regressions to fit
        ax: Axis to plot on
        kwargs: Keyword arguments to pass to elasticnet regressor
    Returns:
        Axis
    """
    ax = ax or plt.subplots()[1]

    alphas = np.linspace(log_alpha_min, log_alpha_max, alpha_step)
    var_names = X.columns
    sc = StandardScaler()
    X_sc = sc.fit_transform(X, y)
    y_sc = (y - y.mean()) / y.std()
    results = np.empty((len(alphas), len(var_names)))
    for i, alpha in enumerate(alphas):
        regressor = ElasticNet(fit_intercept=False, alpha=10**alpha, l1_ratio=l1_ratio, **kwargs)
        regressor.fit(X_sc, y_sc)
        results[i] = regressor.coef_
        _vars = list()

    # For improved readability, linestyle is changed after color palette is exhausted
    color_palette_length = len(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    line_styles = ["-", "--", "-.", ":"]
    line_style_cycler = cycle(line_styles)

    for i, var in enumerate(var_names):
        if i % color_palette_length == 0:
            linestyle = next(line_style_cycler)
        ax.plot(alphas, results[:, i], linestyle=linestyle)
        var = var if var[0] != '_' else var[1:]
        _vars.append(var)
    ax.set_xlim(log_alpha_max, log_alpha_min)
    ax.legend(_vars)
    ax.set_title('ElasticNet coefficient plot')
    ax.set_xlabel('log(alpha)')
    ax.set_ylabel('Coefficient')
    return ax
