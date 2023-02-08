__all__ = ['obtain_independent_variables', 'obtain_independent_variables_vif',
           'obtain_important_variables_loo',
           'obtain_uncorrelated_variables', 'predictor_importance_loo', 'elasticnet_plot']

from itertools import cycle
from typing import List, Optional, Any, Dict

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import t
from sklearn.base import RegressorMixin
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.model_selection import cross_val_score, RepeatedKFold
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools import add_constant
from tqdm import tqdm

kfold = RepeatedKFold(n_splits=10, n_repeats=10, random_state=42)


def _corrected_std(differences, n_train, n_test):
    """Corrects standard deviation using Nadeau and Bengio's approach.

    Ref: https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_stats
    .html#sphx-glr-auto-examples-model-selection-plot-grid-search-stats-py"""

    kr = len(differences)
    corrected_var = np.var(differences, ddof=1) * (1 / kr + n_test / n_train)
    corrected_std = np.sqrt(corrected_var)
    return corrected_std


def _compute_corrected_ttest(differences, deg_f, n_train, n_test):
    """Computes right-tailed paired t-test with corrected variance.

    Ref: https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_stats
    .html#sphx-glr-auto-examples-model-selection-plot-grid-search-stats-py
    """
    mean = np.mean(differences)
    std = _corrected_std(differences, n_train, n_test)
    t_stat = mean / std
    p_val = t.sf(np.abs(t_stat), deg_f)  # right-tailed t-test
    return t_stat, p_val


def _print_if_verbose(message, verbose):
    if verbose:
        print(message)


def obtain_independent_variables(data: pd.DataFrame,
                                 method: str = 'variance_inflation_factor',
                                 threshold: Optional[float] = None,
                                 **kwargs) -> List[str]:
    """Obtains independent variables recursively

    Recursively selects features from the dataset so that the resulting
    dataset does only consist of independent features. Independence is defined
    by method which is passed as a parameter. Note that reasonable values
    for threshold parameter strongly depend on the method used.

    Args:
         data: Dataframe
         method: Method used to measure independence. Currently, supports computing
            "variance_inflation_factor", "leave_one_out", "recursive_inclusion" or "correlation"
         threshold: Threshold used to flag features as dependent (in some way) if
            value of method exceeds this value. Default: 0.5 for "correlation",
            1 for "leave_one_out", 0.05 for "recursive_inclusion" and 5 for "variance_inflation_factor"
        kwargs: Only used for "leave_one_out" and "recursive_inclusion". In that case please specify
         the dependent variable "y=..." as well as the model to use "model=..."

    Returns:
        List of all independent variables
    """
    if threshold == 0:
        threshold = np.finfo(float).eps
    if method in ['variance_inflation_factor', 'vif']:
        threshold = threshold or 5
        return obtain_independent_variables_vif(data, threshold)
    if method == 'correlation':
        threshold = threshold or 0.5
        return obtain_uncorrelated_variables(data, correlation_threshold=threshold)
    if method in ['leave_one_out', 'loo']:
        threshold = threshold or 1
        return obtain_important_variables_loo(X=data, threshold=threshold, **kwargs)
    if method in ['recursive_inclusion', 'ri']:
        threshold = threshold or 0.05
        return obtain_important_variables_ri(X=data, significance=threshold, **kwargs)
    raise ValueError("""Method not understood. Please specify either
        'variance_inflation_factor', 'leave_one_out', 'recursive_inclusion' or 'correlation'.""")


def obtain_independent_variables_vif(data: pd.DataFrame,
                                     threshold: float = 5) -> List[str]:
    """Obtains non-multicollinear variables by recursively computing variance_inflation_factors

    Recursively selects features from the dataset so that the resulting
    dataset does not have features with variance inflation factor larger than
    specified threshold. Note that VIF(x) = \frac{1}{1 - R(x)^2}, where R(x)^2
    is the coefficient of determination computed by regressing the feature x
    onto the other features.

    Args:
         data: Dataframe
         threshold: Features will be excluded if VIF exceeds this value
    Returns:
        List of all independent variables
    """
    indep_variables = list()
    for column_name in data.select_dtypes('number').columns:
        variables = [column_name] + indep_variables
        exog = add_constant(data[variables]).values
        vif = variance_inflation_factor(exog, 1)
        if vif <= threshold:
            indep_variables.append(column_name)
    return indep_variables


def obtain_uncorrelated_variables(data: pd.DataFrame,
                                  correlation_threshold: float = 0.5) -> List[str]:
    """Computes pairwise correlation coefficients and determines uncorrelated variables

    Recursively selects features from the dataset so that the resulting
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


def obtain_important_variables_loo(X: pd.DataFrame,
                                   y: np.ndarray,
                                   threshold: float = 1,
                                   model: RegressorMixin = LinearRegression(),
                                   cvs_kwargs: Optional[Dict[str, Any]] = None,
                                   verbose=False) -> List[str]:
    """Performs a leave-one-out ("loo") analysis to determine important features

    Recursively drops features which the model supplied deems as irrelevant (to be precise,
    the score loss in percent must be less than threshold) until no feature is irrelevant
    enough to be omitted.

    Args:
        X: Data frame containing independent features.
        y: Array containing the dependent feature
        threshold: Minimum percentage that a feature must contribute to model score
            to be contained in the final set
        model: Model to use for the analysis. Note that the results may heavily depend
            on the model chosen.
        cvs_kwargs: Keyword arguments passed down to cross_val_score method
        verbose: Print progress

    Returns:

    """
    if cvs_kwargs is None:
        cvs_kwargs = dict()

    if "cv" not in cvs_kwargs.keys():
        cvs_kwargs["cv"] = kfold

    _print_if_verbose(f"CV score start: {cross_val_score(model, X, y, **cvs_kwargs).mean()}.",
                      verbose)
    while True:
        loo_scores = predictor_importance_loo(X, y, model, verbose=verbose, cvs_kwargs=cvs_kwargs)
        lowest_score = loo_scores.loo_score.iat[-1]
        new_cv_score = loo_scores.new_cv_score.iat[-1]
        least_important_feature = loo_scores.feature.iat[-1]
        if lowest_score > threshold:
            _print_if_verbose(
                f"Not removing feature {least_important_feature} with score {lowest_score} (CV score: {new_cv_score}).",
                verbose)
            _print_if_verbose("Scores last iteration:", verbose)
            _print_if_verbose(loo_scores, verbose)
            break
        _print_if_verbose(
            f"Removed feature {least_important_feature} with score {lowest_score}. New CV score: {new_cv_score}.",
            verbose)
        X = X.drop(columns=least_important_feature)
    _print_if_verbose(f"CV score end: {cross_val_score(model, X, y, **cvs_kwargs).mean()}.",
                      verbose)
    return list(X.columns)


def predictor_importance_loo(X: pd.DataFrame,
                             y: np.ndarray,
                             model: RegressorMixin,
                             cvs_kwargs: Optional[Dict[str, Any]] = None,
                             verbose: bool = True) -> pd.DataFrame:
    """Performs a leave-one-out ("loo") analysis to determine important features

    Iterates over all features (column names in X) and determines the scores (R^2-values)
    of the model fitted on the data without the feature. Features will then be ranked
    according to the score loss that removing this feature implied, relative to the
    score of the model on the full dataset. Scores are computed using k-fold cross-validation
    The result dataset contains values equal to the percentage decrease in score, i.e.
            100*(<score of model with feature> / <score of model without feature> -1)
    (bigger is better).

    Args:
        X: Data frame containing independent features.
        y: Array containing the dependent feature
        model: Model to use for the analysis. Note that the results may heavily depend
            on the model chosen.
        cvs_kwargs: Keyword arguments passed down to cross_val_score method
        verbose: print progress bar

    Returns:
        Data frame containing features and score quotients for both train and test set
    """
    results = {
        "feature": [],
        "loo_score": [],
        "new_cv_score": [],
    }
    if cvs_kwargs is None:
        cvs_kwargs = dict()

    if "cv" not in cvs_kwargs.keys():
        cvs_kwargs["cv"] = kfold

    score_with_feature = cross_val_score(model, X, y, **cvs_kwargs).mean()

    iter_set = tqdm(X.columns) if verbose else X.columns

    for feature_name in iter_set:
        X_wo_feature = X.drop(columns=[feature_name])
        score_without_feature = cross_val_score(model, X_wo_feature, y, **cvs_kwargs).mean()

        if score_without_feature > 0:
            loo_score = 100 * (score_with_feature / score_without_feature - 1)
        else:
            loo_score = np.inf

        results["feature"].append(feature_name)
        results["loo_score"].append(loo_score)
        results["new_cv_score"].append(score_without_feature)
    return pd.DataFrame(data=results).sort_values(by="loo_score", ascending=False)


def obtain_important_variables_ri(X: pd.DataFrame,
                                  y: np.ndarray,
                                  significance: float = 0.05,
                                  model: RegressorMixin = LinearRegression(),
                                  cvs_kwargs: Optional[Dict[str, Any]] = None,
                                  verbose=False) -> List[str]:
    """Performs a recursive inclusion analysis to determine important features

    Recursively includes features which the model supplied deems as most relevant (to be precise,
    the absolute score increase must be more than threshold) until no feature is relevant
    enough to be included.

    Args:
        X: Data frame containing independent features.
        y: Array containing the dependent feature
        significance: Significance (p-value) to accept that a new model is
            better than the old one.
        model: Model to use for the analysis. Note that the results may heavily depend
            on the model chosen.
        cvs_kwargs: Keyword arguments passed down to cross_val_score method
        verbose: Print progress

    Returns:

    """
    if cvs_kwargs is None:
        cvs_kwargs = dict()

    if "cv" not in cvs_kwargs.keys():
        cvs_kwargs["cv"] = kfold

    X_base = X.drop(columns=X.columns)
    candidates = X.copy()
    while len(candidates.columns) > 0:
        incl_scored = predictor_importance_incl(X_base, candidates, y, model, cvs_kwargs,
                                                verbose=verbose)
        highest_score = incl_scored.bonf_p.iat[0]
        new_cv_score = incl_scored.new_cv_score.iat[0]
        most_important_feature = incl_scored.feature.iat[0]
        if highest_score > significance:
            _print_if_verbose(
                f"Not including feature {most_important_feature} with significance {highest_score} (CV score: {new_cv_score}). ",
                verbose)
            _print_if_verbose("Scores last iteration:", verbose)
            _print_if_verbose(incl_scored, verbose)
            break
        _print_if_verbose(
            f"Included feature {most_important_feature} with significance {highest_score}. New CV score: {new_cv_score}.",
            verbose)
        X_base[most_important_feature] = candidates[most_important_feature]
        candidates = candidates.drop(columns=most_important_feature)

    if verbose:
        print(f"CV score end: {cross_val_score(model, X_base, y, **cvs_kwargs).mean()}.")
        print(f"CV score all features: {cross_val_score(model, X, y, **cvs_kwargs).mean()}.")
    return list(X_base.columns)


def predictor_importance_incl(X: pd.DataFrame,
                              candidates: pd.DataFrame,
                              y: np.ndarray,
                              model: RegressorMixin,
                              cvs_kwargs: Optional[Dict[str, Any]] = None,
                              verbose: bool = True) -> pd.DataFrame:
    """Performs a recursive inclusion analysis to determine important features

    Iterates over all features in the candidates dataset and determines the scores (R^2-values)
    of the model fitted on the data in X plus the feature. Features will then be ranked
    according to the t-statistic that including this feature implies a model improvement compared
    to the model on the original dataset. Scores are computed using k-fold cross-validation.

    Args:
        X: Data frame containing independent features.
        candidates: Data frame containing independent feature candidates that may
            be included in the model
        y: Array containing the dependent feature
        model: Model to use for the analysis. Note that the results may heavily depend
            on the model chosen.
        cvs_kwargs: Keyword arguments passed down to cross_val_score method
        verbose: print progress bar

    Returns:
        Data frame containing features and score quotients for both train and test set
    """
    results = {
        "feature": [],
        "p_val": [],
        "bonf_p": [],
        "t_stat": [],
        "new_cv_score": []
    }

    if cvs_kwargs is None:
        cvs_kwargs = dict()

    if "cv" not in cvs_kwargs.keys():
        cvs_kwargs["cv"] = kfold
    cv = cvs_kwargs["cv"]

    first_split = list(cv.split(y, y))[0]

    n_train = len(first_split[0])
    n_test = len(first_split[1])

    if len(X.columns) > 0:
        base_scores = cross_val_score(model, X, y, **cvs_kwargs)
    else:
        base_scores = cross_val_score(DummyRegressor(), y, y, **cvs_kwargs)

    iter_set = tqdm(candidates.columns) if verbose else candidates.columns

    for feature_name in iter_set:
        X_plus_feature = X.copy()
        X_plus_feature[feature_name] = candidates[feature_name]
        scores_with_feature = cross_val_score(model, X_plus_feature, y, **cvs_kwargs)

        differences = scores_with_feature - base_scores

        t_stat, p_val = _compute_corrected_ttest(
            differences,
            differences.shape[0] - 1,
            n_train, n_test)

        results["feature"].append(feature_name)
        results["p_val"].append(p_val)
        results["bonf_p"].append(p_val * len(candidates.columns))
        results["t_stat"].append(t_stat)
        results["new_cv_score"].append(scores_with_feature.mean())
    return pd.DataFrame(data=results).sort_values(by="t_stat", ascending=False)


def elasticnet_plot(X, y, l1_ratio=.5, log_alpha_min=-4, log_alpha_max=1, alpha_step=50, ax=None,
                    **kwargs):
    """Plots ElasticNet regression coefficients against Lagrange multiplier


    Applies ElasticNet regression with different Lagrange multipliers alpha
    and plots the resulting coefficients against them yielding information on
    importance and influence of the variables.

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
        regressor = ElasticNet(fit_intercept=False, alpha=10 ** alpha, l1_ratio=l1_ratio, **kwargs)
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
