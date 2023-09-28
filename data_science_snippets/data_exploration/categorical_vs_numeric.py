import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Any, Literal

from pandas.core.dtypes.common import is_numeric_dtype
from statsmodels.api import stats
from statsmodels.formula.api import ols
import numpy as np
import pandas as pd
import scipy.stats as sp
import seaborn as sns
import matplotlib.pyplot as plt

__all__ = ['anova', 'anova_for_all', 'kruskal', 'kruskal_for_all', 'kruskal_one_vs_all',
           'strip_and_boxplot', 'kruskal_significant_difference', 'KSDResult']


def anova(dataset: pd.DataFrame, test_col: str, target_col: str) -> np.float:
    """Performs a one-way ANOVA F-test for groups in test_col with values in target_col.

    Note that ANOVA tests reqire independently, normally distributed samples with
    homoscedastic groups. If those assumptions are not met, consider using the
    (less powerful) Kruskal-Wallis H-test.

    Args:
        dataset: dataset to check
        test_col: Categorical column containing classes to check
        target_col: numerical column to check categorical column against
    Returns:
        The p-value of the ANOVA F-statistic
    """
    lm = ols(f'{target_col} ~ C({test_col})', data=dataset).fit()
    result = stats.anova_lm(lm)
    return result.iloc[0, -1]


def anova_for_all(dataset: pd.DataFrame, target_col: str,
                  significance: float = 0.05) -> pd.DataFrame:
    """Performs a one-way ANOVA F-test for all categorical columns against target_col.

    Performs a one-way ANOVA F-test to all tuples of the form
    (categorical col, target_col) in order to test whether the medians in each
    of the classes are equal.

    Note that ANOVA tests require independently, normally distributed samples with
    homoscedastic groups. If those assumptions are not met, consider using the
    (less powerful) Kruskal-Wallis H-test.

    Args:
        dataset: dataset to check
        target_col: numerical column to check categorical column against
        significance: If set, only return values with p-value <= significance
    Returns:
        A dataframe consisting of column names and p-values
    """
    result_dict = {}
    col_names = dataset.select_dtypes(object).columns

    for col in col_names:
        try:
            pr_f = anova(dataset, col, target_col)
            if pr_f <= significance:
                result_dict[col] = pr_f
        except Exception as e:
            print(f'Error evaluating column {col}: {e}')

    df = pd.DataFrame(data=result_dict.items(), columns=['Column', 'p-val'])
    df['Bonf_p'] = df['p-val'] * len(df)
    return df.set_index('Column').sort_values(by='p-val')


def kruskal(dataset: pd.DataFrame, test_col: str, target_col: str,
            nan_policy: str = 'propagate') -> np.float:
    """Applies Kruskal-Wallis H-test to a single column


    Applies Kruskal-Wallis H-test to (test col, target_col) in order to
    test whether one class dominates another.

    Args:
        dataset: dataset to check
        test_col: Categorical column containing classes to check
        target_col: numerical column to check categorical column against
        nan_policy: One of {'handle', 'omit', 'propagate', 'raise'}.
            'handle' removes nan values in categorical columns and treats them
            as an own class, then passes 'omit' to scipy.stats.kruskal.
            All other will be passed to scipy.stats.kruskal
    Returns:
        The p-value of the Kruskal-Wallis H-statistic
    """

    column = dataset[test_col]
    if nan_policy == 'handle' and column.dtype.name != 'category':
        column = column.fillna('__n_o_n_e__')

    # From scipi.stats.kruskal:
    # Due to the assumption that H has a chi square distribution, the number of
    # samples in each group must not be too small. A typical rule is that each
    # sample must have at least 5 measurements.
    if column.nunique() == 1:
        warnings.warn(f'Ignoring column {test_col}: Only contains one class.')
        return np.nan
    if len(dataset) / 5 < column.nunique():
        warnings.warn(f'Ignoring column {test_col}: Too few (<5) samples in each class.')
        return np.nan

    samples = [dataset[column == value][target_col] for value in column.unique() if
               not pd.isna(value)]
    _nan_policy = nan_policy if nan_policy != 'handle' else 'omit'
    p_value = sp.kruskal(*samples, nan_policy=_nan_policy).pvalue
    if np.isnan(p_value):
        warnings.warn(
            f"Obtained nan for column {test_col}. This may happen if your input contained "
            f"nan values. In that case, consider setting nan_policy='handle'.")
    return p_value


def kruskal_for_all(dataset: pd.DataFrame,
                    target_col: str,
                    significance: float = 1,
                    nan_policy: str = 'propagate') -> pd.DataFrame:
    """Applies Kruskal-Wallis H-test to all columns

    Applies Kruskal-Wallis H-test to all tuples of the form
    (col, target_col) in order to test whether one class
    dominates another. If target_col is numeric, kruskal
    checks categorical columns and vice versa.

    Args:
        dataset: dataset to check
        target_col: numerical column to check categorical columns against or
            categorical column to check numerical columns against
        significance: If set, only return values with p-value <= significance
        nan_policy: One of {'handle', 'omit', 'propagate', 'raise'}.
            'handle' removes nan values in categorical columns and treats them
            as an own category, then passes 'omit' to scipy.stats.kruskal.
            All other will be passed to scipy.stats.kruskal
    Returns:
        A dataframe consisting of column names and p-values
    """

    result_dict = {}
    if num_vs_cat_mode := is_numeric_dtype(dataset[target_col]):
        col_names = dataset.select_dtypes([object, 'datetime', 'category']).columns
    else:
        col_names = dataset.select_dtypes(np.number).columns

    for col in col_names:
        try:
            if num_vs_cat_mode:
                pr_f = kruskal(dataset, col, target_col, nan_policy=nan_policy)
            else:
                pr_f = kruskal(dataset, target_col, col, nan_policy=nan_policy)
        except ValueError as e:
            warnings.warn(str(e))
            pr_f = 1
        if not significance or pr_f <= significance:
            result_dict[col] = [pr_f, dataset[col].nunique(dropna=(nan_policy != 'handle'))]

    result_col_name = f'p({target_col})'
    df = pd.DataFrame.from_dict(
        result_dict, orient='index', columns=[result_col_name, 'nunique']
    ).astype({result_col_name: float, 'nunique': int})
    df[f'Bonf_{result_col_name}'] = df[result_col_name] * len(df)
    return df.sort_values(by=result_col_name)


def kruskal_one_vs_all(dataset: pd.DataFrame,
                       cat_col: str,
                       target_col: str,
                       significance: float = 1,
                       nan_policy: str = "omit",
                       include_stats: bool = True) -> pd.DataFrame:
    """Applies Kruskal-Wallis H-test to all categories in a specified column

    Applies Kruskal-Wallis H-test to all tuples of the form
    (categorical col == x, categorical col != x) in order to test whether the
    specific category has a significantly different distribution
    of target_col

    Args:
        dataset: dataset to check
        cat_col: Categorical column including different classes for analysis
        target_col: numerical column to check categorical column against
        significance: If set, only return values with p-value <= significance
        nan_policy: passed to scipy.stats.kruskal
        include_stats: Whether to include sample mean and std in the result
    Returns:
        A dataframe consisting of classes and p-values
    """

    result_dict = dict()
    categories = dataset[cat_col].unique()
    num_cat = len(categories)

    for category in categories:
        in_cat = dataset[dataset[cat_col] == category][target_col]
        nin_cat = dataset[dataset[cat_col] != category][target_col]
        pr_f = sp.kruskal(in_cat, nin_cat, nan_policy=nan_policy).pvalue
        if not significance or pr_f <= significance:
            result_dict[category] = [pr_f, pr_f * num_cat, len(in_cat)]
            if include_stats:
                result_dict[category] += [in_cat.mean(), nin_cat.mean(), in_cat.std(),
                                          nin_cat.std()]

    columns = ['p', 'bonf(p)', 'n']
    if include_stats:
        columns += ['in_mean', 'nin_mean', 'in_std', 'nin_std']

    df = pd.DataFrame.from_dict(
        result_dict, orient='index', columns=columns
    )
    return df.sort_values(by='p')


@dataclass
class KSDResult:
    significant_difference: float
    p_value: float
    mean_diff: float
    n_iter_exhausted: bool


def kruskal_significant_difference(x, y, p=0.05, max_iter=1000, tol=1e-3, initial_step_size=0.05):
    """Given two observation vectors x and y, computes the ´maximal difference d such that
    the p-value of the Kruskal-Wallis H-test of x shifted by d and y is at least p."""
    mean_diff = y.mean() - x.mean()
    p_value = sp.kruskal(x, y).pvalue
    if p_value > p:
        return KSDResult(
            significant_difference=np.nan,
            p_value=np.nan,
            mean_diff=mean_diff,
            n_iter_exhausted=False
        )
    else:
        established_diff = 0
        established_p = p
        step_size = initial_step_size
        for _ in range(max_iter):
            x_shift = x + established_diff + step_size * mean_diff
            p_new = sp.kruskal(x_shift, y).pvalue
            if p - tol <= p_new <= p + tol:
                return KSDResult(
                    significant_difference=established_diff + step_size * mean_diff,
                    p_value=p_new,
                    mean_diff=mean_diff,
                    n_iter_exhausted=False
                )
            elif p_new >= p:
                step_size *= 0.5
            elif p_new <= p:
                established_diff = established_diff + step_size * mean_diff
                established_p = p_new
        return KSDResult(
            significant_difference=established_diff,
            p_value=established_p,
            mean_diff=mean_diff,
            n_iter_exhausted=True
        )


def _combined_boxplot(kind: Literal['stripplot', 'swarmplot'],
                      common_kwargs: dict,
                      boxplot_kwargs: dict,
                      pointplot_kwargs: dict,
                      figsize: Optional[Tuple[int, int]] = None):
    ax = common_kwargs.get('ax', None)
    if not ax:
        _, ax = plt.subplots(figsize=figsize)
        common_kwargs['ax'] = ax

    pointplot = getattr(sns, kind)
    pointplot(**common_kwargs, **pointplot_kwargs)
    sns.boxplot(**common_kwargs, **boxplot_kwargs, width=.5, color='white', fliersize=0)
    plt.xticks(rotation=45)


def strip_and_boxplot(data: pd.DataFrame,
                      x: str,
                      y: str,
                      hue: Optional[str] = None,
                      figsize: Tuple[int, int] = (12, 8),
                      alpha: float = 1,
                      ax: Any = None,
                      strip_kwargs: Optional[dict] = None,
                      box_kwargs: Optional[dict] = None) -> None:
    strip_kwargs, box_kwargs = strip_kwargs or dict(), box_kwargs or dict()
    common_kwargs = dict(data=data, x=x, y=y)
    if ax:
        common_kwargs['ax'] = ax
    pointplot_kwargs = dict(hue=hue, alpha=alpha, jitter=.15, **strip_kwargs)
    boxplot_kwargs = box_kwargs
    return _combined_boxplot("stripplot", common_kwargs, boxplot_kwargs, pointplot_kwargs,
                             figsize=figsize)


def swarm_and_boxplot(data: pd.DataFrame,
                      x: str,
                      y: str,
                      hue: Optional[str] = None,
                      figsize: Tuple[int, int] = (12, 8),
                      alpha: float = 1,
                      ax: Any = None,
                      swarm_kwargs: Optional[dict] = None,
                      box_kwargs: Optional[dict] = None) -> None:
    swarm_kwargs, box_kwargs = swarm_kwargs or dict(), box_kwargs or dict()
    common_kwargs = dict(data=data, x=x, y=y, ax=ax)
    if ax:
        common_kwargs['ax'] = ax
    pointplot_kwargs = dict(hue=hue, alpha=alpha, **swarm_kwargs)
    boxplot_kwargs = box_kwargs
    return _combined_boxplot("swarmplot", common_kwargs, boxplot_kwargs, pointplot_kwargs,
                             figsize=figsize)
