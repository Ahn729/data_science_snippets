import warnings
from statsmodels.api import stats
from statsmodels.formula.api import ols
import numpy as np
import pandas as pd
import scipy.stats as sp
import seaborn as sns
import matplotlib.pyplot as plt

def anova(dataset, test_col, target_col):
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


def anova_for_all(dataset, target_col, significance=0.05):
    """Performs a one-way ANOVA F-test for for all categorical columns against target_col.

    Performs a one-way ANOVA F-test to all tuples of the form
    (categorical col, target_col) in order to test whether the medians in each
    of the classes are equal.

    Note that ANOVA tests reqire independently, normally distributed samples with
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
            if (pr_f) <= significance:
                result_dict[col] = pr_f
        except:
            print(f'Error evaluating column {col}')
    df = pd.DataFrame(data=result_dict.items(), columns=['Column', 'p-val'])
    return df.set_index('Column').sort_values(by='p-val')

def kruskal(dataset, test_col, target_col, nan_policy='propagate'):
    """Applies Kruskal-Wallis H-test to a single column


    Applies Kruskal-Wallis H-test to (test col, target_col) in order to
    test whether the medians in each of the classes in test_col are equal.

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
    samples = [dataset[column == value][target_col] for value in column.unique() if not pd.isna(value)]
    _nan_policy = nan_policy if nan_policy != 'handle' else 'omit'
    return sp.kruskal(*samples, nan_policy=_nan_policy).pvalue


def kruskal_for_all(dataset, target_col, significance=1, nan_policy='propagate'):
    """Applies Kruskal-Wallis H-test to all categorical columns

    Applies Kruskal-Wallis H-test to all tuples of the form
    (categorical col, target_col) in order to test whether the medians in each
    of the classes are equal.

    Args:
        dataset: dataset to check
        target_col: numerical column to check categorical columns against
        significance: If set, only return values with p-value <= significance
        nan_policy: One of {'handle', 'omit', 'propagate', 'raise'}.
            'handle' removes nan values in categorical columns and treats them
            as an own category, then passes 'omit' to scipy.stats.kruskal.
            All other will be passed to scipy.stats.kruskal
    Returns:
        A dataframe consisting of column names and p-values
    """

    result_dict = {}
    col_names = dataset.select_dtypes([object, 'datetime', 'category']).columns
    for col in col_names:
        pr_f = kruskal(dataset, col, target_col, nan_policy=nan_policy)
        if significance or pr_f <= significance:
            result_dict[col] = [pr_f, dataset[col].nunique(dropna=(nan_policy != 'handle'))]
    df = pd.DataFrame.from_dict(
        result_dict, orient='index', columns=[f'p({target_col})', 'nunique']
        )
    return df.sort_values(by=f'p({target_col})').astype({f'p({target_col})': float, 'nunique': int})

def strip_and_boxplot(data, x, y, hue=None, figsize=(12, 8), alpha=1, ax=None, strip_kwargs=None, box_kwargs=None):
    strip_kwargs, box_kwargs = strip_kwargs or dict(), box_kwargs or dict()
    if not ax:
        fig, ax = plt.subplots(figsize=figsize)
    sns.stripplot(data=data, x=x, y=y, hue=hue, alpha=alpha, ax=ax, jitter=.15, **strip_kwargs)
    sns.boxplot(data=data, x=x, y=y, color='white', ax=ax, width=.5, fliersize=0, **box_kwargs)
    plt.xticks(rotation=45)
