from math import sqrt
from typing import Optional, Tuple

import pandas as pd
from scipy.stats import norm, t

__all__ = ['std_test', 'grubbs_test', 'recursive_outlier_detection']


def std_test(data: pd.DataFrame, significance: float = 0.01, sigma_threshold: Optional[float] = None) -> pd.DataFrame:
    """Identifies outliers based on standard deviation from sample mean

    Computes deviations of samples from the population mean in terms of
    standard deviations and returns those exceeding sigma_threshold.
    If no explicit threshold is provided, it is computed based on a
    Bonferoni-corrected significance level (p-value) assuming a normal
    distribution.

    Args:
        data: Dataset to check. Must be numeric features only.
        significance: significance (p-value) to identify an outlier
        sigma_threshold: If provided, overrides significance parameter
            to return all samples with deviation > sigma_threshold stds
    Returns:
        DataFrame including max deviation from the mean, measured in stds and
        the column where the max deviation occured.
    """
    sigma_threshold = sigma_threshold or norm.ppf(1 - significance / (data.count().sum() * 2))
    mean = data.mean()
    std = data.std()
    dists = (data.sub(mean).div(std)).abs()
    max_dists = dists.max(axis=1)
    outler_mask = max_dists > sigma_threshold
    outlier_dists = dists[outler_mask]
    outlier_cols = outlier_dists.idxmax(axis=1)
    return pd.DataFrame(data={'column': outlier_cols, 'stds': max_dists[outler_mask]})


def _grubbs_test(series: pd.Series, significance: float) -> Optional[Tuple[int, float]]:
    """Identifies outliers in a single series using Grubbs' test

    https://en.wikipedia.org/wiki/Grubbs%27s_test

    Args:
        series: Series to check. Must be numeric, but nans are okay
        significance: significance (p-value) to identify an outlier
    Returns:
        If an outlier was found, returns a tuple of the corresponding index
        along with the value of Grubbs' test statistic G, else returns None

    """
    if not (std := series.std()):
        return None
    n = series.notna().sum()
    G = max(abs(series - series.mean())) / std
    t_crit = t.ppf(1 - significance / (2 * n), n - 2)
    threshold = (n - 1) / sqrt(n) * sqrt(t_crit**2 / (n - 2 + t_crit**2))
    if G > threshold:
        return abs(series - series.mean()).idxmax(), G
    return None


def grubbs_test(data: pd.DataFrame, significance: float = 0.01) -> pd.DataFrame:
    """Identifies outliers in a dataset based on Grubbs' test

    https://en.wikipedia.org/wiki/Grubbs%27s_test

    For each column, computes Grubbs' test statistic (G) and performs a t-test
    to determine whether there are outliers in the dataset. Grubbs' test
    finds one outlier at a time, so more iterations are needed. Note that
    normality is required for precise results.

    Args:
        data: Dataset to check. Must be numeric features only.
        significance: significance (p-value) to identify an outlier
    Returns:
        DataFrame including value of Grubbs' test statistic ('G') and
        the column which marked the observation as an outlier.
    """
    outliers = dict()
    for col in data.columns:
        idx, G = _grubbs_test(data[col], significance) or (None, None)
        if G:
            outliers[idx] = (col, G)
    return pd.DataFrame.from_dict(outliers, orient='index', columns=['column', 'G'])


METHODS = {
    'std': std_test,
    'grubbs': grubbs_test
}


def recursive_outlier_detection(data: pd.DataFrame, max_iter: int = None, method: str = 'std', **kwargs) -> pd.DataFrame:
    """Recursively identifies and removes outliers from a dataset

    Performs max_iter iterations of outlier detection and repeats recursive
    detection on dataset with outliers removed.

    Args:
        data: Dataset to check. Must be numeric features only.
        max_iter: Maximum number of iterations to perform. If none is provided,
            will use 5 iterations for std, 100 iterations for Grubbs
        method: Method to determine outliers used in each iteration. Must be
            one of {'std', 'grubbs'}. Default: 'std'
        kwargs: Keyword arguments passed down to detection method
    Returns:
        DataFrame including either max deviation from the mean, measured in stds
        (method=="std") or value ob Grubbs' test statistic G (method=="grubbs") and
        the column where the max deviation occured as well as the iteration
        in which the outlier was detected.
    """

    try:
        detection_method = METHODS[method]
    except KeyError:
        raise ValueError(f"Method {method} not understood (must be one of {', '.join(METHODS.keys())})")

    max_iter = max_iter or 5 if detection_method == 'std' else 100

    outliers = pd.DataFrame(columns=['iteration', 'column'])
    for i in range(0, max_iter):
        new_outliers = detection_method(data, **kwargs)
        if new_outliers.empty:
            break
        data = data.drop(new_outliers.index)
        new_outliers['iteration'] = i + 1
        outliers = outliers.append(new_outliers)
    return outliers
