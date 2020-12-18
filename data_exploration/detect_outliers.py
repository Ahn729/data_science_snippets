import pandas as pd
from scipy.stats import norm

def std_check(data, significance=0.01, sigma_threshold=None):
    """Identifies outliers based on standard deviation from sample mean

    Computes deviations of samples from the population mean in terms of
    standard deviations and returns those exceeding sigma_threshold.
    If no explicit threshold is provided, it is computed based on a
    Bonferoni-corrected significance level (p-value).

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


def recursive_outlier_detection(data, max_iter=5, **kwargs):
    """Recursively identifies and removes outliers from a dataset

    Performs max_iter iterations of outlier detection and repeats recursive
    detection on dataset with outliers removed.

    Args:
        data: Dataset to check. Must be numeric features only.
        max_iter: Maximum number of iterations to perform.
        kwargs: Keyword arguments passed down to std_check
    Returns:
        DataFrame including max deviation from the mean, measured in stds and
        the column where the max deviation occured as well as the iteration
        in which the outlier was detected.
    """
    outliers = pd.DataFrame(columns=['iteration', 'column', 'stds'])
    for i in range(0, max_iter):
        new_outliers = std_check(data, **kwargs)
        if new_outliers.empty:
            break
        data = data.drop(new_outliers.index)
        new_outliers['iteration'] = i + 1
        outliers = outliers.append(new_outliers)
    return outliers
